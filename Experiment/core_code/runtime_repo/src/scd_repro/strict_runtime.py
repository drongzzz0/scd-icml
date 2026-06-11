from __future__ import annotations

import gc
import math
import inspect
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from transformers.cache_utils import DynamicCache
from transformers.models.mistral.modeling_mistral import create_causal_mask, create_sliding_window_causal_mask

from .cmrc import CMRCSample, extract_answer_text, score_cmrc_prediction
from .direct_kv_residual_patch import PATCH_FIT_MODE_DIRECT_KV_KNORM_RESIDUAL_MLP_V1, PATCH_FIT_MODE_XNORM_NATIVEKV_RESIDUAL_MLP_V1, PATCH_FIT_MODE_XNORM_NATIVEKV_SHARED_RESIDUAL_MLP_V1, infer_direct_kv_residual_patch_layer_torch, infer_xnorm_nativekv_residual_patch_layer_torch, infer_xnorm_nativekv_shared_residual_patch_torch, load_direct_kv_residual_patch_layer_from_ckpt, load_xnorm_nativekv_residual_patch_layer_from_ckpt, load_xnorm_nativekv_shared_residual_patch_from_ckpt
from .io_utils import load_bundle
from .patch import (
    PATCH_FIT_MODE_DIRECT_KV_V1,
    PATCH_FIT_MODE_DIRECT_KV_KNORM_V1,
    PATCH_FIT_MODE_XNORM_SEMANTIC_V1,
    infer_direct_kv_patch_layer_torch,
    infer_patch_layer_torch,
    load_semantic_patch_layer_from_ckpt,
)
from .reuse import decode_group, encode_group, group_layers_for_gid, move_reuse_weights, pack_group, unpack_group
from .rope import de_rope_k_torch, re_rope_k_torch, remap_rope_k_torch, rope_spec_from_hf_config
from .torch_utils import clear_torch_cache, require_torch, resolve_device, resolve_torch_dtype, tensor_to_cpu


PAPER_TABLE2_ANCHOR = {
    "oracle": {"payload_mb": 0.00, "ttft_ms": 331.9, "speedup_vs_oracle": 1.00, "f1": 0.8105},
    "raw_kv": {"payload_mb": 115.02, "ttft_ms": 145.5, "speedup_vs_oracle": 2.28, "f1": 0.2725},
    "quantized_4bit": {"payload_mb": 28.76, "ttft_ms": 59.2, "speedup_vs_oracle": 5.60, "f1": 0.1289},
    "selective_recompute": {"payload_mb": 127.34, "ttft_ms": 198.7, "speedup_vs_oracle": 1.67, "f1": 0.7052},
    "reuse_only": {"payload_mb": 69.02, "ttft_ms": 118.2, "speedup_vs_oracle": 2.81, "f1": 0.6421},
    "scd_full": {"payload_mb": 87.31, "ttft_ms": 125.2, "speedup_vs_oracle": 2.65, "f1": 0.7850},
}

_METHOD_LABELS = {
    "full_consumer": "Full Consumer",
    "oracle": "Oracle",
    "oracle_xnorm_suffix": "Oracle XNorm Suffix",
    "oracle_state_suffix": "Oracle State Suffix",
    "oracle_kv_patchonly": "Oracle KV Patch-only",
    "patch_only": "Patch-only",
    "raw_kv": "Raw KV (BF16)",
    "quantized_4bit": "Quantized (4-bit)",
    "selective_recompute": "Selective Recompute",
    "reuse_only": "Reuse-only",
    "reuse_fallback": "Reuse + Fallback",
    "reuse_patch": "Reuse + Patch",
    "reuse_patch_recompute": "Reuse + Patch + Short Recompute",
    "scd_full": "SCD",
}

STEP_C_METHODS = [
    "full_consumer",
    "reuse_only",
    "reuse_fallback",
    "patch_only",
    "reuse_patch",
    "reuse_patch_recompute",
]


def _resolve_hidden_states(args, kwargs):
    if args:
        return args[0]
    if kwargs:
        hidden_states = kwargs.get("hidden_states")
        if hidden_states is not None:
            return hidden_states
    return None


@dataclass
class PrefillCapture:
    input_ids: Any
    attention_mask: Any
    logits_last: Any
    past_key_values: list[tuple[Any, Any]]
    xnorm_by_layer: dict[int, Any]
    prefill_ms: float
    layer_ms: list[float]
    vocab_size: int
    residual_by_layer: dict[int, Any] = field(default_factory=dict)


@dataclass
class MethodTiming:
    producer_prefill_ms: float = 0.0
    consumer_prefill_ms: float = 0.0
    encode_ms: float = 0.0
    transfer_ms: float = 0.0
    staging_copy_ms: float = 0.0
    rebuild_ms: float = 0.0
    rebuild_prep_ms: float = 0.0
    rebuild_decode_core_ms: float = 0.0
    rebuild_rerope_ms: float = 0.0
    rebuild_materialize_ms: float = 0.0
    patch_infer_ms: float = 0.0
    patch_shard_load_ms: float = 0.0
    patch_shard_gpu_copy_ms: float = 0.0
    patch_identity_check_ms: float = 0.0
    patch_residual_forward_ms: float = 0.0
    patch_kproj_vproj_ms: float = 0.0
    patch_knorm_ms: float = 0.0
    patch_rotary_ms: float = 0.0
    recompute_ms: float = 0.0
    decode_first_ms: float = 0.0
    decode_rest_ms: float = 0.0
    decode_total_ms: float = 0.0
    ttft_ms: float = 0.0
    total_ms: float = 0.0
    bytes_transferred: int = 0
    raw_cache_bytes: int = 0
    compression_ratio: float | None = None


@dataclass
class MethodSampleResult:
    qid: str
    method: str
    pred_text: str
    f1: float
    em: float
    payload_mb: float
    timing: MethodTiming
    gold_answers: list[str]
    prompt_tokens: int
    prefix_tokens: int
    first_token_id: int | None
    avg_logp: float = 0.0
    avg_nll: float = 0.0
    teacher_logp_sum: float = 0.0
    teacher_token_count: int = 0


@dataclass
class ReuseDecodeTiming:
    encode_ms: float = 0.0
    staging_copy_ms: float = 0.0
    rebuild_ms: float = 0.0
    rebuild_prep_ms: float = 0.0
    rebuild_decode_core_ms: float = 0.0
    rebuild_rerope_ms: float = 0.0
    rebuild_materialize_ms: float = 0.0


class TransformersRuntimeSession:
    def __init__(self, model_path: str, device: str, dtype: str, cpu_fallback: bool = True):
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as exc:
            raise RuntimeError("缺少运行依赖: transformers") from exc

        self.torch = require_torch()
        self.device = str(resolve_device(device, cpu_fallback=cpu_fallback))
        self.dtype = dtype
        self.torch_dtype = resolve_torch_dtype(dtype)
        self.cpu_fallback = cpu_fallback
        self.model_path = model_path

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        load_kwargs = {
            "torch_dtype": self.torch_dtype,
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }
        self.model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
        if self.device != "cpu":
            self.model.to(self.device)
        self.model.eval()
        self.layers = _get_decoder_layers(self.model)
        self.target_device = resolve_device(self.device, cpu_fallback=cpu_fallback)
        self.rope_spec = rope_spec_from_hf_config(self.model.config)

    def _synchronize(self) -> None:
        if str(self.target_device).startswith("cuda"):
            self.torch.cuda.synchronize(self.target_device)

    def tokenize(self, prompt: str, max_length: int | None = None) -> tuple[Any, Any]:
        kwargs = {"return_tensors": "pt", "padding": False}
        if max_length is not None and int(max_length) > 0:
            kwargs["truncation"] = True
            kwargs["max_length"] = int(max_length)
        else:
            kwargs["truncation"] = False
        encoded = self.tokenizer(prompt, **kwargs)
        input_ids = encoded["input_ids"].to(self.target_device)
        attention_mask = encoded.get("attention_mask")
        if attention_mask is None:
            attention_mask = self.torch.ones_like(input_ids, device=self.target_device)
        else:
            attention_mask = attention_mask.to(self.target_device)
        return input_ids, attention_mask

    def prefill(
        self,
        prompt: str,
        max_length: int | None = None,
        capture_xnorm: bool = False,
        capture_residual: bool = False,
        profile_layers: bool = False,
    ) -> PrefillCapture:
        input_ids, attention_mask = self.tokenize(prompt, max_length=max_length)
        xnorm_by_layer: dict[int, Any] = {}
        residual_by_layer: dict[int, Any] = {}
        layer_ms = [0.0 for _ in self.layers]
        starts = [0.0 for _ in self.layers]
        hooks = []

        if capture_xnorm:
            def make_xnorm_hook(layer_idx: int):
                def _hook(_module, args, kwargs):
                    hidden_states = _resolve_hidden_states(args, kwargs)
                    if hidden_states is not None:
                        xnorm_by_layer[layer_idx] = hidden_states.detach().to(dtype=self.torch.float32)
                return _hook

            for idx, layer in enumerate(self.layers):
                attn = getattr(layer, "self_attn", None) or getattr(layer, "attn", None)
                if attn is not None:
                    hooks.append(attn.register_forward_pre_hook(make_xnorm_hook(idx), with_kwargs=True))

        if capture_residual:
            def make_residual_hook(layer_idx: int):
                def _hook(_module, args):
                    hidden_states = args[0] if args else None
                    if hidden_states is not None:
                        residual_by_layer[layer_idx] = hidden_states.detach().to(dtype=self.torch.float32)
                return _hook

            for idx, layer in enumerate(self.layers):
                hooks.append(layer.register_forward_pre_hook(make_residual_hook(idx)))

        if profile_layers:
            def make_pre_hook(layer_idx: int):
                def _hook(_module, _args):
                    self._synchronize()
                    starts[layer_idx] = time.perf_counter()
                return _hook

            def make_post_hook(layer_idx: int):
                def _hook(_module, _args, _out):
                    self._synchronize()
                    layer_ms[layer_idx] += (time.perf_counter() - starts[layer_idx]) * 1000.0
                return _hook

            for idx, layer in enumerate(self.layers):
                hooks.append(layer.register_forward_pre_hook(make_pre_hook(idx)))
                hooks.append(layer.register_forward_hook(make_post_hook(idx)))

        try:
            self._synchronize()
            start = time.perf_counter()
            with self.torch.no_grad():
                out = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=True,
                    output_hidden_states=False,
                    return_dict=True,
                )
            self._synchronize()
            prefill_ms = (time.perf_counter() - start) * 1000.0
        finally:
            for hook in hooks:
                hook.remove()

        past = cache_to_layer_tuples(out.past_key_values)
        logits_last = out.logits[:, -1, :].detach()
        return PrefillCapture(
            input_ids=input_ids.detach(),
            attention_mask=attention_mask.detach(),
            logits_last=logits_last,
            past_key_values=past,
            xnorm_by_layer=xnorm_by_layer,
            prefill_ms=float(prefill_ms),
            layer_ms=[float(x) for x in layer_ms],
            vocab_size=int(logits_last.shape[-1]),
            residual_by_layer=residual_by_layer,
        )

    def generate_from_prefill(
        self,
        capture: PrefillCapture,
        max_new_tokens: int = 32,
        align_first_decode_to_cache: bool = False,
    ) -> tuple[str, float, float, int | None]:
        if align_first_decode_to_cache:
            prefix_tokens = int(capture.input_ids.shape[1]) - 1
            if prefix_tokens <= 0:
                raise ValueError("从 full prefill 对齐首 token 计时时，prompt 至少需要 2 个 token")
            prefix_cache_layers = [
                (
                    _slice_prefix_tensor(k, prefix_tokens).contiguous(),
                    _slice_prefix_tensor(v, prefix_tokens).contiguous(),
                )
                for k, v in capture.past_key_values
            ]
            return self.generate_from_cache(
                cache_layers=prefix_cache_layers,
                prompt_input_ids=capture.input_ids,
                prompt_attention_mask=capture.attention_mask,
                prefix_tokens=prefix_tokens,
                max_new_tokens=max_new_tokens,
            )

        eos_token_id = self.tokenizer.eos_token_id
        generated: list[int] = []
        attention_mask = capture.attention_mask.clone()
        past = build_dynamic_cache(self.model, capture.past_key_values)

        decode_start = time.perf_counter()
        first_decode_ms = 0.0
        first_token_id: int | None = None
        logits = capture.logits_last
        for step in range(max_new_tokens):
            self._synchronize()
            step_start = time.perf_counter()
            token = int(self.torch.argmax(logits, dim=-1).item())
            self._synchronize()
            elapsed = (time.perf_counter() - step_start) * 1000.0
            if step == 0:
                first_decode_ms = elapsed
                first_token_id = token
            if eos_token_id is not None and token == int(eos_token_id):
                break
            generated.append(token)
            text = extract_answer_text(self.tokenizer.decode(generated, skip_special_tokens=True))
            if text and "\n" in text:
                break
            attention_mask = self.torch.cat([attention_mask, self.torch.ones((1, 1), device=attention_mask.device, dtype=attention_mask.dtype)], dim=1)
            with self.torch.no_grad():
                out = self.model(
                    input_ids=self.torch.tensor([[token]], device=self.target_device, dtype=capture.input_ids.dtype),
                    attention_mask=attention_mask,
                    past_key_values=past,
                    use_cache=True,
                    return_dict=True,
                )
            past = out.past_key_values
            logits = out.logits[:, -1, :].detach()

        self._synchronize()
        decode_total_ms = (time.perf_counter() - decode_start) * 1000.0
        text = extract_answer_text(self.tokenizer.decode(generated, skip_special_tokens=True))
        return text, float(first_decode_ms), float(decode_total_ms), first_token_id

    def generate_from_cache(
        self,
        cache_layers: list[tuple[Any, Any]],
        prompt_input_ids: Any,
        prompt_attention_mask: Any,
        prefix_tokens: int,
        max_new_tokens: int = 32,
    ) -> tuple[str, float, float, int | None]:
        total_prompt_tokens = int(prompt_input_ids.shape[1])
        if total_prompt_tokens < 2:
            raise ValueError("从 prefix cache 恢复生成时，prompt 至少需要 2 个 token")
        if prefix_tokens <= 0 or prefix_tokens >= total_prompt_tokens:
            raise ValueError(f"非法的 prefix_tokens: {prefix_tokens}, prompt_tokens={total_prompt_tokens}")
        if not cache_layers:
            raise ValueError("cache_layers 不能为空")

        for layer_idx, (k, v) in enumerate(cache_layers):
            k_tokens = int(k.shape[-2])
            v_tokens = int(v.shape[-2])
            if k_tokens != prefix_tokens or v_tokens != prefix_tokens:
                raise ValueError(
                    f"第 {layer_idx} 层 cache 长度与 prefix_tokens 不一致: "
                    f"k_tokens={k_tokens}, v_tokens={v_tokens}, prefix_tokens={prefix_tokens}"
                )

        if int(prompt_attention_mask.shape[1]) < prefix_tokens + 1:
            raise ValueError(
                f"attention_mask 长度不足: mask_tokens={int(prompt_attention_mask.shape[1])}, 需要至少 {prefix_tokens + 1}"
            )

        past = build_dynamic_cache(self.model, [(k.contiguous(), v.contiguous()) for k, v in cache_layers])
        last_token = prompt_input_ids[:, prefix_tokens : prefix_tokens + 1].to(self.target_device)
        attention_mask = prompt_attention_mask[:, : prefix_tokens + 1].to(self.target_device)
        eos_token_id = self.tokenizer.eos_token_id

        self._synchronize()
        decode_start = time.perf_counter()
        with self.torch.no_grad():
            out = self.model(
                input_ids=last_token,
                attention_mask=attention_mask,
                past_key_values=past,
                use_cache=True,
                return_dict=True,
            )
        self._synchronize()
        first_decode_ms = (time.perf_counter() - decode_start) * 1000.0

        generated: list[int] = []
        logits = out.logits[:, -1, :].detach()
        past = out.past_key_values
        token = int(self.torch.argmax(logits, dim=-1).item())
        first_token_id: int | None = token
        if eos_token_id is None or token != int(eos_token_id):
            generated.append(token)
        else:
            text = extract_answer_text(self.tokenizer.decode(generated, skip_special_tokens=True))
            return text, float(first_decode_ms), float(first_decode_ms), first_token_id

        attention_mask = self.torch.cat([attention_mask, self.torch.ones((1, 1), device=attention_mask.device, dtype=attention_mask.dtype)], dim=1)
        for _ in range(1, max_new_tokens):
            text = extract_answer_text(self.tokenizer.decode(generated, skip_special_tokens=True))
            if text and "\n" in text:
                break
            with self.torch.no_grad():
                out = self.model(
                    input_ids=self.torch.tensor([[generated[-1]]], device=self.target_device, dtype=prompt_input_ids.dtype),
                    attention_mask=attention_mask,
                    past_key_values=past,
                    use_cache=True,
                    return_dict=True,
                )
            past = out.past_key_values
            logits = out.logits[:, -1, :].detach()
            token = int(self.torch.argmax(logits, dim=-1).item())
            if eos_token_id is None or token != int(eos_token_id):
                generated.append(token)
            else:
                break
            attention_mask = self.torch.cat([attention_mask, self.torch.ones((1, 1), device=attention_mask.device, dtype=attention_mask.dtype)], dim=1)

        self._synchronize()
        decode_total_ms = (time.perf_counter() - decode_start) * 1000.0
        text = extract_answer_text(self.tokenizer.decode(generated, skip_special_tokens=True))
        return text, float(first_decode_ms), float(decode_total_ms), first_token_id

    def close(self) -> None:
        graph_cache = getattr(self, "__runtime_consumer_native_proj_graph_cache__", None)
        if isinstance(graph_cache, dict):
            for entry in list(graph_cache.values()):
                _release_consumer_native_proj_graph_entry(entry if isinstance(entry, dict) else None)
            graph_cache.clear()
        stack_cache = getattr(self, "__runtime_consumer_native_proj_stack__", None)
        if isinstance(stack_cache, dict):
            stack_cache.clear()
        del self.model
        clear_torch_cache()


def _get_decoder_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise RuntimeError("无法从模型中定位 decoder layers")


def cache_to_layer_tuples(past_key_values) -> list[tuple[Any, Any]]:
    if hasattr(past_key_values, "layers"):
        return [(layer.keys.detach().clone(), layer.values.detach().clone()) for layer in past_key_values.layers]
    if isinstance(past_key_values, (tuple, list)):
        return [(row[0].detach().clone(), row[1].detach().clone()) for row in past_key_values]
    raise TypeError(f"不支持的 cache 类型: {type(past_key_values)!r}")


def build_dynamic_cache(model, cache_layers: list[tuple[Any, Any]]):
    from transformers.cache_utils import DynamicCache

    return DynamicCache(ddp_cache_data=[(k.contiguous(), v.contiguous()) for k, v in cache_layers], config=model.config)


def assert_same_tokenization(a_capture: PrefillCapture, b_capture: PrefillCapture) -> None:
    t = require_torch()
    if a_capture.input_ids.shape != b_capture.input_ids.shape or not t.equal(a_capture.input_ids.cpu(), b_capture.input_ids.cpu()):
        raise ValueError("Producer 和 Consumer 的 tokenizer 输出不一致，无法进行严格 cache 对齐评测")


def _synchronize_for_device(device: str | Any) -> None:
    t = require_torch()
    if str(device).startswith("cuda"):
        t.cuda.synchronize(device)


def _tensor_bytes(x: Any) -> int:
    t = require_torch()
    if t.is_tensor(x):
        return int(x.numel()) * int(x.element_size())
    return int(x.numel()) * int(x.dtype.itemsize)


def _move_layers_to_device(cache_layers: list[tuple[Any, Any]], device: str, dtype: Any) -> list[tuple[Any, Any]]:
    out = []
    for k, v in cache_layers:
        out.append((k.to(device=device, dtype=dtype).contiguous(), v.to(device=device, dtype=dtype).contiguous()))
    return out


def _quantize_dequantize_int4(x: Any, group_size: int, scale_dtype: Any) -> tuple[Any, int]:
    t = require_torch()
    shape = tuple(x.shape)
    flat = x.reshape(-1).to(dtype=t.float32)
    if flat.numel() == 0:
        return x.to(dtype=t.float32), 0
    pad = (group_size - (flat.numel() % group_size)) % group_size
    if pad:
        flat = t.cat([flat, t.zeros((pad,), device=flat.device, dtype=flat.dtype)], dim=0)
    groups = flat.reshape(-1, group_size)
    scales = t.amax(t.abs(groups), dim=1, keepdim=True) / 7.0
    scales = t.clamp(scales, min=1e-8)
    q = t.round(groups / scales).clamp(-8, 7)
    deq = (q * scales).reshape(-1)[: math.prod(shape)].reshape(shape).to(dtype=t.float32)
    packed_bytes = (int(math.prod(shape)) + 1) // 2
    scale_bytes = int(scales.numel()) * _dtype_size_bytes(scale_dtype)
    return deq, int(packed_bytes + scale_bytes)


def _dtype_size_bytes(dtype: Any) -> int:
    t = require_torch()
    return int(t.tensor([], dtype=dtype).element_size())


def _slice_prefix_tensor(x: Any, prefix_tokens: int) -> Any:
    if x.ndim == 4:
        return x[..., :prefix_tokens, :]
    if x.ndim == 3:
        return x[:, :prefix_tokens, :]
    raise ValueError(f"不支持的 prefix 张量维度: {tuple(x.shape)}")


def _build_layer_forward_inputs(
    consumer_session: TransformersRuntimeSession,
    hidden_states: Any,
    attention_mask: Any,
) -> dict[str, Any]:
    t = require_torch()
    model_dtype = getattr(consumer_session.model, "dtype", t.float32)
    hidden_states = hidden_states.to(device=consumer_session.target_device, dtype=model_dtype)
    attention_mask = attention_mask.to(device=consumer_session.target_device)
    seq_len = int(hidden_states.shape[1])
    cache_position = t.arange(seq_len, device=hidden_states.device)
    position_ids = cache_position.unsqueeze(0)
    mask_fn = create_sliding_window_causal_mask if getattr(consumer_session.model.config, "sliding_window", None) is not None else create_causal_mask
    mask_kwargs = {
        "config": consumer_session.model.config,
        "attention_mask": attention_mask,
        "cache_position": cache_position,
        "past_key_values": None,
        "position_ids": position_ids,
    }
    sig = inspect.signature(mask_fn)
    if "inputs_embeds" in sig.parameters:
        causal_mask = mask_fn(inputs_embeds=hidden_states, **mask_kwargs)
    elif "input_embeds" in sig.parameters:
        causal_mask = mask_fn(input_embeds=hidden_states, **mask_kwargs)
    else:
        raise TypeError(f"{mask_fn.__name__} does not accept inputs_embeds/input_embeds")
    position_embeddings = consumer_session.model.model.rotary_emb(hidden_states, position_ids=position_ids)
    return {
        "attention_mask": causal_mask,
        "position_ids": position_ids,
        "cache_position": cache_position,
        "position_embeddings": position_embeddings,
    }


def _run_consumer_layer_native(
    *,
    consumer_session: TransformersRuntimeSession,
    layer_idx: int,
    hidden_states: Any,
    attention_mask: Any,
    position_ids: Any,
    cache_position: Any,
    position_embeddings: Any,
) -> tuple[Any, Any, Any]:
    t = require_torch()
    model_dtype = getattr(consumer_session.model, "dtype", t.float32)
    layers = consumer_session.layers if hasattr(consumer_session, "layers") else _get_decoder_layers(consumer_session.model)
    layer_module = layers[int(layer_idx)]
    cache = DynamicCache(config=consumer_session.model.config)
    with t.no_grad():
        next_hidden = layer_module(
            hidden_states.to(device=consumer_session.target_device, dtype=model_dtype),
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=cache,
            use_cache=True,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
    layer_cache = cache.layers[int(layer_idx)]
    return next_hidden, layer_cache.keys.detach(), layer_cache.values.detach()


def _run_patch_breakpoint_native(
    *,
    consumer_session: TransformersRuntimeSession,
    layer_idx: int,
    hidden_states: Any,
    attention_mask: Any,
    position_ids: Any,
    cache_position: Any,
    position_embeddings: Any,
) -> tuple[Any, Any, Any]:
    return _run_consumer_layer_native(
        consumer_session=consumer_session,
        layer_idx=layer_idx,
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        cache_position=cache_position,
        position_embeddings=position_embeddings,
    )


def _run_consumer_suffix_native(
    *,
    consumer_session: TransformersRuntimeSession,
    start_layer: int,
    end_layer_exclusive: int,
    hidden_states: Any,
    attention_mask: Any,
    position_ids: Any,
    cache_position: Any,
    position_embeddings: Any,
) -> tuple[Any, dict[int, tuple[Any, Any]]]:
    current_hidden = hidden_states
    layer_caches: dict[int, tuple[Any, Any]] = {}
    for layer_idx in range(int(start_layer), int(end_layer_exclusive)):
        current_hidden, k, v = _run_consumer_layer_native(
            consumer_session=consumer_session,
            layer_idx=layer_idx,
            hidden_states=current_hidden,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
        layer_caches[int(layer_idx)] = (k, v)
    return current_hidden, layer_caches


def _as_int_list(values: Any) -> list[int]:
    if not isinstance(values, list):
        return []
    return sorted({int(x) for x in values})


def _as_int_map(values: Any) -> dict[int, int]:
    if not isinstance(values, dict):
        return {}
    out: dict[int, int] = {}
    for key, value in values.items():
        try:
            layer = int(key)
            seg_len = int(value)
        except Exception:
            continue
        if seg_len > 0:
            out[layer] = seg_len
    return out


def _covered_recompute_layers(num_layers: int, recompute_segment_length_by_layer: dict[int, int]) -> list[int]:
    covered: set[int] = set()
    for layer, seg_len in recompute_segment_length_by_layer.items():
        start = int(layer) + 1
        stop = min(int(num_layers), start + int(seg_len))
        covered.update(range(start, stop))
    return sorted(covered)


def _raw_cache_payload_bytes(capture: PrefillCapture, prefix_tokens: int, transfer_dtype: Any) -> int:
    total = 0
    for k, v in capture.past_key_values:
        k_prefix = _slice_prefix_tensor(k, prefix_tokens)
        v_prefix = _slice_prefix_tensor(v, prefix_tokens)
        total += int(k_prefix.numel()) * _dtype_size_bytes(transfer_dtype)
        total += int(v_prefix.numel()) * _dtype_size_bytes(transfer_dtype)
    return int(total)


def _compression_ratio(raw_cache_bytes: int, bytes_transferred: int) -> float | None:
    if int(bytes_transferred) <= 0:
        return None
    return float(raw_cache_bytes) / float(bytes_transferred)


def _normalize_runtime_route(
    route: dict[str, Any] | None,
    *,
    mode: str,
    num_layers: int,
    default_recompute_segment_length: int = 0,
    override_route_recompute_length: int | None = None,
    fail_on_missing_route_fields: bool = False,
    fail_on_missing_patch_fields: bool = False,
) -> dict[str, Any]:
    route = route or {}
    all_layers = list(range(int(num_layers)))
    route_reuse = _as_int_list(route.get("reuse_layers", []))
    route_patch = _as_int_list(route.get("selected_patch_layers", route.get("patch_layers", [])))
    fallback_explicit = "fallback_layers" in route
    route_fallback = _as_int_list(route.get("fallback_layers", []))
    recompute_map = _as_int_map(route.get("recompute_segment_length_by_layer", {}))
    selected_recompute = _as_int_list(route.get("selected_recompute_layers", []))

    if mode in {"reuse_fallback", "reuse_patch", "reuse_patch_recompute"} and fail_on_missing_route_fields and not route_reuse:
        raise ValueError(f"{mode} 缺少 route.reuse_layers")
    if mode in {"patch_only", "reuse_patch", "reuse_patch_recompute"} and fail_on_missing_patch_fields and not route_patch:
        raise ValueError(f"{mode} 缺少 route.patch_layers/selected_patch_layers")

    if override_route_recompute_length is not None and int(override_route_recompute_length) > 0:
        if recompute_map:
            recompute_map = {int(layer): int(override_route_recompute_length) for layer in recompute_map}
        elif selected_recompute:
            recompute_map = {int(layer): int(override_route_recompute_length) for layer in selected_recompute}
    elif not recompute_map and selected_recompute:
        if int(default_recompute_segment_length) <= 0:
            if fail_on_missing_route_fields and mode == "reuse_patch_recompute":
                raise ValueError("reuse_patch_recompute 缺少 recompute_segment_length_by_layer，且未提供默认长度")
        else:
            recompute_map = {int(layer): int(default_recompute_segment_length) for layer in selected_recompute}
    if mode == "reuse_patch_recompute" and not recompute_map and int(default_recompute_segment_length) > 0:
        recompute_map = {int(layer): int(default_recompute_segment_length) for layer in route_patch}

    if mode == "reuse_only":
        patch_layers: list[int] = []
        fallback_layers: list[int] = []
        recompute_map = {}
        reuse_layers = list(all_layers)
    elif mode == "patch_only":
        patch_layers = sorted(route_patch)
        fallback_layers = []
        recompute_map = {}
        reuse_layers = []
    elif mode == "reuse_fallback":
        patch_layers = []
        recompute_map = {}
        fallback_layers = sorted(layer for layer in all_layers if layer not in set(route_reuse))
        reuse_layers = sorted(route_reuse)
    elif mode == "reuse_patch":
        patch_layers = sorted(route_patch)
        recompute_map = {}
        fallback_layers = sorted(route_fallback if fallback_explicit else [])
        reuse_layers = sorted(layer for layer in all_layers if layer not in set(patch_layers) and layer not in set(fallback_layers))
    elif mode == "reuse_patch_recompute":
        patch_layers = sorted(route_patch)
        recompute_covered = _covered_recompute_layers(int(num_layers), recompute_map)
        fallback_layers = sorted(route_fallback if fallback_explicit else [])
        reuse_layers = sorted(layer for layer in all_layers if layer not in set(patch_layers) and layer not in set(fallback_layers))
    else:
        patch_layers = sorted(route_patch)
        fallback_layers = sorted(route_fallback)
        reuse_layers = sorted(route_reuse)

    recompute_covered_layers = _covered_recompute_layers(int(num_layers), recompute_map)
    layer_sources: dict[str, str] = {}
    patch_set = set(int(layer) for layer in patch_layers)
    fallback_set = set(int(layer) for layer in fallback_layers)
    recompute_covered_set = set(int(layer) for layer in recompute_covered_layers)
    for layer in all_layers:
        if layer in recompute_covered_set:
            source = "recompute"
        elif layer in fallback_set:
            source = "fallback"
        elif layer in patch_set:
            source = "patch"
        else:
            source = "reuse"
        layer_sources[str(layer)] = source

    return {
        "reuse_layers": reuse_layers,
        "patch_layers": patch_layers,
        "fallback_layers": fallback_layers,
        "selected_recompute_layers": sorted(recompute_map),
        "recompute_segment_length_by_layer": {int(layer): int(seg_len) for layer, seg_len in sorted(recompute_map.items())},
        "recompute_covered_layers": recompute_covered_layers,
        "layer_sources": layer_sources,
    }


def _target_answer_ids(session: TransformersRuntimeSession, answers: list[str], max_new_tokens: int) -> Any:
    t = require_torch()
    answer = next((str(x).strip() for x in answers if str(x).strip()), "")
    if not answer:
        return t.zeros((0,), dtype=t.long, device=session.target_device)
    if callable(getattr(session.tokenizer, "__call__", None)):
        encoded = session.tokenizer(
            answer,
            return_tensors="pt",
            add_special_tokens=False,
            truncation=True,
            max_length=max(1, int(max_new_tokens)),
        )
        input_ids = encoded["input_ids"]
    elif hasattr(session.tokenizer, "encode"):
        input_ids = t.tensor(
            session.tokenizer.encode(answer, add_special_tokens=False)[: max(1, int(max_new_tokens))],
            dtype=t.long,
        )
    else:
        return t.zeros((0,), dtype=t.long, device=session.target_device)
    if input_ids.ndim == 2:
        input_ids = input_ids[0]
    return input_ids.to(session.target_device)


def _teacher_forced_from_prefill(
    session: TransformersRuntimeSession,
    capture: PrefillCapture,
    target_ids: Any,
) -> tuple[float, float, int]:
    t = require_torch()
    if int(target_ids.numel()) <= 0:
        return 0.0, 0.0, 0
    past = build_dynamic_cache(session.model, capture.past_key_values)
    attention_mask = capture.attention_mask.clone().to(session.target_device)
    logits = capture.logits_last.to(session.target_device)
    logp_sum = 0.0
    for idx in range(int(target_ids.numel())):
        token = target_ids[idx : idx + 1].reshape(1, 1)
        log_probs = t.log_softmax(logits, dim=-1)
        logp_sum += float(log_probs[0, int(token.item())].item())
        if idx + 1 < int(target_ids.numel()):
            attention_mask = t.cat([attention_mask, t.ones((1, 1), device=attention_mask.device, dtype=attention_mask.dtype)], dim=1)
            with t.no_grad():
                out = session.model(
                    input_ids=token.to(dtype=capture.input_ids.dtype),
                    attention_mask=attention_mask,
                    past_key_values=past,
                    use_cache=True,
                    return_dict=True,
                )
            past = out.past_key_values
            logits = out.logits[:, -1, :].detach()
    avg_logp = float(logp_sum / int(target_ids.numel()))
    return avg_logp, float(-avg_logp), int(target_ids.numel())


def _teacher_forced_from_cache(
    session: TransformersRuntimeSession,
    *,
    cache_layers: list[tuple[Any, Any]],
    prompt_input_ids: Any,
    prompt_attention_mask: Any,
    prefix_tokens: int,
    target_ids: Any,
) -> tuple[float, float, int]:
    t = require_torch()
    if int(target_ids.numel()) <= 0:
        return 0.0, 0.0, 0
    past = build_dynamic_cache(session.model, [(k.contiguous(), v.contiguous()) for k, v in cache_layers])
    attention_mask = prompt_attention_mask[:, : prefix_tokens + 1].to(session.target_device)
    last_token = prompt_input_ids[:, prefix_tokens : prefix_tokens + 1].to(session.target_device)
    with t.no_grad():
        out = session.model(
            input_ids=last_token,
            attention_mask=attention_mask,
            past_key_values=past,
            use_cache=True,
            return_dict=True,
        )
    past = out.past_key_values
    logits = out.logits[:, -1, :].detach()
    logp_sum = 0.0
    for idx in range(int(target_ids.numel())):
        token = target_ids[idx : idx + 1].reshape(1, 1)
        log_probs = t.log_softmax(logits, dim=-1)
        logp_sum += float(log_probs[0, int(token.item())].item())
        if idx + 1 < int(target_ids.numel()):
            attention_mask = t.cat([attention_mask, t.ones((1, 1), device=attention_mask.device, dtype=attention_mask.dtype)], dim=1)
            with t.no_grad():
                out = session.model(
                    input_ids=token.to(dtype=prompt_input_ids.dtype),
                    attention_mask=attention_mask,
                    past_key_values=past,
                    use_cache=True,
                    return_dict=True,
                )
            past = out.past_key_values
            logits = out.logits[:, -1, :].detach()
    avg_logp = float(logp_sum / int(target_ids.numel()))
    return avg_logp, float(-avg_logp), int(target_ids.numel())


def _producer_prefix_layers(capture: PrefillCapture, prefix_tokens: int) -> list[tuple[Any, Any]]:
    out = []
    for k, v in capture.past_key_values:
        out.append((_slice_prefix_tensor(k, prefix_tokens), _slice_prefix_tensor(v, prefix_tokens)))
    return out


def _consumer_prefix_layers(capture: PrefillCapture, prefix_tokens: int) -> list[tuple[Any, Any]]:
    return _producer_prefix_layers(capture, prefix_tokens)


def _map_k_between_sessions(k_src: Any, prefix_tokens: int, producer_session: TransformersRuntimeSession, consumer_session: TransformersRuntimeSession):
    k_prefix = _slice_prefix_tensor(k_src, prefix_tokens).to(dtype=require_torch().float32)
    k_mapped = remap_rope_k_torch(k_prefix, producer_session.rope_spec, consumer_session.rope_spec)
    return k_mapped


def _stack_group_row_vectors(rows: list[Any]) -> Any | None:
    if not rows or any(row is None for row in rows):
        return None
    t = require_torch()
    return t.stack([row.to(dtype=t.float32) for row in rows], dim=0).contiguous()


def _apply_optional_group_center(x: Any, means: Any | None, *, subtract: bool) -> Any:
    if means is None:
        return x
    if subtract:
        return x - means.unsqueeze(1)
    return x + means.unsqueeze(1)


def _can_use_batched_layerwise_reuse_fastpath(reuse_prod, reuse_cons) -> bool:
    if reuse_prod.num_groups != reuse_prod.num_layers or reuse_cons.num_groups != reuse_cons.num_layers:
        return False
    if reuse_prod.num_groups != reuse_cons.num_groups:
        return False
    ref_shapes: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]] | None = None
    for gid in range(int(reuse_prod.num_groups)):
        prod_layers = group_layers_for_gid(reuse_prod, gid)
        cons_layers = group_layers_for_gid(reuse_cons, gid)
        if len(prod_layers) != 1 or len(cons_layers) != 1:
            return False
        w_prod = reuse_prod.weights[gid]
        w_cons = reuse_cons.weights[gid]
        shapes = (
            tuple(w_prod.enc_k.shape),
            tuple(w_prod.enc_v.shape),
            tuple(w_cons.dec_k_b.shape),
            tuple(w_cons.dec_v_b.shape),
        )
        if ref_shapes is None:
            ref_shapes = shapes
        elif shapes != ref_shapes:
            return False
    return ref_shapes is not None


def _selected_reuse_gids(reuse_weights, selected_reuse_layers: list[int] | None) -> list[int]:
    if selected_reuse_layers is None:
        return list(range(int(reuse_weights.num_groups)))
    selected_set = {int(layer) for layer in selected_reuse_layers}
    if not selected_set:
        return []
    gids: list[int] = []
    for gid in range(int(reuse_weights.num_groups)):
        layers = [int(layer) for layer in group_layers_for_gid(reuse_weights, gid)]
        if any(layer in selected_set for layer in layers):
            gids.append(int(gid))
    return gids


def _decode_reuse_layers_batched_layerwise_fastpath(
    producer_capture: PrefillCapture,
    producer_session: TransformersRuntimeSession,
    consumer_session: TransformersRuntimeSession,
    reuse_prod,
    reuse_cons,
    prefix_tokens: int,
    transfer_dtype: Any,
    consumer_dtype: Any,
    consumer_device: str,
    selected_reuse_layers: list[int] | None = None,
) -> tuple[list[tuple[Any, Any]], int, ReuseDecodeTiming]:
    t = require_torch()
    total_num_layers = int(reuse_prod.num_layers)
    decoded: list[tuple[Any, Any]] = [None for _ in range(total_num_layers)]  # type: ignore[list-item]
    selected_gids = _selected_reuse_gids(reuse_prod, selected_reuse_layers)
    if not selected_gids:
        return decoded, 0, ReuseDecodeTiming()
    bsz = int(producer_capture.input_ids.shape[0])
    kvh = int(producer_capture.past_key_values[0][0].shape[1])
    hd = int(producer_capture.past_key_values[0][0].shape[-1])
    flat_rows = int(bsz * kvh * prefix_tokens)

    layer_order = [int(group_layers_for_gid(reuse_prod, gid)[0]) for gid in selected_gids]
    num_selected_layers = len(layer_order)

    k_layers = t.stack(
        [_slice_prefix_tensor(producer_capture.past_key_values[layer][0], prefix_tokens).to(dtype=t.float32) for layer in layer_order],
        dim=0,
    ).contiguous()
    v_layers = t.stack(
        [_slice_prefix_tensor(producer_capture.past_key_values[layer][1], prefix_tokens).to(dtype=t.float32) for layer in layer_order],
        dim=0,
    ).contiguous()

    if reuse_prod.k_de_rope:
        k_layers = de_rope_k_torch(
            k_layers.reshape(num_selected_layers * bsz, kvh, prefix_tokens, hd),
            producer_session.rope_spec,
        ).reshape(num_selected_layers, bsz, kvh, prefix_tokens, hd).contiguous()

    k_mat = k_layers.reshape(num_selected_layers, flat_rows, hd)
    v_mat = v_layers.reshape(num_selected_layers, flat_rows, hd)

    enc_k = t.stack([reuse_prod.weights[gid].enc_k.to(dtype=t.float32) for gid in selected_gids], dim=0).contiguous()
    enc_v = t.stack([reuse_prod.weights[gid].enc_v.to(dtype=t.float32) for gid in selected_gids], dim=0).contiguous()
    mean_a_k = _stack_group_row_vectors([reuse_prod.weights[gid].mean_a_k for gid in selected_gids])
    mean_a_v = _stack_group_row_vectors([reuse_prod.weights[gid].mean_a_v for gid in selected_gids])

    _synchronize_for_device(producer_capture.past_key_values[0][0].device)
    enc_start = time.perf_counter()
    z_k = t.bmm(_apply_optional_group_center(k_mat, mean_a_k, subtract=True), enc_k)
    z_v = t.bmm(_apply_optional_group_center(v_mat, mean_a_v, subtract=True), enc_v)
    z_k = z_k.to(dtype=transfer_dtype).detach()
    z_v = z_v.to(dtype=transfer_dtype).detach()
    _synchronize_for_device(producer_capture.past_key_values[0][0].device)
    encode_ms = (time.perf_counter() - enc_start) * 1000.0
    payload_bytes = _tensor_bytes(z_k) + _tensor_bytes(z_v)

    _synchronize_for_device(consumer_device)
    copy_start = time.perf_counter()
    z_k = z_k.to(device=consumer_device, dtype=t.float32, non_blocking=True)
    z_v = z_v.to(device=consumer_device, dtype=t.float32, non_blocking=True)
    _synchronize_for_device(consumer_device)
    staging_copy_ms = (time.perf_counter() - copy_start) * 1000.0

    rebuild_timing = ReuseDecodeTiming(encode_ms=float(encode_ms), staging_copy_ms=float(staging_copy_ms))

    _synchronize_for_device(consumer_device)
    prep_start = time.perf_counter()
    dec_k = t.stack([reuse_cons.weights[gid].dec_k_b.to(dtype=t.float32) for gid in selected_gids], dim=0).contiguous()
    dec_v = t.stack([reuse_cons.weights[gid].dec_v_b.to(dtype=t.float32) for gid in selected_gids], dim=0).contiguous()
    mean_b_k = _stack_group_row_vectors([reuse_cons.weights[gid].mean_b_k for gid in selected_gids])
    mean_b_v = _stack_group_row_vectors([reuse_cons.weights[gid].mean_b_v for gid in selected_gids])
    _synchronize_for_device(consumer_device)
    rebuild_timing.rebuild_prep_ms = (time.perf_counter() - prep_start) * 1000.0

    _synchronize_for_device(consumer_device)
    core_start = time.perf_counter()
    rec_k = _apply_optional_group_center(t.bmm(z_k, dec_k), mean_b_k, subtract=False)
    rec_v = _apply_optional_group_center(t.bmm(z_v, dec_v), mean_b_v, subtract=False)
    rec_k_layers = rec_k.reshape(num_selected_layers, bsz, kvh, prefix_tokens, hd).contiguous()
    rec_v_layers = rec_v.reshape(num_selected_layers, bsz, kvh, prefix_tokens, hd).contiguous()
    _synchronize_for_device(consumer_device)
    rebuild_timing.rebuild_decode_core_ms = (time.perf_counter() - core_start) * 1000.0

    if reuse_cons.k_re_rope:
        _synchronize_for_device(consumer_device)
        rerope_start = time.perf_counter()
        rec_k_layers = re_rope_k_torch(
            rec_k_layers.reshape(num_selected_layers * bsz, kvh, prefix_tokens, hd),
            consumer_session.rope_spec,
        ).reshape(num_selected_layers, bsz, kvh, prefix_tokens, hd).contiguous()
        _synchronize_for_device(consumer_device)
        rebuild_timing.rebuild_rerope_ms = (time.perf_counter() - rerope_start) * 1000.0

    _synchronize_for_device(consumer_device)
    materialize_start = time.perf_counter()
    for gid, layer in enumerate(layer_order):
        decoded[int(layer)] = (
            rec_k_layers[gid].to(dtype=consumer_dtype).contiguous(),
            rec_v_layers[gid].to(dtype=consumer_dtype).contiguous(),
        )
    _synchronize_for_device(consumer_device)
    rebuild_timing.rebuild_materialize_ms = (time.perf_counter() - materialize_start) * 1000.0
    rebuild_timing.rebuild_ms = (
        float(rebuild_timing.rebuild_prep_ms)
        + float(rebuild_timing.rebuild_decode_core_ms)
        + float(rebuild_timing.rebuild_rerope_ms)
        + float(rebuild_timing.rebuild_materialize_ms)
    )
    return decoded, int(payload_bytes), rebuild_timing


def _decode_reuse_layers(
    producer_capture: PrefillCapture,
    producer_session: TransformersRuntimeSession,
    consumer_session: TransformersRuntimeSession,
    reuse_prod,
    reuse_cons,
    prefix_tokens: int,
    transfer_dtype: Any,
    consumer_dtype: Any,
    consumer_device: str,
    selected_reuse_layers: list[int] | None = None,
) -> tuple[list[tuple[Any, Any]], int, ReuseDecodeTiming]:
    t = require_torch()
    total_num_layers = int(reuse_prod.num_layers)
    decoded: list[tuple[Any, Any]] = [None for _ in range(total_num_layers)]  # type: ignore[list-item]
    selected_gids = _selected_reuse_gids(reuse_prod, selected_reuse_layers)
    if not selected_gids:
        return decoded, 0, ReuseDecodeTiming()
    if _can_use_batched_layerwise_reuse_fastpath(reuse_prod, reuse_cons):
        return _decode_reuse_layers_batched_layerwise_fastpath(
            producer_capture=producer_capture,
            producer_session=producer_session,
            consumer_session=consumer_session,
            reuse_prod=reuse_prod,
            reuse_cons=reuse_cons,
            prefix_tokens=prefix_tokens,
            transfer_dtype=transfer_dtype,
            consumer_dtype=consumer_dtype,
            consumer_device=consumer_device,
            selected_reuse_layers=selected_reuse_layers,
        )
    bsz = int(producer_capture.input_ids.shape[0])
    kvh = int(producer_capture.past_key_values[0][0].shape[1])
    hd = int(producer_capture.past_key_values[0][0].shape[-1])

    transfer_codes: list[tuple[int, Any, Any]] = []
    _synchronize_for_device(producer_capture.past_key_values[0][0].device)
    enc_start = time.perf_counter()
    payload_bytes = 0
    for gid in selected_gids:
        layers = group_layers_for_gid(reuse_prod, gid)
        a_k = pack_group([
            (
                de_rope_k_torch(_slice_prefix_tensor(producer_capture.past_key_values[layer][0], prefix_tokens), producer_session.rope_spec)
                if reuse_prod.k_de_rope
                else _slice_prefix_tensor(producer_capture.past_key_values[layer][0], prefix_tokens)
            ).to(dtype=t.float32)
            for layer in layers
        ])
        a_v = pack_group([_slice_prefix_tensor(producer_capture.past_key_values[layer][1], prefix_tokens).to(dtype=t.float32) for layer in layers])
        z_k, z_v = encode_group(a_k, a_v, reuse_prod.weights[gid])
        z_k = z_k.to(dtype=transfer_dtype).detach()
        z_v = z_v.to(dtype=transfer_dtype).detach()
        transfer_codes.append((int(gid), z_k, z_v))
        payload_bytes += _tensor_bytes(z_k) + _tensor_bytes(z_v)
    _synchronize_for_device(producer_capture.past_key_values[0][0].device)
    encode_ms = (time.perf_counter() - enc_start) * 1000.0

    transfer_codes_on_consumer: dict[int, tuple[Any, Any]] = {}
    _synchronize_for_device(consumer_device)
    copy_start = time.perf_counter()
    for gid, z_k, z_v in transfer_codes:
        z_k = z_k.to(device=consumer_device, dtype=t.float32, non_blocking=True)
        z_v = z_v.to(device=consumer_device, dtype=t.float32, non_blocking=True)
        transfer_codes_on_consumer[int(gid)] = (z_k, z_v)
    _synchronize_for_device(consumer_device)
    staging_copy_ms = (time.perf_counter() - copy_start) * 1000.0

    _synchronize_for_device(consumer_device)
    rebuild_timing = ReuseDecodeTiming(encode_ms=float(encode_ms), staging_copy_ms=float(staging_copy_ms))
    for gid in selected_gids:
        layers = group_layers_for_gid(reuse_cons, gid)
        z_k, z_v = transfer_codes_on_consumer[int(gid)]
        _synchronize_for_device(consumer_device)
        core_start = time.perf_counter()
        rec_k_raw, rec_v = decode_group(z_k, z_v, reuse_cons.weights[gid])
        rec_k_layers_raw = unpack_group(rec_k_raw, len(layers), hd, bsz, kvh, prefix_tokens)
        rec_v_layers = unpack_group(rec_v, len(layers), hd, bsz, kvh, prefix_tokens)
        _synchronize_for_device(consumer_device)
        rebuild_timing.rebuild_decode_core_ms += (time.perf_counter() - core_start) * 1000.0
        for layer, rk_raw, rv in zip(layers, rec_k_layers_raw, rec_v_layers):
            rk = rk_raw
            if reuse_cons.k_re_rope:
                _synchronize_for_device(consumer_device)
                rerope_start = time.perf_counter()
                rk = re_rope_k_torch(rk_raw, consumer_session.rope_spec)
                _synchronize_for_device(consumer_device)
                rebuild_timing.rebuild_rerope_ms += (time.perf_counter() - rerope_start) * 1000.0
            _synchronize_for_device(consumer_device)
            materialize_start = time.perf_counter()
            decoded[layer] = (rk.to(dtype=consumer_dtype).contiguous(), rv.to(dtype=consumer_dtype).contiguous())
            _synchronize_for_device(consumer_device)
            rebuild_timing.rebuild_materialize_ms += (time.perf_counter() - materialize_start) * 1000.0
    rebuild_timing.rebuild_ms = (
        float(rebuild_timing.rebuild_prep_ms)
        + float(rebuild_timing.rebuild_decode_core_ms)
        + float(rebuild_timing.rebuild_rerope_ms)
        + float(rebuild_timing.rebuild_materialize_ms)
    )
    return decoded, int(payload_bytes), rebuild_timing


def _overlay_consumer_layers(
    caches: list[tuple[Any, Any]],
    *,
    consumer_capture: PrefillCapture,
    layers: list[int],
    prefix_tokens: int,
    consumer_device: str,
    consumer_dtype: Any,
) -> None:
    for layer in sorted(set(int(layer) for layer in layers)):
        caches[layer] = (
            _slice_prefix_tensor(consumer_capture.past_key_values[layer][0], prefix_tokens).to(device=consumer_device, dtype=consumer_dtype, non_blocking=True),
            _slice_prefix_tensor(consumer_capture.past_key_values[layer][1], prefix_tokens).to(device=consumer_device, dtype=consumer_dtype, non_blocking=True),
        )


def _oracle_cache_payload_bytes(
    consumer_capture: PrefillCapture,
    *,
    layers: list[int],
    prefix_tokens: int,
    transfer_dtype: Any,
) -> int:
    total = 0
    for layer in sorted(set(int(layer) for layer in layers)):
        total += int(_slice_prefix_tensor(consumer_capture.past_key_values[layer][0], prefix_tokens).numel()) * _dtype_size_bytes(transfer_dtype)
        total += int(_slice_prefix_tensor(consumer_capture.past_key_values[layer][1], prefix_tokens).numel()) * _dtype_size_bytes(transfer_dtype)
    return int(total)


def _resolve_oracle_breakpoint_state(
    consumer_capture: PrefillCapture,
    *,
    layer: int,
    state_kind: str,
) -> Any:
    if state_kind == "xnorm":
        hidden = consumer_capture.xnorm_by_layer.get(int(layer))
    elif state_kind == "residual":
        hidden = consumer_capture.residual_by_layer.get(int(layer))
    else:
        raise ValueError(f"未知 oracle breakpoint state_kind: {state_kind}")
    if hidden is None:
        raise ValueError(f"oracle {state_kind} 诊断缺少 layer={layer} 的 consumer state")
    return hidden


def _apply_oracle_breakpoints_native(
    *,
    caches: list[tuple[Any, Any]],
    patch_layers: list[int],
    consumer_capture: PrefillCapture,
    consumer_session: TransformersRuntimeSession,
    prefix_tokens: int,
    consumer_dtype: Any,
    consumer_device: str,
    transfer_dtype: Any,
    state_kind: str,
) -> tuple[int, float, dict[int, Any]]:
    payload_bytes = 0
    breakpoint_hidden_by_layer: dict[int, Any] = {}
    attn_mask_prefix = consumer_capture.attention_mask[:, :prefix_tokens].to(device=consumer_device)
    _synchronize_for_device(consumer_device)
    start = time.perf_counter()
    for layer in sorted(set(int(layer) for layer in patch_layers)):
        hidden = _resolve_oracle_breakpoint_state(consumer_capture, layer=layer, state_kind=state_kind)
        hidden_prefix = _slice_prefix_tensor(hidden, prefix_tokens).to(device=consumer_device, dtype=require_torch().float32, non_blocking=True)
        payload_bytes += int(hidden_prefix.numel()) * _dtype_size_bytes(transfer_dtype)
        layer_inputs = _build_layer_forward_inputs(consumer_session, hidden_prefix, attn_mask_prefix)
        next_hidden, patch_k, patch_v = _run_patch_breakpoint_native(
            consumer_session=consumer_session,
            layer_idx=layer,
            hidden_states=hidden_prefix,
            attention_mask=layer_inputs["attention_mask"],
            position_ids=layer_inputs["position_ids"],
            cache_position=layer_inputs["cache_position"],
            position_embeddings=layer_inputs["position_embeddings"],
        )
        caches[layer] = (
            patch_k.to(device=consumer_device, dtype=consumer_dtype),
            patch_v.to(device=consumer_device, dtype=consumer_dtype),
        )
        breakpoint_hidden_by_layer[int(layer)] = next_hidden.detach()
    _synchronize_for_device(consumer_device)
    return int(payload_bytes), float((time.perf_counter() - start) * 1000.0), breakpoint_hidden_by_layer


def _apply_xnorm_nativekv_shared_patch_layers_to_cache(
    *,
    caches: list[tuple[Any, Any]],
    patch_layers: list[int],
    producer_capture: PrefillCapture,
    consumer_session: TransformersRuntimeSession,
    patch_bundle: dict[str, Any],
    patch_meta: dict[str, object],
    prefix_tokens: int,
    hidden_dim: int,
    kvh: int,
    hd: int,
    consumer_dtype: Any,
    consumer_device: str,
    transfer_dtype: Any,
) -> tuple[int, float]:
    t = require_torch()
    kv_dim = kvh * hd
    residual_layers = [int(v) for v in patch_meta.get("residual_layers", [])] if isinstance(patch_meta.get("residual_layers", []), list) else []
    residual_patch_layers = [int(layer) for layer in sorted(set(int(layer) for layer in patch_layers)) if int(layer) in set(residual_layers)]
    non_residual_patch_layers = [int(layer) for layer in sorted(set(int(layer) for layer in patch_layers)) if int(layer) not in set(residual_layers)]
    if non_residual_patch_layers:
        raise ValueError(f"shared xnorm patch runtime currently requires all patch layers to be residual layers, got {non_residual_patch_layers}")
    if not residual_patch_layers:
        return 0, 0.0

    patch_breakdown_ms = {
        "patch_shard_load_ms": 0.0,
        "patch_shard_gpu_copy_ms": 0.0,
        "patch_identity_check_ms": 0.0,
        "patch_residual_forward_ms": 0.0,
        "patch_kproj_vproj_ms": 0.0,
        "patch_knorm_ms": 0.0,
        "patch_rotary_ms": 0.0,
    }
    patch_meta["__runtime_patch_breakdown_ms__"] = patch_breakdown_ms
    patch_path_timing_ms = {
        "encode_ms": 0.0,
        "transfer_ms": 0.0,
        "patch_compute_ms": 0.0,
    }
    patch_meta["__runtime_patch_path_timing_ms__"] = patch_path_timing_ms

    shared_module = _load_runtime_xnorm_nativekv_shared_residual_module(
        patch_bundle=patch_bundle,
        patch_meta=patch_meta,
        consumer_session=consumer_session,
    )
    use_fastpath = _shared_nativekv_runtime_fastpath_enabled(patch_meta)
    payload_bytes = 0
    producer_device = str(producer_capture.input_ids.device)
    encoded_xa_parts = []
    _synchronize_for_device(producer_device)
    encode_start = time.perf_counter()
    for layer in residual_patch_layers:
        xa = producer_capture.xnorm_by_layer.get(int(layer))
        if xa is None:
            raise ValueError(f"patch 层 {int(layer)} 缺少 xnorm_a")
        xa_prefix = _slice_prefix_tensor(xa, prefix_tokens).reshape(-1, hidden_dim).to(
            device=producer_capture.input_ids.device,
            dtype=transfer_dtype,
        ).contiguous()
        payload_bytes += int(xa_prefix.numel()) * _dtype_size_bytes(transfer_dtype)
        encoded_xa_parts.append(xa_prefix)
    _synchronize_for_device(producer_device)
    patch_path_timing_ms["encode_ms"] = float((time.perf_counter() - encode_start) * 1000.0)

    xa_parts = []
    _synchronize_for_device(consumer_device)
    transfer_start = time.perf_counter()
    for xa_prefix in encoded_xa_parts:
        xa_parts.append(xa_prefix.to(device=consumer_device, dtype=transfer_dtype, non_blocking=True))
    _synchronize_for_device(consumer_device)
    patch_path_timing_ms["transfer_ms"] = float((time.perf_counter() - transfer_start) * 1000.0)

    identity_positions = []
    non_identity_layers = []
    non_identity_xa_parts = []
    _synchronize_for_device(consumer_device)
    patch_compute_start = time.perf_counter()
    for idx, layer in enumerate(residual_patch_layers):
        xa_prefix = xa_parts[idx].to(dtype=t.float32)
        xa_parts[idx] = xa_prefix
        if _runtime_xnorm_nativekv_shared_residual_is_identity(
            layer=int(layer),
            patch_bundle=patch_bundle,
            patch_meta=patch_meta,
            consumer_session=consumer_session,
        ):
            identity_positions.append(len(xa_parts) - 1)
        else:
            non_identity_layers.append(int(layer))
            non_identity_xa_parts.append(xa_prefix)

    if xa_parts:
        xb_hat_batch = t.stack(xa_parts, dim=0).contiguous()
    else:
        xb_hat_batch = t.empty((0, 0, hidden_dim), device=consumer_device, dtype=t.float32)

    if non_identity_layers:
        non_identity_xa_batch = t.stack(non_identity_xa_parts, dim=0).contiguous()
        layer_ids = t.tensor(non_identity_layers, device=consumer_device, dtype=t.long)
        non_identity_xb_hat_batch, stage_ms = _measure_runtime_stage_ms(
            consumer_device,
            lambda: infer_xnorm_nativekv_shared_residual_patch_torch(non_identity_xa_batch, layer_ids, shared_module),
        )
        patch_breakdown_ms["patch_residual_forward_ms"] += float(stage_ms)
        write_idx = 0
        for pos, layer in enumerate(residual_patch_layers):
            if pos in identity_positions:
                continue
            xb_hat_batch[pos] = non_identity_xb_hat_batch[write_idx]
            write_idx += 1

    if residual_patch_layers:
        if use_fastpath:
            (k_pre_batch, vb_batch), stage_ms = _measure_runtime_stage_ms(
                consumer_device,
                lambda: _project_consumer_native_kv_from_hidden_tensor_batched_runtime(
                    xb_hat=xb_hat_batch,
                    layers=residual_patch_layers,
                    consumer_session=consumer_session,
                    kvh=kvh,
                    hd=hd,
                ),
            )
            patch_breakdown_ms["patch_kproj_vproj_ms"] += float(stage_ms)
            kb_norm_batch = k_pre_batch
            kb_norm = kb_norm_batch.reshape(len(residual_patch_layers), prefix_tokens, kv_dim).reshape(len(residual_patch_layers), prefix_tokens, kvh, hd).permute(0, 2, 1, 3).contiguous()
            vb = vb_batch.reshape(len(residual_patch_layers), prefix_tokens, kv_dim).reshape(len(residual_patch_layers), prefix_tokens, kvh, hd).permute(0, 2, 1, 3).contiguous()
            kb, stage_ms = _measure_runtime_stage_ms(
                consumer_device,
                lambda: _apply_consumer_native_rotary_to_patch_keys_batched(
                    k_states=kb_norm,
                    layers=residual_patch_layers,
                    consumer_session=consumer_session,
                    hidden_dim=hidden_dim,
                ),
            )
            patch_breakdown_ms["patch_rotary_ms"] += float(stage_ms)
            for idx, layer in enumerate(residual_patch_layers):
                caches[int(layer)] = (
                    kb[idx : idx + 1].to(device=consumer_device, dtype=consumer_dtype),
                    vb[idx : idx + 1].to(device=consumer_device, dtype=consumer_dtype),
                )
        else:
            for idx, layer in enumerate(residual_patch_layers):
                xb_hat = xb_hat_batch[idx : idx + 1]
                k_proj, v_proj, k_norm = _get_consumer_attn_modules(consumer_session, int(layer))
                proj_dtype = getattr(getattr(k_proj, "weight", None), "dtype", xb_hat.dtype)

                def _run_layer_proj():
                    xb_proj = xb_hat.to(device=consumer_device, dtype=proj_dtype)
                    k_pre = k_proj(xb_proj).to(dtype=t.float32).reshape(-1, kv_dim)
                    v_pre = v_proj(xb_proj).to(dtype=t.float32).reshape(-1, kv_dim)
                    k_pre = _apply_consumer_k_norm_to_flat_keys(k_pre, k_norm, kvh, hd)
                    return k_pre, v_pre

                (kb_norm_flat, vb_flat), stage_ms = _measure_runtime_stage_ms(
                    consumer_device,
                    _run_layer_proj,
                )
                patch_breakdown_ms["patch_kproj_vproj_ms"] += float(stage_ms)
                kb_norm = kb_norm_flat.reshape(1, prefix_tokens, kv_dim).reshape(1, prefix_tokens, kvh, hd).permute(0, 2, 1, 3).contiguous()
                vb = vb_flat.reshape(1, prefix_tokens, kv_dim).reshape(1, prefix_tokens, kvh, hd).permute(0, 2, 1, 3).contiguous()
                kb, stage_ms = _measure_runtime_stage_ms(
                    consumer_device,
                    lambda: _apply_consumer_native_rotary_to_patch_keys(
                        k_states=kb_norm,
                        layer=int(layer),
                        consumer_session=consumer_session,
                        attention_mask_prefix=None,
                        hidden_dim=hidden_dim,
                    ),
                )
                patch_breakdown_ms["patch_rotary_ms"] += float(stage_ms)
                caches[int(layer)] = (
                    kb.to(device=consumer_device, dtype=consumer_dtype),
                    vb.to(device=consumer_device, dtype=consumer_dtype),
                )

    _synchronize_for_device(consumer_device)
    patch_path_timing_ms["patch_compute_ms"] = float((time.perf_counter() - patch_compute_start) * 1000.0)
    total_patch_ms = (
        float(patch_breakdown_ms["patch_shard_load_ms"])
        + float(patch_breakdown_ms["patch_shard_gpu_copy_ms"])
        + float(patch_breakdown_ms["patch_identity_check_ms"])
        + float(patch_breakdown_ms["patch_residual_forward_ms"])
        + float(patch_breakdown_ms["patch_kproj_vproj_ms"])
        + float(patch_breakdown_ms["patch_knorm_ms"])
        + float(patch_breakdown_ms["patch_rotary_ms"])
    )
    if float(total_patch_ms) > float(patch_path_timing_ms["patch_compute_ms"]):
        patch_path_timing_ms["patch_compute_ms"] = float(total_patch_ms)
    return int(payload_bytes), float(patch_path_timing_ms["patch_compute_ms"])


def _get_consumer_attn_modules(consumer_session: TransformersRuntimeSession, layer: int) -> tuple[Any, Any, Any | None]:
    layers = consumer_session.layers if hasattr(consumer_session, "layers") else _get_decoder_layers(consumer_session.model)
    layer_module = layers[int(layer)]
    attn = getattr(layer_module, "self_attn", None) or getattr(layer_module, "attn", None)
    if attn is None or not hasattr(attn, "k_proj") or not hasattr(attn, "v_proj"):
        raise ValueError(f"layer={layer} 缺少 consumer native k_proj/v_proj")
    return attn.k_proj, attn.v_proj, getattr(attn, "k_norm", None)


def _apply_consumer_k_norm_to_flat_keys(k_pre_flat: Any, k_norm: Any | None, kvh: int, hd: int) -> Any:
    t = require_torch()
    if k_norm is None:
        return k_pre_flat.to(dtype=t.float32)
    k_pre_flat = k_pre_flat.to(dtype=t.float32)
    norm_in = k_pre_flat.reshape(-1, kvh, hd)
    norm_dtype = getattr(getattr(k_norm, "weight", None), "dtype", norm_in.dtype)
    with t.no_grad():
        normed = k_norm(norm_in.to(dtype=norm_dtype)).to(dtype=t.float32)
    return normed.reshape(-1, kvh * hd).contiguous()


def _rotate_half_last_dim(x: Any) -> Any:
    t = require_torch()
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return t.cat((-x2, x1), dim=-1)


def _apply_rotary_key_only(
    *,
    k_states: Any,
    cos: Any,
    sin: Any,
    unsqueeze_dim: int = 1,
) -> Any:
    cos = cos.unsqueeze(int(unsqueeze_dim)).to(dtype=k_states.dtype)
    sin = sin.unsqueeze(int(unsqueeze_dim)).to(dtype=k_states.dtype)
    return (k_states * cos) + (_rotate_half_last_dim(k_states) * sin)


def _consumer_native_rotary_apply_fn(
    *,
    layer: int,
    consumer_session: TransformersRuntimeSession,
):
    cache = getattr(consumer_session, "__runtime_native_rotary_apply_fn__", None)
    if not isinstance(cache, dict):
        cache = {}
        setattr(consumer_session, "__runtime_native_rotary_apply_fn__", cache)
    if int(layer) in cache:
        return cache[int(layer)]
    layers = consumer_session.layers if hasattr(consumer_session, "layers") else _get_decoder_layers(consumer_session.model)
    layer_module = layers[int(layer)]
    attn = getattr(layer_module, "self_attn", None) or getattr(layer_module, "attn", None)
    if attn is None:
        cache[int(layer)] = None
        return None
    try:
        import importlib

        attn_mod = importlib.import_module(attn.__class__.__module__)
        apply_rotary_pos_emb = getattr(attn_mod, "apply_rotary_pos_emb", None)
    except Exception:
        apply_rotary_pos_emb = None
    cache[int(layer)] = apply_rotary_pos_emb
    return apply_rotary_pos_emb


def _consumer_cached_rotary_embeddings(
    *,
    consumer_session: TransformersRuntimeSession,
    seq_len: int,
    hidden_dim: int,
):
    t = require_torch()
    cache = getattr(consumer_session, "__runtime_rotary_embedding_cache__", None)
    if not isinstance(cache, dict):
        cache = {}
        setattr(consumer_session, "__runtime_rotary_embedding_cache__", cache)
    model_dtype = getattr(consumer_session.model, "dtype", t.float32)
    cache_key = (int(seq_len), int(hidden_dim), str(consumer_session.target_device), str(model_dtype))
    if cache_key in cache:
        return cache[cache_key]
    position_ids = t.arange(int(seq_len), device=consumer_session.target_device).unsqueeze(0)
    dummy_hidden = t.zeros((1, int(seq_len), int(hidden_dim)), device=consumer_session.target_device, dtype=model_dtype)
    with t.no_grad():
        position_embeddings = consumer_session.model.model.rotary_emb(dummy_hidden, position_ids=position_ids)
    cache[cache_key] = position_embeddings
    return position_embeddings


def _apply_consumer_native_rotary_to_patch_keys(
    *,
    k_states: Any,
    layer: int,
    consumer_session: TransformersRuntimeSession,
    attention_mask_prefix: Any,
    hidden_dim: int,
) -> Any:
    t = require_torch()
    apply_rotary_pos_emb = _consumer_native_rotary_apply_fn(
        layer=layer,
        consumer_session=consumer_session,
    )
    if apply_rotary_pos_emb is None:
        return re_rope_k_torch(k_states, consumer_session.rope_spec)
    seq_len = int(k_states.shape[2])
    position_embeddings = _consumer_cached_rotary_embeddings(
        consumer_session=consumer_session,
        seq_len=seq_len,
        hidden_dim=hidden_dim,
    )
    if not isinstance(position_embeddings, tuple) or len(position_embeddings) < 2:
        return re_rope_k_torch(k_states, consumer_session.rope_spec)
    cos, sin = position_embeddings
    k_proj, _v_proj, _k_norm = _get_consumer_attn_modules(consumer_session, int(layer))
    rotary_dtype = getattr(getattr(k_proj, "weight", None), "dtype", k_states.dtype)
    k_states = k_states.to(device=consumer_session.target_device, dtype=rotary_dtype)
    with t.no_grad():
        k_rot = _apply_rotary_key_only(k_states=k_states, cos=cos, sin=sin)
    return k_rot.to(dtype=t.float32)


def _consumer_native_proj_stack(
    *,
    consumer_session: TransformersRuntimeSession,
    layers: list[int],
) -> dict[str, Any]:
    t = require_torch()
    layer_key = tuple(int(layer) for layer in layers)
    cache = getattr(consumer_session, "__runtime_consumer_native_proj_stack__", None)
    if not isinstance(cache, dict):
        cache = {}
        setattr(consumer_session, "__runtime_consumer_native_proj_stack__", cache)
    cache_key = (layer_key, str(consumer_session.target_device))
    if cache_key in cache:
        return cache[cache_key]

    k_weights = []
    v_weights = []
    k_biases = []
    v_biases = []
    k_norm_weights = []
    k_norm_eps = []
    proj_dtype = None
    for layer in layer_key:
        k_proj, v_proj, k_norm = _get_consumer_attn_modules(consumer_session, int(layer))
        if proj_dtype is None:
            proj_dtype = getattr(getattr(k_proj, "weight", None), "dtype", t.float32)
        k_weights.append(k_proj.weight.detach().to(device=consumer_session.target_device, dtype=proj_dtype).contiguous())
        v_weights.append(v_proj.weight.detach().to(device=consumer_session.target_device, dtype=proj_dtype).contiguous())
        k_biases.append(k_proj.bias.detach().to(device=consumer_session.target_device, dtype=proj_dtype).contiguous() if getattr(k_proj, "bias", None) is not None else None)
        v_biases.append(v_proj.bias.detach().to(device=consumer_session.target_device, dtype=proj_dtype).contiguous() if getattr(v_proj, "bias", None) is not None else None)
        if k_norm is None or getattr(k_norm, "weight", None) is None:
            k_norm_weights.append(None)
            k_norm_eps.append(1e-6)
        else:
            k_norm_weights.append(k_norm.weight.detach().to(device=consumer_session.target_device, dtype=t.float32).contiguous())
            k_norm_eps.append(float(getattr(k_norm, "variance_epsilon", getattr(k_norm, "eps", 1e-6))))

    k_proj_weight = t.stack(k_weights, dim=0).contiguous()
    v_proj_weight = t.stack(v_weights, dim=0).contiguous()
    k_proj_bias = None if not any(value is not None for value in k_biases) else t.stack(
        [value if value is not None else t.zeros_like(k_weights[0][..., 0]) for value in k_biases],
        dim=0,
    ).contiguous()
    v_proj_bias = None if not any(value is not None for value in v_biases) else t.stack(
        [value if value is not None else t.zeros_like(v_weights[0][..., 0]) for value in v_biases],
        dim=0,
    ).contiguous()

    kv_proj_weight = t.cat((k_proj_weight, v_proj_weight), dim=1).contiguous()
    kv_proj_bias = None if k_proj_bias is None and v_proj_bias is None else t.cat(
        (
            k_proj_bias if k_proj_bias is not None else t.zeros_like(v_proj_bias),
            v_proj_bias if v_proj_bias is not None else t.zeros_like(k_proj_bias),
        ),
        dim=1,
    ).contiguous()
    stack = {
        "proj_dtype": proj_dtype or t.float32,
        "kv_proj_weight_t": kv_proj_weight.transpose(1, 2).contiguous(),
        "kv_proj_bias_row": None if kv_proj_bias is None else kv_proj_bias.unsqueeze(1).contiguous(),
        "kv_dim": int(k_proj_weight.shape[1]),
        "k_norm_weight": None if any(value is None for value in k_norm_weights) else t.stack(k_norm_weights, dim=0).contiguous(),
        "k_norm_eps": t.tensor(k_norm_eps, device=consumer_session.target_device, dtype=t.float32),
    }
    cache[cache_key] = stack
    return stack


def _consumer_native_proj_graph_cache(
    *,
    consumer_session: TransformersRuntimeSession,
) -> dict[tuple[Any, ...], dict[str, Any]]:
    cache = getattr(consumer_session, "__runtime_consumer_native_proj_graph_cache__", None)
    if not isinstance(cache, dict):
        cache = {}
        setattr(consumer_session, "__runtime_consumer_native_proj_graph_cache__", cache)
    return cache


def _consumer_native_proj_graph_slot_key(
    *,
    consumer_session: TransformersRuntimeSession,
    layers: list[int],
    hidden_dim: int,
    kvh: int,
    hd: int,
    proj_dtype: Any,
) -> tuple[Any, ...]:
    return (
        tuple(int(layer) for layer in layers),
        int(hidden_dim),
        int(kvh),
        int(hd),
        str(consumer_session.target_device),
        str(proj_dtype),
    )


def _consumer_native_proj_graph_seq_cap(seq_len: int) -> int:
    seq_cap = 1
    target = max(1, int(seq_len))
    while seq_cap < target:
        seq_cap <<= 1
    return int(seq_cap)


def _release_consumer_native_proj_graph_entry(entry: dict[str, Any] | None) -> None:
    if not isinstance(entry, dict):
        return
    for key in ("graph", "static_xb_proj", "static_k_pre", "static_v_out"):
        value = entry.pop(key, None)
        if value is not None:
            del value
    gc.collect()
    clear_torch_cache()


def _consumer_native_kv_project_batched_kernel(
    *,
    xb_proj: Any,
    stack: dict[str, Any],
    kvh: int,
    hd: int,
) -> tuple[Any, Any]:
    t = require_torch()
    weight_t = stack["kv_proj_weight_t"]
    bias_row = stack.get("kv_proj_bias_row")
    if bias_row is not None:
        kv_out = t.baddbmm(bias_row, xb_proj, weight_t)
    else:
        kv_out = t.bmm(xb_proj, weight_t)
    kv_dim = int(stack["kv_dim"])
    k_pre = kv_out[..., :kv_dim].to(dtype=t.float32)
    weight = stack.get("k_norm_weight")
    if weight is not None:
        norm_in = k_pre.reshape(int(k_pre.shape[0]), -1, kvh, hd)
        eps = stack["k_norm_eps"].view(int(k_pre.shape[0]), 1, 1, 1)
        scale = weight.view(int(k_pre.shape[0]), 1, 1, hd)
        rms = norm_in.pow(2).mean(dim=-1, keepdim=True).add(eps).rsqrt()
        k_pre = (norm_in * rms * scale).reshape(int(k_pre.shape[0]), -1, kv_dim).contiguous()
    else:
        k_pre = k_pre.contiguous()
    v_out = kv_out[..., kv_dim:]
    return k_pre, v_out.to(dtype=t.float32).contiguous()


def _ensure_consumer_native_proj_graph_entry(
    *,
    consumer_session: TransformersRuntimeSession,
    layers: list[int],
    hidden_dim: int,
    kvh: int,
    hd: int,
    stack: dict[str, Any],
    min_seq_len: int,
) -> dict[str, Any] | None:
    t = require_torch()
    device = str(consumer_session.target_device)
    if not device.startswith("cuda") or not layers or int(min_seq_len) <= 0:
        return None
    graph_cache = _consumer_native_proj_graph_cache(consumer_session=consumer_session)
    slot_key = _consumer_native_proj_graph_slot_key(
        consumer_session=consumer_session,
        layers=layers,
        hidden_dim=hidden_dim,
        kvh=kvh,
        hd=hd,
        proj_dtype=stack["proj_dtype"],
    )
    cached = graph_cache.get(slot_key)
    if isinstance(cached, dict) and cached.get("disabled"):
        return None
    if isinstance(cached, dict) and "graph" in cached and int(cached.get("seq_cap", 0)) >= int(min_seq_len):
        return cached

    old_entry = graph_cache.pop(slot_key, None)
    _release_consumer_native_proj_graph_entry(old_entry if isinstance(old_entry, dict) else None)

    seq_cap = _consumer_native_proj_graph_seq_cap(int(min_seq_len))
    stream = t.cuda.Stream(device=consumer_session.target_device)
    static_xb_proj = None
    static_k_pre = None
    static_v_out = None
    graph = None
    try:
        stream.wait_stream(t.cuda.current_stream(device=consumer_session.target_device))
        with t.cuda.stream(stream):
            static_xb_proj = t.empty(
                (len(layers), int(seq_cap), int(hidden_dim)),
                device=consumer_session.target_device,
                dtype=stack["proj_dtype"],
            )
            static_xb_proj.zero_()
            for _ in range(2):
                _consumer_native_kv_project_batched_kernel(
                    xb_proj=static_xb_proj,
                    stack=stack,
                    kvh=kvh,
                    hd=hd,
                )
            stream.synchronize()
            graph = t.cuda.CUDAGraph()
            with t.cuda.graph(graph):
                static_k_pre, static_v_out = _consumer_native_kv_project_batched_kernel(
                    xb_proj=static_xb_proj,
                    stack=stack,
                    kvh=kvh,
                    hd=hd,
                )
        t.cuda.current_stream(device=consumer_session.target_device).wait_stream(stream)
        entry = {
            "graph": graph,
            "static_xb_proj": static_xb_proj,
            "static_k_pre": static_k_pre,
            "static_v_out": static_v_out,
            "seq_cap": int(seq_cap),
            "last_seq_len": 0,
        }
        graph_cache[slot_key] = entry
        return entry
    except Exception as exc:
        _release_consumer_native_proj_graph_entry(
            {
                "graph": graph,
                "static_xb_proj": static_xb_proj,
                "static_k_pre": static_k_pre,
                "static_v_out": static_v_out,
            }
        )
        graph_cache[slot_key] = {
            "disabled": True,
            "error": f"{type(exc).__name__}: {exc}",
        }
        return None


def _prewarm_consumer_native_proj_runtime_artifacts(
    *,
    consumer_session: TransformersRuntimeSession,
    layers: list[int],
    seq_len: int,
    hidden_dim: int,
    kvh: int,
    hd: int,
) -> None:
    stack = _consumer_native_proj_stack(consumer_session=consumer_session, layers=layers)
    _ensure_consumer_native_proj_graph_entry(
        consumer_session=consumer_session,
        layers=layers,
        hidden_dim=hidden_dim,
        kvh=kvh,
        hd=hd,
        stack=stack,
        min_seq_len=int(seq_len),
    )


def _shared_nativekv_runtime_fastpath_enabled(patch_meta: dict[str, object] | None) -> bool:
    if patch_meta is None:
        return False
    return bool(patch_meta.get("runtime_shared_nativekv_fastpath", False))


def _project_consumer_native_kv_from_hidden_tensor_batched_runtime(
    *,
    xb_hat: Any,
    layers: list[int],
    consumer_session: TransformersRuntimeSession,
    kvh: int,
    hd: int,
) -> tuple[Any, Any]:
    t = require_torch()
    stack = _consumer_native_proj_stack(consumer_session=consumer_session, layers=layers)
    proj_dtype = stack["proj_dtype"]
    xb_proj = xb_hat.to(device=consumer_session.target_device, dtype=proj_dtype).contiguous()
    device = str(consumer_session.target_device)
    if device.startswith("cuda"):
        graph_entry = _ensure_consumer_native_proj_graph_entry(
            consumer_session=consumer_session,
            layers=layers,
            hidden_dim=int(xb_proj.shape[2]),
            kvh=kvh,
            hd=hd,
            stack=stack,
            min_seq_len=int(xb_proj.shape[1]),
        )
        if isinstance(graph_entry, dict) and "graph" in graph_entry:
            seq_len = int(xb_proj.shape[1])
            seq_cap = int(graph_entry.get("seq_cap", seq_len))
            last_seq_len = int(graph_entry.get("last_seq_len", 0))
            static_xb_proj = graph_entry["static_xb_proj"]
            static_xb_proj[:, :seq_len, :].copy_(xb_proj, non_blocking=True)
            if seq_len < last_seq_len:
                static_xb_proj[:, seq_len:last_seq_len, :].zero_()
            graph_entry["last_seq_len"] = int(seq_len)
            graph_entry["graph"].replay()
            if seq_len < seq_cap:
                return graph_entry["static_k_pre"][:, :seq_len, :], graph_entry["static_v_out"][:, :seq_len, :]
            return graph_entry["static_k_pre"], graph_entry["static_v_out"]
    return _consumer_native_kv_project_batched_kernel(
        xb_proj=xb_proj,
        stack=stack,
        kvh=kvh,
        hd=hd,
    )


def _apply_consumer_k_norm_to_flat_keys_batched_runtime(
    *,
    k_pre_flat: Any,
    layers: list[int],
    consumer_session: TransformersRuntimeSession,
    kvh: int,
    hd: int,
) -> Any:
    t = require_torch()
    stack = _consumer_native_proj_stack(consumer_session=consumer_session, layers=layers)
    weight = stack.get("k_norm_weight")
    if weight is None:
        return k_pre_flat.to(dtype=t.float32)
    norm_in = k_pre_flat.to(dtype=t.float32).reshape(len(layers), -1, kvh, hd)
    eps = stack["k_norm_eps"].view(len(layers), 1, 1, 1)
    scale = weight.view(len(layers), 1, 1, hd)
    rms = norm_in.pow(2).mean(dim=-1, keepdim=True).add(eps).rsqrt()
    return (norm_in * rms * scale).reshape(len(layers), -1, kvh * hd).contiguous()


def _apply_consumer_native_rotary_to_patch_keys_batched(
    *,
    k_states: Any,
    layers: list[int],
    consumer_session: TransformersRuntimeSession,
    hidden_dim: int,
) -> Any:
    t = require_torch()
    apply_rotary_pos_emb = _consumer_native_rotary_apply_fn(
        layer=int(layers[0]),
        consumer_session=consumer_session,
    )
    if apply_rotary_pos_emb is None:
        return re_rope_k_torch(k_states, consumer_session.rope_spec)
    seq_len = int(k_states.shape[2])
    position_embeddings = _consumer_cached_rotary_embeddings(
        consumer_session=consumer_session,
        seq_len=seq_len,
        hidden_dim=hidden_dim,
    )
    if not isinstance(position_embeddings, tuple) or len(position_embeddings) < 2:
        return re_rope_k_torch(k_states, consumer_session.rope_spec)
    cos, sin = position_embeddings
    rotary_dtype = _consumer_native_proj_stack(consumer_session=consumer_session, layers=layers)["proj_dtype"]
    k_states = k_states.to(device=consumer_session.target_device, dtype=rotary_dtype)
    with t.no_grad():
        k_rot = _apply_rotary_key_only(k_states=k_states, cos=cos, sin=sin)
    return k_rot.to(dtype=t.float32)


def _patch_stream_window_size(patch_meta: dict[str, object] | None) -> int:
    if patch_meta is None:
        return 0
    raw = patch_meta.get("runtime_patch_window_size", 0)
    if raw in {"", None, False}:
        return 0
    try:
        return max(1, int(raw))
    except Exception:
        return 0


def _patch_streaming_enabled(patch_meta: dict[str, object] | None) -> bool:
    return _patch_stream_window_size(patch_meta) > 0


def _patch_shard_manifest(patch_meta: dict[str, object] | None) -> dict[str, Any] | None:
    if patch_meta is None:
        return None
    manifest = patch_meta.get("shard_manifest")
    if isinstance(manifest, dict) and manifest:
        return manifest
    return None


def _patch_persist_shards_enabled(patch_meta: dict[str, object] | None) -> bool:
    if patch_meta is None:
        return False
    return bool(patch_meta.get("runtime_patch_persist_shards", False))


def _patch_persist_modules_enabled(patch_meta: dict[str, object] | None) -> bool:
    if patch_meta is None:
        return False
    return bool(patch_meta.get("runtime_patch_persist_modules", False))


def _patch_layer_runtime_hint(patch_meta: dict[str, object] | None, layer: int) -> dict[str, object]:
    if patch_meta is None:
        return {}
    for source in (
        patch_meta.get("runtime_layer_hints"),
        (_patch_shard_manifest(patch_meta) or {}).get("runtime_layer_hints"),
    ):
        if not isinstance(source, dict):
            continue
        hint = source.get(str(int(layer)))
        if isinstance(hint, dict):
            return hint
        hint = source.get(int(layer))
        if isinstance(hint, dict):
            return hint
    return {}


def _build_runtime_patch_layer_key_index(
    patch_bundle: dict[str, Any],
    patch_meta: dict[str, object],
) -> dict[int, list[str]]:
    cached = patch_meta.setdefault("__runtime_patch_layer_keys__", {})  # type: ignore[assignment]
    if isinstance(cached, dict) and cached:
        return cached  # type: ignore[return-value]
    layer_to_keys: dict[int, list[str]] = {}
    manifest = _patch_shard_manifest(patch_meta)
    if isinstance(manifest, dict):
        raw_layer_keys = manifest.get("layer_keys", {})
        if isinstance(raw_layer_keys, dict):
            for raw_layer, keys in raw_layer_keys.items():
                try:
                    layer = int(raw_layer)
                except Exception:
                    continue
                if not isinstance(keys, list):
                    continue
                layer_to_keys[int(layer)] = [str(key) for key in keys]
    if not layer_to_keys:
        for raw_key in patch_bundle.keys():
            key = str(raw_key)
            parts = key.split(".")
            if len(parts) < 2 or not parts[1].isdigit():
                continue
            layer_to_keys.setdefault(int(parts[1]), []).append(key)
    for keys in layer_to_keys.values():
        keys.sort()
    patch_meta["__runtime_patch_layer_keys__"] = layer_to_keys
    return layer_to_keys


def _move_runtime_patch_value_to_device(value: Any, device: str) -> Any:
    if not hasattr(value, "to"):
        return value
    t = require_torch()
    try:
        is_float = bool(value.is_floating_point())
    except Exception:
        is_float = False
    if is_float:
        moved = value.to(device=device, dtype=t.float32, non_blocking=False)
    else:
        moved = value.to(device=device, non_blocking=False)
    if is_float and hasattr(moved, "contiguous"):
        moved = moved.contiguous()
    return moved


def _clear_runtime_patch_runtime_state(patch_meta: dict[str, object] | None) -> None:
    if patch_meta is None:
        return
    runtime_window = patch_meta.get("__runtime_patch_window__", {})
    if isinstance(runtime_window, dict):
        runtime_window.clear()
    persist_modules = _patch_persist_modules_enabled(patch_meta)
    persist_shards = _patch_persist_shards_enabled(patch_meta)
    keys_to_clear = ["__runtime_patch_loaded_shards__"]
    if not persist_modules:
        keys_to_clear.extend(
            [
                "__runtime_direct_residual_modules__",
                "__runtime_xnorm_nativekv_residual_modules__",
                "__runtime_semantic_modules__",
                "__runtime_xnorm_nativekv_identity__",
            ]
        )
    if not persist_shards:
        keys_to_clear.extend(
            [
                "__runtime_patch_cpu_shard_cache__",
                "__runtime_patch_gpu_shard_cache__",
            ]
        )
    for key in keys_to_clear:
        cached = patch_meta.get(key)
        if isinstance(cached, dict):
            cached.clear()
    patch_meta["__runtime_patch_prewarmed__"] = False
    if persist_modules or persist_shards:
        return
    clear_torch_cache()


def _measure_runtime_cpu_stage_ms(fn):
    start = time.perf_counter()
    out = fn()
    return out, float((time.perf_counter() - start) * 1000.0)


def _prewarm_all_runtime_patch_shards(
    *,
    patch_meta: dict[str, object],
    consumer_session: TransformersRuntimeSession,
) -> None:
    manifest = _patch_shard_manifest(patch_meta)
    if not isinstance(manifest, dict):
        return
    root = Path(str(patch_meta.get("__patch_ckpt_root__") or ""))
    cpu_shard_cache = patch_meta.setdefault("__runtime_patch_cpu_shard_cache__", {})  # type: ignore[assignment]
    if not isinstance(cpu_shard_cache, dict):
        cpu_shard_cache = {}
        patch_meta["__runtime_patch_cpu_shard_cache__"] = cpu_shard_cache
    gpu_shard_cache = patch_meta.setdefault("__runtime_patch_gpu_shard_cache__", {})  # type: ignore[assignment]
    if not isinstance(gpu_shard_cache, dict):
        gpu_shard_cache = {}
        patch_meta["__runtime_patch_gpu_shard_cache__"] = gpu_shard_cache
    device = str(consumer_session.target_device)
    for shard in manifest.get("shards", []):
        shard_relpath = str(shard.get("path") or "")
        if not shard_relpath:
            continue
        if shard_relpath in gpu_shard_cache and isinstance(gpu_shard_cache[shard_relpath], dict):
            continue
        if shard_relpath in cpu_shard_cache and isinstance(cpu_shard_cache[shard_relpath], dict):
            cpu_arrays = cpu_shard_cache[shard_relpath]
        else:
            shard_path = root / shard_relpath
            cpu_bundle = load_bundle(shard_path, device="cpu", dtype="float32", cpu_fallback=consumer_session.cpu_fallback)
            cpu_arrays = {str(key): value for key, value in cpu_bundle.arrays.items()}
            cpu_shard_cache[shard_relpath] = cpu_arrays
        moved_arrays = {str(key): _move_runtime_patch_value_to_device(value, device) for key, value in cpu_arrays.items()}
        gpu_shard_cache[shard_relpath] = moved_arrays


def _ensure_runtime_patch_window(
    *,
    layer: int,
    patch_bundle: dict[str, Any],
    patch_meta: dict[str, object],
    consumer_session: TransformersRuntimeSession,
) -> None:
    if not _patch_streaming_enabled(patch_meta):
        return
    window_size = _patch_stream_window_size(patch_meta)
    layer_idx = int(layer)
    start = (layer_idx // window_size) * window_size
    end = start + window_size
    runtime_window = patch_meta.setdefault("__runtime_patch_window__", {})  # type: ignore[assignment]
    if not isinstance(runtime_window, dict):
        runtime_window = {}
        patch_meta["__runtime_patch_window__"] = runtime_window
    if runtime_window.get("range") == (start, end):
        return

    layer_key_index = _build_runtime_patch_layer_key_index(patch_bundle, patch_meta)
    _clear_runtime_patch_runtime_state(patch_meta)

    device = str(consumer_session.target_device)
    layer_arrays: dict[int, dict[str, Any]] = {}
    gpu_arrays: dict[str, Any] = {}
    patch_breakdown = patch_meta.setdefault("__runtime_patch_breakdown_ms__", {})  # type: ignore[assignment]
    if not isinstance(patch_breakdown, dict):
        patch_breakdown = {}
        patch_meta["__runtime_patch_breakdown_ms__"] = patch_breakdown
    patch_breakdown.setdefault("patch_shard_load_ms", 0.0)
    patch_breakdown.setdefault("patch_shard_gpu_copy_ms", 0.0)

    manifest = _patch_shard_manifest(patch_meta)
    if isinstance(manifest, dict):
        root = Path(str(patch_meta.get("__patch_ckpt_root__") or ""))
        layer_to_shard_raw = manifest.get("layer_to_shard", {})
        layer_to_shard = {
            int(raw_layer): str(shard_path)
            for raw_layer, shard_path in layer_to_shard_raw.items()
            if str(raw_layer).lstrip("-").isdigit() and str(shard_path)
        } if isinstance(layer_to_shard_raw, dict) else {}
        persist_shards = _patch_persist_shards_enabled(patch_meta)
        cpu_shard_cache = patch_meta.setdefault("__runtime_patch_cpu_shard_cache__", {})  # type: ignore[assignment]
        if not isinstance(cpu_shard_cache, dict):
            cpu_shard_cache = {}
            patch_meta["__runtime_patch_cpu_shard_cache__"] = cpu_shard_cache
        gpu_shard_cache = patch_meta.setdefault("__runtime_patch_gpu_shard_cache__", {})  # type: ignore[assignment]
        if not isinstance(gpu_shard_cache, dict):
            gpu_shard_cache = {}
            patch_meta["__runtime_patch_gpu_shard_cache__"] = gpu_shard_cache
        loaded_shards: dict[str, dict[str, Any]] = {}
        seen_shards: set[str] = set()
        for lid in range(start, end):
            shard_relpath = layer_to_shard.get(int(lid), "")
            if not shard_relpath or shard_relpath in seen_shards:
                continue
            seen_shards.add(shard_relpath)
            if shard_relpath in gpu_shard_cache:
                moved_arrays = gpu_shard_cache[shard_relpath]
                if isinstance(moved_arrays, dict):
                    loaded_shards[shard_relpath] = moved_arrays
                    gpu_arrays.update(moved_arrays)
                    continue
            if shard_relpath in cpu_shard_cache:
                cpu_arrays = cpu_shard_cache[shard_relpath]
            else:
                shard_path = root / shard_relpath
                cpu_bundle, load_ms = _measure_runtime_cpu_stage_ms(
                    lambda: load_bundle(shard_path, device="cpu", dtype="float32", cpu_fallback=consumer_session.cpu_fallback)
                )
                cpu_arrays = {str(key): value for key, value in cpu_bundle.arrays.items()}
                patch_breakdown["patch_shard_load_ms"] = float(patch_breakdown.get("patch_shard_load_ms", 0.0)) + float(load_ms)
                if persist_shards:
                    cpu_shard_cache[shard_relpath] = cpu_arrays

            def _move_shard() -> dict[str, Any]:
                moved: dict[str, Any] = {}
                for key, value in cpu_arrays.items():
                    moved[str(key)] = _move_runtime_patch_value_to_device(value, device)
                return moved

            moved_arrays, copy_ms = _measure_runtime_stage_ms(device, _move_shard)
            patch_breakdown["patch_shard_gpu_copy_ms"] = float(patch_breakdown.get("patch_shard_gpu_copy_ms", 0.0)) + float(copy_ms)
            if persist_shards:
                gpu_shard_cache[shard_relpath] = moved_arrays
            loaded_shards[shard_relpath] = moved_arrays
            gpu_arrays.update(moved_arrays)

        for lid in range(start, end):
            keys = layer_key_index.get(int(lid), [])
            if not keys:
                continue
            shard_relpath = layer_to_shard.get(int(lid), "")
            shard_arrays = loaded_shards.get(shard_relpath, {})
            layer_bundle = {key: shard_arrays[key] for key in keys if key in shard_arrays}
            if layer_bundle:
                layer_arrays[int(lid)] = layer_bundle
        patch_meta["__runtime_patch_loaded_shards__"] = loaded_shards
    else:
        for lid in range(start, end):
            keys = layer_key_index.get(int(lid), [])
            if not keys:
                continue
            layer_bundle: dict[str, Any] = {}
            for key in keys:
                moved = _move_runtime_patch_value_to_device(patch_bundle[key], device)
                layer_bundle[key] = moved
                gpu_arrays[key] = moved
            layer_arrays[int(lid)] = layer_bundle
    patch_meta["__runtime_patch_window__"] = {
        "range": (start, end),
        "layer_arrays": layer_arrays,
        "gpu_arrays": gpu_arrays,
    }


def _runtime_patch_layer_bundle(
    *,
    layer: int,
    patch_bundle: dict[str, Any],
    patch_meta: dict[str, object],
    consumer_session: TransformersRuntimeSession,
) -> dict[str, Any]:
    if not _patch_streaming_enabled(patch_meta):
        return patch_bundle
    _ensure_runtime_patch_window(
        layer=layer,
        patch_bundle=patch_bundle,
        patch_meta=patch_meta,
        consumer_session=consumer_session,
    )
    runtime_window = patch_meta.setdefault("__runtime_patch_window__", {})  # type: ignore[assignment]
    if not isinstance(runtime_window, dict):
        raise KeyError(f"runtime patch window missing for layer {int(layer)}")
    layer_arrays = runtime_window.get("layer_arrays", {})
    if not isinstance(layer_arrays, dict):
        raise KeyError(f"runtime patch layer arrays missing for layer {int(layer)}")
    layer_bundle = layer_arrays.get(int(layer), {})
    if not isinstance(layer_bundle, dict):
        raise KeyError(f"runtime patch layer bundle missing for layer {int(layer)}")
    return layer_bundle


def _runtime_patch_key_has_layer_index(key: str) -> bool:
    parts = str(key).split(".")
    return len(parts) >= 2 and str(parts[1]).lstrip("-").isdigit()


def _ensure_runtime_patch_global_arrays(
    *,
    patch_bundle: dict[str, Any],
    patch_meta: dict[str, object],
    consumer_session: TransformersRuntimeSession,
) -> dict[str, Any]:
    runtime_global = patch_meta.setdefault("__runtime_patch_global_arrays__", {})  # type: ignore[assignment]
    if not isinstance(runtime_global, dict):
        runtime_global = {}
        patch_meta["__runtime_patch_global_arrays__"] = runtime_global
    if runtime_global:
        return runtime_global

    device = str(consumer_session.target_device)
    manifest = _patch_shard_manifest(patch_meta)
    if isinstance(manifest, dict):
        global_relpath = str(manifest.get("global_path") or "")
        if global_relpath:
            root = Path(str(patch_meta.get("__patch_ckpt_root__") or ""))
            patch_breakdown = patch_meta.setdefault("__runtime_patch_breakdown_ms__", {})  # type: ignore[assignment]
            if not isinstance(patch_breakdown, dict):
                patch_breakdown = {}
                patch_meta["__runtime_patch_breakdown_ms__"] = patch_breakdown
            patch_breakdown.setdefault("patch_shard_load_ms", 0.0)
            patch_breakdown.setdefault("patch_shard_gpu_copy_ms", 0.0)
            cpu_bundle, load_ms = _measure_runtime_cpu_stage_ms(
                lambda: load_bundle(root / global_relpath, device="cpu", dtype="float32", cpu_fallback=consumer_session.cpu_fallback)
            )
            cpu_arrays = {str(key): value for key, value in cpu_bundle.arrays.items()}
            patch_breakdown["patch_shard_load_ms"] = float(patch_breakdown.get("patch_shard_load_ms", 0.0)) + float(load_ms)

            def _move_global() -> dict[str, Any]:
                moved: dict[str, Any] = {}
                for key, value in cpu_arrays.items():
                    moved[str(key)] = _move_runtime_patch_value_to_device(value, device)
                return moved

            moved_arrays, copy_ms = _measure_runtime_stage_ms(device, _move_global)
            patch_breakdown["patch_shard_gpu_copy_ms"] = float(patch_breakdown.get("patch_shard_gpu_copy_ms", 0.0)) + float(copy_ms)
            runtime_global.update(moved_arrays)
            return runtime_global

    for key, value in patch_bundle.items():
        if _runtime_patch_key_has_layer_index(str(key)):
            continue
        runtime_global[str(key)] = _move_runtime_patch_value_to_device(value, device)
    return runtime_global


def _runtime_patch_tensor(
    *,
    layer: int,
    key: str,
    patch_bundle: dict[str, Any],
    patch_meta: dict[str, object],
    consumer_session: TransformersRuntimeSession,
) -> Any:
    if _patch_streaming_enabled(patch_meta):
        layer_bundle = _runtime_patch_layer_bundle(
            layer=layer,
            patch_bundle=patch_bundle,
            patch_meta=patch_meta,
            consumer_session=consumer_session,
        )
        if key in layer_bundle:
            return layer_bundle[key]
        if key in patch_bundle:
            moved = _move_runtime_patch_value_to_device(patch_bundle[key], str(consumer_session.target_device))
            layer_bundle[key] = moved
            runtime_window = patch_meta.setdefault("__runtime_patch_window__", {})  # type: ignore[assignment]
            if isinstance(runtime_window, dict):
                gpu_arrays = runtime_window.setdefault("gpu_arrays", {})
                if isinstance(gpu_arrays, dict):
                    gpu_arrays[key] = moved
            return moved
        global_arrays = _ensure_runtime_patch_global_arrays(
            patch_bundle=patch_bundle,
            patch_meta=patch_meta,
            consumer_session=consumer_session,
        )
        if key in global_arrays:
            return global_arrays[key]
    value = patch_bundle[key]
    return _move_runtime_patch_value_to_device(value, str(consumer_session.target_device))


def _load_runtime_direct_residual_module(
    *,
    layer: int,
    patch_bundle: dict[str, Any],
    patch_meta: dict[str, object],
    consumer_session: TransformersRuntimeSession,
):
    runtime_modules = patch_meta.setdefault("__runtime_direct_residual_modules__", {})  # type: ignore[assignment]
    if not isinstance(runtime_modules, dict):
        runtime_modules = {}
        patch_meta["__runtime_direct_residual_modules__"] = runtime_modules
    if int(layer) not in runtime_modules:
        layer_bundle = _runtime_patch_layer_bundle(
            layer=layer,
            patch_bundle=patch_bundle,
            patch_meta=patch_meta,
            consumer_session=consumer_session,
        )
        runtime_modules[int(layer)] = load_direct_kv_residual_patch_layer_from_ckpt(
            layer_bundle,
            patch_meta,
            layer=int(layer),
            device=consumer_session.target_device,
            dtype="float32",
        )
    return runtime_modules[int(layer)]


def _load_runtime_xnorm_nativekv_residual_module(
    *,
    layer: int,
    patch_bundle: dict[str, Any],
    patch_meta: dict[str, object],
    consumer_session: TransformersRuntimeSession,
):
    runtime_modules = patch_meta.setdefault("__runtime_xnorm_nativekv_residual_modules__", {})  # type: ignore[assignment]
    if not isinstance(runtime_modules, dict):
        runtime_modules = {}
        patch_meta["__runtime_xnorm_nativekv_residual_modules__"] = runtime_modules
    runtime_identity = patch_meta.setdefault("__runtime_xnorm_nativekv_identity__", {})  # type: ignore[assignment]
    if not isinstance(runtime_identity, dict):
        runtime_identity = {}
        patch_meta["__runtime_xnorm_nativekv_identity__"] = runtime_identity
    if int(layer) not in runtime_modules:
        layer_bundle = _runtime_patch_layer_bundle(
            layer=layer,
            patch_bundle=patch_bundle,
            patch_meta=patch_meta,
            consumer_session=consumer_session,
        )
        residual_module = load_xnorm_nativekv_residual_patch_layer_from_ckpt(
            layer_bundle,
            patch_meta,
            layer=int(layer),
            device=consumer_session.target_device,
            dtype="float32",
        )
        runtime_modules[int(layer)] = residual_module
        delta_head = getattr(residual_module, "delta_h_head", None)
        weight = getattr(delta_head, "weight", None)
        bias = getattr(delta_head, "bias", None)
        weight_max = float(weight.detach().abs().max().item()) if weight is not None else 0.0
        bias_max = float(bias.detach().abs().max().item()) if bias is not None else 0.0
        runtime_identity[int(layer)] = max(weight_max, bias_max) <= 0.0
    elif int(layer) not in runtime_identity:
        residual_module = runtime_modules[int(layer)]
        delta_head = getattr(residual_module, "delta_h_head", None)
        weight = getattr(delta_head, "weight", None)
        bias = getattr(delta_head, "bias", None)
        weight_max = float(weight.detach().abs().max().item()) if weight is not None else 0.0
        bias_max = float(bias.detach().abs().max().item()) if bias is not None else 0.0
        runtime_identity[int(layer)] = max(weight_max, bias_max) <= 0.0
    return runtime_modules[int(layer)]


def _load_runtime_xnorm_nativekv_shared_residual_module(
    *,
    patch_bundle: dict[str, Any],
    patch_meta: dict[str, object],
    consumer_session: TransformersRuntimeSession,
):
    runtime_module = patch_meta.get("__runtime_xnorm_nativekv_shared_residual_module__")
    if runtime_module is None:
        arrays = _ensure_runtime_patch_global_arrays(
            patch_bundle=patch_bundle,
            patch_meta=patch_meta,
            consumer_session=consumer_session,
        )
        if not arrays:
            raise KeyError("shared residual patch checkpoint missing global arrays")
        runtime_module = load_xnorm_nativekv_shared_residual_patch_from_ckpt(
            arrays,
            patch_meta,
            device=consumer_session.target_device,
            dtype="float32",
        )
        patch_meta["__runtime_xnorm_nativekv_shared_residual_module__"] = runtime_module
    return runtime_module


def _runtime_xnorm_nativekv_residual_is_identity(
    *,
    layer: int,
    patch_bundle: dict[str, Any],
    patch_meta: dict[str, object],
    consumer_session: TransformersRuntimeSession,
    atol: float = 0.0,
) -> bool:
    layer_hint = _patch_layer_runtime_hint(patch_meta, int(layer))
    if "xnorm_nativekv_residual_identity" in layer_hint:
        return bool(layer_hint.get("xnorm_nativekv_residual_identity"))
    runtime_identity = patch_meta.setdefault("__runtime_xnorm_nativekv_identity__", {})  # type: ignore[assignment]
    if not isinstance(runtime_identity, dict):
        runtime_identity = {}
        patch_meta["__runtime_xnorm_nativekv_identity__"] = runtime_identity
    if int(layer) not in runtime_identity:
        _load_runtime_xnorm_nativekv_residual_module(
            layer=layer,
            patch_bundle=patch_bundle,
            patch_meta=patch_meta,
            consumer_session=consumer_session,
        )
    cached_identity = runtime_identity[int(layer)]
    if isinstance(cached_identity, bool):
        return cached_identity
    return bool(float(cached_identity) <= float(atol))


def _runtime_xnorm_nativekv_shared_residual_is_identity(
    *,
    layer: int,
    patch_bundle: dict[str, Any],
    patch_meta: dict[str, object],
    consumer_session: TransformersRuntimeSession,
    atol: float = 0.0,
) -> bool:
    layer_hint = _patch_layer_runtime_hint(patch_meta, int(layer))
    if "xnorm_nativekv_shared_residual_identity" in layer_hint:
        return bool(layer_hint.get("xnorm_nativekv_shared_residual_identity"))
    runtime_identity = patch_meta.setdefault("__runtime_xnorm_nativekv_shared_identity__", {})  # type: ignore[assignment]
    if not isinstance(runtime_identity, dict):
        runtime_identity = {}
        patch_meta["__runtime_xnorm_nativekv_shared_identity__"] = runtime_identity
    if int(layer) not in runtime_identity:
        module = _load_runtime_xnorm_nativekv_shared_residual_module(
            patch_bundle=patch_bundle,
            patch_meta=patch_meta,
            consumer_session=consumer_session,
        )
        weight = module.delta_h_weight[int(layer)]
        bias = module.delta_h_bias[int(layer)]
        runtime_identity[int(layer)] = max(
            float(weight.detach().abs().max().item()),
            float(bias.detach().abs().max().item()),
        ) <= float(atol)
    return bool(runtime_identity[int(layer)])


def _measure_runtime_stage_ms(device: str, fn):
    _synchronize_for_device(device)
    start = time.perf_counter()
    out = fn()
    _synchronize_for_device(device)
    return out, float((time.perf_counter() - start) * 1000.0)


def _apply_patch_breakdown_to_timing(
    *,
    timing: MethodTiming,
    patch_meta: dict[str, object] | None,
) -> None:
    if patch_meta is None:
        return
    breakdown = patch_meta.get("__runtime_patch_breakdown_ms__", {})
    if not isinstance(breakdown, dict):
        return
    timing.patch_shard_load_ms = float(breakdown.get("patch_shard_load_ms", 0.0))
    timing.patch_shard_gpu_copy_ms = float(breakdown.get("patch_shard_gpu_copy_ms", 0.0))
    timing.patch_identity_check_ms = float(breakdown.get("patch_identity_check_ms", 0.0))
    timing.patch_residual_forward_ms = float(breakdown.get("patch_residual_forward_ms", 0.0))
    timing.patch_kproj_vproj_ms = float(breakdown.get("patch_kproj_vproj_ms", 0.0))
    timing.patch_knorm_ms = float(breakdown.get("patch_knorm_ms", 0.0))
    timing.patch_rotary_ms = float(breakdown.get("patch_rotary_ms", 0.0))


def _apply_patch_path_timing_to_timing(
    *,
    timing: MethodTiming,
    patch_meta: dict[str, object] | None,
) -> None:
    if patch_meta is None:
        return
    path_timing = patch_meta.get("__runtime_patch_path_timing_ms__", {})
    if not isinstance(path_timing, dict):
        return
    timing.encode_ms += float(path_timing.get("encode_ms", 0.0))
    timing.staging_copy_ms += float(path_timing.get("transfer_ms", 0.0))


def _prewarm_runtime_patch_artifacts(
    *,
    patch_layers: list[int],
    patch_bundle: dict[str, Any] | None,
    patch_meta: dict[str, object] | None,
    consumer_session: TransformersRuntimeSession,
    prefix_tokens: int,
) -> None:
    if not patch_layers or patch_bundle is None or patch_meta is None:
        return
    if bool(patch_meta.get("__runtime_patch_prewarmed__", False)):
        return
    if _patch_streaming_enabled(patch_meta):
        if _patch_persist_shards_enabled(patch_meta):
            _prewarm_all_runtime_patch_shards(
                patch_meta=patch_meta,
                consumer_session=consumer_session,
            )
            patch_meta["__runtime_patch_prewarmed__"] = True
        return
    fit_mode = str(patch_meta.get("fit_mode") or "")
    residual_layers = {
        int(v)
        for v in patch_meta.get("residual_layers", [])
    } if isinstance(patch_meta.get("residual_layers", []), list) else set()
    for layer in sorted(set(int(layer) for layer in patch_layers)):
        if fit_mode == PATCH_FIT_MODE_DIRECT_KV_KNORM_RESIDUAL_MLP_V1 and int(layer) in residual_layers:
            _load_runtime_direct_residual_module(
                layer=layer,
                patch_bundle=patch_bundle,
                patch_meta=patch_meta,
                consumer_session=consumer_session,
            )
        elif fit_mode == PATCH_FIT_MODE_XNORM_NATIVEKV_RESIDUAL_MLP_V1 and int(layer) in residual_layers:
            if not _runtime_xnorm_nativekv_residual_is_identity(
                layer=layer,
                patch_bundle=patch_bundle,
                patch_meta=patch_meta,
                consumer_session=consumer_session,
            ):
                _load_runtime_xnorm_nativekv_residual_module(
                    layer=layer,
                    patch_bundle=patch_bundle,
                    patch_meta=patch_meta,
                    consumer_session=consumer_session,
                )
        elif fit_mode == PATCH_FIT_MODE_XNORM_NATIVEKV_SHARED_RESIDUAL_MLP_V1 and int(layer) in residual_layers:
            if not _runtime_xnorm_nativekv_shared_residual_is_identity(
                layer=layer,
                patch_bundle=patch_bundle,
                patch_meta=patch_meta,
                consumer_session=consumer_session,
            ):
                _load_runtime_xnorm_nativekv_shared_residual_module(
                    patch_bundle=patch_bundle,
                    patch_meta=patch_meta,
                    consumer_session=consumer_session,
                )
        elif fit_mode == PATCH_FIT_MODE_XNORM_SEMANTIC_V1:
            runtime_modules = patch_meta.setdefault("__runtime_semantic_modules__", {})  # type: ignore[assignment]
            if not isinstance(runtime_modules, dict):
                runtime_modules = {}
                patch_meta["__runtime_semantic_modules__"] = runtime_modules
            if int(layer) not in runtime_modules:
                runtime_modules[int(layer)] = load_semantic_patch_layer_from_ckpt(
                    patch_bundle,
                    patch_meta,
                    layer=int(layer),
                    device=consumer_session.target_device,
                    dtype="float32",
                )
    if fit_mode == PATCH_FIT_MODE_XNORM_NATIVEKV_SHARED_RESIDUAL_MLP_V1 and _shared_nativekv_runtime_fastpath_enabled(patch_meta):
        _consumer_native_proj_stack(
            consumer_session=consumer_session,
            layers=sorted(set(int(layer) for layer in patch_layers)),
        )
    patch_meta["__runtime_patch_prewarmed__"] = True


def _apply_semantic_patch_layer(
    *,
    layer: int,
    xa: Any,
    patch_bundle: dict[str, Any],
    patch_meta: dict[str, object],
    consumer_session: TransformersRuntimeSession,
    kvh: int,
    hd: int,
) -> tuple[Any, Any, int]:
    t = require_torch()
    runtime_modules = patch_meta.setdefault("__runtime_semantic_modules__", {})  # type: ignore[assignment]
    if not isinstance(runtime_modules, dict):
        runtime_modules = {}
        patch_meta["__runtime_semantic_modules__"] = runtime_modules
    if int(layer) not in runtime_modules:
        layer_bundle = _runtime_patch_layer_bundle(
            layer=layer,
            patch_bundle=patch_bundle,
            patch_meta=patch_meta,
            consumer_session=consumer_session,
        )
        runtime_modules[int(layer)] = load_semantic_patch_layer_from_ckpt(
            layer_bundle,
            patch_meta,
            layer=int(layer),
            device=consumer_session.target_device,
            dtype="float32",
        )
    patch_layer = runtime_modules[int(layer)]
    k_proj, v_proj, _k_norm = _get_consumer_attn_modules(consumer_session, int(layer))
    with t.no_grad():
        xa = xa.to(device=consumer_session.target_device, dtype=t.float32)
        z_h = patch_layer.encode(xa)
        xb_hat = patch_layer.patch_mlp(z_h)
        proj_dtype = getattr(getattr(k_proj, "weight", None), "dtype", xb_hat.dtype)
        k_raw_flat = k_proj(xb_hat.to(dtype=proj_dtype)).to(dtype=t.float32)
        v_flat = v_proj(xb_hat.to(dtype=proj_dtype)).to(dtype=t.float32)
    kv_dim = int(kvh * hd)
    kb_raw = k_raw_flat.reshape(1, xa.shape[0], kv_dim).reshape(1, xa.shape[0], kvh, hd).permute(0, 2, 1, 3).contiguous()
    vb = v_flat.reshape(1, xa.shape[0], kv_dim).reshape(1, xa.shape[0], kvh, hd).permute(0, 2, 1, 3).contiguous()
    return kb_raw, vb, int(z_h.numel())


def _infer_semantic_patch_hidden(
    *,
    layer: int,
    xa_prefix: Any,
    patch_bundle: dict[str, Any],
    patch_meta: dict[str, object],
    consumer_session: TransformersRuntimeSession,
) -> tuple[Any, int]:
    t = require_torch()
    runtime_modules = patch_meta.setdefault("__runtime_semantic_modules__", {})  # type: ignore[assignment]
    if not isinstance(runtime_modules, dict):
        runtime_modules = {}
        patch_meta["__runtime_semantic_modules__"] = runtime_modules
    if int(layer) not in runtime_modules:
        layer_bundle = _runtime_patch_layer_bundle(
            layer=layer,
            patch_bundle=patch_bundle,
            patch_meta=patch_meta,
            consumer_session=consumer_session,
        )
        runtime_modules[int(layer)] = load_semantic_patch_layer_from_ckpt(
            layer_bundle,
            patch_meta,
            layer=int(layer),
            device=consumer_session.target_device,
            dtype="float32",
        )
    patch_layer = runtime_modules[int(layer)]
    xa_prefix = xa_prefix.to(device=consumer_session.target_device, dtype=t.float32)
    flat = xa_prefix.reshape(-1, xa_prefix.shape[-1])
    with t.no_grad():
        z_h = patch_layer.encode(flat)
        xb_hat = patch_layer.patch_mlp(z_h).reshape_as(xa_prefix)
    return xb_hat, int(z_h.numel())


def _blend_patch_with_reuse(
    *,
    base_k: Any,
    base_v: Any,
    patch_k_raw: Any,
    patch_v: Any,
    consumer_session: TransformersRuntimeSession,
    alpha_k: float,
    alpha_v: float,
    consumer_device: str,
    consumer_dtype: Any,
) -> tuple[Any, Any]:
    t = require_torch()
    base_k_raw = de_rope_k_torch(base_k.to(device=consumer_device, dtype=t.float32), consumer_session.rope_spec)
    base_v = base_v.to(device=consumer_device, dtype=t.float32)
    blended_k_raw = base_k_raw + (float(alpha_k) * (patch_k_raw.to(dtype=t.float32) - base_k_raw))
    blended_v = base_v + (float(alpha_v) * (patch_v.to(dtype=t.float32) - base_v))
    blended_k = re_rope_k_torch(blended_k_raw, consumer_session.rope_spec)
    return blended_k.to(device=consumer_device, dtype=consumer_dtype), blended_v.to(device=consumer_device, dtype=consumer_dtype)


def _blend_native_patch_with_reuse(
    *,
    base_k: Any,
    base_v: Any,
    patch_k: Any,
    patch_v: Any,
    alpha_k: float,
    alpha_v: float,
    consumer_device: str,
    consumer_dtype: Any,
) -> tuple[Any, Any]:
    t = require_torch()
    base_k = base_k.to(device=consumer_device, dtype=t.float32)
    base_v = base_v.to(device=consumer_device, dtype=t.float32)
    patch_k = patch_k.to(device=consumer_device, dtype=t.float32)
    patch_v = patch_v.to(device=consumer_device, dtype=t.float32)
    blended_k = base_k + (float(alpha_k) * (patch_k - base_k))
    blended_v = base_v + (float(alpha_v) * (patch_v - base_v))
    return blended_k.to(device=consumer_device, dtype=consumer_dtype), blended_v.to(device=consumer_device, dtype=consumer_dtype)


def _cache_entry_has_kv(entry: Any) -> bool:
    return (
        isinstance(entry, tuple)
        and len(entry) == 2
        and entry[0] is not None
        and entry[1] is not None
    )


def _apply_patch_breakpoints_native(
    *,
    caches: list[tuple[Any, Any]],
    patch_layers: list[int],
    producer_capture: PrefillCapture,
    consumer_capture: PrefillCapture,
    consumer_session: TransformersRuntimeSession,
    patch_bundle: dict[str, Any],
    patch_meta: dict[str, object],
    prefix_tokens: int,
    hidden_dim: int,
    consumer_dtype: Any,
    consumer_device: str,
    transfer_dtype: Any,
) -> tuple[int, float, dict[int, Any]]:
    t = require_torch()
    payload_bytes = 0
    patch_hidden_by_layer: dict[int, Any] = {}
    attn_mask_prefix = consumer_capture.attention_mask[:, :prefix_tokens].to(device=consumer_device)
    _synchronize_for_device(consumer_device)
    patch_start = time.perf_counter()
    patch_apply_mode = str(patch_meta.get("patch_apply_mode") or "overwrite")
    alpha_k = float(patch_meta.get("patch_blend_alpha_k", 0.25))
    alpha_v = float(patch_meta.get("patch_blend_alpha_v", 0.5))

    for layer in sorted(set(int(layer) for layer in patch_layers)):
        xa = producer_capture.xnorm_by_layer.get(layer)
        if xa is None:
            raise ValueError(f"patch 层 {layer} 缺少 xnorm_a")
        xa_prefix = _slice_prefix_tensor(xa, prefix_tokens).to(device=consumer_device, dtype=t.float32, non_blocking=True)
        xb_hat, payload_units = _infer_semantic_patch_hidden(
            layer=layer,
            xa_prefix=xa_prefix,
            patch_bundle=patch_bundle,
            patch_meta=patch_meta,
            consumer_session=consumer_session,
        )
        payload_bytes += int(payload_units) * _dtype_size_bytes(transfer_dtype)
        layer_inputs = _build_layer_forward_inputs(consumer_session, xb_hat, attn_mask_prefix)
        next_hidden, patch_k, patch_v = _run_patch_breakpoint_native(
            consumer_session=consumer_session,
            layer_idx=layer,
            hidden_states=xb_hat,
            attention_mask=layer_inputs["attention_mask"],
            position_ids=layer_inputs["position_ids"],
            cache_position=layer_inputs["cache_position"],
            position_embeddings=layer_inputs["position_embeddings"],
        )
        if patch_apply_mode == "residual_blend" and _cache_entry_has_kv(caches[layer]):
            base_k, base_v = caches[layer]
            caches[layer] = _blend_native_patch_with_reuse(
                base_k=base_k,
                base_v=base_v,
                patch_k=patch_k,
                patch_v=patch_v,
                alpha_k=alpha_k,
                alpha_v=alpha_v,
                consumer_device=consumer_device,
                consumer_dtype=consumer_dtype,
            )
        else:
            caches[layer] = (patch_k.to(device=consumer_device, dtype=consumer_dtype), patch_v.to(device=consumer_device, dtype=consumer_dtype))
        patch_hidden_by_layer[int(layer)] = next_hidden.detach()

    _synchronize_for_device(consumer_device)
    return int(payload_bytes), float((time.perf_counter() - patch_start) * 1000.0), patch_hidden_by_layer


def _run_true_recompute_suffixes(
    *,
    caches: list[tuple[Any, Any]],
    breakpoint_hidden_by_layer: dict[int, Any],
    recompute_segment_length_by_layer: dict[int, int],
    consumer_session: TransformersRuntimeSession,
    consumer_capture: PrefillCapture,
    prefix_tokens: int,
    consumer_dtype: Any,
    consumer_device: str,
) -> float:
    recompute_ms = 0.0
    attn_mask_prefix = consumer_capture.attention_mask[:, :prefix_tokens].to(device=consumer_device)
    num_layers = len(caches)
    for layer in sorted(recompute_segment_length_by_layer):
        if int(layer) not in breakpoint_hidden_by_layer:
            continue
        seg_len = int(recompute_segment_length_by_layer[layer])
        start_layer = int(layer) + 1
        end_layer = min(int(num_layers), start_layer + seg_len)
        if start_layer >= end_layer:
            continue
        hidden = breakpoint_hidden_by_layer[int(layer)].to(device=consumer_device)
        layer_inputs = _build_layer_forward_inputs(consumer_session, hidden, attn_mask_prefix)
        _synchronize_for_device(consumer_device)
        start = time.perf_counter()
        _final_hidden, layer_caches = _run_consumer_suffix_native(
            consumer_session=consumer_session,
            start_layer=start_layer,
            end_layer_exclusive=end_layer,
            hidden_states=hidden,
            attention_mask=layer_inputs["attention_mask"],
            position_ids=layer_inputs["position_ids"],
            cache_position=layer_inputs["cache_position"],
            position_embeddings=layer_inputs["position_embeddings"],
        )
        _synchronize_for_device(consumer_device)
        recompute_ms += (time.perf_counter() - start) * 1000.0
        for lid, (k, v) in layer_caches.items():
            caches[int(lid)] = (
                k.to(device=consumer_device, dtype=consumer_dtype),
                v.to(device=consumer_device, dtype=consumer_dtype),
            )
    return float(recompute_ms)


def _apply_patch_layers_to_cache(
    *,
    caches: list[tuple[Any, Any]],
    patch_layers: list[int],
    producer_capture: PrefillCapture,
    consumer_capture: PrefillCapture,
    consumer_session: TransformersRuntimeSession,
    patch_bundle: dict[str, Any],
    patch_meta: dict[str, object],
    prefix_tokens: int,
    hidden_dim: int,
    kvh: int,
    hd: int,
    consumer_dtype: Any,
    consumer_device: str,
    transfer_dtype: Any,
) -> tuple[int, float]:
    t = require_torch()
    kv_dim = kvh * hd
    payload_bytes = 0
    patch_breakdown_ms = {
        "patch_shard_load_ms": 0.0,
        "patch_shard_gpu_copy_ms": 0.0,
        "patch_identity_check_ms": 0.0,
        "patch_residual_forward_ms": 0.0,
        "patch_kproj_vproj_ms": 0.0,
        "patch_knorm_ms": 0.0,
        "patch_rotary_ms": 0.0,
    }
    patch_meta["__runtime_patch_breakdown_ms__"] = patch_breakdown_ms
    patch_meta["__runtime_patch_path_timing_ms__"] = {
        "encode_ms": 0.0,
        "transfer_ms": 0.0,
        "patch_compute_ms": 0.0,
    }
    _synchronize_for_device(consumer_device)
    patch_start = time.perf_counter()
    fit_mode = str(patch_meta.get("fit_mode") or "")
    patch_apply_mode = str(patch_meta.get("patch_apply_mode") or "overwrite")
    alpha_k = float(patch_meta.get("patch_blend_alpha_k", 0.25))
    alpha_v = float(patch_meta.get("patch_blend_alpha_v", 0.5))
    state_target_space = str(patch_meta.get("state_target_space") or "xnorm")
    if fit_mode == PATCH_FIT_MODE_XNORM_NATIVEKV_SHARED_RESIDUAL_MLP_V1:
        return _apply_xnorm_nativekv_shared_patch_layers_to_cache(
            caches=caches,
            patch_layers=patch_layers,
            producer_capture=producer_capture,
            consumer_session=consumer_session,
            patch_bundle=patch_bundle,
            patch_meta=patch_meta,
            prefix_tokens=prefix_tokens,
            hidden_dim=hidden_dim,
            kvh=kvh,
            hd=hd,
            consumer_dtype=consumer_dtype,
            consumer_device=consumer_device,
            transfer_dtype=transfer_dtype,
        )
    for layer in sorted(set(int(layer) for layer in patch_layers)):
        _ensure_runtime_patch_window(
            layer=layer,
            patch_bundle=patch_bundle,
            patch_meta=patch_meta,
            consumer_session=consumer_session,
        )
        xa = producer_capture.xnorm_by_layer.get(layer)
        if xa is None:
            raise ValueError(f"patch 层 {layer} 缺少 xnorm_a")
        xa = _slice_prefix_tensor(xa, prefix_tokens).reshape(-1, hidden_dim).to(device=consumer_device, dtype=t.float32, non_blocking=True)
        payload_units = int(xa.numel())
        if fit_mode == PATCH_FIT_MODE_DIRECT_KV_V1:
            wk = _runtime_patch_tensor(layer=layer, key=f"direct_wk.{layer}", patch_bundle=patch_bundle, patch_meta=patch_meta, consumer_session=consumer_session)
            wv = _runtime_patch_tensor(layer=layer, key=f"direct_wv.{layer}", patch_bundle=patch_bundle, patch_meta=patch_meta, consumer_session=consumer_session)
            kb_raw_flat, vb_flat = infer_direct_kv_patch_layer_torch(xa, wk, wv)
            kb_raw = kb_raw_flat.reshape(1, prefix_tokens, kv_dim).reshape(1, prefix_tokens, kvh, hd).permute(0, 2, 1, 3).contiguous()
            vb = vb_flat.reshape(1, prefix_tokens, kv_dim).reshape(1, prefix_tokens, kvh, hd).permute(0, 2, 1, 3).contiguous()
        elif fit_mode == PATCH_FIT_MODE_DIRECT_KV_KNORM_V1:
            wk = _runtime_patch_tensor(layer=layer, key=f"direct_wk.{layer}", patch_bundle=patch_bundle, patch_meta=patch_meta, consumer_session=consumer_session)
            wv = _runtime_patch_tensor(layer=layer, key=f"direct_wv.{layer}", patch_bundle=patch_bundle, patch_meta=patch_meta, consumer_session=consumer_session)
            k_pre_flat, vb_flat = infer_direct_kv_patch_layer_torch(xa, wk, wv)
            _k_proj, _v_proj, k_norm = _get_consumer_attn_modules(consumer_session, int(layer))
            kb_norm_flat = _apply_consumer_k_norm_to_flat_keys(k_pre_flat, k_norm, kvh, hd)
            kb_norm = kb_norm_flat.reshape(1, prefix_tokens, kv_dim).reshape(1, prefix_tokens, kvh, hd).permute(0, 2, 1, 3).contiguous()
            vb = vb_flat.reshape(1, prefix_tokens, kv_dim).reshape(1, prefix_tokens, kvh, hd).permute(0, 2, 1, 3).contiguous()
            payload_bytes += int(payload_units) * _dtype_size_bytes(transfer_dtype)
            kb = _apply_consumer_native_rotary_to_patch_keys(
                k_states=kb_norm,
                layer=layer,
                consumer_session=consumer_session,
                attention_mask_prefix=consumer_capture.attention_mask,
                hidden_dim=hidden_dim,
            )
            caches[layer] = (kb.to(device=consumer_device, dtype=consumer_dtype), vb.to(device=consumer_device, dtype=consumer_dtype))
            continue
        elif fit_mode == PATCH_FIT_MODE_DIRECT_KV_KNORM_RESIDUAL_MLP_V1:
            wk = _runtime_patch_tensor(layer=layer, key=f"direct_wk.{layer}", patch_bundle=patch_bundle, patch_meta=patch_meta, consumer_session=consumer_session)
            wv = _runtime_patch_tensor(layer=layer, key=f"direct_wv.{layer}", patch_bundle=patch_bundle, patch_meta=patch_meta, consumer_session=consumer_session)
            residual_layers = [int(v) for v in patch_meta.get("residual_layers", [])] if isinstance(patch_meta.get("residual_layers", []), list) else []
            if int(layer) in residual_layers:
                residual_module = _load_runtime_direct_residual_module(
                    layer=layer,
                    patch_bundle=patch_bundle,
                    patch_meta=patch_meta,
                    consumer_session=consumer_session,
                )
                k_pre_flat, vb_flat = infer_direct_kv_residual_patch_layer_torch(xa, wk, wv, residual_module)
            else:
                k_pre_flat, vb_flat = infer_direct_kv_patch_layer_torch(xa, wk, wv)
            _k_proj, _v_proj, k_norm = _get_consumer_attn_modules(consumer_session, int(layer))
            kb_norm_flat = _apply_consumer_k_norm_to_flat_keys(k_pre_flat, k_norm, kvh, hd)
            kb_norm = kb_norm_flat.reshape(1, prefix_tokens, kv_dim).reshape(1, prefix_tokens, kvh, hd).permute(0, 2, 1, 3).contiguous()
            vb = vb_flat.reshape(1, prefix_tokens, kv_dim).reshape(1, prefix_tokens, kvh, hd).permute(0, 2, 1, 3).contiguous()
            payload_bytes += int(payload_units) * _dtype_size_bytes(transfer_dtype)
            kb = _apply_consumer_native_rotary_to_patch_keys(
                k_states=kb_norm,
                layer=layer,
                consumer_session=consumer_session,
                attention_mask_prefix=consumer_capture.attention_mask,
                hidden_dim=hidden_dim,
            )
            caches[layer] = (kb.to(device=consumer_device, dtype=consumer_dtype), vb.to(device=consumer_device, dtype=consumer_dtype))
            continue
        elif fit_mode == PATCH_FIT_MODE_XNORM_NATIVEKV_RESIDUAL_MLP_V1:
            residual_layers = [int(v) for v in patch_meta.get("residual_layers", [])] if isinstance(patch_meta.get("residual_layers", []), list) else []
            _k_proj, _v_proj, k_norm = _get_consumer_attn_modules(consumer_session, int(layer))
            if int(layer) in residual_layers:
                residual_is_identity = _runtime_xnorm_nativekv_residual_is_identity(
                    layer=layer,
                    patch_bundle=patch_bundle,
                    patch_meta=patch_meta,
                    consumer_session=consumer_session,
                )
                if residual_is_identity:
                    xb_hat = xa
                else:
                    residual_module = _load_runtime_xnorm_nativekv_residual_module(
                        layer=layer,
                        patch_bundle=patch_bundle,
                        patch_meta=patch_meta,
                        consumer_session=consumer_session,
                    )
                    xb_hat, stage_ms = _measure_runtime_stage_ms(
                        consumer_device,
                        lambda: infer_xnorm_nativekv_residual_patch_layer_torch(xa, residual_module),
                    )
                    patch_breakdown_ms["patch_residual_forward_ms"] += float(stage_ms)
                proj_dtype = getattr(getattr(_k_proj, "weight", None), "dtype", xb_hat.dtype)
                def _run_kv_proj():
                    xb_proj = xb_hat.to(device=consumer_device, dtype=proj_dtype)
                    return _k_proj(xb_proj).to(dtype=t.float32), _v_proj(xb_proj).to(dtype=t.float32)
                (k_pre_flat, vb_flat), stage_ms = _measure_runtime_stage_ms(consumer_device, _run_kv_proj)
                patch_breakdown_ms["patch_kproj_vproj_ms"] += float(stage_ms)
            else:
                wk = _runtime_patch_tensor(layer=layer, key=f"direct_wk.{layer}", patch_bundle=patch_bundle, patch_meta=patch_meta, consumer_session=consumer_session)
                wv = _runtime_patch_tensor(layer=layer, key=f"direct_wv.{layer}", patch_bundle=patch_bundle, patch_meta=patch_meta, consumer_session=consumer_session)
                k_pre_flat, vb_flat = infer_direct_kv_patch_layer_torch(xa, wk, wv)
            kb_norm_flat, stage_ms = _measure_runtime_stage_ms(
                consumer_device,
                lambda: _apply_consumer_k_norm_to_flat_keys(k_pre_flat, k_norm, kvh, hd),
            )
            patch_breakdown_ms["patch_knorm_ms"] += float(stage_ms)
            kb_norm = kb_norm_flat.reshape(1, prefix_tokens, kv_dim).reshape(1, prefix_tokens, kvh, hd).permute(0, 2, 1, 3).contiguous()
            vb = vb_flat.reshape(1, prefix_tokens, kv_dim).reshape(1, prefix_tokens, kvh, hd).permute(0, 2, 1, 3).contiguous()
            payload_bytes += int(payload_units) * _dtype_size_bytes(transfer_dtype)
            kb, stage_ms = _measure_runtime_stage_ms(
                consumer_device,
                lambda: _apply_consumer_native_rotary_to_patch_keys(
                    k_states=kb_norm,
                    layer=layer,
                    consumer_session=consumer_session,
                    attention_mask_prefix=consumer_capture.attention_mask,
                    hidden_dim=hidden_dim,
                ),
            )
            patch_breakdown_ms["patch_rotary_ms"] += float(stage_ms)
            caches[layer] = (kb.to(device=consumer_device, dtype=consumer_dtype), vb.to(device=consumer_device, dtype=consumer_dtype))
            continue
        elif fit_mode == PATCH_FIT_MODE_XNORM_SEMANTIC_V1:
            if state_target_space == "residual_pre_ln":
                xa_prefix = xa.reshape(1, prefix_tokens, hidden_dim)
                resid_hat, payload_units = _infer_semantic_patch_hidden(
                    layer=layer,
                    xa_prefix=xa_prefix,
                    patch_bundle=patch_bundle,
                    patch_meta=patch_meta,
                    consumer_session=consumer_session,
                )
                layer_inputs = _build_layer_forward_inputs(consumer_session, resid_hat, consumer_capture.attention_mask[:, :prefix_tokens])
                _next_hidden, patch_k, patch_v = _run_patch_breakpoint_native(
                    consumer_session=consumer_session,
                    layer_idx=layer,
                    hidden_states=resid_hat,
                    attention_mask=layer_inputs["attention_mask"],
                    position_ids=layer_inputs["position_ids"],
                    cache_position=layer_inputs["cache_position"],
                    position_embeddings=layer_inputs["position_embeddings"],
                )
                kb = patch_k.to(device=consumer_device, dtype=consumer_dtype)
                vb = patch_v.to(device=consumer_device, dtype=consumer_dtype)
                payload_bytes += int(payload_units) * _dtype_size_bytes(transfer_dtype)
                if patch_apply_mode == "residual_blend" and _cache_entry_has_kv(caches[layer]):
                    base_k, base_v = caches[layer]
                    caches[layer] = _blend_native_patch_with_reuse(
                        base_k=base_k,
                        base_v=base_v,
                        patch_k=kb,
                        patch_v=vb,
                        alpha_k=alpha_k,
                        alpha_v=alpha_v,
                        consumer_device=consumer_device,
                        consumer_dtype=consumer_dtype,
                    )
                else:
                    caches[layer] = (kb, vb)
                continue
            kb_raw, vb, payload_units = _apply_semantic_patch_layer(
                layer=layer,
                xa=xa,
                patch_bundle=patch_bundle,
                patch_meta=patch_meta,
                consumer_session=consumer_session,
                kvh=kvh,
                hd=hd,
            )
        else:
            align_w = _runtime_patch_tensor(layer=layer, key=f"align_w.{layer}", patch_bundle=patch_bundle, patch_meta=patch_meta, consumer_session=consumer_session)
            align_b = _runtime_patch_tensor(layer=layer, key=f"align_b.{layer}", patch_bundle=patch_bundle, patch_meta=patch_meta, consumer_session=consumer_session)
            wk = _runtime_patch_tensor(layer=layer, key=f"wk.{layer}", patch_bundle=patch_bundle, patch_meta=patch_meta, consumer_session=consumer_session)
            wv = _runtime_patch_tensor(layer=layer, key=f"wv.{layer}", patch_bundle=patch_bundle, patch_meta=patch_meta, consumer_session=consumer_session)
            xb = infer_patch_layer_torch(xa, type("_W", (), {"w": align_w, "b": align_b})())
            kb_raw_flat = xb @ wk
            vb_flat = xb @ wv
            kb_raw = kb_raw_flat.reshape(1, prefix_tokens, kv_dim).reshape(1, prefix_tokens, kvh, hd).permute(0, 2, 1, 3).contiguous()
            vb = vb_flat.reshape(1, prefix_tokens, kv_dim).reshape(1, prefix_tokens, kvh, hd).permute(0, 2, 1, 3).contiguous()
        payload_bytes += int(payload_units) * _dtype_size_bytes(transfer_dtype)
        if (
            fit_mode == PATCH_FIT_MODE_XNORM_SEMANTIC_V1
            and patch_apply_mode == "residual_blend"
            and _cache_entry_has_kv(caches[layer])
        ):
            base_k, base_v = caches[layer]
            caches[layer] = _blend_patch_with_reuse(
                base_k=base_k,
                base_v=base_v,
                patch_k_raw=kb_raw,
                patch_v=vb,
                consumer_session=consumer_session,
                alpha_k=alpha_k,
                alpha_v=alpha_v,
                consumer_device=consumer_device,
                consumer_dtype=consumer_dtype,
            )
        else:
            kb = re_rope_k_torch(kb_raw, consumer_session.rope_spec)
            caches[layer] = (kb.to(device=consumer_device, dtype=consumer_dtype), vb.to(device=consumer_device, dtype=consumer_dtype))
    _synchronize_for_device(consumer_device)
    patch_infer_ms = (time.perf_counter() - patch_start) * 1000.0
    patch_path_timing = patch_meta.get("__runtime_patch_path_timing_ms__", {})
    if isinstance(patch_path_timing, dict):
        patch_path_timing["patch_compute_ms"] = float(patch_infer_ms)
    return int(payload_bytes), float(patch_infer_ms)


def _assemble_method_cache(
    *,
    method: str,
    producer_capture: PrefillCapture,
    consumer_capture: PrefillCapture,
    producer_session: TransformersRuntimeSession,
    consumer_session: TransformersRuntimeSession,
    route: dict[str, list[int]],
    reuse_prod,
    reuse_cons,
    patch_bundle: dict[str, Any] | None,
    patch_meta: dict[str, object] | None,
    bandwidth_gbps: float,
    quant_group_size: int,
    transfer_dtype: Any,
    default_recompute_segment_length: int = 0,
    override_route_recompute_length: int | None = None,
    fail_on_missing_route_fields: bool = False,
    fail_on_missing_patch_fields: bool = False,
) -> tuple[list[tuple[Any, Any]], MethodTiming, float]:
    t = require_torch()
    prefix_tokens = int(producer_capture.input_ids.shape[1]) - 1
    if prefix_tokens <= 0:
        raise ValueError("严格运行时要求 prompt 至少包含两个 token")

    consumer_device = str(consumer_session.target_device)
    consumer_dtype = consumer_session.model.dtype
    route = route or {"reuse_layers": [], "patch_layers": [], "fallback_layers": []}
    reuse_set = set(int(x) for x in route.get("reuse_layers", []))
    patch_set = set(int(x) for x in route.get("patch_layers", []))
    fallback_set = set(int(x) for x in route.get("fallback_layers", []))
    num_layers = len(producer_capture.past_key_values)
    raw_cache_bytes = _raw_cache_payload_bytes(producer_capture, prefix_tokens, transfer_dtype)
    if method in {"patch_only", "reuse_patch", "scd_full"}:
        _prewarm_runtime_patch_artifacts(
            patch_layers=sorted(patch_set),
            patch_bundle=patch_bundle,
            patch_meta=patch_meta,
            consumer_session=consumer_session,
            prefix_tokens=prefix_tokens,
        )

    timing = MethodTiming()
    caches: list[tuple[Any, Any]] = [None for _ in range(num_layers)]  # type: ignore[list-item]
    payload_bytes = 0
    timing.raw_cache_bytes = int(raw_cache_bytes)

    if method in {"oracle", "full_consumer"}:
        timing.consumer_prefill_ms = float(consumer_capture.prefill_ms)
        timing.ttft_ms = float(consumer_capture.prefill_ms)
        timing.compression_ratio = None
        return _move_layers_to_device(consumer_capture.past_key_values, consumer_device, consumer_dtype), timing, 0.0

    if method == "raw_kv":
        source_layers: list[tuple[Any, Any]] = []
        for layer in range(num_layers):
            k_src = _map_k_between_sessions(producer_capture.past_key_values[layer][0], prefix_tokens, producer_session, consumer_session)
            v_src = _slice_prefix_tensor(producer_capture.past_key_values[layer][1], prefix_tokens)
            source_layers.append((k_src, v_src))
            payload_bytes += k_src.numel() * _dtype_size_bytes(transfer_dtype) + v_src.numel() * _dtype_size_bytes(transfer_dtype)
        _synchronize_for_device(consumer_device)
        copy_start = time.perf_counter()
        for layer in range(num_layers):
            k_src, v_src = source_layers[layer]
            caches[layer] = (
                k_src.to(device=consumer_device, dtype=consumer_dtype, non_blocking=True),
                v_src.to(device=consumer_device, dtype=consumer_dtype, non_blocking=True),
            )
        _synchronize_for_device(consumer_device)
        timing.staging_copy_ms = (time.perf_counter() - copy_start) * 1000.0
    elif method == "quantized_4bit":
        _synchronize_for_device(producer_capture.past_key_values[0][0].device)
        enc_start = time.perf_counter()
        quantized_layers: list[tuple[Any, Any]] = []
        for layer in range(num_layers):
            mapped_k = _map_k_between_sessions(producer_capture.past_key_values[layer][0], prefix_tokens, producer_session, consumer_session)
            qk, bytes_k = _quantize_dequantize_int4(mapped_k, group_size=quant_group_size, scale_dtype=transfer_dtype)
            qv, bytes_v = _quantize_dequantize_int4(_slice_prefix_tensor(producer_capture.past_key_values[layer][1], prefix_tokens), group_size=quant_group_size, scale_dtype=transfer_dtype)
            quantized_layers.append((qk, qv))
            payload_bytes += bytes_k + bytes_v
        _synchronize_for_device(producer_capture.past_key_values[0][0].device)
        timing.encode_ms = (time.perf_counter() - enc_start) * 1000.0

        _synchronize_for_device(consumer_device)
        copy_start = time.perf_counter()
        for layer in range(num_layers):
            qk, qv = quantized_layers[layer]
            caches[layer] = (qk.to(device=consumer_device, dtype=consumer_dtype, non_blocking=True), qv.to(device=consumer_device, dtype=consumer_dtype, non_blocking=True))
        _synchronize_for_device(consumer_device)
        timing.staging_copy_ms = (time.perf_counter() - copy_start) * 1000.0
    elif method == "selective_recompute":
        recompute_set = patch_set | fallback_set
        source_layers: list[tuple[Any, Any]] = []
        for layer in range(num_layers):
            if layer in reuse_set:
                k_src = _map_k_between_sessions(producer_capture.past_key_values[layer][0], prefix_tokens, producer_session, consumer_session)
                v_src = _slice_prefix_tensor(producer_capture.past_key_values[layer][1], prefix_tokens)
                payload_bytes += k_src.numel() * _dtype_size_bytes(transfer_dtype) + v_src.numel() * _dtype_size_bytes(transfer_dtype)
            else:
                k_src = _slice_prefix_tensor(consumer_capture.past_key_values[layer][0], prefix_tokens)
                v_src = _slice_prefix_tensor(consumer_capture.past_key_values[layer][1], prefix_tokens)
            source_layers.append((k_src, v_src))
        _synchronize_for_device(consumer_device)
        copy_start = time.perf_counter()
        for layer in range(num_layers):
            k_src, v_src = source_layers[layer]
            caches[layer] = (
                k_src.to(device=consumer_device, dtype=consumer_dtype, non_blocking=True),
                v_src.to(device=consumer_device, dtype=consumer_dtype, non_blocking=True),
            )
        _synchronize_for_device(consumer_device)
        timing.staging_copy_ms = (time.perf_counter() - copy_start) * 1000.0
        timing.recompute_ms = float(sum(consumer_capture.layer_ms[layer] for layer in recompute_set if 0 <= layer < len(consumer_capture.layer_ms)))
    elif method == "patch_only":
        runtime_route = _normalize_runtime_route(
            route,
            mode="patch_only",
            num_layers=num_layers,
            default_recompute_segment_length=default_recompute_segment_length,
            override_route_recompute_length=override_route_recompute_length,
            fail_on_missing_route_fields=fail_on_missing_route_fields,
            fail_on_missing_patch_fields=fail_on_missing_patch_fields,
        )
        runtime_patch_layers = [int(layer) for layer in runtime_route["patch_layers"]]
        if sorted(runtime_patch_layers) != list(range(num_layers)):
            raise ValueError("patch_only 需要 patch_layers 完整覆盖所有层")
        if patch_bundle is None or patch_meta is None:
            raise ValueError("patch_only 缺少 patch 检查点")
        hidden_dim = int(patch_meta["hidden_dim"])
        kvh = int(patch_meta["num_key_value_heads"])
        hd = int(patch_meta["head_dim"])
        caches = [None for _ in range(num_layers)]  # type: ignore[list-item]
        payload_bytes, patch_infer_ms = _apply_patch_layers_to_cache(
            caches=caches,
            patch_layers=runtime_patch_layers,
            producer_capture=producer_capture,
            consumer_capture=consumer_capture,
            consumer_session=consumer_session,
            patch_bundle=patch_bundle,
            patch_meta=patch_meta,
            prefix_tokens=prefix_tokens,
            hidden_dim=hidden_dim,
            kvh=kvh,
            hd=hd,
            consumer_dtype=consumer_dtype,
            consumer_device=consumer_device,
            transfer_dtype=transfer_dtype,
        )
        timing.patch_infer_ms = float(patch_infer_ms)
        _apply_patch_breakdown_to_timing(timing=timing, patch_meta=patch_meta)
        _apply_patch_path_timing_to_timing(timing=timing, patch_meta=patch_meta)
    elif method in {"reuse_only", "scd_full", "reuse_fallback", "reuse_patch", "reuse_patch_recompute", "oracle_xnorm_suffix", "oracle_state_suffix", "oracle_kv_patchonly"}:
        if reuse_prod is None or reuse_cons is None:
            raise ValueError(f"{method} 缺少 reuse 检查点")
        runtime_route: dict[str, Any] | None = None
        if method == "reuse_only":
            runtime_route = _normalize_runtime_route(
                route,
                mode="reuse_only",
                num_layers=num_layers,
                default_recompute_segment_length=default_recompute_segment_length,
                override_route_recompute_length=override_route_recompute_length,
                fail_on_missing_route_fields=fail_on_missing_route_fields,
                fail_on_missing_patch_fields=fail_on_missing_patch_fields,
            )
        elif method == "reuse_fallback":
            runtime_route = _normalize_runtime_route(
                route,
                mode="reuse_fallback",
                num_layers=num_layers,
                default_recompute_segment_length=default_recompute_segment_length,
                override_route_recompute_length=override_route_recompute_length,
                fail_on_missing_route_fields=fail_on_missing_route_fields,
                fail_on_missing_patch_fields=fail_on_missing_patch_fields,
            )
        elif method in {"scd_full", "reuse_patch", "oracle_kv_patchonly"}:
            runtime_route = _normalize_runtime_route(
                route,
                mode="reuse_patch",
                num_layers=num_layers,
                default_recompute_segment_length=default_recompute_segment_length,
                override_route_recompute_length=override_route_recompute_length,
                fail_on_missing_route_fields=fail_on_missing_route_fields,
                fail_on_missing_patch_fields=fail_on_missing_patch_fields,
            )
        elif method in {"reuse_patch_recompute", "oracle_xnorm_suffix", "oracle_state_suffix"}:
            runtime_route = _normalize_runtime_route(
                route,
                mode="reuse_patch_recompute",
                num_layers=num_layers,
                default_recompute_segment_length=default_recompute_segment_length,
                override_route_recompute_length=override_route_recompute_length,
                fail_on_missing_route_fields=fail_on_missing_route_fields,
                fail_on_missing_patch_fields=fail_on_missing_patch_fields,
            )
        decoded_layers, payload_bytes, reuse_decode_timing = _decode_reuse_layers(
            producer_capture=producer_capture,
            producer_session=producer_session,
            consumer_session=consumer_session,
            reuse_prod=reuse_prod,
            reuse_cons=reuse_cons,
            prefix_tokens=prefix_tokens,
            transfer_dtype=transfer_dtype,
            consumer_dtype=consumer_dtype,
            consumer_device=consumer_device,
            selected_reuse_layers=None if runtime_route is None else [int(layer) for layer in runtime_route["reuse_layers"]],
        )
        timing.encode_ms = reuse_decode_timing.encode_ms
        timing.staging_copy_ms = reuse_decode_timing.staging_copy_ms
        timing.rebuild_ms = reuse_decode_timing.rebuild_ms
        timing.rebuild_prep_ms = reuse_decode_timing.rebuild_prep_ms
        timing.rebuild_decode_core_ms = reuse_decode_timing.rebuild_decode_core_ms
        timing.rebuild_rerope_ms = reuse_decode_timing.rebuild_rerope_ms
        timing.rebuild_materialize_ms = reuse_decode_timing.rebuild_materialize_ms
        caches = decoded_layers
        if method in {"reuse_fallback", "reuse_patch", "reuse_patch_recompute"}:
            assert runtime_route is not None
            runtime_patch_layers = [int(layer) for layer in runtime_route["patch_layers"]]
            runtime_fallback_layers = [int(layer) for layer in runtime_route["fallback_layers"]]
            runtime_recompute_covered = [int(layer) for layer in runtime_route.get("recompute_covered_layers", [])]
            if method in {"reuse_patch", "reuse_patch_recompute"}:
                if patch_bundle is None or patch_meta is None:
                    raise ValueError(f"{method} 缺少 patch 检查点")
                hidden_dim = int(patch_meta["hidden_dim"])
                kvh = int(patch_meta["num_key_value_heads"])
                hd = int(patch_meta["head_dim"])
                fit_mode = str(patch_meta.get("fit_mode") or "")
                if method == "reuse_patch_recompute" and fit_mode == PATCH_FIT_MODE_XNORM_SEMANTIC_V1:
                    extra_payload_bytes, patch_infer_ms, breakpoint_hidden_by_layer = _apply_patch_breakpoints_native(
                        caches=caches,
                        patch_layers=runtime_patch_layers,
                        producer_capture=producer_capture,
                        consumer_capture=consumer_capture,
                        consumer_session=consumer_session,
                        patch_bundle=patch_bundle,
                        patch_meta=patch_meta,
                        prefix_tokens=prefix_tokens,
                        hidden_dim=hidden_dim,
                        consumer_dtype=consumer_dtype,
                        consumer_device=consumer_device,
                        transfer_dtype=transfer_dtype,
                    )
                    timing.recompute_ms = _run_true_recompute_suffixes(
                        caches=caches,
                        breakpoint_hidden_by_layer=breakpoint_hidden_by_layer,
                        recompute_segment_length_by_layer=runtime_route["recompute_segment_length_by_layer"],
                        consumer_session=consumer_session,
                        consumer_capture=consumer_capture,
                        prefix_tokens=prefix_tokens,
                        consumer_dtype=consumer_dtype,
                        consumer_device=consumer_device,
                    )
                else:
                    extra_payload_bytes, patch_infer_ms = _apply_patch_layers_to_cache(
                        caches=caches,
                        patch_layers=runtime_patch_layers,
                        producer_capture=producer_capture,
                        consumer_capture=consumer_capture,
                        consumer_session=consumer_session,
                        patch_bundle=patch_bundle,
                        patch_meta=patch_meta,
                        prefix_tokens=prefix_tokens,
                        hidden_dim=hidden_dim,
                        kvh=kvh,
                        hd=hd,
                        consumer_dtype=consumer_dtype,
                        consumer_device=consumer_device,
                        transfer_dtype=transfer_dtype,
                    )
                    if runtime_recompute_covered:
                        _overlay_consumer_layers(
                            caches,
                            consumer_capture=consumer_capture,
                            layers=runtime_recompute_covered,
                            prefix_tokens=prefix_tokens,
                            consumer_device=consumer_device,
                            consumer_dtype=consumer_dtype,
                        )
                        timing.recompute_ms += float(
                            sum(
                                consumer_capture.layer_ms[layer]
                                for layer in runtime_recompute_covered
                                if 0 <= int(layer) < len(consumer_capture.layer_ms)
                            )
                        )
                payload_bytes += int(extra_payload_bytes)
                timing.patch_infer_ms = float(patch_infer_ms)
                _apply_patch_breakdown_to_timing(timing=timing, patch_meta=patch_meta)
                _apply_patch_path_timing_to_timing(timing=timing, patch_meta=patch_meta)
            if runtime_fallback_layers:
                _overlay_consumer_layers(
                    caches,
                    consumer_capture=consumer_capture,
                    layers=runtime_fallback_layers,
                    prefix_tokens=prefix_tokens,
                    consumer_device=consumer_device,
                    consumer_dtype=consumer_dtype,
                )
                timing.recompute_ms += float(
                    sum(
                        consumer_capture.layer_ms[layer]
                        for layer in runtime_fallback_layers
                        if 0 <= int(layer) < len(consumer_capture.layer_ms)
                    )
                )
        elif method == "scd_full":
            assert runtime_route is not None
            if patch_bundle is None or patch_meta is None:
                raise ValueError("scd_full 缺少 patch 检查点")
            hidden_dim = int(patch_meta["hidden_dim"])
            kvh = int(patch_meta["num_key_value_heads"])
            hd = int(patch_meta["head_dim"])
            runtime_patch_layers = [int(layer) for layer in runtime_route["patch_layers"]]
            runtime_fallback_layers = [int(layer) for layer in runtime_route["fallback_layers"]]
            extra_payload_bytes, patch_infer_ms = _apply_patch_layers_to_cache(
                caches=caches,
                patch_layers=runtime_patch_layers,
                producer_capture=producer_capture,
                consumer_capture=consumer_capture,
                consumer_session=consumer_session,
                patch_bundle=patch_bundle,
                patch_meta=patch_meta,
                prefix_tokens=prefix_tokens,
                hidden_dim=hidden_dim,
                kvh=kvh,
                hd=hd,
                consumer_dtype=consumer_dtype,
                consumer_device=consumer_device,
                transfer_dtype=transfer_dtype,
            )
            payload_bytes += int(extra_payload_bytes)
            timing.patch_infer_ms = float(patch_infer_ms)
            _apply_patch_breakdown_to_timing(timing=timing, patch_meta=patch_meta)
            _apply_patch_path_timing_to_timing(timing=timing, patch_meta=patch_meta)
            _overlay_consumer_layers(
                caches,
                consumer_capture=consumer_capture,
                layers=runtime_fallback_layers,
                prefix_tokens=prefix_tokens,
                consumer_device=consumer_device,
                consumer_dtype=consumer_dtype,
            )
            timing.recompute_ms = float(sum(consumer_capture.layer_ms[layer] for layer in runtime_fallback_layers if 0 <= layer < len(consumer_capture.layer_ms)))
        elif method == "oracle_kv_patchonly":
            assert runtime_route is not None
            runtime_patch_layers = [int(layer) for layer in runtime_route["patch_layers"]]
            _overlay_consumer_layers(
                caches,
                consumer_capture=consumer_capture,
                layers=runtime_patch_layers,
                prefix_tokens=prefix_tokens,
                consumer_device=consumer_device,
                consumer_dtype=consumer_dtype,
            )
            payload_bytes += _oracle_cache_payload_bytes(
                consumer_capture,
                layers=runtime_patch_layers,
                prefix_tokens=prefix_tokens,
                transfer_dtype=transfer_dtype,
            )
        elif method in {"oracle_xnorm_suffix", "oracle_state_suffix"}:
            assert runtime_route is not None
            runtime_patch_layers = [int(layer) for layer in runtime_route["patch_layers"]]
            state_kind = "xnorm" if method == "oracle_xnorm_suffix" else "residual"
            extra_payload_bytes, _breakpoint_ms, breakpoint_hidden_by_layer = _apply_oracle_breakpoints_native(
                caches=caches,
                patch_layers=runtime_patch_layers,
                consumer_capture=consumer_capture,
                consumer_session=consumer_session,
                prefix_tokens=prefix_tokens,
                consumer_dtype=consumer_dtype,
                consumer_device=consumer_device,
                transfer_dtype=transfer_dtype,
                state_kind=state_kind,
            )
            payload_bytes += int(extra_payload_bytes)
            timing.recompute_ms = _run_true_recompute_suffixes(
                caches=caches,
                breakpoint_hidden_by_layer=breakpoint_hidden_by_layer,
                recompute_segment_length_by_layer=runtime_route["recompute_segment_length_by_layer"],
                consumer_session=consumer_session,
                consumer_capture=consumer_capture,
                prefix_tokens=prefix_tokens,
                consumer_dtype=consumer_dtype,
                consumer_device=consumer_device,
            )
    else:
        raise ValueError(f"未知方法: {method}")

    timing.transfer_ms = float(timing.staging_copy_ms)
    timing.bytes_transferred = int(payload_bytes)
    timing.compression_ratio = _compression_ratio(raw_cache_bytes, payload_bytes)
    payload_mb = float(payload_bytes) / (1024.0 * 1024.0)
    return caches, timing, payload_mb


def evaluate_cmrc_method(
    *,
    method: str,
    sample: CMRCSample,
    producer_capture: PrefillCapture,
    consumer_capture: PrefillCapture,
    producer_session: TransformersRuntimeSession,
    consumer_session: TransformersRuntimeSession,
    route: dict[str, list[int]],
    reuse_prod,
    reuse_cons,
    patch_bundle: dict[str, Any] | None,
    patch_meta: dict[str, object] | None,
    bandwidth_gbps: float,
    max_new_tokens: int,
    quant_group_size: int,
    transfer_dtype: Any,
    default_recompute_segment_length: int = 0,
    override_route_recompute_length: int | None = None,
    fail_on_missing_route_fields: bool = False,
    fail_on_missing_patch_fields: bool = False,
    measure_teacher_forced_metrics: bool = True,
    align_full_consumer_first_decode: bool = False,
) -> MethodSampleResult:
    prefix_tokens = int(consumer_capture.input_ids.shape[1]) - 1
    target_ids = _target_answer_ids(consumer_session, sample.answers, max_new_tokens=max_new_tokens)
    if method in {"oracle", "full_consumer"}:
        answer, first_decode_ms, decode_total_ms, first_token_id = consumer_session.generate_from_prefill(
            consumer_capture,
            max_new_tokens=max_new_tokens,
            align_first_decode_to_cache=align_full_consumer_first_decode,
        )
        avg_logp, avg_nll, teacher_token_count = _teacher_forced_from_prefill(consumer_session, consumer_capture, target_ids) if measure_teacher_forced_metrics else (0.0, 0.0, 0)
        timing = MethodTiming(
            producer_prefill_ms=0.0,
            consumer_prefill_ms=float(consumer_capture.prefill_ms),
            encode_ms=0.0,
            transfer_ms=0.0,
            rebuild_ms=0.0,
            rebuild_prep_ms=0.0,
            rebuild_decode_core_ms=0.0,
            rebuild_rerope_ms=0.0,
            rebuild_materialize_ms=0.0,
            patch_infer_ms=0.0,
            recompute_ms=0.0,
            decode_first_ms=float(first_decode_ms),
            decode_rest_ms=max(0.0, float(decode_total_ms - first_decode_ms)),
            decode_total_ms=float(decode_total_ms),
            ttft_ms=float(consumer_capture.prefill_ms + first_decode_ms),
            total_ms=float(consumer_capture.prefill_ms + decode_total_ms),
            bytes_transferred=0,
            raw_cache_bytes=_raw_cache_payload_bytes(consumer_capture, prefix_tokens, transfer_dtype),
            compression_ratio=None,
        )
        payload_mb = 0.0
    else:
        cache_layers, timing, payload_mb = _assemble_method_cache(
            method=method,
            producer_capture=producer_capture,
            consumer_capture=consumer_capture,
            producer_session=producer_session,
            consumer_session=consumer_session,
            route=route,
            reuse_prod=reuse_prod,
            reuse_cons=reuse_cons,
            patch_bundle=patch_bundle,
            patch_meta=patch_meta,
            bandwidth_gbps=bandwidth_gbps,
            quant_group_size=quant_group_size,
            transfer_dtype=transfer_dtype,
            default_recompute_segment_length=default_recompute_segment_length,
            override_route_recompute_length=override_route_recompute_length,
            fail_on_missing_route_fields=fail_on_missing_route_fields,
            fail_on_missing_patch_fields=fail_on_missing_patch_fields,
        )
        answer, first_decode_ms, decode_total_ms, first_token_id = consumer_session.generate_from_cache(
            cache_layers=cache_layers,
            prompt_input_ids=consumer_capture.input_ids,
            prompt_attention_mask=consumer_capture.attention_mask,
            prefix_tokens=prefix_tokens,
            max_new_tokens=max_new_tokens,
        )
        avg_logp, avg_nll, teacher_token_count = (
            _teacher_forced_from_cache(
                consumer_session,
                cache_layers=cache_layers,
                prompt_input_ids=consumer_capture.input_ids,
                prompt_attention_mask=consumer_capture.attention_mask,
                prefix_tokens=prefix_tokens,
                target_ids=target_ids,
            )
            if measure_teacher_forced_metrics
            else (0.0, 0.0, 0)
        )
        timing.decode_first_ms = float(first_decode_ms)
        timing.decode_rest_ms = max(0.0, float(decode_total_ms - first_decode_ms))
        timing.decode_total_ms = float(decode_total_ms)
        timing.ttft_ms = float(timing.encode_ms + timing.transfer_ms + timing.rebuild_ms + timing.patch_infer_ms + timing.recompute_ms + timing.decode_first_ms)
        timing.total_ms = float(timing.encode_ms + timing.transfer_ms + timing.rebuild_ms + timing.patch_infer_ms + timing.recompute_ms + timing.decode_total_ms)

    score = score_cmrc_prediction(answer, sample.answers)
    return MethodSampleResult(
        qid=sample.qid,
        method=method,
        pred_text=answer,
        f1=float(score["f1"]),
        em=float(score["em"]),
        payload_mb=float(payload_mb),
        timing=timing,
        gold_answers=list(sample.answers),
        prompt_tokens=int(consumer_capture.input_ids.shape[1]),
        prefix_tokens=prefix_tokens,
        first_token_id=first_token_id,
        avg_logp=float(avg_logp),
        avg_nll=float(avg_nll),
        teacher_logp_sum=float(avg_logp * max(1, int(teacher_token_count))) if int(teacher_token_count) > 0 else 0.0,
        teacher_token_count=int(teacher_token_count),
    )


def summarize_method_results(results: list[MethodSampleResult]) -> dict[str, Any]:
    if not results:
        raise ValueError("method results 为空")
    n = len(results)
    mean = lambda fn: float(sum(fn(x) for x in results) / n)
    total_teacher_tokens = int(sum(int(x.teacher_token_count) for x in results))
    total_teacher_logp = float(sum(float(x.teacher_logp_sum) for x in results))
    avg_logp = float(total_teacher_logp / total_teacher_tokens) if total_teacher_tokens > 0 else 0.0
    avg_nll = float(-avg_logp)
    total_raw_bytes = int(sum(int(x.timing.raw_cache_bytes) for x in results))
    total_transfer_bytes = int(sum(int(x.timing.bytes_transferred) for x in results))
    return {
        "method": results[0].method,
        "method_label": _METHOD_LABELS.get(results[0].method, results[0].method),
        "num_samples": n,
        "f1": mean(lambda x: x.f1),
        "em": mean(lambda x: x.em),
        "avg_logp": avg_logp,
        "avg_nll": avg_nll,
        "payload_mb": mean(lambda x: x.payload_mb),
        "ttft_ms": mean(lambda x: x.timing.ttft_ms),
        "producer_prefill_ms": mean(lambda x: x.timing.producer_prefill_ms),
        "consumer_prefill_ms": mean(lambda x: x.timing.consumer_prefill_ms),
        "encode_ms": mean(lambda x: x.timing.encode_ms),
        "transfer_ms": mean(lambda x: x.timing.transfer_ms),
        "staging_copy_ms": mean(lambda x: x.timing.staging_copy_ms),
        "rebuild_ms": mean(lambda x: x.timing.rebuild_ms),
        "rebuild_prep_ms": mean(lambda x: x.timing.rebuild_prep_ms),
        "rebuild_decode_core_ms": mean(lambda x: x.timing.rebuild_decode_core_ms),
        "rebuild_rerope_ms": mean(lambda x: x.timing.rebuild_rerope_ms),
        "rebuild_materialize_ms": mean(lambda x: x.timing.rebuild_materialize_ms),
        "patch_infer_ms": mean(lambda x: x.timing.patch_infer_ms),
        "patch_shard_load_ms": mean(lambda x: x.timing.patch_shard_load_ms),
        "patch_shard_gpu_copy_ms": mean(lambda x: x.timing.patch_shard_gpu_copy_ms),
        "patch_identity_check_ms": mean(lambda x: x.timing.patch_identity_check_ms),
        "patch_residual_forward_ms": mean(lambda x: x.timing.patch_residual_forward_ms),
        "patch_kproj_vproj_ms": mean(lambda x: x.timing.patch_kproj_vproj_ms),
        "patch_knorm_ms": mean(lambda x: x.timing.patch_knorm_ms),
        "patch_rotary_ms": mean(lambda x: x.timing.patch_rotary_ms),
        "recompute_ms": mean(lambda x: x.timing.recompute_ms),
        "decode_first_ms": mean(lambda x: x.timing.decode_first_ms),
        "decode_rest_ms": mean(lambda x: x.timing.decode_rest_ms),
        "decode_total_ms": mean(lambda x: x.timing.decode_total_ms),
        "total_ms": mean(lambda x: x.timing.total_ms),
        "bytes_transferred": float(total_transfer_bytes / max(1, n)),
        "raw_cache_bytes": float(total_raw_bytes / max(1, n)),
        "compression_ratio": None if total_transfer_bytes <= 0 else float(total_raw_bytes / total_transfer_bytes),
        "teacher_token_count": total_teacher_tokens,
    }
