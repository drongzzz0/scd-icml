from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterator, Protocol

from .torch_utils import clear_torch_cache, require_torch, resolve_device, resolve_torch_dtype


@dataclass
class CaptureRequest:
    model_path: str
    prompt: str | list[str]
    device: str
    dtype: str
    max_length: int | None = None


class KVAdapter(Protocol):
    def capture(self, req: CaptureRequest):
        """抓取每层的 k/v/xnorm。"""


def _get_decoder_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise RuntimeError("无法从模型中定位 decoder layers")


def _resolve_prompts(prompt: str | list[str]) -> list[str]:
    if isinstance(prompt, list):
        out = [str(x).strip() for x in prompt if str(x).strip()]
        if out:
            return out
    else:
        s = str(prompt).strip()
        if s:
            return [s]
    return ["请解释语义缓存蒸馏。"]


def is_cuda_oom(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "out of memory" in text or "cuda oom" in text


def _resolve_hidden_states(args, kwargs):
    if args:
        return args[0]
    if kwargs:
        hidden_states = kwargs.get("hidden_states")
        if hidden_states is not None:
            return hidden_states
    return None


class TransformersCaptureSession:
    def __init__(self, model_path: str, device: str, dtype: str, cpu_fallback: bool = True):
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as exc:
            raise RuntimeError("缺少运行依赖: transformers") from exc

        self.torch = require_torch()
        self.device = str(resolve_device(device, cpu_fallback=cpu_fallback))
        self.dtype = dtype
        self.torch_dtype = resolve_torch_dtype(dtype)
        self.model_path = model_path
        self.cpu_fallback = cpu_fallback
        self.capture_store_device = str(os.environ.get("SCD_CAPTURE_STORE_DEVICE", "model") or "model").strip().lower()

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        load_kwargs = {
            "torch_dtype": self.torch_dtype,
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }
        if self.device.startswith("cuda"):
            load_kwargs["device_map"] = {"": self.device}

        self.model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
        if self.device == "cpu":
            self.model.to(self.device)
        self.model.eval()
        self.layers = _get_decoder_layers(self.model)
        self.xnorm_by_layer: dict[int, object] = {}
        self.residual_by_layer: dict[int, object] = {}
        self.hooks = []
        self.target_device = resolve_device(self.device, cpu_fallback=cpu_fallback)

        def make_hook(layer_idx: int):
            def _hook(_module, args, kwargs):
                hidden_states = _resolve_hidden_states(args, kwargs)
                if hidden_states is None:
                    return
                self.xnorm_by_layer[layer_idx] = self._materialize_capture_tensor(hidden_states)

            return _hook

        def make_residual_hook(layer_idx: int):
            def _hook(_module, args):
                hidden_states = args[0] if args else None
                if hidden_states is None:
                    return
                self.residual_by_layer[layer_idx] = self._materialize_capture_tensor(hidden_states)

            return _hook

        for idx, layer in enumerate(self.layers):
            self.hooks.append(layer.register_forward_pre_hook(make_residual_hook(idx)))
            if hasattr(layer, "self_attn"):
                self.hooks.append(layer.self_attn.register_forward_pre_hook(make_hook(idx), with_kwargs=True))
            elif hasattr(layer, "attn"):
                self.hooks.append(layer.attn.register_forward_pre_hook(make_hook(idx), with_kwargs=True))

    def _materialize_capture_tensor(self, value):
        target = "cpu" if self.capture_store_device == "cpu" else self.target_device
        return value.detach().to(device=target, dtype=self.torch.float32).contiguous()

    def estimate_prompt_tokens(self, prompt: str, max_length: int | None = None) -> int:
        encoded = self.tokenizer(prompt, truncation=bool(max_length and max_length > 0), max_length=max_length)
        return int(len(encoded.get("input_ids", [])))

    def capture(self, prompt: str | list[str], max_length: int | None = None):
        prompts = _resolve_prompts(prompt)
        self.xnorm_by_layer.clear()
        self.residual_by_layer.clear()

        tokenizer_kwargs = {"return_tensors": "pt", "padding": True}
        if max_length is not None and int(max_length) > 0:
            tokenizer_kwargs["truncation"] = True
            tokenizer_kwargs["max_length"] = int(max_length)
        else:
            tokenizer_kwargs["truncation"] = False

        encoded = self.tokenizer(prompts, **tokenizer_kwargs)
        input_ids = encoded["input_ids"].to(self.target_device, non_blocking=True)
        attention_mask = encoded.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.target_device, non_blocking=True)

        with self.torch.no_grad():
            out = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                output_hidden_states=False,
                return_dict=True,
            )

        if out.past_key_values is None:
            raise RuntimeError("模型没有返回 past_key_values")

        payload = {}
        for layer_idx, kv in enumerate(out.past_key_values):
            if not isinstance(kv, (tuple, list)) or len(kv) < 2:
                raise RuntimeError(f"层 {layer_idx} 的 past_key_values 格式不合法")
            x_tensor = self.xnorm_by_layer.get(layer_idx)
            resid_tensor = self.residual_by_layer.get(layer_idx)
            if x_tensor is None:
                raise RuntimeError(f"层 {layer_idx} 无法提取 xnorm（未命中 hook）")
            if resid_tensor is None:
                raise RuntimeError(f"层 {layer_idx} 无法提取 residual state（未命中 hook）")
            payload[layer_idx] = {
                "k": self._materialize_capture_tensor(kv[0]),
                "v": self._materialize_capture_tensor(kv[1]),
                "xnorm": x_tensor.contiguous(),
                "resid": resid_tensor.contiguous(),
            }

        del out, encoded, input_ids, attention_mask
        clear_torch_cache()
        return payload

    def close(self) -> None:
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        for name in ("layers", "model", "tokenizer"):
            if hasattr(self, name):
                delattr(self, name)
        clear_torch_cache()


def iter_capture_batches_with_transformers(req: CaptureRequest, prompt_batches: list[list[str]], cpu_fallback: bool = True) -> Iterator[dict[int, dict[str, object]]]:
    session = TransformersCaptureSession(req.model_path, req.device, req.dtype, cpu_fallback=cpu_fallback)
    try:
        for batch_prompts in prompt_batches:
            yield session.capture(batch_prompts, req.max_length)
    finally:
        session.close()


def capture_batches_with_transformers(req: CaptureRequest, prompt_batches: list[list[str]], cpu_fallback: bool = True):
    return list(iter_capture_batches_with_transformers(req, prompt_batches, cpu_fallback=cpu_fallback))


def capture_with_transformers(req: CaptureRequest, cpu_fallback: bool = True):
    return capture_batches_with_transformers(req, [_resolve_prompts(req.prompt)], cpu_fallback=cpu_fallback)[0]

