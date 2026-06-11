#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
import types
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from scd_repro.cmrc import extract_answer_text, load_cmrc2018_samples, score_cmrc_prediction


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run official xKV method on Qwen3 self-self CMRC samples")
    p.add_argument("--xkv-root", required=True, help="Path to the official xKV repo root")
    p.add_argument("--data-root", required=True, help="Path to CMRC data root")
    p.add_argument("--model-name-or-path", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--num-samples", type=int, default=20)
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--mode", choices=["full", "xkv"], default="full")
    p.add_argument("--group-size", type=int, default=1)
    p.add_argument("--rank-k", type=int, default=128)
    p.add_argument("--rank-v", type=int, default=192)
    p.add_argument("--flash2", action="store_true")
    return p


def _wrap_fake_svd_identity_fastpath(fake_svd_fn):
    def _lowrank_fake_svd(tensor: torch.Tensor, rank: int) -> torch.Tensor:
        bs, nh, sl, hd = tensor.shape
        tensor_reshaped = tensor.transpose(1, 2).reshape(bs, sl, nh * hd)
        q = max(1, min(int(rank), sl, nh * hd))
        approx_batches: list[torch.Tensor] = []
        for batch_idx in range(bs):
            mat = tensor_reshaped[batch_idx]
            u, s, v = torch.svd_lowrank(mat, q=q, niter=2)
            approx = (u * s.unsqueeze(0)) @ v.transpose(0, 1)
            approx_batches.append(approx)
        approx_tensor = torch.stack(approx_batches, dim=0)
        return approx_tensor.view(bs, sl, nh, hd).transpose(1, 2)

    def wrapped(tensor: torch.Tensor, rank: int, apply_hadamard_transform: bool = False, quantizer=None):
        if not apply_hadamard_transform and quantizer is None:
            seq_len = int(tensor.shape[-2])
            feature_dim = int(tensor.shape[1] * tensor.shape[-1])
            max_rank = min(seq_len, feature_dim)
            if rank is None or int(rank) >= max_rank:
                return tensor
            return _lowrank_fake_svd(tensor, int(rank))
        return fake_svd_fn(
            tensor,
            rank,
            apply_hadamard_transform=apply_hadamard_transform,
            quantizer=quantizer,
        )

    return wrapped


def _install_xkv_qwen3_patch(xkv_root: Path) -> tuple[Any, Any]:
    if str(xkv_root) not in sys.path:
        sys.path.insert(0, str(xkv_root))

    from transformers.cache_utils import Cache
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
    from transformers.models.qwen3.modeling_qwen3 import (
        Qwen3Attention,
        apply_rotary_pos_emb,
        eager_attention_forward,
    )
    from transformers.processing_utils import Unpack

    from xKV.configurations import generate_consecutive_xKV_config
    from xKV.customized_cache import fake_layer_merge_dynamic_cache as xkv_cache_mod
    from xKV.customized_cache.fake_layer_merge_dynamic_cache import FakeLayerMergingCache
    from xKV.patch import prepare_cache

    xkv_cache_mod.fake_svd = _wrap_fake_svd_identity_fastpath(xkv_cache_mod.fake_svd)

    def _cache_getitem(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        layer = self.layers[idx]
        return layer.keys, layer.values

    def _update_cache(self, layer_idx: int, key_approx: torch.Tensor, value_approx: torch.Tensor) -> None:
        layer = self.layers[layer_idx]
        layer.keys = key_approx
        layer.values = value_approx
        layer.is_initialized = True

    def _update(self, key, value, layer_idx, mode="prefill", cos=None, sin=None, re_apply_rope=True):
        super(FakeLayerMergingCache, self).update(key, value, layer_idx)

        if mode == "prefill":
            self.num_heads = key.shape[1]
            self.head_dim = key.shape[3]

            if self._should_merge(layer_idx):
                self.grouped_layer_merging(layer_idx)

            group_info = self.merge_setup.get_group_for_layer(layer_idx)
            if group_info is not None:
                if layer_idx == group_info.layers[-1]:
                    for layer_idx_in_group in group_info.layers:
                        pre_rope_key, _ = self.__getitem__(layer_idx_in_group)
                        if re_apply_rope:
                            _, rope_key = apply_rotary_pos_emb(pre_rope_key, pre_rope_key, cos, sin)
                            self.update_cache(layer_idx_in_group, rope_key, self.__getitem__(layer_idx_in_group)[1])
            else:
                pre_rope_key, rope_value = self.__getitem__(layer_idx)
                if re_apply_rope:
                    _, rope_key = apply_rotary_pos_emb(pre_rope_key, pre_rope_key, cos, sin)
                    self.update_cache(layer_idx, rope_key, rope_value)
        return self.__getitem__(layer_idx)

    FakeLayerMergingCache.__getitem__ = _cache_getitem
    FakeLayerMergingCache.update_cache = _update_cache
    FakeLayerMergingCache.update = _update

    def xkv_qwen3_forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        is_prefill = past_key_values is None or past_key_values.get_seq_length(self.layer_idx) == 0

        # Follow official xKV Qwen patch semantics: keep K in pre-RoPE space during prefill merging.
        query_states, _ = apply_rotary_pos_emb(query_states, query_states, cos, sin)

        if past_key_values is not None:
            if is_prefill:
                if not isinstance(past_key_values, FakeLayerMergingCache):
                    raise TypeError(f"expected FakeLayerMergingCache, got {type(past_key_values)!r}")
                past_key_values.update(key_states, value_states, self.layer_idx, mode="prefill", cos=cos, sin=sin)
                _, key_states = apply_rotary_pos_emb(key_states, key_states, cos, sin)
            else:
                _, key_states = apply_rotary_pos_emb(key_states, key_states, cos, sin)
                key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, mode="decode")

        attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation,
            eager_attention_forward,
        )
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

    def enable_qwen3_xkv_eval(model: Any) -> Any:
        for layer in model.model.layers:
            module = layer.self_attn
            if not isinstance(module, Qwen3Attention):
                raise TypeError(f"expected Qwen3Attention, got {type(module)!r}")
            module.forward = types.MethodType(xkv_qwen3_forward, module)
        return model

    return generate_consecutive_xKV_config, prepare_cache, enable_qwen3_xkv_eval


def load_model_and_tokenizer(model_name_or_path: str, flash2: bool) -> tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="cuda",
        attn_implementation="flash_attention_2" if flash2 else "sdpa",
    )
    model.eval()
    return model, tokenizer


def _maybe_synchronize(device: Any) -> None:
    device_type = getattr(device, "type", str(device).split(":")[0])
    if device_type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device=device)


def _measure_ttft_ms(model: Any, inputs: dict[str, torch.Tensor], device: Any) -> float:
    model_inputs = dict(inputs)
    if hasattr(model, "kv_compress_config") and hasattr(model, "_prepare_cache_for_generation"):
        model._prepare_cache_for_generation(getattr(model, "generation_config", None), model_inputs)

    _maybe_synchronize(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        outputs = model(**model_inputs, use_cache=True, return_dict=True)
        _ = torch.argmax(outputs.logits[:, -1, :], dim=-1)
    _maybe_synchronize(device)
    return (time.perf_counter() - t0) * 1000.0


def apply_xkv_patch(
    model: Any,
    xkv_root: Path,
    group_size: int,
    rank_k: int,
    rank_v: int,
) -> Any:
    generate_consecutive_xKV_config, prepare_cache, enable_qwen3_xkv_eval = _install_xkv_qwen3_patch(xkv_root)
    xkv_config = generate_consecutive_xKV_config(
        num_layers=model.config.num_hidden_layers,
        rank_k=rank_k,
        rank_v=rank_v,
        group_size=group_size,
        layer_merge_impl="svd",
        slerp_t=0.5,
        slerp_gamma=0.05,
        merge_key=True,
        merge_value=True,
        start_layer=0,
        end_layer=model.config.num_hidden_layers - 1,
        kv_bits=16,
        quantizer_group_size=0,
        sym=False,
        clip_ratio=1.0,
        hadamard=False,
    )
    model = enable_qwen3_xkv_eval(model)
    model.kv_compress_config = xkv_config
    model._prepare_cache_for_generation = prepare_cache("xKV", xkv_config).__get__(model, model.__class__)
    return model


def main() -> None:
    args = build_parser().parse_args()
    xkv_root = Path(args.xkv_root).resolve()
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_model_and_tokenizer(args.model_name_or_path, flash2=bool(args.flash2))
    if args.mode == "xkv":
        model = apply_xkv_patch(
            model,
            xkv_root=xkv_root,
            group_size=int(args.group_size),
            rank_k=int(args.rank_k),
            rank_v=int(args.rank_v),
        )

    samples = load_cmrc2018_samples(args.data_root, limit=int(args.num_samples))
    rows: list[dict[str, Any]] = []
    total_f1 = 0.0
    total_em = 0.0
    total_ttft_ms = 0.0
    total_full_generate_ms = 0.0

    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    device = next(model.parameters()).device

    for idx, sample in enumerate(samples):
        inputs = tokenizer(sample.prompt, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        input_len = int(inputs["input_ids"].shape[-1])

        ttft_ms = _measure_ttft_ms(model, inputs, device)

        _maybe_synchronize(device)
        t0 = time.perf_counter()
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=int(args.max_new_tokens),
                do_sample=False,
                use_cache=True,
                pad_token_id=pad_token_id,
            )
        _maybe_synchronize(device)
        full_generate_ms = (time.perf_counter() - t0) * 1000.0
        answer_text = tokenizer.decode(
            output_ids[0][input_len:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        answer = extract_answer_text(answer_text)
        score = score_cmrc_prediction(answer, sample.answers)

        total_f1 += float(score["f1"])
        total_em += float(score["em"])
        total_ttft_ms += ttft_ms
        total_full_generate_ms += full_generate_ms
        rows.append(
            {
                "idx": idx,
                "qid": sample.qid,
                "prediction": answer,
                "raw_generation": answer_text,
                "answers": sample.answers,
                "f1": float(score["f1"]),
                "em": float(score["em"]),
                "ttft_ms": ttft_ms,
                "full_generate_ms": full_generate_ms,
                "elapsed_ms": full_generate_ms,
            }
        )

    summary = {
        "mode": args.mode,
        "model_name_or_path": args.model_name_or_path,
        "num_samples": len(rows),
        "group_size": int(args.group_size) if args.mode == "xkv" else 0,
        "rank_k": int(args.rank_k) if args.mode == "xkv" else 0,
        "rank_v": int(args.rank_v) if args.mode == "xkv" else 0,
        "mean_f1": total_f1 / max(1, len(rows)),
        "mean_em": total_em / max(1, len(rows)),
        "mean_ttft_ms": total_ttft_ms / max(1, len(rows)),
        "mean_full_generate_ms": total_full_generate_ms / max(1, len(rows)),
        "mean_elapsed_ms": total_full_generate_ms / max(1, len(rows)),
        "rows": rows,
    }
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
