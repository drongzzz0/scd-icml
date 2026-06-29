"""从 Reuse/Patch checkpoint 重建 SCD KV bundle。

该模块把离线训练得到的 Reuse checkpoint、Patch checkpoint 和路线文件
应用到一个固定 prefix 上，导出可校验的 KV bundle。它是 vLLM smoke
路径的上游，不直接启动 vLLM。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from scd_repro.io_utils import load_json
from scd_repro.pipeline import load_patch_ckpt, load_reuse_weights
from scd_repro.reuse import move_reuse_weights
from scd_repro.strict_runtime import (
    TransformersRuntimeSession,
    _assemble_method_cache,
    assert_same_tokenization,
)

from .bundle import make_manifest, payload_nbytes, stable_json_hash, write_manifest
from .hf_export import load_prompt
from .kv_spec import KVSpec, kv_spec_from_hf_config


class ReconstructedExportError(RuntimeError):
    """Raised when a reconstructed SCD cache bundle cannot be exported."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a reconstructed reuse+patch SCD KV bundle")
    parser.add_argument("--producer-model", required=True)
    parser.add_argument("--consumer-model", required=True)
    parser.add_argument("--tokenizer", default="", help="Recorded tokenizer id; defaults to consumer model")
    parser.add_argument("--reuse-ckpt", required=True)
    parser.add_argument("--patch-ckpt", required=True)
    parser.add_argument("--route", required=True)
    parser.add_argument("--prompt", default="")
    parser.add_argument("--prefix-file", default="")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--transfer-dtype", default="", help="Defaults to --dtype")
    parser.add_argument("--producer-device", default="cuda:0")
    parser.add_argument("--consumer-device", default="cuda:1")
    parser.add_argument("--patch-ckpt-device", default="cpu")
    parser.add_argument("--max-prefix-tokens", type=int, default=256)
    parser.add_argument("--bandwidth-gbps", type=float, default=200.0)
    parser.add_argument("--quant-group-size", type=int, default=128)
    parser.add_argument("--runtime-patch-window-size", type=int, default=0)
    parser.add_argument("--runtime-patch-persist-shards", action="store_true")
    parser.add_argument("--runtime-patch-persist-modules", action="store_true")
    parser.add_argument("--out", required=True)
    return parser.parse_args(argv)


def export_reconstructed_bundle(
    *,
    producer_model: str,
    consumer_model: str,
    tokenizer_id: str | None,
    reuse_ckpt: str | Path,
    patch_ckpt: str | Path,
    route_path: str | Path,
    prompt: str,
    dtype: str,
    transfer_dtype: str,
    producer_device: str,
    consumer_device: str,
    patch_ckpt_device: str,
    max_prefix_tokens: int,
    bandwidth_gbps: float,
    quant_group_size: int,
    runtime_patch_window_size: int = 0,
    runtime_patch_persist_shards: bool = False,
    runtime_patch_persist_modules: bool = False,
    out_dir: str | Path = "runs/scd_vllm/reconstructed_bundle",
) -> dict[str, Any]:
    torch, save_file = _load_runtime_dependencies()
    route = load_json(route_path)
    reuse_cpu, reuse_meta = load_reuse_weights(reuse_ckpt, device="cpu", dtype="float32")
    patch_bundle, patch_meta = load_patch_ckpt(patch_ckpt, device=patch_ckpt_device, dtype="float32")
    _apply_patch_runtime_overrides(
        patch_meta,
        runtime_patch_window_size=runtime_patch_window_size,
        runtime_patch_persist_shards=runtime_patch_persist_shards,
        runtime_patch_persist_modules=runtime_patch_persist_modules,
    )
    reuse_prod = move_reuse_weights(reuse_cpu, producer_device, dtype="float32", cpu_fallback=True)
    reuse_cons = move_reuse_weights(reuse_cpu, consumer_device, dtype="float32", cpu_fallback=True)
    transfer_torch_dtype = _torch_dtype(torch, transfer_dtype or dtype)

    producer = TransformersRuntimeSession(producer_model, producer_device, dtype)
    consumer = TransformersRuntimeSession(consumer_model, consumer_device, dtype)
    try:
        producer_capture = producer.prefill(
            prompt,
            max_length=max_prefix_tokens if max_prefix_tokens > 0 else None,
            capture_xnorm=True,
            profile_layers=False,
        )
        consumer_capture = consumer.prefill(
            prompt,
            max_length=max_prefix_tokens if max_prefix_tokens > 0 else None,
            capture_xnorm=False,
            capture_residual=False,
            profile_layers=True,
        )
        assert_same_tokenization(producer_capture, consumer_capture)
        caches, timing, payload_mb = _assemble_method_cache(
            method="scd_full",
            producer_capture=producer_capture,
            consumer_capture=consumer_capture,
            producer_session=producer,
            consumer_session=consumer,
            route=route,
            reuse_prod=reuse_prod,
            reuse_cons=reuse_cons,
            patch_bundle=patch_bundle,
            patch_meta=patch_meta,
            bandwidth_gbps=bandwidth_gbps,
            quant_group_size=quant_group_size,
            transfer_dtype=transfer_torch_dtype,
            default_recompute_segment_length=0,
            override_route_recompute_length=None,
            fail_on_missing_route_fields=True,
            fail_on_missing_patch_fields=True,
        )
        token_ids = consumer_capture.input_ids.detach().cpu().numpy().astype(np.int64, copy=False)
        attention_mask = consumer_capture.attention_mask.detach().cpu().numpy().astype(np.int64, copy=False)
    finally:
        producer.close()
        consumer.close()

    out = Path(out_dir)
    kv_dir = out / "kv"
    kv_dir.mkdir(parents=True, exist_ok=True)
    num_tokens = int(token_ids.shape[-1])
    cache_tokens = num_tokens - 1
    if cache_tokens <= 0:
        raise ReconstructedExportError("reconstructed export requires at least two prompt tokens")
    spec = kv_spec_from_hf_config(consumer_model, dtype)
    spec = KVSpec.from_dict({**spec.to_dict(), "tokenizer_id": tokenizer_id or consumer_model, "dtype": dtype})
    if int(spec.num_layers) != len(caches):
        raise ReconstructedExportError(f"cache layer count mismatch: {len(caches)} vs spec {spec.num_layers}")

    layer_files: list[str] = []
    kv_bytes = 0
    for layer_idx, (key, value) in enumerate(caches):
        key_norm = _normalize_cache_tensor(key, cache_tokens, torch).detach().cpu().contiguous()
        value_norm = _normalize_cache_tensor(value, cache_tokens, torch).detach().cpu().contiguous()
        _validate_layer(layer_idx, key_norm, value_norm, spec, cache_tokens)
        layer_path = kv_dir / f"layer_{layer_idx:03d}.safetensors"
        save_file({"k": key_norm, "v": value_norm}, str(layer_path))
        layer_files.append(str(layer_path.relative_to(out)).replace("\\", "/"))
        kv_bytes += payload_nbytes(key_norm, value_norm)

    np.save(out / "token_ids.npy", token_ids)
    np.save(out / "attention_mask.npy", attention_mask)
    (out / "prefix.txt").write_text(prompt + "\n", encoding="utf-8")
    manifest = make_manifest(
        consumer_model=consumer_model,
        producer_model=producer_model,
        tokenizer_id=tokenizer_id or consumer_model,
        token_ids=token_ids,
        bundle_type="reuse_patch",
        dtype=dtype,
        kv_spec=spec,
        prefix_text=prompt,
        route=route,
        route_id=str(route.get("name") or Path(route_path).stem),
        device_origin=f"producer={producer_device};consumer={consumer_device}",
        quality_gate={
            "gate_name": "strict_runtime_scd_full_reconstructed_export",
            "thresholds": {
                "fallback_layers": 0,
                "recompute_layers": 0,
                "cache_tokens": cache_tokens,
            },
        },
    )
    write_manifest(out / "manifest.json", manifest)
    route_out = out / "route.json"
    route_out.write_text(json.dumps(route, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    metrics = {
        "mode": "reuse_patch_reconstructed_bundle",
        "bundle_type": "reuse_patch",
        "producer_model": producer_model,
        "consumer_model": consumer_model,
        "tokenizer_id": tokenizer_id or consumer_model,
        "reuse_ckpt": str(reuse_ckpt),
        "patch_ckpt": str(patch_ckpt),
        "route": str(route_path),
        "route_name": route.get("name"),
        "route_hash": manifest.route_hash,
        "num_tokens": num_tokens,
        "cache_tokens": cache_tokens,
        "num_layers": len(layer_files),
        "dtype": dtype,
        "transfer_dtype": transfer_dtype or dtype,
        "token_hash": manifest.token_hash,
        "prefix_text_hash": manifest.prefix_text_hash,
        "kv_layout": "hf_contiguous",
        "kv_layer_files": layer_files,
        "payload_bytes": int(kv_bytes),
        "timing_ms": _timing_to_dict(timing),
        "payload_mb_runtime": float(payload_mb),
        "reuse_meta": _small_meta(reuse_meta),
        "patch_meta": _small_meta(patch_meta),
        "manifest_hash": stable_json_hash(manifest.to_dict()),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return metrics


def _apply_patch_runtime_overrides(
    patch_meta: dict[str, Any],
    *,
    runtime_patch_window_size: int,
    runtime_patch_persist_shards: bool,
    runtime_patch_persist_modules: bool,
) -> None:
    if runtime_patch_window_size > 0:
        patch_meta["runtime_patch_window_size"] = int(runtime_patch_window_size)
    patch_meta["runtime_patch_persist_shards"] = bool(runtime_patch_persist_shards)
    patch_meta["runtime_patch_persist_modules"] = bool(runtime_patch_persist_modules)


def _normalize_cache_tensor(tensor: Any, cache_tokens: int, torch: Any) -> Any:
    if not torch.is_tensor(tensor):
        tensor = torch.as_tensor(tensor)
    if tensor.dim() == 4 and int(tensor.shape[0]) == 1:
        if int(tensor.shape[2]) < cache_tokens:
            raise ReconstructedExportError(f"KV tensor has {int(tensor.shape[2])} tokens, needs {cache_tokens}")
        return tensor[0, :, :cache_tokens, :].permute(1, 0, 2).contiguous()
    if tensor.dim() == 3:
        if int(tensor.shape[0]) >= cache_tokens:
            return tensor[:cache_tokens].contiguous()
        if int(tensor.shape[1]) >= cache_tokens:
            return tensor[:, :cache_tokens, :].permute(1, 0, 2).contiguous()
    raise ReconstructedExportError(f"unsupported reconstructed KV tensor shape: {tuple(tensor.shape)}")


def _validate_layer(layer_idx: int, key: Any, value: Any, spec: KVSpec, cache_tokens: int) -> None:
    expected = (cache_tokens, spec.num_kv_heads, spec.head_dim)
    if tuple(key.shape) != expected or tuple(value.shape) != expected:
        raise ReconstructedExportError(
            f"layer {layer_idx} KV shape mismatch: key={tuple(key.shape)} value={tuple(value.shape)} expected={expected}"
        )


def _timing_to_dict(timing: Any) -> dict[str, Any]:
    data = getattr(timing, "__dict__", {})
    return {str(k): _json_scalar(v) for k, v in data.items()}


def _json_scalar(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    try:
        return float(value)
    except Exception:
        return str(value)


def _small_meta(meta: dict[str, Any]) -> dict[str, Any]:
    keep = [
        "mode",
        "fit_mode",
        "selected_layers",
        "residual_layers",
        "num_layers",
        "group_size",
        "rank",
        "rank_k",
        "rank_v",
        "num_groups",
        "semantic_source",
        "patch_apply_mode",
        "runtime_patch_window_size",
        "runtime_patch_persist_shards",
        "runtime_patch_persist_modules",
    ]
    return {key: meta.get(key) for key in keep if key in meta}


def _load_runtime_dependencies() -> tuple[Any, Any]:
    try:
        import torch
        from safetensors.torch import save_file
    except Exception as exc:  # pragma: no cover - optional runtime
        raise ReconstructedExportError("torch and safetensors are required for reconstructed export") from exc
    return torch, save_file


def _torch_dtype(torch: Any, dtype: str) -> Any:
    normalized = dtype.lower().replace("torch.", "")
    aliases = {
        "bf16": "bfloat16",
        "bfloat16": "bfloat16",
        "fp16": "float16",
        "float16": "float16",
        "half": "float16",
        "fp32": "float32",
        "float32": "float32",
    }
    attr = aliases.get(normalized)
    if attr is None or not hasattr(torch, attr):
        raise ReconstructedExportError(f"unsupported torch dtype: {dtype}")
    return getattr(torch, attr)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    prompt = load_prompt(args.prompt, args.prefix_file)
    metrics = export_reconstructed_bundle(
        producer_model=args.producer_model,
        consumer_model=args.consumer_model,
        tokenizer_id=args.tokenizer or None,
        reuse_ckpt=args.reuse_ckpt,
        patch_ckpt=args.patch_ckpt,
        route_path=args.route,
        prompt=prompt,
        dtype=args.dtype,
        transfer_dtype=args.transfer_dtype or args.dtype,
        producer_device=args.producer_device,
        consumer_device=args.consumer_device,
        patch_ckpt_device=args.patch_ckpt_device,
        max_prefix_tokens=args.max_prefix_tokens,
        bandwidth_gbps=args.bandwidth_gbps,
        quant_group_size=args.quant_group_size,
        runtime_patch_window_size=args.runtime_patch_window_size,
        runtime_patch_persist_shards=args.runtime_patch_persist_shards,
        runtime_patch_persist_modules=args.runtime_patch_persist_modules,
        out_dir=args.out,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
