from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .bundle import make_manifest, normalize_kv_tensor, payload_nbytes, stable_json_hash, write_manifest
from .kv_spec import KVSpec, kv_spec_from_hf_config


class HFExportError(RuntimeError):
    """Raised when an HF reference cache bundle cannot be exported safely."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a Hugging Face full-consumer KV bundle for SCD-vLLM")
    parser.add_argument("--consumer-model", required=True, help="HF model id or local path for the consumer model")
    parser.add_argument("--producer-model", default="", help="Optional producer model id/path recorded in manifest")
    parser.add_argument("--tokenizer", default="", help="Tokenizer id/path; defaults to the consumer model")
    parser.add_argument("--prompt", default="", help="Prefix prompt text")
    parser.add_argument("--prefix-file", default="", help="File containing prefix prompt text")
    parser.add_argument("--mode", default="full_consumer", choices=["full_consumer"])
    parser.add_argument("--dtype", default="bfloat16", help="Model/cache dtype, e.g. bfloat16, float16, float32")
    parser.add_argument("--device", default="cuda:0", help="Torch device, e.g. cuda:0 or cpu")
    parser.add_argument("--max-prefix-tokens", type=int, default=128, help="Truncate prefix to this many tokens; <=0 disables")
    parser.add_argument("--out", required=True, help="Output bundle directory")
    parser.add_argument("--trust-remote-code", action="store_true", help="Pass trust_remote_code=True to HF loaders")
    parser.add_argument("--no-special-tokens", action="store_true", help="Do not add tokenizer special tokens")
    return parser.parse_args(argv)


def load_prompt(prompt: str, prefix_file: str) -> str:
    if prompt and prefix_file:
        raise HFExportError("provide either --prompt or --prefix-file, not both")
    if prefix_file:
        text = Path(prefix_file).read_text(encoding="utf-8")
    else:
        text = prompt
    text = text.strip()
    if not text:
        raise HFExportError("prefix prompt is empty")
    return text


def build_position_ids(attention_mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(attention_mask, dtype=np.int64)
    if mask.ndim != 2:
        raise HFExportError(f"attention_mask must be [B,T], got {mask.shape}")
    position_ids = np.cumsum(mask, axis=-1, dtype=np.int64) - 1
    position_ids = np.where(mask > 0, position_ids, 0)
    return np.ascontiguousarray(position_ids.astype(np.int64, copy=False))


def iter_cache_layers(past_key_values: Any) -> Iterable[tuple[int, Any, Any]]:
    if past_key_values is None:
        raise HFExportError("model output did not include past_key_values")

    legacy = _to_legacy_cache(past_key_values)
    if legacy is not past_key_values:
        yield from iter_cache_layers(legacy)
        return

    key_cache = _noncallable_attr(past_key_values, ("key_cache", "keys"))
    value_cache = _noncallable_attr(past_key_values, ("value_cache", "values"))
    if isinstance(key_cache, (list, tuple)) and isinstance(value_cache, (list, tuple)):
        for idx, (key, value) in enumerate(zip(key_cache, value_cache, strict=True)):
            yield idx, key, value
        return

    layers = _noncallable_attr(past_key_values, ("layers", "cache_layers"))
    if isinstance(layers, (list, tuple)):
        for idx, layer in enumerate(layers):
            yield idx, *_cache_pair_from_layer(layer, idx)
        return

    if isinstance(past_key_values, (list, tuple)):
        for idx, layer in enumerate(past_key_values):
            yield idx, *_cache_pair_from_layer(layer, idx)
        return

    if hasattr(past_key_values, "__iter__"):
        try:
            for idx, layer in enumerate(past_key_values):
                yield idx, *_cache_pair_from_layer(layer, idx)
            return
        except TypeError:
            pass

    raise HFExportError(f"unsupported cache object type: {type(past_key_values).__name__}")


def normalize_cache_tensor(tensor: Any) -> Any:
    if _looks_like_torch_tensor(tensor):
        return _normalize_torch_cache_tensor(tensor)
    return normalize_kv_tensor(np.asarray(tensor))


def export_full_consumer_bundle(
    *,
    consumer_model: str,
    producer_model: str | None,
    tokenizer_id: str | None,
    prompt: str,
    dtype: str,
    device: str,
    max_prefix_tokens: int,
    out_dir: str | Path,
    trust_remote_code: bool = False,
    add_special_tokens: bool = True,
) -> dict[str, Any]:
    torch, AutoModelForCausalLM, AutoTokenizer, save_file = _load_runtime_dependencies()
    torch_dtype = _torch_dtype(torch, dtype)
    tokenizer_name = tokenizer_id or consumer_model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=trust_remote_code, use_fast=True)
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=add_special_tokens,
        truncation=max_prefix_tokens > 0,
        max_length=max_prefix_tokens if max_prefix_tokens > 0 else None,
    )
    input_ids = encoded["input_ids"]
    attention_mask = encoded.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    if int(input_ids.shape[0]) != 1:
        raise HFExportError("first MVP supports batch size 1 only")
    if int(input_ids.shape[-1]) <= 0:
        raise HFExportError("tokenized prefix is empty")

    spec = kv_spec_from_hf_config(consumer_model, dtype)
    spec = KVSpec.from_dict({**spec.to_dict(), "tokenizer_id": tokenizer_name, "dtype": dtype})
    model = AutoModelForCausalLM.from_pretrained(
        consumer_model,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()

    inputs = {
        "input_ids": input_ids.to(device),
        "attention_mask": attention_mask.to(device),
    }
    with torch.inference_mode():
        output = model(**inputs, use_cache=True, return_dict=True)

    out = Path(out_dir)
    kv_dir = out / "kv"
    kv_dir.mkdir(parents=True, exist_ok=True)
    token_ids_np = input_ids.detach().cpu().numpy().astype(np.int64, copy=False)
    attention_mask_np = attention_mask.detach().cpu().numpy().astype(np.int64, copy=False)
    position_ids_np = build_position_ids(attention_mask_np)

    np.save(out / "token_ids.npy", token_ids_np)
    np.save(out / "attention_mask.npy", attention_mask_np)
    np.save(out / "position_ids.npy", position_ids_np)
    (out / "prefix.txt").write_text(prompt + "\n", encoding="utf-8")

    layer_files: list[str] = []
    kv_bytes = 0
    layers = list(iter_cache_layers(output.past_key_values))
    if spec.num_layers and len(layers) != spec.num_layers:
        raise HFExportError(f"cache layer count mismatch: got {len(layers)}, expected {spec.num_layers}")
    for layer_idx, key, value in layers:
        key_norm = normalize_cache_tensor(key).detach().cpu().contiguous()
        value_norm = normalize_cache_tensor(value).detach().cpu().contiguous()
        _validate_normalized_layer(layer_idx, key_norm, value_norm, spec, int(token_ids_np.shape[-1]))
        layer_path = kv_dir / f"layer_{layer_idx:03d}.safetensors"
        save_file({"k": key_norm, "v": value_norm}, str(layer_path))
        layer_files.append(str(layer_path.relative_to(out)).replace("\\", "/"))
        kv_bytes += payload_nbytes(key_norm, value_norm)

    logits = output.logits[:, -1, :].detach().cpu().contiguous()
    save_file({"logits": logits}, str(out / "first_token_logits.safetensors"))
    manifest = make_manifest(
        consumer_model=consumer_model,
        producer_model=producer_model or None,
        tokenizer_id=tokenizer_name,
        token_ids=token_ids_np,
        bundle_type="full_consumer",
        dtype=dtype,
        kv_spec=spec,
        prefix_text=prompt,
        device_origin=device,
        quality_gate={"gate_name": "hf_full_consumer_export", "thresholds": {"batch_size": 1}},
    )
    write_manifest(out / "manifest.json", manifest)
    metrics = {
        "mode": "full_consumer",
        "consumer_model": consumer_model,
        "producer_model": producer_model or None,
        "tokenizer_id": tokenizer_name,
        "num_tokens": int(token_ids_np.shape[-1]),
        "num_layers": len(layer_files),
        "dtype": dtype,
        "device_origin": device,
        "token_hash": manifest.token_hash,
        "prefix_text_hash": manifest.prefix_text_hash,
        "route_hash": manifest.route_hash,
        "kv_layout": "hf_contiguous",
        "kv_layer_files": layer_files,
        "raw_consumer_kv_bytes": int(kv_bytes),
        "payload_bytes": int(kv_bytes),
        "token_payload_bytes": payload_nbytes(token_ids_np, attention_mask_np, position_ids_np),
        "first_token_logits_bytes": payload_nbytes(logits),
        "manifest_hash": stable_json_hash(manifest.to_dict()),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return metrics


def _to_legacy_cache(cache: Any) -> Any:
    to_legacy = getattr(cache, "to_legacy_cache", None)
    if callable(to_legacy):
        converted = to_legacy()
        if converted is not None:
            return converted
    return cache


def _cache_pair_from_layer(layer: Any, idx: int) -> tuple[Any, Any]:
    if isinstance(layer, dict):
        key = _dict_value(layer, ("key", "k", "keys", "key_states", "key_cache"))
        value = _dict_value(layer, ("value", "v", "values", "value_states", "value_cache"))
    elif isinstance(layer, (list, tuple)) and len(layer) >= 2:
        key, value = layer[0], layer[1]
    else:
        key = _noncallable_attr(layer, ("key", "k", "keys", "key_states", "key_cache"))
        value = _noncallable_attr(layer, ("value", "v", "values", "value_states", "value_cache"))
    if key is None or value is None:
        raise HFExportError(f"could not read key/value tensors from cache layer {idx}")
    return key, value


def _dict_value(data: dict[str, Any], names: tuple[str, ...]) -> Any:
    for name in names:
        if name in data:
            return data[name]
    return None


def _noncallable_attr(obj: Any, names: tuple[str, ...]) -> Any:
    for name in names:
        if hasattr(obj, name):
            value = getattr(obj, name)
            if value is not None and not callable(value):
                return value
    return None


def _looks_like_torch_tensor(value: Any) -> bool:
    return hasattr(value, "detach") and hasattr(value, "dim") and hasattr(value, "permute")


def _normalize_torch_cache_tensor(tensor: Any) -> Any:
    if tensor.dim() == 3:
        return tensor.contiguous()
    if tensor.dim() == 4 and int(tensor.shape[0]) == 1:
        return tensor[0].permute(1, 0, 2).contiguous()
    raise HFExportError(f"unsupported torch KV tensor shape for normalization: {tuple(tensor.shape)}")


def _validate_normalized_layer(layer_idx: int, key: Any, value: Any, spec: KVSpec, num_tokens: int) -> None:
    key_shape = tuple(int(x) for x in key.shape)
    value_shape = tuple(int(x) for x in value.shape)
    expected = (num_tokens, spec.num_kv_heads, spec.head_dim)
    if key_shape != expected or value_shape != expected:
        raise HFExportError(
            f"layer {layer_idx} normalized KV shape mismatch: key={key_shape} value={value_shape} expected={expected}"
        )


def _load_runtime_dependencies() -> tuple[Any, Any, Any, Any]:
    try:
        import torch
        from safetensors.torch import save_file
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:  # pragma: no cover - depends on optional runtime environment
        raise HFExportError("torch, transformers, and safetensors are required for HF export") from exc
    return torch, AutoModelForCausalLM, AutoTokenizer, save_file


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
        raise HFExportError(f"unsupported torch dtype: {dtype}")
    return getattr(torch, attr)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    prompt = load_prompt(args.prompt, args.prefix_file)
    metrics = export_full_consumer_bundle(
        consumer_model=args.consumer_model,
        producer_model=args.producer_model or None,
        tokenizer_id=args.tokenizer or None,
        prompt=prompt,
        dtype=args.dtype,
        device=args.device,
        max_prefix_tokens=args.max_prefix_tokens,
        out_dir=args.out,
        trust_remote_code=args.trust_remote_code,
        add_special_tokens=not args.no_special_tokens,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
