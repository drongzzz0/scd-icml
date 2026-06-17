from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from .bundle import read_manifest, stable_json_hash
from .hf_export import load_prompt


class SCDInjectionError(RuntimeError):
    """Raised when an SCD-vLLM injection smoke run cannot proceed safely."""


DEFAULT_LAYER_NAME_TEMPLATE = "model.layers.{layer}.self_attn.attn"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare and smoke-test SCD bundles through vLLM KV connectors")
    sub = parser.add_subparsers(dest="command", required=True)
    convert_parser = sub.add_parser("convert", help="Convert an SCD cache bundle to ExampleConnector storage")
    add_convert_args(convert_parser)
    run_parser = sub.add_parser("run", help="Run a vLLM baseline/load smoke test for a converted SCD bundle")
    add_convert_args(run_parser)
    add_runtime_args(run_parser)
    return parser.parse_args(argv)


def add_convert_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--bundle-dir", required=True, help="SCD cache bundle directory")
    parser.add_argument("--storage-dir", required=True, help="Output storage directory for vLLM ExampleConnector")
    parser.add_argument("--block-size", type=int, default=16, help="vLLM cache block size")
    parser.add_argument("--layer-name-template", default=DEFAULT_LAYER_NAME_TEMPLATE)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--out", default="", help="Output metrics JSON or run directory")


def add_runtime_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", required=True, help="vLLM model id or local path")
    parser.add_argument("--tokenizer", default="", help="Tokenizer id/path; defaults to --model")
    parser.add_argument("--prompt", default="", help="Prefix prompt text; defaults to bundle prefix.txt")
    parser.add_argument("--prefix-file", default="", help="File containing prefix prompt text")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--enable-prefix-caching", action="store_true")
    parser.add_argument("--enable-chunked-prefill", action="store_true")
    parser.add_argument("--disable-hybrid-kv-cache-manager", action="store_true")


def align_to_vllm_block(num_tokens: int, block_size: int) -> int:
    """Match vLLM ExampleConnector's current block-alignment rule."""

    if block_size <= 0:
        raise SCDInjectionError(f"block_size must be positive, got {block_size}")
    if num_tokens <= 0:
        return 0
    return (num_tokens - 1) // block_size * block_size


def example_connector_folder_hash(token_ids: np.ndarray, aligned_tokens: int, mm_hashes: list[str] | None = None) -> str:
    ids = np.asarray(token_ids, dtype=np.int64).reshape(-1)[:aligned_tokens]
    data = ids.tobytes(order="C")
    if mm_hashes:
        data += "-".join(mm_hashes).encode("utf-8")
    return hashlib.md5(data, usedforsecurity=False).hexdigest()


def convert_bundle_to_example_storage(
    *,
    bundle_dir: str | Path,
    storage_dir: str | Path,
    block_size: int = 16,
    layer_name_template: str = DEFAULT_LAYER_NAME_TEMPLATE,
    overwrite: bool = False,
) -> dict[str, Any]:
    torch, load_file, save_file = _load_tensor_runtime()
    bundle = Path(bundle_dir)
    storage = Path(storage_dir)
    manifest = read_manifest(bundle / "manifest.json")
    token_ids = _load_token_ids(bundle)
    aligned_tokens = align_to_vllm_block(int(token_ids.size), block_size)
    if aligned_tokens <= 0:
        raise SCDInjectionError(
            f"bundle has {int(token_ids.size)} tokens; ExampleConnector needs at least one aligned block"
        )
    if int(manifest.num_tokens) != int(token_ids.size):
        raise SCDInjectionError(f"manifest/token_ids length mismatch: {manifest.num_tokens} vs {token_ids.size}")
    if overwrite and storage.exists():
        shutil.rmtree(storage)
    storage.mkdir(parents=True, exist_ok=True)
    folder_hash = example_connector_folder_hash(token_ids, aligned_tokens)
    folder = storage / folder_hash
    folder.mkdir(parents=True, exist_ok=True)

    kv_spec = manifest.kv_spec
    num_layers = int(kv_spec.get("num_layers", 0) or 0)
    if num_layers <= 0:
        raise SCDInjectionError("manifest kv_spec.num_layers must be positive")
    files: list[dict[str, Any]] = []
    for layer_idx in range(num_layers):
        source = bundle / "kv" / f"layer_{layer_idx:03d}.safetensors"
        if not source.is_file():
            raise SCDInjectionError(f"missing bundle layer file: {source}")
        tensors = load_file(str(source), device="cpu")
        if "k" not in tensors or "v" not in tensors:
            raise SCDInjectionError(f"bundle layer must contain k and v tensors: {source}")
        k = _normalize_bundle_tensor(tensors["k"], aligned_tokens, torch)
        v = _normalize_bundle_tensor(tensors["v"], aligned_tokens, torch)
        if tuple(k.shape) != tuple(v.shape):
            raise SCDInjectionError(f"layer {layer_idx} k/v shape mismatch: {tuple(k.shape)} vs {tuple(v.shape)}")
        kv_cache = torch.stack([k.reshape(aligned_tokens, -1), v.reshape(aligned_tokens, -1)], dim=0).contiguous()
        target = folder / f"{layer_name_template.format(layer=layer_idx)}.safetensors"
        save_file({"kv_cache": kv_cache}, str(target))
        files.append(
            {
                "layer": layer_idx,
                "source": str(source),
                "target": str(target),
                "kv_cache_shape": [int(x) for x in kv_cache.shape],
                "dtype": str(kv_cache.dtype).replace("torch.", ""),
                "bytes": int(target.stat().st_size),
            }
        )
    metrics = {
        "mode": "scd_example_connector_storage",
        "bundle_dir": str(bundle),
        "storage_dir": str(storage),
        "folder_hash": folder_hash,
        "folder": str(folder),
        "schema_version": manifest.schema_version,
        "bundle_type": manifest.bundle_type,
        "consumer_model": manifest.consumer_model,
        "producer_model": manifest.producer_model,
        "tokenizer_id": manifest.tokenizer_id,
        "token_hash": manifest.token_hash,
        "route_id": manifest.route_id,
        "route_hash": manifest.route_hash,
        "num_tokens": int(token_ids.size),
        "aligned_tokens": int(aligned_tokens),
        "block_size": int(block_size),
        "layer_name_template": layer_name_template,
        "num_layers": len(files),
        "payload_bytes": sum(item["bytes"] for item in files),
        "files": files,
        "manifest_hash": stable_json_hash(manifest.to_dict()),
    }
    return metrics


def run_injection_smoke(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out or "runs/scd_vllm/scd_injection_smoke")
    out_dir.mkdir(parents=True, exist_ok=True)
    storage_dir = Path(args.storage_dir)
    conversion = convert_bundle_to_example_storage(
        bundle_dir=args.bundle_dir,
        storage_dir=storage_dir,
        block_size=args.block_size,
        layer_name_template=args.layer_name_template,
        overwrite=args.overwrite,
    )
    (out_dir / "conversion.json").write_text(json.dumps(conversion, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    prompt = _resolve_prompt_for_bundle(Path(args.bundle_dir), args.prompt, args.prefix_file)
    common = [
        "--model",
        args.model,
        "--tokenizer",
        args.tokenizer or args.model,
        "--prompt",
        prompt,
        "--storage-dir",
        str(storage_dir),
        "--dtype",
        args.dtype,
        "--max-model-len",
        str(args.max_model_len),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--seed",
        str(args.seed),
        "--temperature",
        str(args.temperature),
        "--top-p",
        str(args.top_p),
    ]
    for flag, enabled in (
        ("--enforce-eager", args.enforce_eager),
        ("--enable-prefix-caching", args.enable_prefix_caching),
        ("--enable-chunked-prefill", args.enable_chunked_prefill),
        ("--disable-hybrid-kv-cache-manager", args.disable_hybrid_kv_cache_manager),
    ):
        if enabled:
            common.append(flag)
    records: dict[str, dict[str, Any]] = {}
    for stage in ("baseline", "load"):
        stage_out = out_dir / f"{stage}.json"
        cmd = [
            sys.executable,
            "-m",
            "scd_vllm.truekv_oracle",
            "_generate",
            "--stage",
            stage,
            "--out",
            str(stage_out),
            *common,
        ]
        subprocess.run(cmd, check=True)
        records[stage] = json.loads(stage_out.read_text(encoding="utf-8"))
    metrics = build_injection_metrics(
        model=args.model,
        tokenizer=args.tokenizer or args.model,
        prompt=prompt,
        conversion=conversion,
        records=records,
        max_new_tokens=args.max_new_tokens,
    )
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out_dir / "prompt.txt").write_text(prompt + "\n", encoding="utf-8")
    return metrics


def build_injection_metrics(
    *,
    model: str,
    tokenizer: str,
    prompt: str,
    conversion: dict[str, Any],
    records: dict[str, dict[str, Any]],
    max_new_tokens: int,
) -> dict[str, Any]:
    baseline = records["baseline"]
    load = records["load"]
    prompt_token_hash = baseline.get("prompt_token_hash")
    bundle_token_match = bool(prompt_token_hash) and prompt_token_hash == conversion.get("token_hash")
    same_token_ids = baseline.get("prompt_token_ids") == load.get("prompt_token_ids")
    first_token_baseline = _first_token(baseline, "one_token")
    first_token_loaded = _first_token(load, "one_token")
    load_has_first_token = first_token_loaded is not None
    greedy_1tok_match = first_token_baseline is not None and first_token_baseline == first_token_loaded
    baseline_n = baseline.get("n_token", {}).get("output_token_ids", [])
    load_n = load.get("n_token", {}).get("output_token_ids", [])
    load_has_ntok = bool(load_n)
    greedy_ntok_match = bool(baseline_n) and baseline_n == load_n
    bundle_type = str(conversion.get("bundle_type") or "")
    reconstructed_scd = bundle_type not in {"", "full_consumer"}
    loaded_generation_ok = bool(load_has_first_token and load_has_ntok)
    exact_match_required = not reconstructed_scd
    adapter_smoke_pass = bool(
        conversion.get("num_layers")
        and bundle_token_match
        and same_token_ids
        and loaded_generation_ok
        and ((greedy_1tok_match and greedy_ntok_match) if exact_match_required else True)
    )
    return {
        "mode": "scd_vllm_injection_smoke",
        "adapter_smoke_pass": adapter_smoke_pass,
        "scd_injection_pass": bool(adapter_smoke_pass and reconstructed_scd),
        "reconstructed_scd": reconstructed_scd,
        "exact_match_required": exact_match_required,
        "loaded_generation_ok": loaded_generation_ok,
        "bundle_type": bundle_type,
        "model": model,
        "tokenizer": tokenizer,
        "prompt_token_count": baseline.get("prompt_token_count"),
        "prompt_token_hash": prompt_token_hash,
        "bundle_token_hash": conversion.get("token_hash"),
        "bundle_token_match": bundle_token_match,
        "same_token_ids": same_token_ids,
        "aligned_tokens": conversion.get("aligned_tokens"),
        "block_size": conversion.get("block_size"),
        "num_layers": conversion.get("num_layers"),
        "payload_bytes": conversion.get("payload_bytes"),
        "max_new_tokens": max_new_tokens,
        "first_token_id_baseline": first_token_baseline,
        "first_token_id_loaded": first_token_loaded,
        "greedy_1tok_match": greedy_1tok_match,
        "greedy_ntok_match": greedy_ntok_match,
        "conversion": conversion,
        "records": records,
        "notes": [
            "This smoke test proves the SCD bundle adapter can populate vLLM ExampleConnector storage.",
            "A full_consumer bundle is a bridge correctness test, not an approximate reconstructed SCD KV result.",
        ],
    }


def _load_token_ids(bundle: Path) -> np.ndarray:
    path = bundle / "token_ids.npy"
    if not path.is_file():
        raise SCDInjectionError(f"missing token_ids.npy: {path}")
    token_ids = np.load(path)
    if token_ids.ndim == 2 and token_ids.shape[0] == 1:
        token_ids = token_ids[0]
    if token_ids.ndim != 1:
        raise SCDInjectionError(f"token_ids must be [T] or [1,T], got {token_ids.shape}")
    return np.ascontiguousarray(token_ids.astype(np.int64, copy=False))


def _normalize_bundle_tensor(tensor: Any, aligned_tokens: int, torch: Any) -> Any:
    if int(tensor.shape[0]) < aligned_tokens:
        raise SCDInjectionError(f"KV tensor has {int(tensor.shape[0])} tokens, needs {aligned_tokens}")
    out = tensor[:aligned_tokens].contiguous()
    if out.dim() != 3:
        raise SCDInjectionError(f"expected normalized [T,H,D] bundle tensor, got {tuple(out.shape)}")
    return out.to(dtype=out.dtype)


def _resolve_prompt_for_bundle(bundle: Path, prompt: str, prefix_file: str) -> str:
    if prompt or prefix_file:
        return load_prompt(prompt, prefix_file)
    prefix_path = bundle / "prefix.txt"
    if not prefix_path.is_file():
        raise SCDInjectionError("no prompt/prefix-file supplied and bundle prefix.txt is missing")
    return load_prompt(prefix_path.read_text(encoding="utf-8"), "")


def _first_token(record: dict[str, Any], key: str) -> int | None:
    token_ids = record.get(key, {}).get("output_token_ids", [])
    return int(token_ids[0]) if token_ids else None


def _load_tensor_runtime() -> tuple[Any, Any, Any]:
    try:
        import torch
        from safetensors.torch import load_file, save_file
    except Exception as exc:  # pragma: no cover - depends on optional runtime environment
        raise SCDInjectionError("torch and safetensors are required for SCD-vLLM injection") from exc
    return torch, load_file, save_file


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "convert":
        metrics = convert_bundle_to_example_storage(
            bundle_dir=args.bundle_dir,
            storage_dir=args.storage_dir,
            block_size=args.block_size,
            layer_name_template=args.layer_name_template,
            overwrite=args.overwrite,
        )
        if args.out:
            out = Path(args.out)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
        return 0
    if args.command == "run":
        metrics = run_injection_smoke(args)
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
        return 0
    raise SCDInjectionError(f"unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
