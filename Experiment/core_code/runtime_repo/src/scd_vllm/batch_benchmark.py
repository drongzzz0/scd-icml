from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence

from .bundle import array_sha256, read_manifest, text_sha256
from .hf_export import load_prompt
from .injection import SCDInjectionError
from .truekv_oracle import _cuda_memory_info, _cuda_synchronize, _load_optional, summarize_storage


class SCDBatchBenchmarkError(RuntimeError):
    """Raised when the SCD-vLLM batch benchmark cannot proceed safely."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a repeated-prefix vLLM batch benchmark for an existing SCD bundle/storage pair."
    )
    parser.add_argument("--model", required=True, help="vLLM model id or local path")
    parser.add_argument("--tokenizer", default="", help="Tokenizer id/path; defaults to --model")
    parser.add_argument("--bundle-dir", required=True, help="SCD bundle directory used for prompt and manifest metadata")
    parser.add_argument("--storage-dir", required=True, help="ExampleConnector storage generated from the bundle")
    parser.add_argument("--prompt", default="", help="Prefix prompt text; defaults to bundle prefix.txt")
    parser.add_argument("--prefix-file", default="", help="File containing prefix prompt text")
    parser.add_argument("--out", required=True, help="Output directory for metrics")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-batches", type=int, default=3)
    parser.add_argument("--warmup-batches", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--enable-prefix-caching", action="store_true")
    parser.add_argument("--enable-chunked-prefill", action="store_true")
    parser.add_argument("--disable-hybrid-kv-cache-manager", action="store_true")
    parser.add_argument(
        "--include-prefix-cache-baseline",
        action="store_true",
        help="Also run a plain-vLLM stage with vLLM prefix caching enabled.",
    )
    parser.add_argument("--skip-baseline", action="store_true", help="Run only the ExampleConnector load stage")
    parser.add_argument("--stage", choices=["baseline", "prefix-cache", "load"], default="", help=argparse.SUPPRESS)
    return parser.parse_args(list(argv) if argv is not None else None)


def run_batch_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    if args.batch_size <= 0:
        raise SCDBatchBenchmarkError(f"--batch-size must be positive, got {args.batch_size}")
    if args.num_batches <= 0:
        raise SCDBatchBenchmarkError(f"--num-batches must be positive, got {args.num_batches}")
    if args.warmup_batches < 0:
        raise SCDBatchBenchmarkError(f"--warmup-batches must be non-negative, got {args.warmup_batches}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    common = [
        "--model",
        args.model,
        "--tokenizer",
        args.tokenizer or args.model,
        "--bundle-dir",
        args.bundle_dir,
        "--storage-dir",
        args.storage_dir,
        "--prompt",
        args.prompt,
        "--prefix-file",
        args.prefix_file,
        "--batch-size",
        str(args.batch_size),
        "--num-batches",
        str(args.num_batches),
        "--warmup-batches",
        str(args.warmup_batches),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--dtype",
        args.dtype,
        "--max-model-len",
        str(args.max_model_len),
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
    stages = ["load"] if args.skip_baseline else ["baseline", "load"]
    if args.include_prefix_cache_baseline and not args.skip_baseline:
        stages.insert(1, "prefix-cache")
    for stage in stages:
        stage_out = out_dir / f"{stage}.json"
        cmd = [
            sys.executable,
            "-m",
            "scd_vllm.batch_benchmark",
            "--stage",
            stage,
            "--out",
            str(stage_out),
            *common,
        ]
        subprocess.run(cmd, check=True)
        records[stage] = json.loads(stage_out.read_text(encoding="utf-8"))

    metrics = build_batch_benchmark_metrics(
        model=args.model,
        tokenizer=args.tokenizer or args.model,
        bundle_dir=Path(args.bundle_dir),
        storage_dir=Path(args.storage_dir),
        records=records,
        batch_size=args.batch_size,
        num_batches=args.num_batches,
        warmup_batches=args.warmup_batches,
        max_new_tokens=args.max_new_tokens,
    )
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return metrics


def run_stage(args: argparse.Namespace) -> dict[str, Any]:
    prompt = _resolve_prompt(Path(args.bundle_dir), args.prompt, args.prefix_file)
    prompts = [prompt] * int(args.batch_size)
    vllm = importlib.import_module("vllm")
    torch = _load_optional("torch")
    sampling_params_cls = getattr(vllm, "SamplingParams")
    llm_cls = getattr(vllm, "LLM")
    llm_kwargs: dict[str, Any] = {
        "model": args.model,
        "tokenizer": args.tokenizer or args.model,
        "dtype": args.dtype,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "seed": args.seed,
        "enforce_eager": args.enforce_eager,
        "enable_prefix_caching": bool(args.enable_prefix_caching or args.stage == "prefix-cache"),
        "enable_chunked_prefill": args.enable_chunked_prefill,
        "disable_hybrid_kv_cache_manager": args.disable_hybrid_kv_cache_manager,
    }
    if args.stage == "load":
        if not args.storage_dir:
            raise SCDBatchBenchmarkError("--storage-dir is required for load stage")
        llm_kwargs["kv_transfer_config"] = {
            "kv_connector": "ExampleConnector",
            "kv_role": "kv_both",
            "kv_load_failure_policy": "fail",
            "kv_connector_extra_config": {"shared_storage_path": args.storage_dir},
        }

    memory_before = _cuda_memory_info(torch)
    start_load = time.perf_counter()
    llm = llm_cls(**llm_kwargs)
    load_seconds = time.perf_counter() - start_load
    memory_after_load = _cuda_memory_info(torch)
    sampling = sampling_params_cls(
        max_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
    )
    for _ in range(int(args.warmup_batches)):
        llm.generate(prompts, sampling, use_tqdm=False)
    _cuda_synchronize(torch)

    batch_records: list[dict[str, Any]] = []
    total_generation_seconds = 0.0
    total_output_tokens = 0
    total_requests = 0
    prompt_token_ids: list[int] | None = None
    for batch_idx in range(int(args.num_batches)):
        _cuda_synchronize(torch)
        start = time.perf_counter()
        outputs = llm.generate(prompts, sampling, use_tqdm=False)
        _cuda_synchronize(torch)
        elapsed = time.perf_counter() - start
        total_generation_seconds += elapsed
        batch_output_tokens = 0
        first_output_ids: list[int] = []
        for request_idx, request_output in enumerate(outputs):
            completion = request_output.outputs[0] if request_output.outputs else None
            token_ids = [int(x) for x in (getattr(completion, "token_ids", []) or [])]
            batch_output_tokens += len(token_ids)
            if request_idx == 0:
                first_output_ids = token_ids
                prompt_token_ids = [int(x) for x in (getattr(request_output, "prompt_token_ids", []) or [])]
        total_output_tokens += batch_output_tokens
        total_requests += len(outputs)
        batch_records.append(
            {
                "batch_index": batch_idx,
                "request_count": len(outputs),
                "generation_seconds": elapsed,
                "output_tokens": batch_output_tokens,
                "tokens_per_second": _safe_div(batch_output_tokens, elapsed),
                "first_output_token_ids": first_output_ids,
            }
        )
    if prompt_token_ids is None:
        raise SCDBatchBenchmarkError("no prompt token ids returned by vLLM")
    record = {
        "stage": args.stage,
        "mode": "scd_vllm_batch_benchmark_stage",
        "vllm_version": getattr(vllm, "__version__", None),
        "model": args.model,
        "tokenizer": args.tokenizer or args.model,
        "bundle_dir": args.bundle_dir,
        "storage_dir": args.storage_dir,
        "prompt_text_hash": text_sha256(prompt),
        "prompt_token_count": len(prompt_token_ids),
        "prompt_token_ids": prompt_token_ids,
        "prompt_token_hash": array_sha256(_np_array_2d(prompt_token_ids)),
        "batch_size": int(args.batch_size),
        "num_batches": int(args.num_batches),
        "warmup_batches": int(args.warmup_batches),
        "max_new_tokens": int(args.max_new_tokens),
        "load_seconds": load_seconds,
        "generation_seconds": total_generation_seconds,
        "request_count": total_requests,
        "output_tokens": total_output_tokens,
        "requests_per_second": _safe_div(total_requests, total_generation_seconds),
        "tokens_per_second": _safe_div(total_output_tokens, total_generation_seconds),
        "memory_before": memory_before,
        "memory_after_load": memory_after_load,
        "storage": summarize_storage(args.storage_dir) if args.storage_dir else {"exists": False},
        "batches": batch_records,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return record


def build_batch_benchmark_metrics(
    *,
    model: str,
    tokenizer: str,
    bundle_dir: Path,
    storage_dir: Path,
    records: dict[str, dict[str, Any]],
    batch_size: int,
    num_batches: int,
    warmup_batches: int,
    max_new_tokens: int,
) -> dict[str, Any]:
    load = records.get("load", {})
    baseline = records.get("baseline", {})
    prefix_cache = records.get("prefix-cache", {})
    manifest = _load_manifest_metadata(bundle_dir)
    storage = load.get("storage", {})
    expected_requests = int(batch_size) * int(num_batches)
    load_generation_ok = bool(
        load.get("request_count") == expected_requests
        and int(load.get("output_tokens", 0) or 0) > 0
        and float(load.get("tokens_per_second", 0.0) or 0.0) > 0.0
    )
    token_match = bool(load.get("prompt_token_hash") and load.get("prompt_token_hash") == manifest.get("token_hash"))
    storage_ready = bool(storage.get("exists") and int(storage.get("safetensor_files", 0) or 0) > 0)
    benchmark_pass = bool(load_generation_ok and token_match and storage_ready)
    speedup_tokens_per_second: float | None = None
    speedup_requests_per_second: float | None = None
    if baseline:
        speedup_tokens_per_second = _safe_div(
            float(load.get("tokens_per_second", 0.0) or 0.0),
            float(baseline.get("tokens_per_second", 0.0) or 0.0),
        )
        speedup_requests_per_second = _safe_div(
            float(load.get("requests_per_second", 0.0) or 0.0),
            float(baseline.get("requests_per_second", 0.0) or 0.0),
        )
    prefix_cache_generation_ok = bool(
        prefix_cache
        and prefix_cache.get("request_count") == expected_requests
        and int(prefix_cache.get("output_tokens", 0) or 0) > 0
        and float(prefix_cache.get("tokens_per_second", 0.0) or 0.0) > 0.0
        and prefix_cache.get("prompt_token_hash") == manifest.get("token_hash")
    )
    prefix_cache_vs_baseline_tokens_per_second: float | None = None
    prefix_cache_vs_baseline_requests_per_second: float | None = None
    load_vs_prefix_cache_tokens_per_second: float | None = None
    load_vs_prefix_cache_requests_per_second: float | None = None
    if prefix_cache and baseline:
        prefix_cache_vs_baseline_tokens_per_second = _safe_div(
            float(prefix_cache.get("tokens_per_second", 0.0) or 0.0),
            float(baseline.get("tokens_per_second", 0.0) or 0.0),
        )
        prefix_cache_vs_baseline_requests_per_second = _safe_div(
            float(prefix_cache.get("requests_per_second", 0.0) or 0.0),
            float(baseline.get("requests_per_second", 0.0) or 0.0),
        )
    if prefix_cache:
        load_vs_prefix_cache_tokens_per_second = _safe_div(
            float(load.get("tokens_per_second", 0.0) or 0.0),
            float(prefix_cache.get("tokens_per_second", 0.0) or 0.0),
        )
        load_vs_prefix_cache_requests_per_second = _safe_div(
            float(load.get("requests_per_second", 0.0) or 0.0),
            float(prefix_cache.get("requests_per_second", 0.0) or 0.0),
        )
    return {
        "mode": "scd_vllm_batch_benchmark",
        "benchmark_pass": benchmark_pass,
        "claim_boundary": "repeated-prefix trusted microbenchmark; compares plain vLLM, optional vLLM prefix caching, and SCD load; not a multi-tenant or general throughput claim",
        "model": model,
        "tokenizer": tokenizer,
        "bundle_dir": str(bundle_dir),
        "storage_dir": str(storage_dir),
        "bundle_type": manifest.get("bundle_type"),
        "route_id": manifest.get("route_id"),
        "route_hash": manifest.get("route_hash"),
        "batch_size": int(batch_size),
        "num_batches": int(num_batches),
        "warmup_batches": int(warmup_batches),
        "max_new_tokens": int(max_new_tokens),
        "expected_requests": expected_requests,
        "load_generation_ok": load_generation_ok,
        "bundle_token_match": token_match,
        "storage_ready": storage_ready,
        "load_tokens_per_second": load.get("tokens_per_second"),
        "load_requests_per_second": load.get("requests_per_second"),
        "baseline_tokens_per_second": baseline.get("tokens_per_second") if baseline else None,
        "baseline_requests_per_second": baseline.get("requests_per_second") if baseline else None,
        "prefix_cache_generation_ok": prefix_cache_generation_ok if prefix_cache else None,
        "prefix_cache_tokens_per_second": prefix_cache.get("tokens_per_second") if prefix_cache else None,
        "prefix_cache_requests_per_second": prefix_cache.get("requests_per_second") if prefix_cache else None,
        "speedup_tokens_per_second": speedup_tokens_per_second,
        "speedup_requests_per_second": speedup_requests_per_second,
        "prefix_cache_vs_baseline_tokens_per_second": prefix_cache_vs_baseline_tokens_per_second,
        "prefix_cache_vs_baseline_requests_per_second": prefix_cache_vs_baseline_requests_per_second,
        "load_vs_prefix_cache_tokens_per_second": load_vs_prefix_cache_tokens_per_second,
        "load_vs_prefix_cache_requests_per_second": load_vs_prefix_cache_requests_per_second,
        "records": records,
    }


def _resolve_prompt(bundle_dir: Path, prompt: str, prefix_file: str) -> str:
    if prompt or prefix_file:
        return load_prompt(prompt, prefix_file)
    prefix_path = bundle_dir / "prefix.txt"
    if not prefix_path.is_file():
        raise SCDInjectionError("no prompt/prefix-file supplied and bundle prefix.txt is missing")
    return load_prompt(prefix_path.read_text(encoding="utf-8"), "")


def _load_manifest_metadata(bundle_dir: Path) -> dict[str, Any]:
    manifest_path = bundle_dir / "manifest.json"
    if not manifest_path.is_file():
        return {}
    manifest = read_manifest(manifest_path)
    return {
        "bundle_type": manifest.bundle_type,
        "route_id": manifest.route_id,
        "route_hash": manifest.route_hash,
        "token_hash": manifest.token_hash,
        "num_tokens": manifest.num_tokens,
    }


def _safe_div(numerator: float | int, denominator: float | int) -> float:
    denominator_f = float(denominator)
    if denominator_f <= 0.0:
        return 0.0
    return float(numerator) / denominator_f


def _np_array_2d(token_ids: list[int]) -> Any:
    np = importlib.import_module("numpy")
    return np.asarray([token_ids], dtype=np.int64)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.stage:
        payload = run_stage(args)
    else:
        payload = run_batch_benchmark(args)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload.get("benchmark_pass", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
