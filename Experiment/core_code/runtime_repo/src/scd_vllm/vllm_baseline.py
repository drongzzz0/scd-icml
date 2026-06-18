from __future__ import annotations

import argparse
import importlib
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from .bundle import array_sha256
from .cache_env import configure_model_cache
from .hf_export import HFExportError, load_prompt


class VLLMBaselineError(RuntimeError):
    """Raised when the plain vLLM baseline cannot be measured safely."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a plain vLLM baseline for SCD-vLLM")
    parser.add_argument("--model", required=True, help="vLLM model id or local path")
    parser.add_argument("--tokenizer", default="", help="Tokenizer id/path; defaults to --model")
    parser.add_argument(
        "--cache-root",
        default="",
        help="Storage directory for HF/Transformers/ModelScope/Torch/vLLM caches and vLLM downloads.",
    )
    parser.add_argument("--prompt", default="", help="Prefix prompt text")
    parser.add_argument("--prefix-file", default="", help="File containing prefix prompt text")
    parser.add_argument("--hf-bundle", default="", help="Optional HF full_consumer bundle directory for token-hash check")
    parser.add_argument("--out", required=True, help="Output directory for metrics and generated text")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--ttft-proxy-repeat", type=int, default=3)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--enable-prefix-caching", action="store_true")
    parser.add_argument("--allow-token-mismatch", action="store_true")
    return parser.parse_args(argv)


def run_vllm_baseline(
    *,
    model: str,
    tokenizer: str | None,
    prompt: str,
    out_dir: str | Path,
    cache_root: str | Path | None = None,
    hf_bundle: str | None = None,
    dtype: str = "bfloat16",
    max_model_len: int = 512,
    max_new_tokens: int = 16,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = 0,
    seed: int = 0,
    warmup: int = 1,
    repeat: int = 3,
    ttft_proxy_repeat: int = 3,
    gpu_memory_utilization: float = 0.85,
    tensor_parallel_size: int = 1,
    trust_remote_code: bool = False,
    enforce_eager: bool = False,
    enable_prefix_caching: bool = False,
    allow_token_mismatch: bool = False,
) -> dict[str, Any]:
    if repeat <= 0:
        raise VLLMBaselineError("--repeat must be positive")
    if max_new_tokens <= 0:
        raise VLLMBaselineError("--max-new-tokens must be positive")
    if ttft_proxy_repeat < 0:
        raise VLLMBaselineError("--ttft-proxy-repeat must be non-negative")
    if tensor_parallel_size <= 0:
        raise VLLMBaselineError("--tensor-parallel-size must be positive")

    cache_env = configure_model_cache(cache_root, override=True)
    LLM, SamplingParams, vllm_version = _load_vllm()
    torch = _load_torch()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tokenizer_id = tokenizer or model

    memory_before = _cuda_memory_info(torch)
    load_start = time.perf_counter()
    llm_kwargs: dict[str, Any] = {
        "model": model,
        "tokenizer": tokenizer_id,
        "dtype": dtype,
        "seed": seed,
        "tensor_parallel_size": tensor_parallel_size,
        "gpu_memory_utilization": gpu_memory_utilization,
        "trust_remote_code": trust_remote_code,
        "enforce_eager": enforce_eager,
        "enable_prefix_caching": enable_prefix_caching,
    }
    if max_model_len > 0:
        llm_kwargs["max_model_len"] = max_model_len
    if cache_env.get("download_dir"):
        llm_kwargs["download_dir"] = str(cache_env["download_dir"])
    llm = LLM(**llm_kwargs)
    load_seconds = time.perf_counter() - load_start
    memory_after_load = _cuda_memory_info(torch)

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=max_new_tokens,
        seed=seed,
    )

    warmup_records: list[dict[str, Any]] = []
    for idx in range(max(0, warmup)):
        warmup_records.append(_generate_once(llm, sampling_params, prompt, torch, idx, kind="warmup"))

    ttft_proxy_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=1,
        seed=seed,
    )
    ttft_proxy_records = [
        _generate_once(llm, ttft_proxy_params, prompt, torch, idx, kind="ttft_proxy")
        for idx in range(ttft_proxy_repeat)
    ]
    records = [_generate_once(llm, sampling_params, prompt, torch, idx, kind="measure") for idx in range(repeat)]
    memory_after_generate = _cuda_memory_info(torch)
    prompt_token_ids = _prompt_token_ids_from_records(records) or _tokenize_prompt(tokenizer_id, prompt, trust_remote_code)
    token_ids_np = np.asarray([prompt_token_ids], dtype=np.int64)
    token_hash = array_sha256(token_ids_np)
    hf_check = _check_hf_bundle_token_hash(hf_bundle, token_hash, allow_token_mismatch)
    generated_outputs = [
        {
            "repeat_index": int(record["index"]),
            "text": record.get("text", ""),
            "output_token_ids": record.get("output_token_ids", []),
        }
        for record in records
    ]
    metrics = {
        "mode": "vllm_plain_baseline",
        "vllm_version": vllm_version,
        "model_cache": cache_env,
        "model": model,
        "tokenizer_id": tokenizer_id,
        "dtype": dtype,
        "max_model_len": max_model_len,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "seed": seed,
        "warmup": warmup,
        "repeat": repeat,
        "ttft_proxy_repeat": ttft_proxy_repeat,
        "enable_prefix_caching": enable_prefix_caching,
        "load_seconds": load_seconds,
        "prompt_token_count": len(prompt_token_ids),
        "prompt_token_hash": token_hash,
        "hf_bundle_check": hf_check,
        "memory_before": memory_before,
        "memory_after_load": memory_after_load,
        "memory_after_generate": memory_after_generate,
        "warmup_runs": warmup_records,
        "ttft_proxy_runs": ttft_proxy_records,
        "runs": records,
        "summary": summarize_records(records, ttft_proxy_records=ttft_proxy_records),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out / "outputs.json").write_text(json.dumps(generated_outputs, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out / "prompt.txt").write_text(prompt + "\n", encoding="utf-8")
    return metrics


def summarize_records(
    records: list[dict[str, Any]], *, ttft_proxy_records: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    generation_seconds = [float(record["generation_seconds"]) for record in records]
    output_token_counts = [int(record["output_token_count"]) for record in records]
    ttft_seconds = [float(record["ttft_seconds"]) for record in records if record.get("ttft_seconds") is not None]
    ttft_proxy_generation_seconds = [
        float(record["generation_seconds"]) for record in (ttft_proxy_records or []) if record.get("generation_seconds")
    ]
    mean_output_tokens = _mean(output_token_counts)
    mean_generation = _mean(generation_seconds)
    summary = {
        "generation_seconds_mean": mean_generation,
        "generation_seconds_min": min(generation_seconds),
        "generation_seconds_max": max(generation_seconds),
        "output_token_count_mean": mean_output_tokens,
        "tokens_per_second_mean": mean_output_tokens / mean_generation if mean_generation > 0 else None,
        "ttft_seconds_mean": _mean(ttft_seconds) if ttft_seconds else None,
        "ttft_seconds_min": min(ttft_seconds) if ttft_seconds else None,
        "ttft_source": "vllm_request_metrics" if ttft_seconds else "unavailable",
        "ttft_proxy_seconds_mean": _mean(ttft_proxy_generation_seconds) if ttft_proxy_generation_seconds else None,
        "ttft_proxy_seconds_min": min(ttft_proxy_generation_seconds) if ttft_proxy_generation_seconds else None,
        "ttft_proxy_source": "one_token_generate_wall_time" if ttft_proxy_generation_seconds else "unavailable",
    }
    return summary


def extract_request_output(record_output: Any, generation_seconds: float, index: int, kind: str) -> dict[str, Any]:
    output = record_output.outputs[0] if getattr(record_output, "outputs", None) else None
    output_token_ids = list(getattr(output, "token_ids", []) or [])
    metrics_obj = getattr(record_output, "metrics", None)
    raw_metrics = object_to_plain(metrics_obj)
    ttft_seconds = _timing_delta(raw_metrics, "arrival_time", "first_token_time")
    if ttft_seconds is None:
        ttft_seconds = _timing_delta(raw_metrics, "scheduled_time", "first_token_time")
    return {
        "kind": kind,
        "index": index,
        "generation_seconds": generation_seconds,
        "ttft_seconds": ttft_seconds,
        "output_token_count": len(output_token_ids),
        "output_token_ids": output_token_ids,
        "finish_reason": getattr(output, "finish_reason", None),
        "text": getattr(output, "text", "") if output is not None else "",
        "prompt_token_ids": list(getattr(record_output, "prompt_token_ids", []) or []),
        "request_metrics": raw_metrics,
    }


def object_to_plain(value: Any, depth: int = 0) -> Any:
    if depth > 4:
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): object_to_plain(v, depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [object_to_plain(v, depth + 1) for v in value]
    if hasattr(value, "__dict__"):
        return {
            str(k): object_to_plain(v, depth + 1)
            for k, v in vars(value).items()
            if not str(k).startswith("_")
        }
    return str(value)


def _generate_once(llm: Any, sampling_params: Any, prompt: str, torch: Any, index: int, kind: str) -> dict[str, Any]:
    _cuda_synchronize(torch)
    start = time.perf_counter()
    outputs = llm.generate([prompt], sampling_params, use_tqdm=False)
    _cuda_synchronize(torch)
    elapsed = time.perf_counter() - start
    if len(outputs) != 1:
        raise VLLMBaselineError(f"expected one vLLM output, got {len(outputs)}")
    return extract_request_output(outputs[0], elapsed, index, kind)


def _load_vllm() -> tuple[Any, Any, str | None]:
    try:
        vllm = importlib.import_module("vllm")
        return getattr(vllm, "LLM"), getattr(vllm, "SamplingParams"), getattr(vllm, "__version__", None)
    except Exception as exc:  # pragma: no cover - local Windows env does not have vLLM
        raise VLLMBaselineError("vLLM is required for Phase 3 plain baseline") from exc


def _load_torch() -> Any:
    try:
        return importlib.import_module("torch")
    except Exception:
        return None


def _load_transformers() -> Any:
    try:
        return importlib.import_module("transformers")
    except Exception as exc:  # pragma: no cover - transformers is required in runtime env
        raise HFExportError("transformers is required to tokenize the baseline prompt") from exc


def _tokenize_prompt(tokenizer_id: str, prompt: str, trust_remote_code: bool) -> list[int]:
    transformers = _load_transformers()
    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_id, trust_remote_code=trust_remote_code, use_fast=True)
    encoded = tokenizer(prompt, return_tensors=None, add_special_tokens=True)
    return [int(token) for token in encoded["input_ids"]]


def _prompt_token_ids_from_records(records: list[dict[str, Any]]) -> list[int]:
    for record in records:
        token_ids = record.get("prompt_token_ids")
        if token_ids:
            return [int(token) for token in token_ids]
    return []


def _check_hf_bundle_token_hash(hf_bundle: str | None, token_hash: str, allow_token_mismatch: bool) -> dict[str, Any]:
    if not hf_bundle:
        return {"enabled": False}
    manifest_path = Path(hf_bundle) / "manifest.json"
    if not manifest_path.exists():
        raise VLLMBaselineError(f"HF bundle manifest does not exist: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = str(manifest.get("token_hash") or "")
    matched = expected == token_hash
    if not matched and not allow_token_mismatch:
        raise VLLMBaselineError(f"vLLM prompt token hash differs from HF bundle: {token_hash} != {expected}")
    return {
        "enabled": True,
        "bundle": str(Path(hf_bundle)),
        "expected_token_hash": expected,
        "actual_token_hash": token_hash,
        "matched": matched,
    }


def _timing_delta(metrics: Any, start_key: str, end_key: str) -> float | None:
    if not isinstance(metrics, dict):
        return None
    start = _as_float(metrics.get(start_key))
    end = _as_float(metrics.get(end_key))
    if start is None or end is None or end < start:
        return None
    return end - start


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _mean(values: list[float] | list[int]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _cuda_synchronize(torch: Any) -> None:
    if torch is not None and getattr(torch, "cuda", None) is not None and torch.cuda.is_available():
        torch.cuda.synchronize()


def _cuda_memory_info(torch: Any) -> dict[str, Any]:
    if torch is None or getattr(torch, "cuda", None) is None or not torch.cuda.is_available():
        return {"cuda_available": False}
    free, total = torch.cuda.mem_get_info()
    return {
        "cuda_available": True,
        "device": int(torch.cuda.current_device()),
        "device_name": torch.cuda.get_device_name(torch.cuda.current_device()),
        "free_bytes": int(free),
        "total_bytes": int(total),
        "used_bytes": int(total - free),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    prompt = load_prompt(args.prompt, args.prefix_file)
    metrics = run_vllm_baseline(
        model=args.model,
        tokenizer=args.tokenizer or None,
        cache_root=args.cache_root or None,
        prompt=prompt,
        out_dir=args.out,
        hf_bundle=args.hf_bundle or None,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        seed=args.seed,
        warmup=args.warmup,
        repeat=args.repeat,
        ttft_proxy_repeat=args.ttft_proxy_repeat,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=args.trust_remote_code,
        enforce_eager=args.enforce_eager,
        enable_prefix_caching=args.enable_prefix_caching,
        allow_token_mismatch=args.allow_token_mismatch,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
