from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Iterable

from scd_repro.cmrc import extract_answer_text, load_cmrc2018_samples, score_cmrc_prediction

from .vllm_baseline import _cuda_memory_info, _cuda_synchronize, _load_torch, _load_vllm


class VLLMDatasetEvalError(RuntimeError):
    """Raised when a vLLM dataset evaluation cannot be run safely."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full-dataset quality evals through official vLLM.")
    parser.add_argument("--dataset", default="cmrc2018", choices=["cmrc2018"])
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split", default="dev")
    parser.add_argument("--limit", type=int, default=0, help="0 means full split")
    parser.add_argument("--qid-file", default="", help="Optional qid allowlist, one qid per line")
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", default="")
    parser.add_argument("--out", required=True)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--enable-prefix-caching", action="store_true")
    parser.add_argument("--log-every", type=int, default=50)
    return parser.parse_args(argv)


def run_cmrc2018_vllm_eval(
    *,
    data_root: str | Path,
    split: str,
    model: str,
    tokenizer: str | None,
    out_dir: str | Path,
    limit: int = 0,
    qid_file: str | Path | None = None,
    dtype: str = "bfloat16",
    max_model_len: int = 2048,
    max_new_tokens: int = 32,
    batch_size: int = 16,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = 0,
    seed: int = 0,
    gpu_memory_utilization: float = 0.85,
    tensor_parallel_size: int = 1,
    trust_remote_code: bool = False,
    enforce_eager: bool = False,
    enable_prefix_caching: bool = False,
    log_every: int = 50,
) -> dict[str, Any]:
    if batch_size <= 0:
        raise VLLMDatasetEvalError("--batch-size must be positive")
    if max_new_tokens <= 0:
        raise VLLMDatasetEvalError("--max-new-tokens must be positive")
    if tensor_parallel_size <= 0:
        raise VLLMDatasetEvalError("--tensor-parallel-size must be positive")

    load_limit = None if int(limit or 0) <= 0 else int(limit)
    samples = load_cmrc2018_samples(data_root, limit=load_limit, split=split)
    qid_allowlist = _read_qid_allowlist(qid_file)
    if qid_allowlist is not None:
        samples = [sample for sample in samples if sample.qid in qid_allowlist]
    if not samples:
        raise VLLMDatasetEvalError("no CMRC2018 samples selected")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    per_sample_path = out / "per_sample_metrics.jsonl"

    LLM, SamplingParams, vllm_version = _load_vllm()
    torch = _load_torch()
    tokenizer_id = tokenizer or model
    llm_kwargs: dict[str, Any] = {
        "model": model,
        "tokenizer": tokenizer_id,
        "dtype": dtype,
        "seed": int(seed),
        "tensor_parallel_size": int(tensor_parallel_size),
        "gpu_memory_utilization": float(gpu_memory_utilization),
        "trust_remote_code": bool(trust_remote_code),
        "enforce_eager": bool(enforce_eager),
        "enable_prefix_caching": bool(enable_prefix_caching),
    }
    if max_model_len > 0:
        llm_kwargs["max_model_len"] = int(max_model_len)

    memory_before = _cuda_memory_info(torch)
    load_start = time.perf_counter()
    llm = LLM(**llm_kwargs)
    load_seconds = time.perf_counter() - load_start
    memory_after_load = _cuda_memory_info(torch)

    sampling = SamplingParams(
        temperature=float(temperature),
        top_p=float(top_p),
        top_k=int(top_k),
        max_tokens=int(max_new_tokens),
        seed=int(seed),
    )

    total_f1 = 0.0
    total_em = 0.0
    total_output_tokens = 0
    generation_seconds_total = 0.0
    first_prediction = ""
    first_reference = ""
    start_eval = time.perf_counter()

    with per_sample_path.open("w", encoding="utf-8") as fh:
        for offset in range(0, len(samples), int(batch_size)):
            batch = samples[offset : offset + int(batch_size)]
            prompts = [sample.prompt for sample in batch]
            _cuda_synchronize(torch)
            batch_start = time.perf_counter()
            outputs = llm.generate(prompts, sampling, use_tqdm=False)
            _cuda_synchronize(torch)
            batch_seconds = time.perf_counter() - batch_start
            generation_seconds_total += batch_seconds
            if len(outputs) != len(batch):
                raise VLLMDatasetEvalError(f"expected {len(batch)} vLLM outputs, got {len(outputs)}")

            for sample, output in zip(batch, outputs, strict=True):
                completion = output.outputs[0] if getattr(output, "outputs", None) else None
                raw_text = str(getattr(completion, "text", "") or "")
                prediction = extract_answer_text(raw_text)
                score = score_cmrc_prediction(prediction, sample.answers)
                output_token_ids = [int(x) for x in (getattr(completion, "token_ids", []) or [])]
                total_output_tokens += len(output_token_ids)
                total_f1 += float(score["f1"])
                total_em += float(score["em"])
                if not first_prediction:
                    first_prediction = prediction
                    first_reference = str(score.get("best_answer", ""))
                row = {
                    "qid": sample.qid,
                    "prediction": prediction,
                    "raw_generation": raw_text,
                    "answers": sample.answers,
                    "best_answer": score.get("best_answer", ""),
                    "f1": float(score["f1"]),
                    "em": float(score["em"]),
                    "output_token_count": len(output_token_ids),
                    "prompt_token_count": len(getattr(output, "prompt_token_ids", []) or []),
                }
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

            completed = min(offset + len(batch), len(samples))
            if log_every > 0 and (completed == len(samples) or completed % int(log_every) == 0):
                print(
                    json.dumps(
                        {
                            "progress": completed,
                            "total": len(samples),
                            "f1": total_f1 / completed,
                            "em": total_em / completed,
                            "batch_seconds": batch_seconds,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

    eval_seconds = time.perf_counter() - start_eval
    memory_after_generate = _cuda_memory_info(torch)
    sample_count = len(samples)
    summary = {
        "mode": "vllm_dataset_eval",
        "dataset": "cmrc2018",
        "split": split,
        "sample_count": sample_count,
        "f1": total_f1 / sample_count,
        "em": total_em / sample_count,
        "model": model,
        "tokenizer_id": tokenizer_id,
        "vllm_version": vllm_version,
        "dtype": dtype,
        "max_model_len": int(max_model_len),
        "max_new_tokens": int(max_new_tokens),
        "batch_size": int(batch_size),
        "tensor_parallel_size": int(tensor_parallel_size),
        "gpu_memory_utilization": float(gpu_memory_utilization),
        "temperature": float(temperature),
        "top_p": float(top_p),
        "top_k": int(top_k),
        "seed": int(seed),
        "enable_prefix_caching": bool(enable_prefix_caching),
        "load_seconds": load_seconds,
        "eval_seconds": eval_seconds,
        "generation_seconds_total": generation_seconds_total,
        "output_tokens": total_output_tokens,
        "output_tokens_per_second": (
            total_output_tokens / generation_seconds_total if generation_seconds_total > 0 else None
        ),
        "first_prediction": first_prediction,
        "first_reference": first_reference,
        "memory_before": memory_before,
        "memory_after_load": memory_after_load,
        "memory_after_generate": memory_after_generate,
        "per_sample_metrics": str(per_sample_path),
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def _read_qid_allowlist(path: str | Path | None) -> set[str] | None:
    if not path:
        return None
    qids = {line.strip() for line in Path(path).read_text(encoding="utf-8-sig").splitlines() if line.strip()}
    return qids


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.dataset != "cmrc2018":
        raise VLLMDatasetEvalError(f"unsupported dataset: {args.dataset}")
    summary = run_cmrc2018_vllm_eval(
        data_root=args.data_root,
        split=args.split,
        model=args.model,
        tokenizer=args.tokenizer or None,
        out_dir=args.out,
        limit=args.limit,
        qid_file=args.qid_file or None,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        seed=args.seed,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=args.trust_remote_code,
        enforce_eager=args.enforce_eager,
        enable_prefix_caching=args.enable_prefix_caching,
        log_every=args.log_every,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
