from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Iterable

from scd_repro.cmrc import extract_answer_text, load_cmrc2018_samples, score_cmrc_prediction
from scd_repro.text_benchmarks import load_text_benchmark_samples

from .vllm_baseline import _cuda_memory_info, _cuda_synchronize, _load_torch, _load_transformers, _load_vllm


class VLLMDatasetEvalError(RuntimeError):
    """Raised when a vLLM dataset evaluation cannot be run safely."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full-dataset quality evals through official vLLM.")
    parser.add_argument(
        "--dataset",
        default="cmrc2018",
        choices=["cmrc2018", "hotpotqa", "hotpotqa_compressed", "wikitext2", "wikitext-2"],
    )
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
    parser.add_argument("--apply-chat-template", action="store_true")
    parser.add_argument("--system-prompt", default="")
    parser.add_argument("--extra-instruction", default="")
    parser.add_argument("--compute-prompt-nll", action="store_true")
    parser.add_argument("--skip-prompt-nll", action="store_true")
    parser.add_argument("--log-every", type=int, default=50)
    return parser.parse_args(argv)


def run_cmrc2018_vllm_eval(
    *,
    dataset: str = "cmrc2018",
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
    apply_chat_template: bool = False,
    system_prompt: str = "",
    extra_instruction: str = "",
    compute_prompt_nll: bool | None = None,
    log_every: int = 50,
) -> dict[str, Any]:
    if batch_size <= 0:
        raise VLLMDatasetEvalError("--batch-size must be positive")
    if max_new_tokens <= 0:
        raise VLLMDatasetEvalError("--max-new-tokens must be positive")
    if tensor_parallel_size <= 0:
        raise VLLMDatasetEvalError("--tensor-parallel-size must be positive")

    dataset_key = str(dataset).strip().lower()
    load_limit = None if int(limit or 0) <= 0 else int(limit)
    if dataset_key == "cmrc2018":
        samples = load_cmrc2018_samples(data_root, limit=load_limit, split=split)
    else:
        samples = load_text_benchmark_samples(dataset_key, data_root, limit=load_limit, cmrc_split=split)
    qid_allowlist = _read_qid_allowlist(qid_file)
    if qid_allowlist is not None:
        samples = [sample for sample in samples if sample.qid in qid_allowlist]
    if not samples:
        raise VLLMDatasetEvalError(f"no {dataset_key} samples selected")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    per_sample_path = out / "per_sample_metrics.jsonl"

    LLM, SamplingParams, vllm_version = _load_vllm()
    torch = _load_torch()
    tokenizer_id = tokenizer or model
    chat_tokenizer = _load_chat_tokenizer(tokenizer_id, trust_remote_code) if apply_chat_template else None
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
            prompts = [
                _format_prompt(
                    sample.prompt,
                    chat_tokenizer=chat_tokenizer,
                    system_prompt=system_prompt,
                    extra_instruction=extra_instruction,
                )
                for sample in batch
            ]
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
    nll_summary = _maybe_compute_prompt_nll(
        llm=llm,
        SamplingParams=SamplingParams,
        samples=samples,
        dataset_key=dataset_key,
        tokenizer_id=tokenizer_id,
        trust_remote_code=trust_remote_code,
        batch_size=int(batch_size),
        seed=int(seed),
        compute_prompt_nll=compute_prompt_nll,
        apply_chat_template=apply_chat_template,
        extra_instruction=extra_instruction,
        torch=torch,
        log_every=int(log_every),
    )
    memory_after_generate = _cuda_memory_info(torch)
    sample_count = len(samples)
    summary = {
        "mode": "vllm_dataset_eval",
        "dataset": dataset_key,
        "split": split,
        "sample_count": sample_count,
        "f1": total_f1 / sample_count,
        "em": total_em / sample_count,
        "primary_metric": _primary_metric_for_dataset(dataset_key),
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
        "apply_chat_template": bool(apply_chat_template),
        "system_prompt": system_prompt,
        "extra_instruction": extra_instruction,
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
    summary.update(nll_summary)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def _read_qid_allowlist(path: str | Path | None) -> set[str] | None:
    if not path:
        return None
    qids = {line.strip() for line in Path(path).read_text(encoding="utf-8-sig").splitlines() if line.strip()}
    return qids


def _primary_metric_for_dataset(dataset_key: str) -> str:
    if dataset_key in {"wikitext2", "wikitext-2"}:
        return "ppl"
    return "f1_em"


def _auto_compute_prompt_nll(dataset_key: str) -> bool:
    return dataset_key in {"wikitext2", "wikitext-2"}


def _maybe_compute_prompt_nll(
    *,
    llm: Any,
    SamplingParams: Any,
    samples: list[Any],
    dataset_key: str,
    tokenizer_id: str,
    trust_remote_code: bool,
    batch_size: int,
    seed: int,
    compute_prompt_nll: bool | None,
    apply_chat_template: bool,
    extra_instruction: str,
    torch: Any,
    log_every: int,
) -> dict[str, Any]:
    should_compute = _auto_compute_prompt_nll(dataset_key) if compute_prompt_nll is None else bool(compute_prompt_nll)
    if not should_compute:
        return {"avg_nll": None, "ppl": None, "teacher_token_count": 0}
    if apply_chat_template or extra_instruction.strip():
        return {
            "avg_nll": None,
            "ppl": None,
            "teacher_token_count": 0,
            "nll_error": "prompt NLL scoring is disabled for chat-template or extra-instruction prompts",
        }

    transformers = _load_transformers()
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        tokenizer_id,
        trust_remote_code=trust_remote_code,
        use_fast=True,
    )
    try:
        sampling = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            max_tokens=1,
            seed=int(seed),
            prompt_logprobs=1,
        )
    except TypeError as exc:
        return {
            "avg_nll": None,
            "ppl": None,
            "teacher_token_count": 0,
            "nll_error": f"vLLM SamplingParams does not support prompt_logprobs: {exc}",
        }

    scoring_rows = [_build_nll_scoring_prompt(sample, tokenizer) for sample in samples]
    scoring_rows = [row for row in scoring_rows if row["target_token_count"] > 0]
    if not scoring_rows:
        return {
            "avg_nll": None,
            "ppl": None,
            "teacher_token_count": 0,
            "nll_error": "no target tokens available for prompt NLL scoring",
        }

    total_logp = 0.0
    total_tokens = 0
    missing_logprobs = 0
    start = time.perf_counter()
    for offset in range(0, len(scoring_rows), max(1, int(batch_size))):
        batch = scoring_rows[offset : offset + max(1, int(batch_size))]
        prompts = [str(row["text"]) for row in batch]
        _cuda_synchronize(torch)
        outputs = llm.generate(prompts, sampling, use_tqdm=False)
        _cuda_synchronize(torch)
        if len(outputs) != len(batch):
            raise VLLMDatasetEvalError(f"expected {len(batch)} vLLM NLL outputs, got {len(outputs)}")
        for row, output in zip(batch, outputs, strict=True):
            prompt_ids = [int(x) for x in (getattr(output, "prompt_token_ids", []) or [])]
            prompt_logprobs = getattr(output, "prompt_logprobs", None) or []
            target_start = int(row["target_start"])
            target_stop = min(len(prompt_ids), int(row["target_stop"]))
            for pos in range(max(1, target_start), target_stop):
                entry = prompt_logprobs[pos] if pos < len(prompt_logprobs) else None
                logprob = _lookup_prompt_logprob(entry, prompt_ids[pos])
                if logprob is None:
                    missing_logprobs += 1
                    continue
                total_logp += float(logprob)
                total_tokens += 1
        completed = min(offset + len(batch), len(scoring_rows))
        if log_every > 0 and (completed == len(scoring_rows) or completed % int(log_every) == 0):
            avg_nll = float(-total_logp / total_tokens) if total_tokens > 0 else None
            print(
                json.dumps(
                    {
                        "nll_progress": completed,
                        "total": len(scoring_rows),
                        "avg_nll": avg_nll,
                        "teacher_token_count": total_tokens,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    if total_tokens <= 0:
        return {
            "avg_nll": None,
            "ppl": None,
            "teacher_token_count": 0,
            "prompt_logprob_missing_count": missing_logprobs,
            "nll_error": "vLLM did not return logprobs for target prompt tokens",
        }
    avg_nll = float(-total_logp / total_tokens)
    return {
        "avg_nll": avg_nll,
        "ppl": float(math.exp(avg_nll)) if avg_nll < 50 else float("inf"),
        "teacher_token_count": int(total_tokens),
        "prompt_logprob_missing_count": int(missing_logprobs),
        "nll_seconds": float(time.perf_counter() - start),
    }


def _build_nll_scoring_prompt(sample: Any, tokenizer: Any) -> dict[str, Any]:
    prompt = str(getattr(sample, "prompt", "") or "")
    answers = list(getattr(sample, "answers", []) or [])
    target = str(answers[0]) if answers else ""
    sep = "" if prompt.endswith((" ", "\n", "\t")) else " "
    text = f"{prompt}{sep}{target}" if target else prompt
    prefix_ids = tokenizer(prompt, add_special_tokens=True).get("input_ids", [])
    full_ids = tokenizer(text, add_special_tokens=True).get("input_ids", [])
    target_start = min(len(prefix_ids), len(full_ids))
    return {
        "qid": str(getattr(sample, "qid", "")),
        "text": text,
        "target_start": target_start,
        "target_stop": len(full_ids),
        "target_token_count": max(0, len(full_ids) - target_start),
    }


def _lookup_prompt_logprob(entry: Any, token_id: int) -> float | None:
    if entry is None:
        return None
    value = None
    if isinstance(entry, dict):
        value = entry.get(token_id)
        if value is None:
            value = entry.get(str(token_id))
    if value is None:
        return None
    if hasattr(value, "logprob"):
        return float(value.logprob)
    if isinstance(value, dict) and "logprob" in value:
        return float(value["logprob"])
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _load_chat_tokenizer(tokenizer_id: str, trust_remote_code: bool) -> Any:
    transformers = _load_transformers()
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        tokenizer_id,
        trust_remote_code=trust_remote_code,
        use_fast=True,
    )
    if not hasattr(tokenizer, "apply_chat_template"):
        raise VLLMDatasetEvalError(f"tokenizer does not support apply_chat_template: {tokenizer_id}")
    return tokenizer


def _format_prompt(
    prompt: str,
    *,
    chat_tokenizer: Any | None,
    system_prompt: str = "",
    extra_instruction: str = "",
) -> str:
    text = str(prompt or "").rstrip()
    if extra_instruction.strip():
        text = f"{text}\n{extra_instruction.strip()}"
    if chat_tokenizer is None:
        return text
    messages: list[dict[str, str]] = []
    if system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt.strip()})
    messages.append({"role": "user", "content": text})
    return str(
        chat_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = run_cmrc2018_vllm_eval(
        dataset=args.dataset,
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
        apply_chat_template=args.apply_chat_template,
        system_prompt=args.system_prompt,
        extra_instruction=args.extra_instruction,
        compute_prompt_nll=False if args.skip_prompt_nll else (True if args.compute_prompt_nll else None),
        log_every=args.log_every,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
