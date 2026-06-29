from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from ..io_utils import load_json, save_json
from ..pipeline import load_patch_ckpt, load_reuse_weights
from ..prompting import apply_prompt_style_to_samples, normalize_prompt_style
from ..reuse import move_reuse_weights
from ..seed import set_global_seed
from ..strict_runtime import (
    MethodSampleResult,
    TransformersRuntimeSession,
    assert_same_tokenization,
    evaluate_cmrc_method,
    summarize_method_results,
)
from ..text_benchmarks import load_text_benchmark_samples
from ..torch_utils import clear_torch_cache, configure_torch_acceleration, resolve_torch_dtype
from ..tracking import command_from_argv, write_manifest
from .common import add_accel_args, load_cfg, resolve_accel_options, resolve_logger
from .eval_cmrc_strict import (
    _apply_patch_runtime_overrides,
    _comparison_row,
    _consumer_capture_flags,
    _patch_ckpt_load_device,
    _resolve_qid_allowlist,
    _route_counts,
    _runtime_eval_cfg,
    _sample_result_to_dict,
    _select_runtime_methods,
    _write_jsonl,
    _write_summary_csv,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Strict text benchmark runtime comparison for SCD")
    p.add_argument("--dataset", required=True, choices=["cmrc2018", "hotpotqa", "hotpotqa_compressed", "wikitext2", "wikitext-2"])
    p.add_argument("--models-config", required=True)
    p.add_argument("--eval-config", default="")
    p.add_argument("--data-root", required=True)
    p.add_argument("--route", required=True)
    p.add_argument("--reuse-ckpt", required=True)
    p.add_argument("--patch-ckpt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int, default=200, help="0 means use all loadable samples")
    p.add_argument("--max-length", type=int, default=0)
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--bandwidth-gbps", type=float, default=200.0)
    p.add_argument("--quant-group-size", type=int, default=128)
    p.add_argument("--producer-device", default="")
    p.add_argument("--consumer-device", default="")
    p.add_argument("--dtype", default="")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log", default="")
    p.add_argument("--cmrc-split", default="", help="CMRC2018 split for eval; defaults to runtime_eval.cmrc_split or dev")
    p.add_argument("--qids", default="")
    p.add_argument("--qid-file", default="")
    add_accel_args(p)
    return p


def _add_ppl(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        nll = float(row.get("avg_nll", 0.0))
        row["ppl"] = float(math.exp(nll)) if nll < 50 else float("inf")


def _write_summary_md(path: Path, dataset: str, rows: list[dict[str, object]]) -> None:
    lines = [
        f"# Strict Runtime Comparison: {dataset}",
        "",
        "| mode | avg_nll | ppl | f1 | em | payload_mb | compression | ttft_ms | total_ms | delta_ttft_vs_full |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        compression = row.get("compression_ratio")
        delta_ttft = row.get("delta_ttft_vs_full")
        lines.append(
            f"| {row['method_label']} | "
            f"{float(row['avg_nll']):.4f} | "
            f"{float(row.get('ppl', 0.0)):.4f} | "
            f"{float(row['f1']):.4f} | "
            f"{float(row['em']):.4f} | "
            f"{float(row['payload_mb']):.4f} | "
            f"{0.0 if compression is None else float(compression):.3f} | "
            f"{float(row['ttft_ms']):.2f} | "
            f"{float(row['total_ms']):.2f} | "
            f"{0.0 if delta_ttft is None else float(delta_ttft):.2f} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _methods_need_patch(methods: list[str]) -> bool:
    return any(
        method
        in {
            "patch_only",
            "reuse_patch",
            "reuse_patch_recompute",
            "scd_full",
            "oracle_kv_patchonly",
        }
        for method in methods
    )


def _methods_need_reuse(methods: list[str]) -> bool:
    return any(
        method
        in {
            "reuse_only",
            "reuse_fallback",
            "reuse_patch",
            "reuse_patch_recompute",
            "scd_full",
            "oracle_xnorm_suffix",
            "oracle_state_suffix",
            "oracle_kv_patchonly",
        }
        for method in methods
    )


def _guard_hotpot_prompt_truncation(
    *,
    dataset: str,
    samples: list[Any],
    tokenizer_path: str,
    max_length: int,
) -> None:
    dataset_key = str(dataset).strip().lower()
    if dataset_key not in {"hotpotqa", "hotpotqa_compressed"} or int(max_length) <= 0:
        return
    try:
        from transformers import AutoTokenizer
    except Exception as exc:
        raise RuntimeError("HotpotQA truncation guard requires transformers tokenizer support") from exc

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    bad: list[tuple[str, int, int]] = []
    for sample in samples:
        full_ids = tokenizer(sample.prompt, truncation=False, return_tensors=None)["input_ids"]
        trunc_ids = tokenizer(
            sample.prompt,
            truncation=True,
            max_length=int(max_length),
            return_tensors=None,
        )["input_ids"]
        trunc_text = tokenizer.decode(trunc_ids, skip_special_tokens=False)
        if "Answer:" not in trunc_text:
            bad.append((str(sample.qid), len(full_ids), len(trunc_ids)))
            if len(bad) >= 5:
                break
    if bad:
        examples = ", ".join(f"{qid}:raw={raw},trunc={trunc}" for qid, raw, trunc in bad)
        raise ValueError(
            "HotpotQA prompt truncation removed the final Answer marker; this makes generation continue "
            "inside the evidence and invalidates F1/EM. Use --dataset hotpotqa_compressed, increase "
            f"--max-length, or disable truncation. Examples: {examples}"
        )


def _prefill_capture_to_cpu(capture):
    torch = __import__("torch")

    def move(value):
        if hasattr(value, "detach"):
            return value.detach().to(device="cpu").contiguous()
        return value

    capture.input_ids = move(capture.input_ids)
    capture.attention_mask = move(capture.attention_mask)
    capture.logits_last = move(capture.logits_last)
    capture.past_key_values = [(move(k), move(v)) for k, v in capture.past_key_values]
    capture.xnorm_by_layer = {int(layer): move(value) for layer, value in capture.xnorm_by_layer.items()}
    capture.residual_by_layer = {int(layer): move(value) for layer, value in capture.residual_by_layer.items()}
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return capture


def _sample_slices(num_samples: int, chunk_size: int):
    if chunk_size <= 0 or chunk_size >= num_samples:
        yield 0, num_samples
        return
    for start in range(0, num_samples, chunk_size):
        yield start, min(num_samples, start + chunk_size)


def main() -> None:
    args = build_parser().parse_args()
    logger = resolve_logger(args.log)
    set_global_seed(args.seed)

    models_cfg = load_cfg(args.models_config)
    eval_cfg = load_cfg(args.eval_config) if args.eval_config else {}
    runtime_eval_cfg = _runtime_eval_cfg(eval_cfg)
    methods = _select_runtime_methods(runtime_eval_cfg)
    prompt_style = normalize_prompt_style(runtime_eval_cfg.get("prompt_style", ""))

    accel = resolve_accel_options(models_cfg, args)
    configure_torch_acceleration(str(accel["accel_mode"]))
    logger.info("accel_mode=%s", accel["accel_mode"])

    model_a_path = str(models_cfg["model_a"]["path"])
    model_b_path = str(models_cfg["model_b"]["path"])
    runtime_cfg = models_cfg.get("runtime", {})
    compute_cfg = models_cfg.get("compute", {})
    network_cfg = eval_cfg.get("network", {}) if isinstance(eval_cfg.get("network", {}), dict) else {}

    producer_device = str(args.producer_device or compute_cfg.get("model_a_device") or runtime_cfg.get("device") or "cuda:0")
    consumer_device = str(args.consumer_device or compute_cfg.get("model_b_device") or runtime_cfg.get("device") or "cuda:1")
    dtype = str(args.dtype or runtime_cfg.get("dtype") or "bfloat16")
    transfer_dtype = resolve_torch_dtype(dtype)
    bandwidth_gbps = float(network_cfg.get("bandwidth_gbps", args.bandwidth_gbps))
    max_eval_samples = int(runtime_eval_cfg.get("max_eval_samples", args.limit))
    load_limit = None if max_eval_samples <= 0 else max_eval_samples
    default_recompute_segment_length = int(runtime_eval_cfg.get("default_recompute_segment_length", 0) or 0)
    override_route_recompute_length = runtime_eval_cfg.get("override_route_recompute_length", None)
    override_route_recompute_length = None if override_route_recompute_length in {"", None} else int(override_route_recompute_length)
    measure_teacher_forced_metrics = bool(runtime_eval_cfg.get("measure_teacher_forced_metrics", True))
    align_full_consumer_first_decode = bool(runtime_eval_cfg.get("align_full_consumer_first_decode", False))
    generation_stop_cfg = runtime_eval_cfg.get("generation_stop", {})
    if not isinstance(generation_stop_cfg, dict):
        generation_stop_cfg = {}
    save_per_sample_metrics = bool(runtime_eval_cfg.get("save_per_sample_metrics", True))
    save_generation_outputs = bool(runtime_eval_cfg.get("save_generation_outputs", True))
    fail_on_missing_route_fields = bool(runtime_eval_cfg.get("fail_on_missing_route_fields", True))
    fail_on_missing_patch_fields = bool(runtime_eval_cfg.get("fail_on_missing_patch_fields", True))
    same_device_capture_chunk_size = int(runtime_eval_cfg.get("same_device_capture_chunk_size", 0) or 0)
    producer_capture_log_every = int(runtime_eval_cfg.get("producer_capture_log_every", 25) or 0)
    qid_allowlist = _resolve_qid_allowlist(args, runtime_eval_cfg)

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    route = load_json(args.route)
    same_runtime_device = producer_device == consumer_device
    if _methods_need_reuse(methods):
        reuse_cpu, reuse_meta = load_reuse_weights(args.reuse_ckpt, device="cpu", dtype="float32")
    else:
        reuse_cpu, reuse_meta = None, {}
    patch_ckpt_device = _patch_ckpt_load_device(runtime_eval_cfg, consumer_device)
    if _methods_need_patch(methods):
        patch_bundle, patch_meta = load_patch_ckpt(args.patch_ckpt, device=patch_ckpt_device, dtype="float32")
        _apply_patch_runtime_overrides(patch_meta, runtime_eval_cfg)
    else:
        patch_bundle, patch_meta = None, {}
    if reuse_cpu is not None:
        reuse_prod = move_reuse_weights(reuse_cpu, "cpu" if same_runtime_device else producer_device, dtype="float32", cpu_fallback=True)
        reuse_cons = move_reuse_weights(reuse_cpu, consumer_device, dtype="float32", cpu_fallback=True)
    else:
        reuse_prod = None
        reuse_cons = None

    logger.info(
        "strict text runtime: dataset=%s route=%s patch_ckpt=%s reuse_ckpt=%s methods=%s fit_mode=%s semantic_source=%s",
        args.dataset,
        args.route,
        args.patch_ckpt,
        args.reuse_ckpt,
        methods,
        patch_meta.get("fit_mode"),
        patch_meta.get("semantic_source"),
    )
    logger.info(
        "patch runtime: ckpt_device=%s window_size=%s persist_shards=%s persist_modules=%s",
        patch_ckpt_device,
        patch_meta.get("runtime_patch_window_size", 0),
        patch_meta.get("runtime_patch_persist_shards", False),
        patch_meta.get("runtime_patch_persist_modules", False),
    )
    if reuse_cpu is not None:
        logger.info(
            "reuse mainline: mode=%s translator.default_path=%s num_groups=%s group_size=%s rank_k=%s rank_v=%s",
            reuse_meta.get("mode", "xkv_group"),
            reuse_meta.get("translator", {}).get("default_path", "shared_z"),
            reuse_meta.get("num_groups"),
            reuse_meta.get("group_size"),
            reuse_meta.get("rank_k", reuse_meta.get("rank")),
            reuse_meta.get("rank_v", reuse_meta.get("rank")),
        )
    else:
        logger.info("reuse mainline: skipped for methods=%s", methods)

    cmrc_split = str(args.cmrc_split or runtime_eval_cfg.get("cmrc_split") or "dev").strip().lower()
    samples = load_text_benchmark_samples(args.dataset, args.data_root, limit=load_limit, cmrc_split=cmrc_split)
    logger.info(
        "loaded samples: dataset=%s cmrc_split=%s n=%d",
        args.dataset,
        cmrc_split if str(args.dataset).strip().lower() == "cmrc2018" else "",
        len(samples),
    )
    if qid_allowlist:
        qid_set = set(qid_allowlist)
        samples = [sample for sample in samples if sample.qid in qid_set]
        logger.info("after qid filter: n=%d / requested=%d", len(samples), len(qid_allowlist))
        if not samples:
            raise ValueError("qid filter removed all samples")
    if prompt_style not in {"", "default", "cmrc_default"}:
        from transformers import AutoTokenizer

        prompt_tokenizer = AutoTokenizer.from_pretrained(model_b_path, trust_remote_code=True)
        samples = apply_prompt_style_to_samples(
            samples,
            style=prompt_style,
            tokenizer=prompt_tokenizer,
            dataset=args.dataset,
        )
        logger.info("applied prompt_style=%s to %d samples", prompt_style, len(samples))
    _guard_hotpot_prompt_truncation(
        dataset=args.dataset,
        samples=samples,
        tokenizer_path=model_b_path,
        max_length=int(args.max_length),
    )
    consumer_capture_xnorm, consumer_capture_residual = _consumer_capture_flags(methods)

    producer = None
    consumer = None
    try:
        all_results: dict[str, list[MethodSampleResult]] = {method: [] for method in methods}
        sample_logs: dict[str, list[dict[str, object]]] = {method: [] for method in methods}
        combined_sample_logs: list[dict[str, object]] = []

        def evaluate_one(idx: int, sample, producer_capture, consumer_capture, producer_session, consumer_session) -> None:
            assert_same_tokenization(producer_capture, consumer_capture)

            for method in methods:
                result = evaluate_cmrc_method(
                    method=method,
                    sample=sample,
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
                    max_new_tokens=int(args.max_new_tokens),
                    quant_group_size=int(args.quant_group_size),
                    transfer_dtype=transfer_dtype,
                    default_recompute_segment_length=default_recompute_segment_length,
                    override_route_recompute_length=override_route_recompute_length,
                    fail_on_missing_route_fields=fail_on_missing_route_fields,
                    fail_on_missing_patch_fields=fail_on_missing_patch_fields,
                    measure_teacher_forced_metrics=measure_teacher_forced_metrics,
                    align_full_consumer_first_decode=align_full_consumer_first_decode,
                    generation_stop=generation_stop_cfg,
                )
                all_results[method].append(result)
                row = _sample_result_to_dict(result)
                sample_logs[method].append(row)
                combined_sample_logs.append(row)

            logger.info("strict text progress: %d/%d qid=%s", idx, len(samples), sample.qid)

        if same_runtime_device:
            logger.info(
                "same-device strict eval: staging producer captures on CPU before loading consumer "
                "capture_chunk_size=%s producer_capture_log_every=%s",
                same_device_capture_chunk_size,
                producer_capture_log_every,
            )
            for chunk_start, chunk_end in _sample_slices(len(samples), same_device_capture_chunk_size):
                chunk_samples = samples[chunk_start:chunk_end]
                logger.info(
                    "same-device capture chunk start: samples %d-%d / %d",
                    chunk_start + 1,
                    chunk_end,
                    len(samples),
                )
                logger.info("same-device producer session load start: device=%s", producer_device)
                producer = TransformersRuntimeSession(model_a_path, producer_device, dtype)
                logger.info("same-device producer session load done: device=%s", producer_device)
                producer_session_proxy = SimpleNamespace(rope_spec=producer.rope_spec)
                producer_captures = []
                try:
                    for local_idx, sample in enumerate(chunk_samples, start=1):
                        global_idx = chunk_start + local_idx
                        producer_capture = producer.prefill(
                            sample.prompt,
                            max_length=args.max_length,
                            capture_xnorm=True,
                            profile_layers=False,
                        )
                        producer_captures.append(_prefill_capture_to_cpu(producer_capture))
                        del producer_capture
                        clear_torch_cache()
                        if producer_capture_log_every > 0 and (
                            global_idx == 1
                            or global_idx == len(samples)
                            or local_idx == len(chunk_samples)
                            or local_idx % producer_capture_log_every == 0
                        ):
                            logger.info(
                                "same-device producer capture progress: %d/%d qid=%s",
                                global_idx,
                                len(samples),
                                sample.qid,
                            )
                finally:
                    producer.close()
                    producer = None
                    clear_torch_cache()

                logger.info("same-device consumer session load start: device=%s", consumer_device)
                consumer = TransformersRuntimeSession(model_b_path, consumer_device, dtype)
                logger.info("same-device consumer session load done: device=%s", consumer_device)
                try:
                    for local_idx, sample in enumerate(chunk_samples, start=1):
                        global_idx = chunk_start + local_idx
                        producer_capture = producer_captures[local_idx - 1]
                        consumer_capture = consumer.prefill(
                            sample.prompt,
                            max_length=args.max_length,
                            capture_xnorm=consumer_capture_xnorm,
                            capture_residual=consumer_capture_residual,
                            profile_layers=True,
                        )
                        evaluate_one(global_idx, sample, producer_capture, consumer_capture, producer_session_proxy, consumer)
                        del producer_capture, consumer_capture
                        clear_torch_cache()
                finally:
                    consumer.close()
                    consumer = None
                    del producer_captures
                    clear_torch_cache()
        else:
            producer = TransformersRuntimeSession(model_a_path, producer_device, dtype)
            consumer = TransformersRuntimeSession(model_b_path, consumer_device, dtype)
            for idx, sample in enumerate(samples, start=1):
                producer_capture = producer.prefill(sample.prompt, max_length=args.max_length, capture_xnorm=True, profile_layers=False)
                consumer_capture = consumer.prefill(
                    sample.prompt,
                    max_length=args.max_length,
                    capture_xnorm=consumer_capture_xnorm,
                    capture_residual=consumer_capture_residual,
                    profile_layers=True,
                )
                evaluate_one(idx, sample, producer_capture, consumer_capture, producer, consumer)
                del producer_capture, consumer_capture
                clear_torch_cache()

        summary_rows = [summarize_method_results(all_results[method]) for method in methods]
        _add_ppl(summary_rows)
        summary_map = {str(row["method"]): row for row in summary_rows}
        full_baseline = summary_map.get("full_consumer")
        reuse_only_baseline = summary_map.get("reuse_only")
        for row in summary_rows:
            delta_full = _comparison_row(row, full_baseline)
            delta_reuse = _comparison_row(row, reuse_only_baseline)
            row["delta_ttft_vs_full"] = delta_full.get("delta_ttft_ms")
            row["delta_logp_vs_full"] = delta_full.get("delta_logp")
            row["delta_nll_vs_full"] = delta_full.get("delta_nll")
            row["delta_ttft_vs_reuse_only"] = delta_reuse.get("delta_ttft_ms")
            row["delta_logp_vs_reuse_only"] = delta_reuse.get("delta_logp")
            row["delta_nll_vs_reuse_only"] = delta_reuse.get("delta_nll")

        per_mode_summary = {str(row["method"]): row for row in summary_rows}
        final_comparison = {
            "meta": {
                "dataset": args.dataset,
                "data_root": args.data_root,
                "cmrc_split": cmrc_split if str(args.dataset).strip().lower() == "cmrc2018" else None,
                "route_path": args.route,
                "patch_ckpt_path": args.patch_ckpt,
                "reuse_ckpt_path": args.reuse_ckpt,
                "eval_config_path": args.eval_config,
                "models_config_path": args.models_config,
                "producer_model": model_a_path,
                "consumer_model": model_b_path,
                "producer_device": producer_device,
                "consumer_device": consumer_device,
                "bandwidth_gbps": bandwidth_gbps,
                "max_new_tokens": int(args.max_new_tokens),
                "max_length": int(args.max_length),
                "route_counts": _route_counts(route),
                "fit_mode": patch_meta.get("fit_mode"),
                "semantic_source": patch_meta.get("semantic_source"),
                "methods": methods,
                "strict_mode": bool(runtime_eval_cfg.get("strict_mode", True)),
                "prompt_style": prompt_style or "default",
                "align_full_consumer_first_decode": align_full_consumer_first_decode,
                "generation_stop": generation_stop_cfg,
                "same_device_capture_chunk_size": same_device_capture_chunk_size,
                "qid_allowlist": qid_allowlist,
            },
            "modes": per_mode_summary,
            "delta_vs_full_consumer": {method: _comparison_row(row, full_baseline) for method, row in per_mode_summary.items()},
            "delta_vs_reuse_only": {method: _comparison_row(row, reuse_only_baseline) for method, row in per_mode_summary.items()},
        }

        save_json(out_root / "summary.json", {"meta": final_comparison["meta"], "summary": summary_rows})
        save_json(out_root / "per_mode_summary.json", per_mode_summary)
        save_json(out_root / "final_comparison.json", final_comparison)
        _write_summary_csv(out_root / "summary.csv", summary_rows)
        _write_summary_md(out_root / "summary.md", args.dataset, summary_rows)

        if save_per_sample_metrics:
            _write_jsonl(out_root / "per_sample_metrics.jsonl", combined_sample_logs)
        if save_generation_outputs:
            for method in methods:
                _write_jsonl(out_root / "samples" / f"{method}.jsonl", sample_logs[method])

        write_manifest(
            out_dir=out_root,
            command=command_from_argv(sys.argv),
            config_paths={
                "models": args.models_config,
                "eval": args.eval_config,
                "route": args.route,
                "reuse": args.reuse_ckpt,
                "patch": args.patch_ckpt,
            },
            seed=args.seed,
        )
        logger.info("strict text results saved to %s", out_root)
    finally:
        if producer is not None:
            producer.close()
        if consumer is not None:
            consumer.close()


if __name__ == "__main__":
    main()
