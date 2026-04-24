from __future__ import annotations

import argparse
from pathlib import Path

from ..io_utils import load_json
from ..pipeline import load_patch_ckpt, load_reuse_weights
from ..reuse import move_reuse_weights
from ..seed import set_global_seed
from ..strict_runtime import (
    MethodSampleResult,
    STEP_C_METHODS,
    TransformersRuntimeSession,
    assert_same_tokenization,
    evaluate_wikitext2_method,
)
from ..torch_utils import clear_torch_cache, configure_torch_acceleration, resolve_torch_dtype
from ..wikitext2 import load_wikitext2_samples
from .common import add_accel_args, load_cfg, resolve_accel_options, resolve_logger
from .eval_cmrc_strict import (
    ProducerSessionView,
    _consumer_capture_flags,
    _evaluate_sample_methods_with,
    _finalize_eval_outputs,
    _load_prefill_capture,
    _method_dependencies,
    _producer_prefill_path,
    _resolve_max_eval_samples,
    _resolve_qid_allowlist,
    _runtime_eval_cfg,
    _save_prefill_capture,
    _select_runtime_methods,
    _should_use_sequential_runtime,
)


METHODS = list(STEP_C_METHODS)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Strict WikiText-2 evaluation: baseline / reuse-only / patch-only")
    p.add_argument("--models-config", required=True)
    p.add_argument("--eval-config", default="")
    p.add_argument("--data-root", required=True)
    p.add_argument("--route", required=True)
    p.add_argument("--reuse-ckpt", required=True)
    p.add_argument("--patch-ckpt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int, default=200)
    p.add_argument("--max-length", type=int, default=0)
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--bandwidth-gbps", type=float, default=200.0)
    p.add_argument("--quant-group-size", type=int, default=128)
    p.add_argument("--producer-device", default="")
    p.add_argument("--consumer-device", default="")
    p.add_argument("--dtype", default="")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log", default="")
    p.add_argument("--sequential-runtime", action="store_true")
    p.add_argument("--split", default="")
    p.add_argument("--qids", default="")
    p.add_argument("--qid-file", default="")
    add_accel_args(p)
    return p


def main() -> None:
    args = build_parser().parse_args()
    logger = resolve_logger(args.log)
    set_global_seed(args.seed)

    models_cfg = load_cfg(args.models_config)
    eval_cfg = load_cfg(args.eval_config) if args.eval_config else {}
    runtime_eval_cfg = _runtime_eval_cfg(eval_cfg)
    methods = _select_runtime_methods(runtime_eval_cfg)

    accel = resolve_accel_options(models_cfg, args)
    configure_torch_acceleration(str(accel["accel_mode"]))
    logger.info("Acceleration mode: %s", accel["accel_mode"])

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
    max_eval_samples = _resolve_max_eval_samples(args.limit, runtime_eval_cfg)
    default_recompute_segment_length = int(runtime_eval_cfg.get("default_recompute_segment_length", 0) or 0)
    override_route_recompute_length = runtime_eval_cfg.get("override_route_recompute_length", None)
    override_route_recompute_length = None if override_route_recompute_length in {"", None} else int(override_route_recompute_length)
    measure_teacher_forced_metrics = bool(runtime_eval_cfg.get("measure_teacher_forced_metrics", True))
    align_full_consumer_first_decode = bool(runtime_eval_cfg.get("align_full_consumer_first_decode", False))
    save_per_sample_metrics = bool(runtime_eval_cfg.get("save_per_sample_metrics", True))
    save_generation_outputs = bool(runtime_eval_cfg.get("save_generation_outputs", True))
    fail_on_missing_route_fields = bool(runtime_eval_cfg.get("fail_on_missing_route_fields", True))
    fail_on_missing_patch_fields = bool(runtime_eval_cfg.get("fail_on_missing_patch_fields", True))
    qid_allowlist = _resolve_qid_allowlist(args, runtime_eval_cfg)
    data_split = str(args.split or runtime_eval_cfg.get("data_split") or runtime_eval_cfg.get("split") or "test")

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    route = load_json(args.route)
    deps = _method_dependencies(methods)
    sequential_runtime = _should_use_sequential_runtime(producer_device, consumer_device, force=bool(args.sequential_runtime))
    reuse_cpu, reuse_meta = (
        load_reuse_weights(args.reuse_ckpt, device="cpu", dtype="float32")
        if deps["needs_reuse"]
        else (None, {})
    )
    patch_bundle, patch_meta = (
        load_patch_ckpt(args.patch_ckpt, device="cpu", dtype="float32")
        if deps["needs_patch"]
        else (None, {})
    )
    reuse_prod = (
        reuse_cpu if sequential_runtime else move_reuse_weights(reuse_cpu, producer_device, dtype="float32", cpu_fallback=True)
    ) if deps["needs_reuse"] else None
    reuse_cons = (
        reuse_cpu if sequential_runtime else move_reuse_weights(reuse_cpu, consumer_device, dtype="float32", cpu_fallback=True)
    ) if deps["needs_reuse"] else None

    logger.info(
        "strict WikiText-2 runtime: route=%s patch_ckpt=%s reuse_ckpt=%s methods=%s fit_mode=%s sequential=%s split=%s",
        args.route,
        args.patch_ckpt,
        args.reuse_ckpt,
        methods,
        patch_meta.get("fit_mode"),
        sequential_runtime,
        data_split,
    )

    samples = load_wikitext2_samples(args.data_root, limit=max_eval_samples, split=data_split)
    logger.info("WikiText-2 samples loaded: %d, split=%s", len(samples), data_split)
    if qid_allowlist:
        qid_set = set(qid_allowlist)
        samples = [sample for sample in samples if sample.qid in qid_set]
        logger.info("Applied qid filter: kept %d / %d", len(samples), len(qid_allowlist))
        if not samples:
            raise ValueError("no WikiText-2 samples remain after qid filtering")

    consumer_capture_xnorm, consumer_capture_residual = _consumer_capture_flags(methods)
    all_results: dict[str, list[MethodSampleResult]] = {method: [] for method in methods}
    sample_logs: dict[str, list[dict[str, object]]] = {method: [] for method in methods}
    combined_sample_logs: list[dict[str, object]] = []

    if sequential_runtime:
        prefill_root = out_root / ".producer_prefill"
        producer = TransformersRuntimeSession(model_a_path, producer_device, dtype)
        producer_view = ProducerSessionView(rope_spec=producer.rope_spec)
        try:
            for idx, sample in enumerate(samples, start=1):
                producer_capture = producer.prefill(sample.prompt, max_length=args.max_length, capture_xnorm=True, profile_layers=False)
                _save_prefill_capture(_producer_prefill_path(prefill_root, idx, sample.qid), producer_capture)
                logger.info("producer sequential prefill progress: %d/%d, qid=%s", idx, len(samples), sample.qid)
                del producer_capture
                clear_torch_cache()
        finally:
            producer.close()

        consumer = TransformersRuntimeSession(model_b_path, consumer_device, dtype)
        try:
            for idx, sample in enumerate(samples, start=1):
                prefill_path = _producer_prefill_path(prefill_root, idx, sample.qid)
                producer_capture = _load_prefill_capture(prefill_path)
                consumer_capture = consumer.prefill(
                    sample.prompt,
                    max_length=args.max_length,
                    capture_xnorm=consumer_capture_xnorm,
                    capture_residual=consumer_capture_residual,
                    profile_layers=True,
                )
                assert_same_tokenization(producer_capture, consumer_capture)
                _evaluate_sample_methods_with(
                    eval_method=evaluate_wikitext2_method,
                    methods=methods,
                    sample=sample,
                    producer_capture=producer_capture,
                    consumer_capture=consumer_capture,
                    producer_session=producer_view,
                    consumer_session=consumer,
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
                    all_results=all_results,
                    sample_logs=sample_logs,
                    combined_sample_logs=combined_sample_logs,
                )
                logger.info("strict WikiText-2 eval progress: %d/%d, qid=%s", idx, len(samples), sample.qid)
                prefill_path.unlink(missing_ok=True)
                del producer_capture, consumer_capture
                clear_torch_cache()
        finally:
            consumer.close()
    else:
        producer = TransformersRuntimeSession(model_a_path, producer_device, dtype)
        consumer = TransformersRuntimeSession(model_b_path, consumer_device, dtype)
        try:
            for idx, sample in enumerate(samples, start=1):
                producer_capture = producer.prefill(sample.prompt, max_length=args.max_length, capture_xnorm=True, profile_layers=False)
                consumer_capture = consumer.prefill(
                    sample.prompt,
                    max_length=args.max_length,
                    capture_xnorm=consumer_capture_xnorm,
                    capture_residual=consumer_capture_residual,
                    profile_layers=True,
                )
                assert_same_tokenization(producer_capture, consumer_capture)
                _evaluate_sample_methods_with(
                    eval_method=evaluate_wikitext2_method,
                    methods=methods,
                    sample=sample,
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
                    max_new_tokens=int(args.max_new_tokens),
                    quant_group_size=int(args.quant_group_size),
                    transfer_dtype=transfer_dtype,
                    default_recompute_segment_length=default_recompute_segment_length,
                    override_route_recompute_length=override_route_recompute_length,
                    fail_on_missing_route_fields=fail_on_missing_route_fields,
                    fail_on_missing_patch_fields=fail_on_missing_patch_fields,
                    measure_teacher_forced_metrics=measure_teacher_forced_metrics,
                    align_full_consumer_first_decode=align_full_consumer_first_decode,
                    all_results=all_results,
                    sample_logs=sample_logs,
                    combined_sample_logs=combined_sample_logs,
                )
                logger.info("strict WikiText-2 eval progress: %d/%d, qid=%s", idx, len(samples), sample.qid)
                del producer_capture, consumer_capture
                clear_torch_cache()
        finally:
            producer.close()
            consumer.close()

    _finalize_eval_outputs(
        out_root=out_root,
        all_results=all_results,
        methods=methods,
        route=route,
        args=args,
        runtime_eval_cfg=runtime_eval_cfg,
        model_a_path=model_a_path,
        model_b_path=model_b_path,
        producer_device=producer_device,
        consumer_device=consumer_device,
        bandwidth_gbps=bandwidth_gbps,
        patch_meta=patch_meta,
        align_full_consumer_first_decode=align_full_consumer_first_decode,
        qid_allowlist=qid_allowlist,
        save_per_sample_metrics=save_per_sample_metrics,
        save_generation_outputs=save_generation_outputs,
        combined_sample_logs=combined_sample_logs,
        sample_logs=sample_logs,
    )
    logger.info("strict WikiText-2 runtime evaluation saved to %s", out_root)


if __name__ == "__main__":
    main()
