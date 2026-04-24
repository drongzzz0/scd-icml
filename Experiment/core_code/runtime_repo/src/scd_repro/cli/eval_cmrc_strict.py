from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..cmrc import load_cmrc2018_samples
from ..io_utils import load_json, save_json
from ..pipeline import load_patch_ckpt, load_reuse_weights
from ..reuse import move_reuse_weights
from ..seed import set_global_seed
from ..strict_runtime import (
    MethodSampleResult,
    PrefillCapture,
    STEP_C_METHODS,
    TransformersRuntimeSession,
    assert_same_tokenization,
    evaluate_cmrc_method,
    summarize_method_results,
)
from ..torch_utils import clear_torch_cache, configure_torch_acceleration, require_torch, resolve_device, resolve_torch_dtype
from ..tracking import command_from_argv, write_manifest
from .common import add_accel_args, load_cfg, resolve_accel_options, resolve_logger


METHODS = list(STEP_C_METHODS)


@dataclass
class ProducerSessionView:
    rope_spec: Any


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="严格 CMRC2018 评测：四种最终 serving mode 的严格 runtime 对比")
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
    p.add_argument("--sequential-runtime", action="store_true", help="强制使用单卡顺序 runtime，避免 producer/consumer 同时常驻显存")
    p.add_argument("--qids", default="", help="逗号分隔的 qid 子集，仅在当前加载样本窗口内过滤")
    p.add_argument("--qid-file", default="", help="每行一个 qid 的文件，仅在当前加载样本窗口内过滤")
    add_accel_args(p)
    return p


def _should_use_sequential_runtime(producer_device: str, consumer_device: str, force: bool = False) -> bool:
    if force:
        return True
    producer_norm = str(resolve_device(producer_device, cpu_fallback=True))
    consumer_norm = str(resolve_device(consumer_device, cpu_fallback=True))
    return producer_norm == consumer_norm


def _tensor_to_cpu(value: Any) -> Any:
    t = require_torch()
    if isinstance(value, t.Tensor):
        return value.detach().to("cpu")
    return value


def _prefill_capture_to_payload(capture: PrefillCapture) -> dict[str, Any]:
    return {
        "input_ids": _tensor_to_cpu(capture.input_ids),
        "attention_mask": _tensor_to_cpu(capture.attention_mask),
        "logits_last": _tensor_to_cpu(capture.logits_last),
        "past_key_values": [(_tensor_to_cpu(k), _tensor_to_cpu(v)) for k, v in capture.past_key_values],
        "xnorm_by_layer": {int(layer): _tensor_to_cpu(value) for layer, value in capture.xnorm_by_layer.items()},
        "prefill_ms": float(capture.prefill_ms),
        "layer_ms": [float(x) for x in capture.layer_ms],
        "vocab_size": int(capture.vocab_size),
        "residual_by_layer": {int(layer): _tensor_to_cpu(value) for layer, value in capture.residual_by_layer.items()},
    }


def _payload_to_prefill_capture(payload: dict[str, Any]) -> PrefillCapture:
    return PrefillCapture(
        input_ids=payload["input_ids"],
        attention_mask=payload["attention_mask"],
        logits_last=payload["logits_last"],
        past_key_values=[(k, v) for k, v in payload.get("past_key_values", [])],
        xnorm_by_layer={int(layer): value for layer, value in dict(payload.get("xnorm_by_layer", {})).items()},
        prefill_ms=float(payload.get("prefill_ms", 0.0)),
        layer_ms=[float(x) for x in payload.get("layer_ms", [])],
        vocab_size=int(payload.get("vocab_size", 0)),
        residual_by_layer={int(layer): value for layer, value in dict(payload.get("residual_by_layer", {})).items()},
    )


def _save_prefill_capture(path: Path, capture: PrefillCapture) -> None:
    t = require_torch()
    path.parent.mkdir(parents=True, exist_ok=True)
    t.save(_prefill_capture_to_payload(capture), path)


def _load_prefill_capture(path: Path) -> PrefillCapture:
    t = require_torch()
    payload = t.load(path, map_location="cpu", weights_only=False)
    return _payload_to_prefill_capture(payload)


def _producer_prefill_path(root: Path, idx: int, qid: str) -> Path:
    safe_qid = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in str(qid)).strip("._")
    if not safe_qid:
        safe_qid = "sample"
    return root / f"{int(idx):05d}_{safe_qid}.pt"


def _method_dependencies(methods: list[str]) -> dict[str, bool]:
    reuse_methods = {
        "reuse_only",
        "reuse_fallback",
        "reuse_patch",
        "reuse_patch_recompute",
        "scd_full",
        "oracle_xnorm_suffix",
        "oracle_state_suffix",
        "oracle_kv_patchonly",
    }
    patch_methods = {
        "patch_only",
        "reuse_patch",
        "reuse_patch_recompute",
        "scd_full",
        "oracle_xnorm_suffix",
        "oracle_state_suffix",
        "oracle_kv_patchonly",
    }
    active = {str(method) for method in methods}
    return {
        "needs_reuse": bool(active & reuse_methods),
        "needs_patch": bool(active & patch_methods),
    }


def _resolve_max_eval_samples(limit: int, runtime_eval_cfg: dict[str, Any]) -> int:
    cli_limit = int(limit)
    if cli_limit > 0:
        return cli_limit
    return int(runtime_eval_cfg.get("max_eval_samples", 200))


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _sample_result_to_dict(row: MethodSampleResult) -> dict[str, object]:
    return {
        "qid": row.qid,
        "method": row.method,
        "pred_text": row.pred_text,
        "gold_answers": row.gold_answers,
        "f1": row.f1,
        "em": row.em,
        "avg_logp": row.avg_logp,
        "avg_nll": row.avg_nll,
        "teacher_token_count": row.teacher_token_count,
        "payload_mb": row.payload_mb,
        "prompt_tokens": row.prompt_tokens,
        "prefix_tokens": row.prefix_tokens,
        "first_token_id": row.first_token_id,
        "bytes_transferred": row.timing.bytes_transferred,
        "raw_cache_bytes": row.timing.raw_cache_bytes,
        "compression_ratio": row.timing.compression_ratio,
        "timing_ms": {
            "producer_prefill": row.timing.producer_prefill_ms,
            "consumer_prefill": row.timing.consumer_prefill_ms,
            "encode": row.timing.encode_ms,
            "transfer": row.timing.transfer_ms,
            "staging_copy": row.timing.staging_copy_ms,
            "rebuild": row.timing.rebuild_ms,
            "rebuild_prep": row.timing.rebuild_prep_ms,
            "rebuild_decode_core": row.timing.rebuild_decode_core_ms,
            "rebuild_rerope": row.timing.rebuild_rerope_ms,
            "rebuild_materialize": row.timing.rebuild_materialize_ms,
            "patch_infer": row.timing.patch_infer_ms,
            "patch_shard_load": row.timing.patch_shard_load_ms,
            "patch_shard_gpu_copy": row.timing.patch_shard_gpu_copy_ms,
            "patch_identity_check": row.timing.patch_identity_check_ms,
            "patch_residual_forward": row.timing.patch_residual_forward_ms,
            "patch_kproj_vproj": row.timing.patch_kproj_vproj_ms,
            "patch_knorm": row.timing.patch_knorm_ms,
            "patch_rotary": row.timing.patch_rotary_ms,
            "recompute": row.timing.recompute_ms,
            "decode_first": row.timing.decode_first_ms,
            "decode_rest": row.timing.decode_rest_ms,
            "decode_total": row.timing.decode_total_ms,
            "ttft": row.timing.ttft_ms,
            "total": row.timing.total_ms,
        },
    }


def _split_qids(raw: str) -> list[str]:
    text = str(raw or "").replace(",", "\n")
    items = [part.strip() for part in text.splitlines()]
    return [item for item in items if item]


def _resolve_qid_allowlist(args: argparse.Namespace, runtime_eval_cfg: dict[str, Any]) -> list[str]:
    resolved: list[str] = []
    seen: set[str] = set()

    cfg_qids = runtime_eval_cfg.get("qid_allowlist", [])
    if isinstance(cfg_qids, str):
        cfg_items = _split_qids(cfg_qids)
    elif isinstance(cfg_qids, list):
        cfg_items = [str(item).strip() for item in cfg_qids if str(item).strip()]
    else:
        cfg_items = []

    file_items: list[str] = []
    if str(args.qid_file or "").strip():
        path = Path(str(args.qid_file)).expanduser()
        file_items = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    cli_items = _split_qids(str(args.qids or ""))
    for qid in [*cfg_items, *file_items, *cli_items]:
        if qid not in seen:
            seen.add(qid)
            resolved.append(qid)
    return resolved


def _runtime_eval_cfg(eval_cfg: dict[str, Any]) -> dict[str, Any]:
    return eval_cfg.get("runtime_eval", {}) if isinstance(eval_cfg.get("runtime_eval", {}), dict) else {}


def _select_runtime_methods(runtime_eval_cfg: dict[str, Any]) -> list[str]:
    mapping = [
        ("run_full_consumer_baseline", "full_consumer", True),
        ("run_reuse_only", "reuse_only", True),
        ("run_reuse_fallback", "reuse_fallback", True),
        ("run_patch_only", "patch_only", False),
        ("run_reuse_patch", "reuse_patch", True),
        ("run_reuse_patch_recompute", "reuse_patch_recompute", True),
        ("run_oracle_xnorm_suffix", "oracle_xnorm_suffix", False),
        ("run_oracle_state_suffix", "oracle_state_suffix", False),
        ("run_oracle_kv_patchonly", "oracle_kv_patchonly", False),
    ]
    methods = [method for key, method, default in mapping if bool(runtime_eval_cfg.get(key, default))]
    return methods or list(METHODS)


def _consumer_capture_flags(methods: list[str]) -> tuple[bool, bool]:
    capture_xnorm = any(method in {"oracle_xnorm_suffix", "oracle_state_suffix"} for method in methods)
    capture_residual = any(method == "oracle_state_suffix" for method in methods)
    return capture_xnorm, capture_residual


def _route_counts(route: dict[str, list[int]]) -> dict[str, int]:
    return {
        "reuse_layers": len(route.get("reuse_layers", [])),
        "patch_layers": len(route.get("patch_layers", route.get("selected_patch_layers", []))),
        "fallback_layers": len(route.get("fallback_layers", [])),
        "selected_recompute_layers": len(route.get("selected_recompute_layers", [])),
    }


def _comparison_row(row: dict[str, Any], baseline: dict[str, Any] | None) -> dict[str, float | None]:
    if baseline is None:
        return {}
    return {
        "delta_ttft_ms": None if row.get("ttft_ms") is None or baseline.get("ttft_ms") is None else float(row["ttft_ms"]) - float(baseline["ttft_ms"]),
        "delta_logp": None if row.get("avg_logp") is None or baseline.get("avg_logp") is None else float(row["avg_logp"]) - float(baseline["avg_logp"]),
        "delta_nll": None if row.get("avg_nll") is None or baseline.get("avg_nll") is None else float(row["avg_nll"]) - float(baseline["avg_nll"]),
        "delta_ppl": None if row.get("avg_ppl") is None or baseline.get("avg_ppl") is None else float(row["avg_ppl"]) - float(baseline["avg_ppl"]),
        "delta_em": None if row.get("em") is None or baseline.get("em") is None else float(row["em"]) - float(baseline["em"]),
        "delta_f1": None if row.get("f1") is None or baseline.get("f1") is None else float(row["f1"]) - float(baseline["f1"]),
    }


def _write_summary_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "method",
        "method_label",
        "num_samples",
        "avg_logp",
        "avg_nll",
        "avg_ppl",
        "f1",
        "em",
        "bytes_transferred",
        "raw_cache_bytes",
        "compression_ratio",
        "payload_mb",
        "ttft_ms",
        "delta_ttft_vs_full",
        "delta_logp_vs_full",
        "delta_nll_vs_full",
        "delta_ppl_vs_full",
        "producer_prefill_ms",
        "consumer_prefill_ms",
        "encode_ms",
        "transfer_ms",
        "rebuild_ms",
        "patch_infer_ms",
        "recompute_ms",
        "decode_first_ms",
        "decode_rest_ms",
        "decode_total_ms",
        "total_ms",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _write_summary_md(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        "# Strict Runtime Comparison",
        "",
        "| mode | avg_logp | avg_nll | avg_ppl | f1 | em | bytes | compression | ttft_ms | delta_ttft_vs_full |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        compression = row.get("compression_ratio")
        delta_ttft = row.get("delta_ttft_vs_full")
        lines.append(
            f"| {row['method_label']} | {float(row['avg_logp']):.4f} | {float(row['avg_nll']):.4f} | {float(row['avg_ppl']):.4f} | {float(row['f1']):.4f} | {float(row['em']):.4f} | {float(row['bytes_transferred']):.1f} | {0.0 if compression is None else float(compression):.3f} | {float(row['ttft_ms']):.2f} | {0.0 if delta_ttft is None else float(delta_ttft):.2f} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _evaluate_sample_methods_with(
    *,
    eval_method,
    methods: list[str],
    sample,
    producer_capture: PrefillCapture,
    consumer_capture: PrefillCapture,
    producer_session: Any,
    consumer_session: TransformersRuntimeSession,
    route: dict[str, list[int]],
    reuse_prod,
    reuse_cons,
    patch_bundle,
    patch_meta,
    bandwidth_gbps: float,
    max_new_tokens: int,
    quant_group_size: int,
    transfer_dtype: Any,
    default_recompute_segment_length: int,
    override_route_recompute_length: int | None,
    fail_on_missing_route_fields: bool,
    fail_on_missing_patch_fields: bool,
    measure_teacher_forced_metrics: bool,
    align_full_consumer_first_decode: bool,
    all_results: dict[str, list[MethodSampleResult]],
    sample_logs: dict[str, list[dict[str, object]]],
    combined_sample_logs: list[dict[str, object]],
) -> None:
    for method in methods:
        result = eval_method(
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
            max_new_tokens=int(max_new_tokens),
            quant_group_size=int(quant_group_size),
            transfer_dtype=transfer_dtype,
            default_recompute_segment_length=default_recompute_segment_length,
            override_route_recompute_length=override_route_recompute_length,
            fail_on_missing_route_fields=fail_on_missing_route_fields,
            fail_on_missing_patch_fields=fail_on_missing_patch_fields,
            measure_teacher_forced_metrics=measure_teacher_forced_metrics,
            align_full_consumer_first_decode=align_full_consumer_first_decode,
        )
        all_results[method].append(result)
        row = _sample_result_to_dict(result)
        sample_logs[method].append(row)
        combined_sample_logs.append(row)


def _evaluate_sample_methods(
    *,
    methods: list[str],
    sample,
    producer_capture: PrefillCapture,
    consumer_capture: PrefillCapture,
    producer_session: Any,
    consumer_session: TransformersRuntimeSession,
    route: dict[str, list[int]],
    reuse_prod,
    reuse_cons,
    patch_bundle,
    patch_meta,
    bandwidth_gbps: float,
    max_new_tokens: int,
    quant_group_size: int,
    transfer_dtype: Any,
    default_recompute_segment_length: int,
    override_route_recompute_length: int | None,
    fail_on_missing_route_fields: bool,
    fail_on_missing_patch_fields: bool,
    measure_teacher_forced_metrics: bool,
    align_full_consumer_first_decode: bool,
    all_results: dict[str, list[MethodSampleResult]],
    sample_logs: dict[str, list[dict[str, object]]],
    combined_sample_logs: list[dict[str, object]],
) -> None:
    _evaluate_sample_methods_with(
        eval_method=evaluate_cmrc_method,
        methods=methods,
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
        max_new_tokens=max_new_tokens,
        quant_group_size=quant_group_size,
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


def _finalize_eval_outputs(
    *,
    out_root: Path,
    all_results: dict[str, list[MethodSampleResult]],
    methods: list[str],
    route: dict[str, list[int]],
    args: argparse.Namespace,
    runtime_eval_cfg: dict[str, Any],
    model_a_path: str,
    model_b_path: str,
    producer_device: str,
    consumer_device: str,
    bandwidth_gbps: float,
    patch_meta: dict[str, Any] | None,
    align_full_consumer_first_decode: bool,
    qid_allowlist: list[str],
    save_per_sample_metrics: bool,
    save_generation_outputs: bool,
    combined_sample_logs: list[dict[str, object]],
    sample_logs: dict[str, list[dict[str, object]]],
) -> None:
    summary_rows = [summarize_method_results(all_results[method]) for method in methods]
    summary_map = {str(row["method"]): row for row in summary_rows}
    full_baseline = summary_map.get("full_consumer")
    reuse_only_baseline = summary_map.get("reuse_only")
    for row in summary_rows:
        delta_full = _comparison_row(row, full_baseline)
        delta_reuse = _comparison_row(row, reuse_only_baseline)
        row["delta_ttft_vs_full"] = delta_full.get("delta_ttft_ms")
        row["delta_logp_vs_full"] = delta_full.get("delta_logp")
        row["delta_nll_vs_full"] = delta_full.get("delta_nll")
        row["delta_ppl_vs_full"] = delta_full.get("delta_ppl")
        row["delta_ttft_vs_reuse_only"] = delta_reuse.get("delta_ttft_ms")
        row["delta_logp_vs_reuse_only"] = delta_reuse.get("delta_logp")
        row["delta_nll_vs_reuse_only"] = delta_reuse.get("delta_nll")

    per_mode_summary = {str(row["method"]): row for row in summary_rows}
    patch_meta = patch_meta or {}
    final_comparison = {
        "meta": {
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
            "align_full_consumer_first_decode": align_full_consumer_first_decode,
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
    _write_summary_md(out_root / "summary.md", summary_rows)

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
    logger.info("计算加速模式: %s", accel["accel_mode"])

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
        reuse_cpu
        if sequential_runtime
        else move_reuse_weights(reuse_cpu, producer_device, dtype="float32", cpu_fallback=True)
    ) if deps["needs_reuse"] else None
    reuse_cons = (
        reuse_cpu
        if sequential_runtime
        else move_reuse_weights(reuse_cpu, consumer_device, dtype="float32", cpu_fallback=True)
    ) if deps["needs_reuse"] else None

    logger.info(
        "strict runtime Step C: route=%s patch_ckpt=%s reuse_ckpt=%s methods=%s fit_mode=%s semantic_source=%s sequential=%s",
        args.route,
        args.patch_ckpt,
        args.reuse_ckpt,
        methods,
        patch_meta.get("fit_mode"),
        patch_meta.get("semantic_source"),
        sequential_runtime,
    )
    logger.info(
        "route summary: reuse=%s patch=%s fallback=%s recompute=%s",
        route.get("reuse_layers", []),
        route.get("selected_patch_layers", route.get("patch_layers", [])),
        route.get("fallback_layers", []),
        route.get("recompute_segment_length_by_layer", {}),
    )
    logger.info(
        "reuse mainline: mode=%s translator.default_path=%s num_groups=%s group_size=%s rank_k=%s rank_v=%s",
        reuse_meta.get("mode", "xkv_group"),
        reuse_meta.get("translator", {}).get("default_path", "shared_z"),
        reuse_meta.get("num_groups"),
        reuse_meta.get("group_size"),
        reuse_meta.get("rank_k", reuse_meta.get("rank")),
        reuse_meta.get("rank_v", reuse_meta.get("rank")),
    )

    samples = load_cmrc2018_samples(args.data_root, limit=max_eval_samples)
    logger.info("CMRC 样本已加载: %d", len(samples))
    if qid_allowlist:
        qid_set = set(qid_allowlist)
        samples = [sample for sample in samples if sample.qid in qid_set]
        logger.info("应用 qid 子集过滤后保留样本: %d / %d", len(samples), len(qid_allowlist))
        if not samples:
            raise ValueError("qid 子集过滤后没有剩余样本；注意当前过滤只作用在已加载样本窗口内")
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
                _evaluate_sample_methods(
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
                logger.info("涓ユ牸璇勬祴杩涘害: %d/%d, qid=%s", idx, len(samples), sample.qid)
                prefill_path.unlink(missing_ok=True)
                del producer_capture, consumer_capture
                clear_torch_cache()
        finally:
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
        logger.info("Step C 涓ユ牸 runtime 璇勬祴缁撴灉宸蹭繚瀛樺埌 %s", out_root)
        return

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

            _evaluate_sample_methods(
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

            logger.info("严格评测进度: %d/%d, qid=%s", idx, len(samples), sample.qid)
            del producer_capture, consumer_capture
            clear_torch_cache()

        summary_rows = [summarize_method_results(all_results[method]) for method in methods]
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
                "align_full_consumer_first_decode": align_full_consumer_first_decode,
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
        _write_summary_md(out_root / "summary.md", summary_rows)

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
        logger.info("Step C 严格 runtime 评测结果已保存到 %s", out_root)
    finally:
        producer.close()
        consumer.close()


if __name__ == "__main__":
    main()
