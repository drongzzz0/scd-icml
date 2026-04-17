from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

from scd_repro.io_utils import load_bundle, load_json, save_json


def _dir_size_bytes(path: Path) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += (Path(root) / name).stat().st_size
            except FileNotFoundError:
                continue
    return int(total)


def _sum_bundle_numel(bundle_path: Path) -> int:
    payload = load_bundle(bundle_path, as_numpy=True)
    total = 0
    for value in payload.arrays.values():
        total += int(getattr(value, "size", 0))
    return int(total)


def _sum_patch_params(patch_dir: Path) -> int:
    weights_path = patch_dir / "weights.pt"
    if weights_path.exists():
        return _sum_bundle_numel(weights_path)
    shards_dir = patch_dir / "shards"
    if shards_dir.exists():
        total = 0
        for pt_path in sorted(shards_dir.glob("*.pt")):
            total += _sum_bundle_numel(pt_path)
        return int(total)
    raise FileNotFoundError(f"no patch weights found under {patch_dir}")


def _sum_model_params(model_dir: Path) -> int:
    files = sorted({path.resolve() for path in model_dir.rglob("*.safetensors")})
    if files:
        from safetensors import safe_open

        total = 0
        for file_path in files:
            with safe_open(str(file_path), framework="pt", device="cpu") as handle:
                for key in handle.keys():
                    shape = tuple(int(v) for v in handle.get_slice(key).get_shape())
                    total += int(math.prod(shape))
        return int(total)

    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(str(model_dir), trust_remote_code=True)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    return int(sum(int(param.numel()) for param in model.parameters()))


def _load_stage_summary(measure_root: Path, name: str) -> dict[str, Any]:
    return load_json(measure_root / name / "stage_summary.json")


def _maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _safe_ratio(num: float | int | None, den: float | int | None) -> float | None:
    if num is None or den is None:
        return None
    den_f = float(den)
    if den_f == 0.0:
        return None
    return float(num) / den_f


def _row_from_mode(mode_row: dict[str, Any], baseline_row: dict[str, Any]) -> dict[str, Any]:
    f1 = _maybe_float(mode_row.get("f1"))
    ttft = _maybe_float(mode_row.get("ttft_ms"))
    total = _maybe_float(mode_row.get("total_ms"))
    baseline_f1 = _maybe_float(baseline_row.get("f1"))
    baseline_ttft = _maybe_float(baseline_row.get("ttft_ms"))
    baseline_total = _maybe_float(baseline_row.get("total_ms"))
    return {
        "F1": f1,
        "EM": _maybe_float(mode_row.get("em")),
        "TTFT_ms": ttft,
        "total_ms": total,
        "payload_mb": _maybe_float(mode_row.get("payload_mb")),
        "bytes_transferred": _maybe_float(mode_row.get("bytes_transferred")),
        "compression_ratio": _maybe_float(mode_row.get("compression_ratio")),
        "encode_ms": _maybe_float(mode_row.get("encode_ms")),
        "transfer_ms": _maybe_float(mode_row.get("transfer_ms")),
        "rebuild_ms": _maybe_float(mode_row.get("rebuild_ms")),
        "patch_infer_ms": _maybe_float(mode_row.get("patch_infer_ms")),
        "recompute_ms": _maybe_float(mode_row.get("recompute_ms")),
        "decode_first_ms": _maybe_float(mode_row.get("decode_first_ms")),
        "f1_retention_vs_full": _safe_ratio(f1, baseline_f1),
        "f1_absolute_drop_vs_full": None if f1 is None or baseline_f1 is None else float(baseline_f1 - f1),
        "ttft_ratio_method_over_full": _safe_ratio(ttft, baseline_ttft),
        "ttft_speedup_full_over_method": _safe_ratio(baseline_ttft, ttft),
        "ttft_absolute_delta_ms_vs_full": None if ttft is None or baseline_ttft is None else float(ttft - baseline_ttft),
        "total_latency_ratio_method_over_full": _safe_ratio(total, baseline_total),
        "payload_ratio_vs_raw_cache": _safe_ratio(
            _maybe_float(mode_row.get("bytes_transferred")),
            _maybe_float(mode_row.get("raw_cache_bytes")),
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize tuned fresh MistralLite->Mistral-7B rerun")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--models-config", required=True)
    parser.add_argument("--reuse-train-config", required=True)
    parser.add_argument("--patch-train-config", required=True)
    parser.add_argument("--route-template", required=True)
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--reuse-ckpt", required=True)
    parser.add_argument("--patch-ckpt", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--search-out", required=True)
    parser.add_argument("--baseline-reuse-out", required=True)
    parser.add_argument("--hybrid-out", required=True)
    parser.add_argument("--measure-root", required=True)
    parser.add_argument("--code-version", default="")
    args = parser.parse_args()

    run_root = Path(args.run_root)
    measure_root = Path(args.measure_root)
    pairs_dir = Path(args.pairs)
    reuse_dir = Path(args.reuse_ckpt)
    patch_dir = Path(args.patch_ckpt)
    profile_path = Path(args.profile)
    search_out = Path(args.search_out)
    baseline_reuse_out = Path(args.baseline_reuse_out)
    hybrid_out = Path(args.hybrid_out)

    models_cfg = load_json(args.models_config)
    reuse_train_cfg = load_json(args.reuse_train_config)
    patch_train_cfg = load_json(args.patch_train_config)
    route_template = load_json(args.route_template)
    manifest = load_json(pairs_dir / "manifest.json")
    reuse_meta = load_json(reuse_dir / "meta.json")
    patch_meta = load_json(patch_dir / "meta.json")
    search_report = load_json(search_out / "search_report.json")
    selected_route = load_json(search_out / "selected_hybrid_route.json")
    baseline_per_mode = load_json(baseline_reuse_out / "per_mode_summary.json")
    hybrid_per_mode = load_json(hybrid_out / "per_mode_summary.json")
    patch_anchor_per_mode = load_json(search_out / "eval_anchor_patch_only" / "per_mode_summary.json")
    gpu_assignment = load_json(measure_root / "gpu_assignment.json")

    prefix_tokens = [int(row.get("prefix_tokens", 0)) for row in manifest.get("chunks", [])]
    calibration_samples = int(len(prefix_tokens))
    total_prefix_tokens = int(sum(prefix_tokens))
    prefix_mean = float(total_prefix_tokens / calibration_samples) if calibration_samples > 0 else 0.0

    reuse_params = _sum_bundle_numel(reuse_dir / "weights.pt")
    patch_params = _sum_patch_params(patch_dir)
    total_extra_params = int(reuse_params + patch_params)

    producer_model_path = Path(str(models_cfg["model_a"]["path"]))
    consumer_model_path = Path(str(models_cfg["model_b"]["path"]))
    producer_backbone_params = _sum_model_params(producer_model_path)
    consumer_backbone_params = _sum_model_params(consumer_model_path)

    offline_stage_costs = {
        name: _load_stage_summary(measure_root, name)
        for name in [
            "capture_pair",
            "train_reuse",
            "profile",
            "train_patch",
            "reshard_patch",
            "blockwise_search",
            "strict_baseline_reuse_only",
            "strict_hybrid",
        ]
    }

    baseline_row = baseline_per_mode["full_consumer"]
    online_metrics = {
        "baseline": _row_from_mode(baseline_row, baseline_row),
        "reuse_only": _row_from_mode(baseline_per_mode["reuse_only"], baseline_row),
        "patch_only": _row_from_mode(patch_anchor_per_mode["patch_only"], baseline_row),
        "hybrid": _row_from_mode(hybrid_per_mode["reuse_patch"], baseline_row),
    }

    summary = {
        "status": {
            "run_root": str(run_root),
            "all_stages_success": bool(all(bool(row.get("success")) for row in offline_stage_costs.values())),
            "code_version": str(args.code_version or "unknown"),
        },
        "paths": {
            "run_root": str(run_root),
            "pairs_dir": str(pairs_dir),
            "manifest_json": str(pairs_dir / "manifest.json"),
            "measure_root": str(measure_root),
            "reuse_ckpt_dir": str(reuse_dir),
            "patch_ckpt_dir": str(patch_dir),
            "profile_json": str(profile_path),
            "search_out": str(search_out),
            "search_report_json": str(search_out / "search_report.json"),
            "selected_route_json": str(search_out / "selected_hybrid_route.json"),
            "patch_anchor_out": str(search_out / "eval_anchor_patch_only"),
            "baseline_reuse_out": str(baseline_reuse_out),
            "hybrid_out": str(hybrid_out),
            "baseline_reuse_per_mode_json": str(baseline_reuse_out / "per_mode_summary.json"),
            "patch_anchor_per_mode_json": str(search_out / "eval_anchor_patch_only" / "per_mode_summary.json"),
            "hybrid_per_mode_json": str(hybrid_out / "per_mode_summary.json"),
        },
        "calibration": {
            "dataset": "CMRC2018 train-only subset",
            "calibration_samples": calibration_samples,
            "prefix_tokens_mean": prefix_mean,
            "prefix_tokens_min": int(min(prefix_tokens)) if prefix_tokens else 0,
            "prefix_tokens_max": int(max(prefix_tokens)) if prefix_tokens else 0,
            "total_calibration_prefix_tokens": total_prefix_tokens,
            "pairs_dir_size_bytes": _dir_size_bytes(pairs_dir),
        },
        "offline_stage_costs": offline_stage_costs,
        "module_overhead": {
            "reuse_ckpt_dir_size_bytes": _dir_size_bytes(reuse_dir),
            "patch_ckpt_dir_size_bytes": _dir_size_bytes(patch_dir),
            "reuse_module_params": reuse_params,
            "patch_module_params": patch_params,
            "total_extra_params": total_extra_params,
            "producer_backbone_params": producer_backbone_params,
            "consumer_backbone_params": consumer_backbone_params,
            "total_extra_params_ratio_vs_consumer_backbone": _safe_ratio(total_extra_params, consumer_backbone_params),
        },
        "route_and_engineering": {
            "producer_model": str(models_cfg["model_a"]["name"]),
            "consumer_model": str(models_cfg["model_b"]["name"]),
            "dataset": "CMRC2018 train-only subset",
            "group_size": int(reuse_meta.get("group_size", reuse_train_cfg.get("xkv_group", {}).get("group_size", 0))),
            "rank_k": int(reuse_meta.get("rank_k", reuse_train_cfg.get("xkv_group", {}).get("rank_k", 0))),
            "rank_v": int(reuse_meta.get("rank_v", reuse_train_cfg.get("xkv_group", {}).get("rank_v", 0))),
            "patch_budget": int(patch_meta.get("patch_budget", patch_train_cfg.get("patch", {}).get("patch_budget", 0))),
            "patch_num_epochs": int(patch_train_cfg.get("patch", {}).get("patch_num_epochs", 0)),
            "patch_batch_size": int(patch_train_cfg.get("patch", {}).get("patch_batch_size", 0)),
            "patch_residual_hidden_dim": int(patch_train_cfg.get("patch", {}).get("patch_residual_hidden_dim", 0)),
            "patch_shared_bottleneck_dim": int(patch_train_cfg.get("patch", {}).get("patch_shared_bottleneck_dim", 0)),
            "patch_residual_num_blocks": int(patch_train_cfg.get("patch", {}).get("patch_residual_num_blocks", 0)),
            "reuse_layers": [int(v) for v in selected_route.get("reuse_layers", [])],
            "patch_layers": [int(v) for v in selected_route.get("patch_layers", [])],
            "fallback_layers": [int(v) for v in selected_route.get("fallback_layers", [])],
            "selected_recompute_layers": [int(v) for v in selected_route.get("selected_recompute_layers", [])],
            "accepted_block_ids": search_report.get("accepted_block_ids", []),
            "selected_hybrid_patch_layers": search_report.get("selected_hybrid_patch_layers", []),
            "search_block_size": int(search_report.get("block_size", 0) or 0),
            "search_replay_top_k": int(search_report.get("replay_top_k", 0) or 0),
            "search_replay_selection_mode": str(search_report.get("replay_selection_mode", "")),
            "search_replay_decision_max_f1_drop_ratio": _maybe_float(search_report.get("replay_decision_max_f1_drop_ratio")),
            "backbone_frozen": True,
            "fit_mode": str(patch_meta.get("fit_mode")),
            "code_version": str(args.code_version or "unknown"),
            "gpu_assignment": gpu_assignment,
            "route_template_name": str(route_template.get("name", "")),
        },
        "online_metrics": online_metrics,
        "definitions": {
            "ttft_ratio_method_over_full": "method_ttft_ms / baseline_ttft_ms; smaller is better",
            "ttft_speedup_full_over_method": "baseline_ttft_ms / method_ttft_ms; larger than 1 means faster than baseline",
            "total_latency_ratio_method_over_full": "method_total_ms / baseline_total_ms; smaller is better",
            "payload_ratio_vs_raw_cache": "bytes_transferred / raw_cache_bytes",
        },
    }

    save_json(measure_root / "tuned_rebuttal_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
