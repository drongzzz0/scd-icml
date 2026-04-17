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
        try:
            from safetensors import safe_open
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("safetensors is required to count backbone parameters") from exc

        total = 0
        for file_path in files:
            with safe_open(str(file_path), framework="pt", device="cpu") as handle:
                for key in handle.keys():
                    shape = tuple(int(v) for v in handle.get_slice(key).get_shape())
                    total += int(math.prod(shape))
        return int(total)

    try:
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("transformers is required to count backbone parameters for non-safetensors checkpoints") from exc

    config = AutoConfig.from_pretrained(str(model_dir), trust_remote_code=True)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    return int(sum(int(param.numel()) for param in model.parameters()))


def _load_stage_summary(measure_root: Path, name: str) -> dict[str, Any]:
    path = measure_root / name / "stage_summary.json"
    return load_json(path)


def _safe_ratio(num: float | int | None, den: float | int | None) -> float | None:
    if num is None or den is None:
        return None
    den_f = float(den)
    if abs(den_f) <= 0.0:
        return None
    return float(num) / den_f


def _maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _mode_table(per_mode: dict[str, Any], baseline_key: str = "full_consumer") -> dict[str, Any]:
    baseline = per_mode.get(baseline_key, {})
    baseline_f1 = _maybe_float(baseline.get("f1"))
    baseline_ttft = _maybe_float(baseline.get("ttft_ms"))
    baseline_total = _maybe_float(baseline.get("total_ms"))
    rows: dict[str, Any] = {}
    for mode, row in per_mode.items():
        f1 = _maybe_float(row.get("f1"))
        ttft = _maybe_float(row.get("ttft_ms"))
        total = _maybe_float(row.get("total_ms"))
        payload_mb = _maybe_float(row.get("payload_mb"))
        bytes_transferred = _maybe_float(row.get("bytes_transferred"))
        raw_cache_bytes = _maybe_float(row.get("raw_cache_bytes"))
        rows[mode] = {
            "F1": f1,
            "EM": _maybe_float(row.get("em")),
            "TTFT_ms": ttft,
            "total_ms": total,
            "payload_mb": payload_mb,
            "bytes_transferred": bytes_transferred,
            "compression_ratio": _maybe_float(row.get("compression_ratio")),
            "encode_ms": _maybe_float(row.get("encode_ms")),
            "transfer_ms": _maybe_float(row.get("transfer_ms")),
            "rebuild_ms": _maybe_float(row.get("rebuild_ms")),
            "patch_infer_ms": _maybe_float(row.get("patch_infer_ms")),
            "recompute_ms": _maybe_float(row.get("recompute_ms")),
            "decode_first_ms": _maybe_float(row.get("decode_first_ms")),
            "f1_retention_vs_full": _safe_ratio(f1, baseline_f1),
            "f1_absolute_drop_vs_full": None if f1 is None or baseline_f1 is None else float(baseline_f1 - f1),
            "ttft_ratio_method_over_full": _safe_ratio(ttft, baseline_ttft),
            "ttft_speedup_full_over_method": _safe_ratio(baseline_ttft, ttft),
            "ttft_absolute_delta_ms_vs_full": None if ttft is None or baseline_ttft is None else float(ttft - baseline_ttft),
            "total_latency_ratio_method_over_full": _safe_ratio(total, baseline_total),
            "total_latency_speedup_full_over_method": _safe_ratio(baseline_total, total),
            "payload_ratio_vs_raw_cache": _safe_ratio(bytes_transferred, raw_cache_bytes),
        }
    return rows


def _build_markdown(summary: dict[str, Any]) -> str:
    calib = summary["calibration"]
    lines = [
        "# Q2 Fresh Rerun Summary",
        "",
        "## Calibration",
        "",
        "| item | value |",
        "|---|---:|",
        f"| calibration_samples | {calib['calibration_samples']} |",
        f"| prefix_tokens_mean | {calib['prefix_tokens_mean']:.2f} |",
        f"| prefix_tokens_min | {calib['prefix_tokens_min']} |",
        f"| prefix_tokens_max | {calib['prefix_tokens_max']} |",
        f"| total_calibration_prefix_tokens | {calib['total_calibration_prefix_tokens']} |",
        f"| pairs_dir_size_bytes | {calib['pairs_dir_size_bytes']} |",
        "",
        "## Offline Stages",
        "",
        "| stage | success | wall_clock_sec | peak_gpu_memory_mb | max_cpu_rss_kb | gpu_ids |",
        "|---|---|---:|---:|---:|---|",
    ]
    for name in ["capture_pair", "train_reuse", "profile", "build_route", "train_patch"]:
        row = summary["offline_stage_costs"][name]
        lines.append(
            f"| {name} | {row['success']} | {row['wall_clock_sec']:.3f} | {row['peak_gpu_memory_mb'] if row['peak_gpu_memory_mb'] is not None else ''} | {row['max_cpu_rss_kb'] if row['max_cpu_rss_kb'] is not None else ''} | {','.join(str(x) for x in row['gpu_ids'])} |"
        )

    lines.extend(
        [
            "",
            "## Online Metrics",
            "",
            "| mode | F1 | EM | TTFT_ms | total_ms | payload_mb | compression_ratio | encode_ms | transfer_ms | rebuild_ms | patch_infer_ms | recompute_ms |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for mode, row in summary["online_metrics"].items():
        lines.append(
            f"| {mode} | {row['F1']:.6f} | {row['EM']:.6f} | {row['TTFT_ms']:.3f} | {row['total_ms']:.3f} | {row['payload_mb']:.3f} | {row['compression_ratio'] if row['compression_ratio'] is not None else ''} | {row['encode_ms']:.3f} | {row['transfer_ms']:.3f} | {row['rebuild_ms']:.3f} | {row['patch_infer_ms']:.3f} | {row['recompute_ms']:.3f} |"
        )
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize a fresh Q2 Mistral-7B -> MistralLite run")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--models-config", required=True)
    parser.add_argument("--train-config", required=True)
    parser.add_argument("--eval-config", required=True)
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--reuse-ckpt", required=True)
    parser.add_argument("--patch-ckpt", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--route", required=True)
    parser.add_argument("--strict-out", required=True)
    parser.add_argument("--measure-root", required=True)
    parser.add_argument("--code-version", default="")
    args = parser.parse_args()

    run_root = Path(args.run_root)
    measure_root = Path(args.measure_root)
    pairs_dir = Path(args.pairs)
    reuse_dir = Path(args.reuse_ckpt)
    patch_dir = Path(args.patch_ckpt)
    profile_path = Path(args.profile)
    route_path = Path(args.route)
    strict_out = Path(args.strict_out)

    models_cfg = load_json(args.models_config)
    train_cfg = load_json(args.train_config)
    eval_cfg = load_json(args.eval_config)
    manifest = load_json(pairs_dir / "manifest.json")
    reuse_meta = load_json(reuse_dir / "meta.json")
    patch_meta = load_json(patch_dir / "meta.json")
    route = load_json(route_path)
    strict_summary = load_json(strict_out / "summary.json")
    per_mode = load_json(strict_out / "per_mode_summary.json")
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
        for name in ["capture_pair", "train_reuse", "profile", "build_route", "train_patch", "strict_eval"]
    }

    online_metrics = _mode_table(per_mode, baseline_key="full_consumer")

    summary = {
        "status": {
            "run_root": str(run_root),
            "all_stages_success": bool(all(bool(row.get("success")) for row in offline_stage_costs.values())),
            "strict_eval_success": bool(offline_stage_costs["strict_eval"].get("success")),
            "code_version": str(args.code_version or "unknown"),
        },
        "paths": {
            "run_root": str(run_root),
            "pairs_dir": str(pairs_dir),
            "manifest_json": str(pairs_dir / "manifest.json"),
            "measure_root": str(measure_root),
            "reuse_ckpt_dir": str(reuse_dir),
            "reuse_meta_json": str(reuse_dir / "meta.json"),
            "patch_ckpt_dir": str(patch_dir),
            "patch_meta_json": str(patch_dir / "meta.json"),
            "profile_json": str(profile_path),
            "route_json": str(route_path),
            "strict_eval_dir": str(strict_out),
            "summary_json": str(strict_out / "summary.json"),
            "per_mode_summary_json": str(strict_out / "per_mode_summary.json"),
        },
        "calibration": {
            "dataset": "CMRC2018 train -> dev strict eval",
            "calibration_samples": calibration_samples,
            "prefix_tokens_mean": prefix_mean,
            "prefix_tokens_min": int(min(prefix_tokens)) if prefix_tokens else 0,
            "prefix_tokens_max": int(max(prefix_tokens)) if prefix_tokens else 0,
            "prefix_tokens_per_sample": prefix_tokens,
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
            "producer_model_path": str(producer_model_path),
            "consumer_model_path": str(consumer_model_path),
            "dataset": "CMRC2018",
            "group_size": int(reuse_meta.get("group_size", train_cfg.get("xkv_group", {}).get("group_size", 0))),
            "rank_k": int(reuse_meta.get("rank_k", train_cfg.get("xkv_group", {}).get("rank_k", 0))),
            "rank_v": int(reuse_meta.get("rank_v", train_cfg.get("xkv_group", {}).get("rank_v", 0))),
            "patch_budget": int(eval_cfg.get("route", {}).get("patch_budget", train_cfg.get("patch", {}).get("patch_budget", 0))),
            "patch_num_epochs": int(train_cfg.get("patch", {}).get("patch_num_epochs", 0)),
            "patch_batch_size": int(train_cfg.get("patch", {}).get("patch_batch_size", 0)),
            "patch_mlp_hidden_dim": int(train_cfg.get("patch", {}).get("patch_mlp_hidden_dim", 0)),
            "reuse_layers": [int(v) for v in route.get("reuse_layers", [])],
            "patch_layers": [int(v) for v in route.get("patch_layers", [])],
            "fallback_layers": [int(v) for v in route.get("fallback_layers", [])],
            "selected_recompute_layers": [int(v) for v in route.get("selected_recompute_layers", [])],
            "backbone_frozen": True,
            "semantic_probe_shared": bool(patch_meta.get("semantic_probe_shared", train_cfg.get("patch", {}).get("semantic_probe_shared", True))),
            "semantic_probe_frozen": bool(patch_meta.get("semantic_probe_frozen", train_cfg.get("patch", {}).get("semantic_probe_frozen", True))),
            "fit_mode": str(patch_meta.get("fit_mode")),
            "code_version": str(args.code_version or "unknown"),
            "gpu_assignment": gpu_assignment,
        },
        "online_metrics": online_metrics,
        "definitions": {
            "ttft_ratio_method_over_full": "method_ttft_ms / full_consumer_ttft_ms; smaller is better",
            "ttft_speedup_full_over_method": "full_consumer_ttft_ms / method_ttft_ms; larger than 1 means faster than full_consumer",
            "total_latency_ratio_method_over_full": "method_total_ms / full_consumer_total_ms; smaller is better",
            "payload_ratio_vs_raw_cache": "bytes_transferred / raw_cache_bytes; this is the payload compression ratio field aligned with strict_eval",
        },
        "raw_meta": {
            "strict_summary_meta": strict_summary.get("meta", {}),
        },
    }

    out_json = measure_root / "rebuttal_summary.json"
    out_md = measure_root / "rebuttal_summary.md"
    save_json(out_json, summary)
    out_md.write_text(_build_markdown(summary), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
