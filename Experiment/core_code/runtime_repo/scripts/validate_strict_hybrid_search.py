#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


EPS = 1e-9


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate strict-stop hybrid search artifacts.")
    parser.add_argument("--search-root", required=True, help="Directory containing search_report.json and selected_hybrid_route.json")
    parser.add_argument(
        "--max-f1-drop-ratio",
        type=float,
        default=0.05,
        help="Largest allowed configured F1-drop ratio versus patch-only.",
    )
    parser.add_argument(
        "--require-mixed",
        action="store_true",
        help="Require at least one accepted Reuse block and a non-all-patch selected route.",
    )
    return parser


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def int_list(payload: dict[str, Any], key: str) -> list[int]:
    return sorted({int(value) for value in payload.get(key, [])})


def metric(payload: dict[str, Any], key: str) -> float:
    value = payload.get(key)
    if value is None:
        raise ValueError(f"missing metric {key}")
    return float(value)


def route_errors(route: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    num_layers = int(route.get("num_layers", 0) or 0)
    patch_layers = int_list(route, "selected_patch_layers") or int_list(route, "patch_layers")
    reuse_layers = int_list(route, "reuse_layers")
    fallback_layers = int_list(route, "fallback_layers")
    recompute_layers = int_list(route, "selected_recompute_layers")
    recompute_map = route.get("recompute_segment_length_by_layer", {})

    if fallback_layers:
        errors.append(f"fallback_layers must be empty, got {fallback_layers}")
    if recompute_layers or recompute_map:
        errors.append(f"recompute must be disabled, got layers={recompute_layers} map={recompute_map}")
    if num_layers <= 0:
        errors.append(f"num_layers must be positive, got {num_layers}")
    else:
        all_layers = set(range(num_layers))
        patch_set = set(patch_layers)
        reuse_set = set(reuse_layers)
        if patch_set & reuse_set:
            errors.append(f"patch/reuse overlap: {sorted(patch_set & reuse_set)}")
        if patch_set | reuse_set != all_layers:
            errors.append(f"patch+reuse must cover all layers 0..{num_layers - 1}")
    return errors


def main() -> None:
    args = build_parser().parse_args()
    search_root = Path(args.search_root)
    report_path = search_root / "search_report.json"
    route_path = search_root / "selected_hybrid_route.json"
    errors: list[str] = []

    if not report_path.is_file():
        errors.append(f"missing search report: {report_path}")
    if not route_path.is_file():
        errors.append(f"missing selected route: {route_path}")
    if errors:
        raise SystemExit("\n".join(errors))

    report = load_json(report_path)
    route = load_json(route_path)
    errors.extend(route_errors(route))

    search_cfg = report.get("search_config", {})
    route_policy = route.get("search_policy", {})
    if route_policy.get("anchor_method") != "patch_only":
        errors.append(f"search_policy.anchor_method must be patch_only, got {route_policy.get('anchor_method')!r}")
    if route_policy.get("hybrid_method") != "reuse_patch":
        errors.append(f"search_policy.hybrid_method must be reuse_patch, got {route_policy.get('hybrid_method')!r}")

    replay_top_k = int(search_cfg.get("replay_top_k", route_policy.get("replay_top_k", 0)) or 0)
    if replay_top_k <= 0:
        errors.append(f"strict-stop hybrid search must replay candidates, got replay_top_k={replay_top_k}")
    selection_mode = str(search_cfg.get("replay_selection_mode", route_policy.get("replay_selection_mode", "")))
    if not selection_mode.startswith("quality_"):
        errors.append(f"replay_selection_mode must be quality-gated, got {selection_mode!r}")

    cfg_ratio = float(search_cfg.get("replay_decision_max_f1_drop_ratio", route_policy.get("replay_decision_max_f1_drop_ratio", -1.0)))
    if cfg_ratio < 0.0:
        errors.append("replay_decision_max_f1_drop_ratio must be enabled")
    elif cfg_ratio > float(args.max_f1_drop_ratio) + EPS:
        errors.append(f"configured F1-drop ratio {cfg_ratio} exceeds allowed {args.max_f1_drop_ratio}")

    anchor_metrics = report.get("anchor_patch_only", {}).get("metrics", {})
    selected_metrics = report.get("selected_hybrid_metrics", {})
    try:
        anchor_f1 = metric(anchor_metrics, "f1")
        selected_f1 = metric(selected_metrics, "f1")
        f1_drop = anchor_f1 - selected_f1
        allowed_values: list[float] = []
        cfg_abs = float(search_cfg.get("replay_decision_max_f1_drop", route_policy.get("replay_decision_max_f1_drop", -1.0)))
        if cfg_abs >= 0.0:
            allowed_values.append(cfg_abs)
        if cfg_ratio >= 0.0:
            allowed_values.append(anchor_f1 * cfg_ratio)
        if not allowed_values:
            errors.append("no enabled replay F1 gate found")
        else:
            allowed = min(allowed_values)
            if f1_drop > allowed + EPS:
                errors.append(f"selected route exceeds replay F1 gate: drop={f1_drop} allowed={allowed}")
    except (TypeError, ValueError) as exc:
        errors.append(str(exc))

    report_patch_layers = sorted({int(layer) for layer in report.get("selected_hybrid_patch_layers", [])})
    route_patch_layers = int_list(route, "selected_patch_layers") or int_list(route, "patch_layers")
    if report_patch_layers and report_patch_layers != route_patch_layers:
        errors.append(f"report/route patch layer mismatch: report={report_patch_layers} route={route_patch_layers}")

    accepted_blocks = [int(value) for value in report.get("accepted_block_ids", [])]
    if args.require_mixed:
        if not accepted_blocks:
            errors.append("--require-mixed was set but no accepted blocks are recorded")
        num_layers = int(route.get("num_layers", 0) or 0)
        if num_layers > 0 and set(route_patch_layers) == set(range(num_layers)):
            errors.append("--require-mixed was set but selected route is still all-patch")

    for row in report.get("search_trace", []):
        if "accepted_block_id" not in row:
            continue
        gate = row.get("decision_quality_gate")
        if not isinstance(gate, dict):
            errors.append(f"accepted step {row.get('step')} missing decision_quality_gate")
            continue
        if gate.get("max_f1_drop") is None:
            errors.append(f"accepted step {row.get('step')} missing max_f1_drop in decision_quality_gate")

    if errors:
        print("strict hybrid search validation failed:")
        for err in errors:
            print(f"- {err}")
        raise SystemExit(1)

    print(f"strict hybrid search validation passed: {search_root}")


if __name__ == "__main__":
    main()
