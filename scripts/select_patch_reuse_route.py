#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from scd_repro.route_selection.config import load_config, write_json
from scd_repro.route_selection.layer_mapping import estimate_layer_mapping, load_mapping
from scd_repro.route_selection.metrics import RouteObjectiveWeights
from scd_repro.route_selection.schema import RoutePlan
from scd_repro.route_selection.selectors import (
    ConditionalReusePruner,
    JsonlMetricEvaluator,
    SelectorConstraints,
    load_metric_rows,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Select a teacher-aligned Patch/Reuse route.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--selector", default="", choices=["", "conditional_pruning", "beam"])
    parser.add_argument("--initial-route", default="", help="Optional all-patch or high-quality route JSON")
    parser.add_argument("--metrics", default="", help="Route-level metrics JSON/JSONL from completed evals")
    parser.add_argument("--out-dir", default="")
    return parser


def _constraints(selector_cfg: dict) -> SelectorConstraints:
    def opt_float(key: str) -> float | None:
        value = selector_cfg.get(key)
        return None if value in {"", None} else float(value)

    return SelectorConstraints(
        max_total_drift=opt_float("max_total_drift"),
        max_delta_per_step=opt_float("max_delta_per_step"),
        max_logit_kl=opt_float("max_logit_kl"),
        max_ppl_delta=opt_float("max_ppl_delta"),
        max_avg_nll_delta=opt_float("max_avg_nll_delta"),
        max_kv_drift=opt_float("max_kv_drift"),
        max_hidden_drift=opt_float("max_hidden_drift"),
        min_f1_retention=opt_float("min_f1_retention"),
        max_estimated_ttft_ms=opt_float("max_estimated_ttft_ms"),
    )


def _source_mapping(cfg: dict) -> list[int]:
    model_cfg = cfg.get("model", {})
    mapping_cfg = cfg.get("mapping", {})
    mapping_path = mapping_cfg.get("mapping_path") or ""
    if mapping_path and Path(mapping_path).is_file():
        return load_mapping(mapping_path)
    return estimate_layer_mapping(
        strategy=str(mapping_cfg.get("strategy", "proportional")),
        producer_layers=int(model_cfg.get("producer_num_layers")),
        consumer_layers=int(model_cfg.get("consumer_num_layers")),
        mapping_path=mapping_path or None,
    )


def main() -> None:
    args = build_parser().parse_args()
    cfg = load_config(args.config)
    selector_cfg = cfg.get("selector", {})
    out_dir = Path(args.out_dir or cfg.get("output", {}).get("dir", "outputs/route_selection"))
    out_dir.mkdir(parents=True, exist_ok=True)
    selector_name = args.selector or str(selector_cfg.get("name", "conditional_pruning"))
    if selector_name != "conditional_pruning":
        raise SystemExit("beam selector is provided as a proxy module but this CLI currently materializes conditional_pruning only")

    metrics_path = args.metrics or selector_cfg.get("metrics_path") or ""
    if not metrics_path:
        report = {
            "status": "needs_metrics",
            "reason": "selector requires route-level teacher metrics; no metrics_path was supplied",
            "expected_metrics_schema": {
                "route_fingerprint": "optional if patch_layers is present",
                "patch_layers": [0],
                "kv_drift_mean": 0.0,
                "hidden_drift_mean": 0.0,
                "logit_kl": 0.0,
                "ppl_delta": 0.0,
                "payload_bytes": 0,
                "estimated_ttft_ms": 0.0,
            },
        }
        write_json(out_dir / "selection_report.json", report)
        print(f"selection_report={out_dir / 'selection_report.json'}")
        raise SystemExit("missing route-level metrics; wrote schema-only report")

    model_cfg = cfg.get("model", {})
    source_mapping = _source_mapping(cfg)
    if args.initial_route:
        initial_route = RoutePlan.from_dict(__import__("json").loads(Path(args.initial_route).read_text(encoding="utf-8-sig")))
    else:
        initial_route = RoutePlan.all_patch(
            num_layers=int(model_cfg.get("consumer_num_layers")),
            source_mapping=source_mapping,
            name="all_patch_initial_route",
        )
    metric_rows = load_metric_rows(metrics_path)
    evaluator = JsonlMetricEvaluator(metric_rows)
    weights = RouteObjectiveWeights(**{key: float(value) for key, value in cfg.get("metrics", {}).get("weights", {}).items()})
    pruner = ConditionalReusePruner(
        evaluator,
        weights=weights,
        constraints=_constraints(selector_cfg),
    )
    result = pruner.run(initial_route)
    write_json(out_dir / "selection_report.json", result)
    write_json(out_dir / "selected_route.json", result["selected_route"])
    steps_dir = out_dir / "selected_route_steps"
    for item in result.get("trace", []):
        if isinstance(item, dict) and isinstance(item.get("route"), dict):
            steps_dir.mkdir(parents=True, exist_ok=True)
            write_json(steps_dir / f"selected_route_step_{int(item['step']):03d}.json", item["route"])
    if result.get("pending_routes"):
        pending_dir = out_dir / "pending_routes"
        pending_dir.mkdir(parents=True, exist_ok=True)
        for idx, route in enumerate(result["pending_routes"], start=1):
            write_json(pending_dir / f"pending_route_{idx:03d}.json", route)
    print(f"selection_report={out_dir / 'selection_report.json'}")
    print(f"selected_route={out_dir / 'selected_route.json'}")
    if result.get("pending_routes"):
        print(f"pending_routes={out_dir / 'pending_routes'} count={len(result['pending_routes'])}")


if __name__ == "__main__":
    main()
