#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
from pathlib import Path
from typing import Any

from scd_repro.route_selection.selectors import load_metric_rows


FIELDS = [
    "layer_i",
    "layer_j",
    "status",
    "objective_field",
    "interaction",
    "base_objective",
    "objective_i",
    "objective_j",
    "objective_ij",
    "base_reuse_layers",
    "reuse_i",
    "reuse_j",
    "reuse_ij",
    "f1_retention_ij",
    "avg_nll_delta_ij",
    "estimated_ttft_ms_ij",
    "missing_sets",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Report pairwise Patch/Reuse route interactions from completed route metrics. "
            "Rows with missing routes are reported as missing rather than imputed."
        )
    )
    parser.add_argument("--metrics", action="append", default=[], help="Metrics JSON/JSONL; may be repeated")
    parser.add_argument("--metrics-glob", default="", help="Glob for metrics JSON/JSONL files")
    parser.add_argument("--candidate-layers", required=True, help="Comma/range list of layers to check")
    parser.add_argument("--base-reuse-layers", default="", help="Base Reuse set; defaults to all-Patch")
    parser.add_argument("--objective-field", default="avg_nll_delta")
    parser.add_argument("--base-objective", type=float, default=None, help="Optional objective for the base route")
    parser.add_argument("--conflict-threshold", type=float, default=0.0)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-csv", default="")
    return parser


def parse_layers(raw: str) -> list[int]:
    layers: list[int] = []
    seen: set[int] = set()
    for item in str(raw or "").replace(";", ",").split(","):
        part = item.strip()
        if not part:
            continue
        if "-" in part:
            left, right = part.split("-", 1)
            start = int(left.strip())
            end = int(right.strip())
            if end < start:
                raise ValueError(f"invalid descending layer range: {part}")
            values = range(start, end + 1)
        else:
            values = [int(part)]
        for layer in values:
            if layer not in seen:
                seen.add(layer)
                layers.append(layer)
    return sorted(layers)


def collect_metric_paths(args: argparse.Namespace) -> list[Path]:
    paths = [Path(value) for value in args.metrics]
    if args.metrics_glob:
        paths.extend(Path(value) for value in glob.glob(args.metrics_glob))
    unique: dict[str, Path] = {}
    for path in paths:
        unique[str(path.resolve())] = path
    return [unique[key] for key in sorted(unique)]


def route_key(layers: list[int] | set[int]) -> tuple[int, ...]:
    return tuple(sorted(int(layer) for layer in layers))


def objective(row: dict[str, Any], field: str) -> float | None:
    value = row.get(field)
    if value in {"", None}:
        return None
    value_f = float(value)
    if not math.isfinite(value_f):
        return None
    return value_f


def main() -> None:
    args = build_parser().parse_args()
    metric_paths = collect_metric_paths(args)
    if not metric_paths:
        raise SystemExit("no metrics files supplied")

    by_reuse: dict[tuple[int, ...], dict[str, Any]] = {}
    source_files: dict[tuple[int, ...], str] = {}
    for path in metric_paths:
        for row in load_metric_rows(path):
            if row.get("status") not in {None, "ok"}:
                continue
            key = route_key(row.get("reuse_layers") or [])
            by_reuse[key] = row
            source_files[key] = str(path)

    base_key = route_key(parse_layers(args.base_reuse_layers))
    candidate_layers = parse_layers(args.candidate_layers)
    rows: list[dict[str, Any]] = []
    base_row = by_reuse.get(base_key)
    base_obj = args.base_objective
    if base_obj is None and base_row is not None:
        base_obj = objective(base_row, args.objective_field)

    for idx, left in enumerate(candidate_layers):
        for right in candidate_layers[idx + 1 :]:
            key_i = route_key(set(base_key) | {left})
            key_j = route_key(set(base_key) | {right})
            key_ij = route_key(set(base_key) | {left, right})
            row_i = by_reuse.get(key_i)
            row_j = by_reuse.get(key_j)
            row_ij = by_reuse.get(key_ij)
            missing = [
                ",".join(str(v) for v in key)
                for key, row in [(base_key, base_row), (key_i, row_i), (key_j, row_j), (key_ij, row_ij)]
                if row is None and not (key == base_key and base_obj is not None)
            ]
            obj_i = None if row_i is None else objective(row_i, args.objective_field)
            obj_j = None if row_j is None else objective(row_j, args.objective_field)
            obj_ij = None if row_ij is None else objective(row_ij, args.objective_field)
            if base_obj is None or obj_i is None or obj_j is None or obj_ij is None or missing:
                status = "missing"
                interaction = None
            else:
                interaction = float(obj_ij - obj_i - obj_j + base_obj)
                status = "conflict" if interaction > float(args.conflict_threshold) else "ok"
            rows.append(
                {
                    "layer_i": int(left),
                    "layer_j": int(right),
                    "status": status,
                    "objective_field": args.objective_field,
                    "interaction": interaction,
                    "base_objective": base_obj,
                    "objective_i": obj_i,
                    "objective_j": obj_j,
                    "objective_ij": obj_ij,
                    "base_reuse_layers": list(base_key),
                    "reuse_i": list(key_i),
                    "reuse_j": list(key_j),
                    "reuse_ij": list(key_ij),
                    "f1_retention_ij": None if row_ij is None else row_ij.get("f1_retention"),
                    "avg_nll_delta_ij": None if row_ij is None else row_ij.get("avg_nll_delta"),
                    "estimated_ttft_ms_ij": None if row_ij is None else row_ij.get("estimated_ttft_ms"),
                    "missing_sets": missing,
                    "metric_sources": {
                        ",".join(str(v) for v in key): source_files.get(key)
                        for key in [base_key, key_i, key_j, key_ij]
                    },
                }
            )

    payload = {
        "objective_field": args.objective_field,
        "conflict_threshold": float(args.conflict_threshold),
        "base_reuse_layers": list(base_key),
        "candidate_layers": candidate_layers,
        "rows": rows,
    }
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    out_csv = Path(args.out_csv) if args.out_csv else out_json.with_suffix(".csv")
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in FIELDS})
    print(f"pairwise_json={out_json}")
    print(f"pairwise_csv={out_csv}")
    print(f"rows={len(rows)} missing={sum(1 for row in rows if row['status'] == 'missing')}")


if __name__ == "__main__":
    main()
