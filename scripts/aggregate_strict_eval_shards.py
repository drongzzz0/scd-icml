#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


SUMMARY_COLUMNS = [
    "method",
    "method_label",
    "num_samples",
    "f1",
    "em",
    "avg_nll",
    "ppl",
    "ttft_ms",
    "total_ms",
    "payload_mb",
    "compression_ratio",
]

WEIGHTED_COLUMNS = [
    "f1",
    "em",
    "avg_nll",
    "ttft_ms",
    "total_ms",
    "payload_mb",
    "compression_ratio",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate strict eval summary.csv files from qid shards. "
            "Rows are grouped by method and weighted by num_samples."
        )
    )
    parser.add_argument("--out-dir", required=True, help="Directory for aggregate summary artifacts")
    parser.add_argument("summary_csv", nargs="+", help="Shard summary.csv files")
    return parser


def parse_float(value: str | None) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return float(text)


def read_summary(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty summary file: {path}")
    for row in rows:
        row["_source"] = str(path)
    return rows


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        method = str(row.get("method") or "").strip()
        if not method:
            raise ValueError(f"summary row without method: {row}")
        groups[method].append(row)

    out: list[dict[str, Any]] = []
    for method, method_rows in groups.items():
        total_n = 0
        sums = {col: 0.0 for col in WEIGHTED_COLUMNS}
        weights = {col: 0 for col in WEIGHTED_COLUMNS}
        method_label = str(method_rows[0].get("method_label") or method)
        sources = []
        for row in method_rows:
            n = int(float(str(row.get("num_samples") or "0")))
            if n <= 0:
                raise ValueError(f"invalid num_samples in {row.get('_source')}: {row}")
            total_n += n
            sources.append(row["_source"])
            for col in WEIGHTED_COLUMNS:
                value = parse_float(row.get(col))
                if value is not None and math.isfinite(value):
                    sums[col] += value * n
                    weights[col] += n
        if total_n <= 0:
            raise ValueError(f"no samples for method {method}")
        agg: dict[str, Any] = {
            "method": method,
            "method_label": method_label,
            "num_samples": total_n,
            "sources": sources,
        }
        for col in WEIGHTED_COLUMNS:
            agg[col] = sums[col] / weights[col] if weights[col] else None
        nll = agg.get("avg_nll")
        agg["ppl"] = float(math.exp(nll)) if isinstance(nll, float) and nll < 50 else None
        out.append(agg)
    return sorted(out, key=lambda row: str(row["method"]))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in SUMMARY_COLUMNS})


def write_md(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Aggregated Strict Runtime Comparison",
        "",
        "| mode | n | avg_nll | ppl | f1 | em | payload_mb | ttft_ms | total_ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['method_label']} | {int(row['num_samples'])} | "
            f"{float(row['avg_nll']):.6f} | {float(row['ppl']):.6f} | "
            f"{float(row['f1']):.6f} | {float(row['em']):.6f} | "
            f"{float(row['payload_mb']):.6f} | {float(row['ttft_ms']):.3f} | "
            f"{float(row['total_ms']):.3f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    summary_paths = [Path(value) for value in args.summary_csv]
    all_rows: list[dict[str, Any]] = []
    for path in summary_paths:
        if not path.exists():
            raise FileNotFoundError(path)
        all_rows.extend(read_summary(path))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = aggregate(all_rows)
    write_csv(out_dir / "summary.csv", rows)
    write_md(out_dir / "summary.md", rows)
    (out_dir / "summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"aggregate_summary={out_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
