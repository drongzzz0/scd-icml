from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


METHOD_FLAGS = [
    ("run_full_consumer_baseline", "full_consumer"),
    ("run_raw_kv", "raw_kv"),
    ("run_quantized_4bit", "quantized_4bit"),
    ("run_selective_recompute", "selective_recompute"),
    ("run_reuse_only", "reuse_only"),
    ("run_reuse_fallback", "reuse_fallback"),
    ("run_patch_only", "patch_only"),
    ("run_reuse_patch", "reuse_patch"),
    ("run_reuse_patch_recompute", "reuse_patch_recompute"),
    ("run_scd_full", "scd_full"),
    ("run_oracle_xnorm_suffix", "oracle_xnorm_suffix"),
    ("run_oracle_state_suffix", "oracle_state_suffix"),
    ("run_oracle_kv_patchonly", "oracle_kv_patchonly"),
]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Summarize strict SCD-ICML Mistral reproduction outputs")
    p.add_argument("--run-root", required=True, help="Run root containing <direction>/strict_eval/<dataset> outputs")
    p.add_argument("--out-json", default="", help="Defaults to <run-root>/scd_icml_mistral_summary.json")
    p.add_argument("--out-md", default="", help="Defaults to <run-root>/scd_icml_mistral_summary.md")
    p.add_argument("--eval-config", default="", help="Optional eval config used to derive expected method names")
    p.add_argument("--expected-directions", default="", help="Comma-separated directions expected in the run")
    p.add_argument("--expected-datasets", default="", help="Comma-separated datasets expected in the run")
    p.add_argument("--expected-methods", default="", help="Comma-separated method names expected in each dataset")
    p.add_argument("--allow-missing", action="store_true", help="Write a partial summary instead of failing")
    return p


def _split_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _expected_methods_from_eval_config(path: str) -> list[str]:
    if not path:
        return []
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"eval config not found: {cfg_path}")
    cfg = _load_json(cfg_path)
    runtime_eval = cfg.get("runtime_eval", {})
    if not isinstance(runtime_eval, dict):
        return []
    return [method for flag, method in METHOD_FLAGS if bool(runtime_eval.get(flag, False))]


def _iter_summary_paths(root: Path) -> list[Path]:
    return sorted(root.glob("*/strict_eval/*/per_mode_summary.json"))


def _row_from_method(direction: str, dataset: str, method: str, row: dict[str, Any]) -> dict[str, Any]:
    return {
        "direction": direction,
        "dataset": dataset,
        "method": method,
        "method_label": row.get("method_label"),
        "f1": row.get("f1"),
        "em": row.get("em"),
        "avg_nll": row.get("avg_nll"),
        "ppl": row.get("ppl"),
        "ttft_ms": row.get("ttft_ms"),
        "total_ms": row.get("total_ms"),
        "payload_mb": row.get("payload_mb"),
        "compression_ratio": row.get("compression_ratio"),
        "delta_ttft_vs_full": row.get("delta_ttft_vs_full"),
        "delta_ttft_vs_reuse_only": row.get("delta_ttft_vs_reuse_only"),
    }


def collect_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for summary_path in _iter_summary_paths(root):
        rel = summary_path.relative_to(root)
        if len(rel.parts) < 4:
            continue
        direction = rel.parts[0]
        dataset = rel.parts[2]
        payload = _load_json(summary_path)
        if not isinstance(payload, dict):
            raise ValueError(f"expected object in {summary_path}")
        for method, row in payload.items():
            if not isinstance(row, dict):
                raise ValueError(f"expected object for method {method} in {summary_path}")
            rows.append(_row_from_method(direction, dataset, str(method), row))
    return rows


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and (math.isinf(value) or math.isnan(value)):
        return str(value)
    if isinstance(value, (int, float)):
        return f"{float(value):.4f}"
    return str(value)


def write_outputs(root: Path, rows: list[dict[str, Any]], out_json: Path, out_md: Path) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# SCD-ICML Mistral Reproduction Summary",
        "",
        f"- Run root: `{root}`",
        f"- Rows: {len(rows)}",
        "",
        "| direction | dataset | method | f1 | em | avg_nll | ppl | ttft_ms | total_ms | payload_mb | compression |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['direction']} | {row['dataset']} | {row['method']} | "
            f"{_fmt(row.get('f1'))} | {_fmt(row.get('em'))} | "
            f"{_fmt(row.get('avg_nll'))} | {_fmt(row.get('ppl'))} | "
            f"{_fmt(row.get('ttft_ms'))} | {_fmt(row.get('total_ms'))} | "
            f"{_fmt(row.get('payload_mb'))} | {_fmt(row.get('compression_ratio'))} |"
        )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def find_missing(
    rows: list[dict[str, Any]],
    expected_directions: list[str],
    expected_datasets: list[str],
    expected_methods: list[str],
) -> list[str]:
    if not expected_directions or not expected_datasets or not expected_methods:
        return []
    present = {(str(row["direction"]), str(row["dataset"]), str(row["method"])) for row in rows}
    missing = []
    for direction in expected_directions:
        for dataset in expected_datasets:
            for method in expected_methods:
                key = (direction, dataset, method)
                if key not in present:
                    missing.append("/".join(key))
    return missing


def main() -> None:
    args = build_parser().parse_args()
    root = Path(args.run_root)
    if not root.exists():
        raise SystemExit(f"run root not found: {root}")

    rows = collect_rows(root)
    if not rows:
        raise SystemExit(f"no per_mode_summary.json files found under {root}")

    expected_directions = _split_csv(args.expected_directions)
    expected_datasets = _split_csv(args.expected_datasets)
    expected_methods = _split_csv(args.expected_methods) or _expected_methods_from_eval_config(args.eval_config)
    missing = find_missing(rows, expected_directions, expected_datasets, expected_methods)

    out_json = Path(args.out_json) if args.out_json else root / "scd_icml_mistral_summary.json"
    out_md = Path(args.out_md) if args.out_md else root / "scd_icml_mistral_summary.md"
    write_outputs(root, rows, out_json, out_md)

    print(f"summary_json={out_json}")
    print(f"summary_md={out_md}")
    print(f"rows={len(rows)}")
    if missing:
        print("missing expected results:", file=sys.stderr)
        for item in missing:
            print(f"- {item}", file=sys.stderr)
        if not args.allow_missing:
            raise SystemExit(2)


if __name__ == "__main__":
    main()
