from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


BASELINE_SPECS = [
    ("svdllm", "SVD-LLM-style adaptation", ["svdllm_style_kv"]),
    ("slicegpt", "SliceGPT-style adapted KV transfer", ["slicegpt_style_kv"]),
    ("tensorllm", "TensorLLM-style adapted KV transfer", ["tensorllm_style_kv"]),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the adapted neighboring-compression baseline table from completed strict-runtime summaries."
    )
    parser.add_argument("--svdllm-summary", required=True, help="summary.json from run_svdllm_style_kv_cmrc.py")
    parser.add_argument("--slicegpt-summary", required=True, help="summary.json from run_slicegpt_style_kv_cmrc.py")
    parser.add_argument("--tensorllm-summary", required=True, help="summary.json from run_tensorllm_style_kv_cmrc.py")
    parser.add_argument("--scd-summary", required=True, help="summary JSON containing full_consumer and SCD rows")
    parser.add_argument("--out-dir", required=True, help="Output directory for CSV/Markdown/LaTeX artifacts")
    parser.add_argument(
        "--require-compatible-protocol",
        action="store_true",
        help="Fail instead of writing outputs when source protocols or full-consumer baselines disagree.",
    )
    return parser.parse_args()


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def coerce_rows(obj: Any) -> list[dict[str, Any]]:
    if isinstance(obj, dict) and isinstance(obj.get("summary"), list):
        return [dict(row) for row in obj["summary"]]
    if isinstance(obj, dict) and isinstance(obj.get("rows"), list):
        return [dict(row) for row in obj["rows"]]
    if isinstance(obj, dict) and all(isinstance(value, dict) for value in obj.values()):
        rows: list[dict[str, Any]] = []
        for key, value in obj.items():
            row = dict(value)
            row.setdefault("method", key)
            rows.append(row)
        return rows
    if isinstance(obj, list):
        return [dict(row) for row in obj]
    raise ValueError("summary JSON must be a list, a {summary:[...]} object, a {rows:[...]} object, or a method map")


def find_method(rows: list[dict[str, Any]], candidates: list[str]) -> dict[str, Any]:
    candidate_set = {name.lower() for name in candidates}
    for row in rows:
        method = str(row.get("method") or row.get("mode") or "").lower()
        label = str(row.get("method_label") or "").lower()
        if method in candidate_set or label in candidate_set:
            return row
    raise ValueError(f"missing method row, expected one of: {', '.join(candidates)}")


def optional_float(row: dict[str, Any], *keys: str) -> float:
    for key in keys:
        value = row.get(key)
        if value is not None:
            return float(value)
    raise ValueError(f"row is missing numeric keys: {', '.join(keys)}")


def summarize_method(label: str, rows: list[dict[str, Any]], candidates: list[str]) -> dict[str, Any]:
    full = find_method(rows, ["full_consumer", "oracle"])
    method = find_method(rows, candidates)
    full_f1 = optional_float(full, "f1", "full_f1")
    method_f1 = optional_float(method, "f1")
    full_ttft = optional_float(full, "ttft_ms", "full_ttft_ms")
    method_ttft = optional_float(method, "ttft_ms")
    return {
        "method": label,
        "quality_retention": method_f1 / full_f1 if full_f1 > 0 else None,
        "ttft_speedup": full_ttft / method_ttft if method_ttft > 0 else None,
        "f1": method_f1,
        "full_f1": full_f1,
        "ttft_ms": method_ttft,
        "full_ttft_ms": full_ttft,
    }


def summarize_scd(rows: list[dict[str, Any]]) -> dict[str, Any]:
    candidates = [
        "scd_full",
        "reuse_patch",
        "reuse_patch_scd",
        "scd",
        "hotpotqa_compressed_full7405_selected_scd",
    ]
    return summarize_method("SCD", rows, candidates)


def extract_meta(obj: Any) -> dict[str, Any]:
    if isinstance(obj, dict) and isinstance(obj.get("meta"), dict):
        return dict(obj["meta"])
    return {}


def protocol_subset(meta: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "limit",
        "max_eval_samples",
        "max_length",
        "max_new_tokens",
        "bandwidth_gbps",
        "dataset",
        "data_root",
    ]
    return {key: meta.get(key) for key in keys if key in meta}


def protocol_warnings(protocols: dict[str, dict[str, Any]], rows: list[dict[str, Any]]) -> list[str]:
    warnings: list[str] = []
    comparable_keys = ["limit", "max_eval_samples", "max_length", "max_new_tokens", "bandwidth_gbps"]
    for key in comparable_keys:
        values = {
            name: proto[key]
            for name, proto in protocols.items()
            if key in proto and proto[key] is not None
        }
        if len({json.dumps(value, sort_keys=True) for value in values.values()}) > 1:
            warnings.append(
                f"Protocol mismatch for {key}: "
                + ", ".join(f"{name}={value}" for name, value in sorted(values.items()))
            )

    full_f1_values = {
        str(row["method"]): float(row["full_f1"])
        for row in rows
        if row.get("full_f1") is not None
    }
    if full_f1_values:
        low = min(full_f1_values.values())
        high = max(full_f1_values.values())
        if high - low > 0.05:
            warnings.append(
                "Full-consumer F1 differs across sources: "
                + ", ".join(f"{name}={value:.6f}" for name, value in sorted(full_f1_values.items()))
            )
    return warnings


def fmt_percent(value: float | None) -> str:
    return "NA" if value is None else f"{value * 100.0:.1f}%"


def fmt_speedup(value: float | None) -> str:
    return "NA" if value is None else f"{value:.2f}x"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = ["method", "quality_retention", "ttft_speedup", "f1", "full_f1", "ttft_ms", "full_ttft_ms"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def write_markdown(path: Path, rows: list[dict[str, Any]], warnings: list[str] | None = None) -> None:
    lines = [
        "# Adapted Neighboring Compression Baselines",
        "",
        "| method | quality retention | TTFT speedup | F1 | TTFT ms |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['method']} | {fmt_percent(row['quality_retention'])} | "
            f"{fmt_speedup(row['ttft_speedup'])} | {float(row['f1']):.4f} | {float(row['ttft_ms']):.2f} |"
        )
    if warnings:
        lines.extend(["", "> Protocol warning: " + " ".join(warnings)])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def escape_latex(text: str) -> str:
    return text.replace("\\", r"\textbackslash{}").replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def write_latex(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"\textbf{Method} & \textbf{Quality Retention} & \textbf{TTFT Speedup} \\",
        r"\midrule",
    ]
    for row in rows:
        method = escape_latex(str(row["method"]))
        lines.append(f"{method} & {escape_latex(fmt_percent(row['quality_retention']))} & {fmt_speedup(row['ttft_speedup'])} " + r"\\")
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    source_map = {
        "svdllm": Path(args.svdllm_summary),
        "slicegpt": Path(args.slicegpt_summary),
        "tensorllm": Path(args.tensorllm_summary),
    }
    rows: list[dict[str, Any]] = []
    protocols: dict[str, dict[str, Any]] = {}
    for key, label, candidates in BASELINE_SPECS:
        summary_obj = load_json(source_map[key])
        summary_rows = coerce_rows(summary_obj)
        protocols[key] = protocol_subset(extract_meta(summary_obj))
        row = summarize_method(label, summary_rows, candidates)
        row["source"] = str(source_map[key])
        rows.append(row)
    scd_obj = load_json(args.scd_summary)
    scd_rows = coerce_rows(scd_obj)
    protocols["scd"] = protocol_subset(extract_meta(scd_obj))
    scd_row = summarize_scd(scd_rows)
    scd_row["source"] = str(Path(args.scd_summary))
    rows.append(scd_row)
    warnings = protocol_warnings(protocols, rows)
    if args.require_compatible_protocol and warnings:
        raise SystemExit("incompatible neighboring-baseline protocols: " + " | ".join(warnings))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "neighboring_baselines.csv", rows)
    write_markdown(out_dir / "neighboring_baselines.md", rows, warnings)
    write_latex(out_dir / "neighboring_baselines.tex", rows)
    manifest = {
        "status": "ok",
        "warnings": warnings,
        "protocols": protocols,
        "outputs": [
            "neighboring_baselines.csv",
            "neighboring_baselines.md",
            "neighboring_baselines.tex",
        ],
        "sources": {key: str(value) for key, value in source_map.items()} | {"scd": str(Path(args.scd_summary))},
        "rows": rows,
    }
    (out_dir / "neighboring_baselines_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"neighboring_baselines={out_dir}")


if __name__ == "__main__":
    main()
