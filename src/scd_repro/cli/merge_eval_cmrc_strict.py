from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


METHOD_ORDER = [
    "full_consumer",
    "reuse_only",
    "reuse_fallback",
    "patch_only",
    "reuse_patch",
    "reuse_patch_recompute",
    "oracle_xnorm_suffix",
    "oracle_state_suffix",
    "oracle_kv_patchonly",
]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="合并多次 strict CMRC eval 产物为单一对比目录")
    p.add_argument("--input-dir", dest="input_dirs", action="append", required=True, help="输入目录，可重复传入")
    p.add_argument("--out", required=True, help="合并后的输出目录")
    return p


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _comparison_row(row: dict[str, Any], baseline: dict[str, Any] | None) -> dict[str, float | None]:
    if baseline is None:
        return {}
    return {
        "delta_ttft_ms": None if row.get("ttft_ms") is None or baseline.get("ttft_ms") is None else float(row["ttft_ms"]) - float(baseline["ttft_ms"]),
        "delta_logp": None if row.get("avg_logp") is None or baseline.get("avg_logp") is None else float(row["avg_logp"]) - float(baseline["avg_logp"]),
        "delta_nll": None if row.get("avg_nll") is None or baseline.get("avg_nll") is None else float(row["avg_nll"]) - float(baseline["avg_nll"]),
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
        "| mode | avg_logp | avg_nll | f1 | em | bytes | compression | ttft_ms | delta_ttft_vs_full |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        compression = row.get("compression_ratio")
        delta_ttft = row.get("delta_ttft_vs_full")
        lines.append(
            f"| {row['method_label']} | {float(row['avg_logp']):.4f} | {float(row['avg_nll']):.4f} | {float(row['f1']):.4f} | {float(row['em']):.4f} | {float(row['bytes_transferred']):.1f} | {0.0 if compression is None else float(compression):.3f} | {float(row['ttft_ms']):.2f} | {0.0 if delta_ttft is None else float(delta_ttft):.2f} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load_rows(path: Path) -> dict[str, dict[str, Any]]:
    rows = _load_json(path / "per_mode_summary.json")
    if not isinstance(rows, dict):
        raise ValueError(f"{path}/per_mode_summary.json 格式非法")
    return {str(k): dict(v) for k, v in rows.items()}


def _load_meta(path: Path) -> dict[str, Any]:
    final_path = path / "final_comparison.json"
    if final_path.exists():
        data = _load_json(final_path)
        if isinstance(data, dict) and isinstance(data.get("meta"), dict):
            return dict(data["meta"])
    summary_path = path / "summary.json"
    if summary_path.exists():
        data = _load_json(summary_path)
        if isinstance(data, dict) and isinstance(data.get("meta"), dict):
            return dict(data["meta"])
    return {}


def _load_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _method_sort_key(method: str) -> tuple[int, str]:
    try:
        return (METHOD_ORDER.index(method), method)
    except ValueError:
        return (len(METHOD_ORDER), method)


def _write_manifest(out_dir: Path, input_dirs: list[Path]) -> None:
    manifest = {
        "timestamp_utc": None,
        "command": " ".join(sys.argv),
        "seed": None,
        "config_paths": {"inputs": [str(p) for p in input_dirs]},
        "git_commit": "unknown",
        "hostname": "unknown",
    }
    _save_json(out_dir / "run_manifest.json", manifest)


def main() -> None:
    args = build_parser().parse_args()
    input_dirs = [Path(p).resolve() for p in args.input_dirs]
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    per_mode_summary: dict[str, dict[str, Any]] = {}
    sample_rows_by_method: dict[str, list[dict[str, Any]]] = {}
    sample_seen: set[tuple[str, str]] = set()
    input_meta: list[dict[str, Any]] = []

    for input_dir in input_dirs:
        rows = _load_rows(input_dir)
        meta = _load_meta(input_dir)
        input_meta.append(
            {
                "dir": str(input_dir),
                "methods": sorted(rows.keys(), key=_method_sort_key),
                "meta": meta,
            }
        )
        for method, row in rows.items():
            if method in per_mode_summary:
                raise ValueError(f"重复 method: {method} 来自 {input_dir}")
            per_mode_summary[method] = dict(row)

        sample_rows = _load_jsonl_rows(input_dir / "per_sample_metrics.jsonl")
        for row in sample_rows:
            method = str(row.get("method"))
            qid = str(row.get("qid"))
            key = (method, qid)
            if key in sample_seen:
                raise ValueError(f"重复 sample: method={method} qid={qid}")
            sample_seen.add(key)
            sample_rows_by_method.setdefault(method, []).append(row)

    ordered_methods = sorted(per_mode_summary.keys(), key=_method_sort_key)
    summary_rows = [per_mode_summary[method] for method in ordered_methods]
    full_baseline = per_mode_summary.get("full_consumer")
    reuse_only_baseline = per_mode_summary.get("reuse_only")

    for row in summary_rows:
        delta_full = _comparison_row(row, full_baseline)
        delta_reuse = _comparison_row(row, reuse_only_baseline)
        row["delta_ttft_vs_full"] = delta_full.get("delta_ttft_ms")
        row["delta_logp_vs_full"] = delta_full.get("delta_logp")
        row["delta_nll_vs_full"] = delta_full.get("delta_nll")
        row["delta_ttft_vs_reuse_only"] = delta_reuse.get("delta_ttft_ms")
        row["delta_logp_vs_reuse_only"] = delta_reuse.get("delta_logp")
        row["delta_nll_vs_reuse_only"] = delta_reuse.get("delta_nll")

    base_meta = dict(input_meta[0]["meta"]) if input_meta else {}
    merged_meta = {
        **base_meta,
        "methods": ordered_methods,
        "merge_inputs": input_meta,
        "isolated_patch_only": "patch_only" in ordered_methods and len(input_dirs) > 1,
    }

    final_comparison = {
        "meta": merged_meta,
        "modes": {method: per_mode_summary[method] for method in ordered_methods},
        "delta_vs_full_consumer": {method: _comparison_row(per_mode_summary[method], full_baseline) for method in ordered_methods},
        "delta_vs_reuse_only": {method: _comparison_row(per_mode_summary[method], reuse_only_baseline) for method in ordered_methods},
    }

    _save_json(out_dir / "summary.json", {"meta": merged_meta, "summary": summary_rows})
    _save_json(out_dir / "per_mode_summary.json", {method: per_mode_summary[method] for method in ordered_methods})
    _save_json(out_dir / "final_comparison.json", final_comparison)
    _write_summary_csv(out_dir / "summary.csv", summary_rows)
    _write_summary_md(out_dir / "summary.md", summary_rows)

    combined_sample_rows: list[dict[str, Any]] = []
    samples_dir = out_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    for method in ordered_methods:
        method_rows = sample_rows_by_method.get(method, [])
        if method_rows:
            _write_jsonl(samples_dir / f"{method}.jsonl", method_rows)
            combined_sample_rows.extend(method_rows)
    if combined_sample_rows:
        _write_jsonl(out_dir / "per_sample_metrics.jsonl", combined_sample_rows)

    _write_manifest(out_dir, input_dirs)


if __name__ == "__main__":
    main()
