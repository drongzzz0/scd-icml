#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scd_repro.strict_runtime import MethodSampleResult, MethodTiming, summarize_method_results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Re-aggregate a strict text eval directory after including or "
            "excluding qids. This is intended for leakage audits where the "
            "GPU run saved per-sample metrics and only the reporting subset "
            "changes."
        )
    )
    parser.add_argument("--strict-eval-dir", required=True, help="Directory containing per_sample_metrics.jsonl")
    parser.add_argument("--out-dir", required=True, help="Output directory for filtered summary artifacts")
    parser.add_argument("--include-qid-file", default="", help="Optional newline-separated qids to keep")
    parser.add_argument("--exclude-qid-file", default="", help="Optional newline-separated qids to remove")
    parser.add_argument("--direction", default="lite_to_7b")
    parser.add_argument("--dataset", default="hotpotqa_compressed")
    parser.add_argument(
        "--summary-list-out",
        default="",
        help="Optional path for a route-level scd_icml_mistral_summary.json-style list",
    )
    return parser


def load_qids(path: str) -> set[str]:
    if not path:
        return set()
    qid_path = Path(path)
    return {line.strip() for line in qid_path.read_text(encoding="utf-8").splitlines() if line.strip()}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def row_to_result(row: dict[str, Any]) -> MethodSampleResult:
    timing_row = row.get("timing_ms") if isinstance(row.get("timing_ms"), dict) else {}
    timing = MethodTiming(
        producer_prefill_ms=float(timing_row.get("producer_prefill", 0.0) or 0.0),
        consumer_prefill_ms=float(timing_row.get("consumer_prefill", 0.0) or 0.0),
        encode_ms=float(timing_row.get("encode", 0.0) or 0.0),
        transfer_ms=float(timing_row.get("transfer", 0.0) or 0.0),
        staging_copy_ms=float(timing_row.get("staging_copy", 0.0) or 0.0),
        rebuild_ms=float(timing_row.get("rebuild", 0.0) or 0.0),
        rebuild_prep_ms=float(timing_row.get("rebuild_prep", 0.0) or 0.0),
        rebuild_decode_core_ms=float(timing_row.get("rebuild_decode_core", 0.0) or 0.0),
        rebuild_rerope_ms=float(timing_row.get("rebuild_rerope", 0.0) or 0.0),
        rebuild_materialize_ms=float(timing_row.get("rebuild_materialize", 0.0) or 0.0),
        patch_infer_ms=float(timing_row.get("patch_infer", 0.0) or 0.0),
        patch_shard_load_ms=float(timing_row.get("patch_shard_load", 0.0) or 0.0),
        patch_shard_gpu_copy_ms=float(timing_row.get("patch_shard_gpu_copy", 0.0) or 0.0),
        patch_identity_check_ms=float(timing_row.get("patch_identity_check", 0.0) or 0.0),
        patch_residual_forward_ms=float(timing_row.get("patch_residual_forward", 0.0) or 0.0),
        patch_kproj_vproj_ms=float(timing_row.get("patch_kproj_vproj", 0.0) or 0.0),
        patch_knorm_ms=float(timing_row.get("patch_knorm", 0.0) or 0.0),
        patch_rotary_ms=float(timing_row.get("patch_rotary", 0.0) or 0.0),
        recompute_ms=float(timing_row.get("recompute", 0.0) or 0.0),
        decode_first_ms=float(timing_row.get("decode_first", 0.0) or 0.0),
        decode_rest_ms=float(timing_row.get("decode_rest", 0.0) or 0.0),
        decode_total_ms=float(timing_row.get("decode_total", 0.0) or 0.0),
        ttft_ms=float(timing_row.get("ttft", row.get("ttft_ms", 0.0)) or 0.0),
        total_ms=float(timing_row.get("total", row.get("total_ms", 0.0)) or 0.0),
        bytes_transferred=int(row.get("bytes_transferred", 0) or 0),
        raw_cache_bytes=int(row.get("raw_cache_bytes", 0) or 0),
        compression_ratio=row.get("compression_ratio"),
    )
    teacher_tokens = int(row.get("teacher_token_count", 0) or 0)
    avg_logp = float(row.get("avg_logp", 0.0) or 0.0)
    return MethodSampleResult(
        qid=str(row["qid"]),
        method=str(row["method"]),
        pred_text=str(row.get("pred_text", "")),
        f1=float(row.get("f1", 0.0) or 0.0),
        em=float(row.get("em", 0.0) or 0.0),
        payload_mb=float(row.get("payload_mb", 0.0) or 0.0),
        timing=timing,
        gold_answers=[str(value) for value in row.get("gold_answers", [])],
        prompt_tokens=int(row.get("prompt_tokens", 0) or 0),
        prefix_tokens=int(row.get("prefix_tokens", 0) or 0),
        first_token_id=row.get("first_token_id"),
        avg_logp=avg_logp,
        avg_nll=float(row.get("avg_nll", 0.0) or 0.0),
        teacher_logp_sum=float(avg_logp * teacher_tokens),
        teacher_token_count=teacher_tokens,
    )


def add_ppl(row: dict[str, Any]) -> None:
    nll = float(row.get("avg_nll", 0.0) or 0.0)
    row["ppl"] = float(math.exp(nll)) if nll < 50 else float("inf")


def comparison(row: dict[str, Any], baseline: dict[str, Any] | None) -> dict[str, Any]:
    if not baseline:
        return {}
    out: dict[str, Any] = {}
    for key in ("ttft_ms", "avg_logp", "avg_nll", "f1", "em"):
        if key in row and key in baseline and row[key] is not None and baseline[key] is not None:
            out[f"delta_{key.replace('_ms', '')}"] = float(row[key]) - float(baseline[key])
    return out


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    cols = [
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
    lines = [",".join(cols)]
    for row in rows:
        values = []
        for col in cols:
            value = row.get(col, "")
            values.append("" if value is None else str(value))
        lines.append(",".join(values))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary_md(path: Path, dataset: str, rows: list[dict[str, Any]]) -> None:
    lines = [
        f"# Filtered Strict Runtime Comparison: {dataset}",
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
    strict_eval_dir = Path(args.strict_eval_dir)
    rows = load_jsonl(strict_eval_dir / "per_sample_metrics.jsonl")
    include_qids = load_qids(args.include_qid_file)
    exclude_qids = load_qids(args.exclude_qid_file)
    if include_qids:
        rows = [row for row in rows if str(row.get("qid")) in include_qids]
    if exclude_qids:
        rows = [row for row in rows if str(row.get("qid")) not in exclude_qids]
    if not rows:
        raise SystemExit("no rows left after qid filtering")

    by_method: dict[str, list[MethodSampleResult]] = {}
    for row in rows:
        by_method.setdefault(str(row.get("method")), []).append(row_to_result(row))
    summary_rows = [summarize_method_results(items) for method, items in sorted(by_method.items())]
    for row in summary_rows:
        add_ppl(row)

    summary_map = {str(row["method"]): row for row in summary_rows}
    full = summary_map.get("full_consumer")
    reuse = summary_map.get("reuse_only")
    for row in summary_rows:
        delta_full = comparison(row, full)
        delta_reuse = comparison(row, reuse)
        row["delta_ttft_vs_full"] = delta_full.get("delta_ttft")
        row["delta_logp_vs_full"] = delta_full.get("delta_avg_logp")
        row["delta_nll_vs_full"] = delta_full.get("delta_avg_nll")
        row["delta_ttft_vs_reuse_only"] = delta_reuse.get("delta_ttft")
        row["delta_logp_vs_reuse_only"] = delta_reuse.get("delta_avg_logp")
        row["delta_nll_vs_reuse_only"] = delta_reuse.get("delta_avg_nll")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "source_strict_eval_dir": str(strict_eval_dir),
        "direction": args.direction,
        "dataset": args.dataset,
        "include_qid_file": args.include_qid_file or None,
        "exclude_qid_file": args.exclude_qid_file or None,
        "num_rows": len(rows),
        "num_qids": len({str(row.get("qid")) for row in rows}),
        "methods": sorted(by_method),
    }
    final_comparison = {
        "meta": meta,
        "modes": summary_map,
        "delta_vs_full_consumer": {method: comparison(row, full) for method, row in summary_map.items()},
        "delta_vs_reuse_only": {method: comparison(row, reuse) for method, row in summary_map.items()},
    }
    (out_dir / "summary.json").write_text(
        json.dumps({"meta": meta, "summary": summary_rows}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (out_dir / "per_mode_summary.json").write_text(json.dumps(summary_map, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out_dir / "final_comparison.json").write_text(json.dumps(final_comparison, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_summary_csv(out_dir / "summary.csv", summary_rows)
    write_summary_md(out_dir / "summary.md", args.dataset, summary_rows)

    if args.summary_list_out:
        route_rows = []
        for row in summary_rows:
            item = {
                "direction": args.direction,
                "dataset": args.dataset,
                "method": row["method"],
                "method_label": row["method_label"],
                "f1": row["f1"],
                "em": row["em"],
                "avg_nll": row["avg_nll"],
                "ppl": row["ppl"],
                "ttft_ms": row["ttft_ms"],
                "total_ms": row["total_ms"],
                "payload_mb": row["payload_mb"],
                "compression_ratio": row["compression_ratio"],
                "delta_ttft_vs_full": row.get("delta_ttft_vs_full"),
                "delta_ttft_vs_reuse_only": row.get("delta_ttft_vs_reuse_only"),
            }
            route_rows.append(item)
        summary_list_out = Path(args.summary_list_out)
        summary_list_out.parent.mkdir(parents=True, exist_ok=True)
        summary_list_out.write_text(json.dumps(route_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"filtered_summary={out_dir} qids={meta['num_qids']} rows={meta['num_rows']}")


if __name__ == "__main__":
    main()
