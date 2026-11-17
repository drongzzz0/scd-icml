#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any

from scd_repro.route_selection.profiler import baseline_row_from_eval


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Collect completed strict-eval summaries into route-level metrics for "
            "teacher-aligned Patch/Reuse selection."
        )
    )
    parser.add_argument("--eval-root", action="append", default=[], help="Strict eval run root; may be repeated")
    parser.add_argument("--eval-glob", default="", help="Glob for strict eval run roots")
    parser.add_argument("--direction", default="lite_to_7b")
    parser.add_argument("--dataset", default="cmrc2018")
    parser.add_argument("--method", default="scd_full")
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-jsonl", default="")
    return parser


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def collect_roots(args: argparse.Namespace) -> list[Path]:
    roots = [Path(value) for value in args.eval_root]
    if args.eval_glob:
        roots.extend(Path(value) for value in glob.glob(args.eval_glob))
    unique: dict[str, Path] = {}
    for root in roots:
        unique[str(root.resolve())] = root
    return [unique[key] for key in sorted(unique)]


def route_path_for_eval(root: Path, direction: str) -> Path:
    return root / direction / "route" / "route_cmrc2018_seed42.json"


def summary_path_for_eval(root: Path) -> Path:
    return root / "scd_icml_mistral_summary.json"


def enrich_row(row: dict[str, Any], eval_root: Path, direction: str, dataset: str) -> dict[str, Any]:
    summary_path = summary_path_for_eval(eval_root)
    route_path = route_path_for_eval(eval_root, direction)
    if not summary_path.is_file() or not route_path.is_file():
        return row
    summary = load_json(summary_path)
    if not isinstance(summary, list):
        return row
    methods: dict[str, dict[str, Any]] = {}
    for item in summary:
        if str(item.get("direction")) == direction and str(item.get("dataset")) == dataset:
            methods[str(item.get("method"))] = dict(item)
    full = methods.get("full_consumer", {})
    selected = methods.get(str(row.get("method") or "scd_full"), methods.get("scd_full", {}))
    full_f1 = float(full.get("f1") or 0.0)
    selected_f1 = float(selected.get("f1") or 0.0)
    full_em = float(full.get("em") or 0.0)
    selected_em = float(selected.get("em") or 0.0)
    full_nll = full.get("avg_nll")
    selected_nll = selected.get("avg_nll")
    row.update(
        {
            "eval_root": str(eval_root),
            "direction": direction,
            "dataset": dataset,
            "full_f1": full_f1,
            "f1_retention": selected_f1 / full_f1 if full_f1 > 0 else None,
            "full_em": full_em,
            "em": selected.get("em"),
            "em_retention": selected_em / full_em if full_em > 0 else None,
            "em_delta": selected_em - full_em,
            "full_avg_nll": full_nll,
            "avg_nll": selected_nll,
            "avg_nll_delta": None if full_nll is None or selected_nll is None else float(selected_nll) - float(full_nll),
            "full_ttft_ms": full.get("ttft_ms"),
            "ttft_ms": selected.get("ttft_ms"),
            "method": str(row.get("method") or "scd_full"),
        }
    )
    return row


def main() -> None:
    args = build_parser().parse_args()
    rows: list[dict[str, Any]] = []
    for root in collect_roots(args):
        route_path = route_path_for_eval(root, args.direction)
        summary_path = summary_path_for_eval(root)
        route_name = root.name
        try:
            row = baseline_row_from_eval(
                route_name=route_name,
                route_path=route_path,
                summary_path=summary_path,
                method=args.method,
                direction=args.direction,
                dataset=args.dataset,
            )
        except Exception as exc:
            row = {
                "route_name": route_name,
                "status": "error",
                "error": str(exc),
                "route_path": str(route_path),
                "summary_path": str(summary_path),
                "eval_root": str(root),
            }
        row = enrich_row(row, root, args.direction, args.dataset)
        rows.append(row)

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps({"rows": rows}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.out_jsonl:
        out_jsonl = Path(args.out_jsonl)
        out_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with out_jsonl.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"metrics_jsonl={out_jsonl}")
    print(f"metrics_json={out_json} rows={len(rows)}")


if __name__ == "__main__":
    main()
