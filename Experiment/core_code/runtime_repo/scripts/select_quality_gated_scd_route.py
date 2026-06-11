#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import shutil
from pathlib import Path
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Select the maximum-Reuse paper-SCD route that passes the primary "
            "quality gate. F1/EM use retention; PPL uses relative increase. "
            "Avg NLL is advisory by default."
        )
    )
    parser.add_argument("--eval-root", action="append", default=[], help="Candidate eval run root; may be repeated")
    parser.add_argument("--eval-glob", default="", help="Glob for candidate eval run roots")
    parser.add_argument("--direction", default="lite_to_7b")
    parser.add_argument("--dataset", default="cmrc2018")
    parser.add_argument("--metric", default="f1", choices=["f1", "em", "ppl"])
    parser.add_argument("--retention-threshold", type=float, default=0.95)
    parser.add_argument(
        "--max-metric-drop",
        type=float,
        default=None,
        help=(
            "Hard absolute-drop gate for F1/EM: full_metric - scd_metric must "
            "be at most this value. When set for F1/EM, this replaces the "
            "retention threshold as the primary gate."
        ),
    )
    parser.add_argument(
        "--max-ppl-relative-increase",
        type=float,
        default=None,
        help=(
            "Hard PPL gate: scd_ppl/full_ppl - 1 must be at most this value. "
            "When --metric ppl is used and this is omitted, defaults to 0.10."
        ),
    )
    parser.add_argument(
        "--max-avg-nll-delta",
        type=float,
        default=None,
        help=(
            "Advisory avg-NLL risk threshold by default. It becomes a hard gate "
            "only when --enforce-avg-nll-delta is set."
        ),
    )
    parser.add_argument(
        "--enforce-avg-nll-delta",
        action="store_true",
        help="Treat --max-avg-nll-delta as a hard rejection gate instead of advisory metadata.",
    )
    parser.add_argument("--max-scd-ttft-ms", type=float, default=None)
    parser.add_argument("--out-report", required=True)
    parser.add_argument("--out-route", default="", help="Optional path to copy the selected route JSON")
    return parser


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def route_path_for_eval(eval_root: Path, direction: str) -> Path:
    return eval_root / direction / "route" / "route_cmrc2018_seed42.json"


def summary_path_for_eval(eval_root: Path) -> Path:
    return eval_root / "scd_icml_mistral_summary.json"


def row_by_method(rows: list[dict[str, Any]], direction: str, dataset: str) -> dict[str, dict[str, Any]]:
    selected = {}
    for row in rows:
        if str(row.get("direction")) == direction and str(row.get("dataset")) == dataset:
            selected[str(row.get("method"))] = row
    return selected


def collect_eval_roots(args: argparse.Namespace) -> list[Path]:
    roots = [Path(value) for value in args.eval_root]
    if args.eval_glob:
        roots.extend(Path(value) for value in glob.glob(args.eval_glob))
    unique: dict[str, Path] = {}
    for root in roots:
        unique[str(root.resolve())] = root
    return [unique[key] for key in sorted(unique)]


def primary_gate(
    metric: str,
    full_metric: float,
    scd_metric: float,
    retention_threshold: float,
    max_ppl_relative_increase: float | None,
    max_metric_drop: float | None,
) -> dict[str, Any]:
    if full_metric <= 0:
        return {
            "primary_ratio": None,
            "metric_drop": None,
            "ppl_relative_increase": None,
            "passes_primary_metric": False,
            "primary_gate": "missing_or_zero_full_metric",
        }
    ratio = scd_metric / full_metric
    metric_drop = full_metric - scd_metric
    if metric == "ppl":
        gate = 0.10 if max_ppl_relative_increase is None else float(max_ppl_relative_increase)
        relative_increase = ratio - 1.0
        return {
            "primary_ratio": ratio,
            "metric_drop": metric_drop,
            "ppl_relative_increase": relative_increase,
            "passes_primary_metric": relative_increase <= gate,
            "primary_gate": f"ppl_relative_increase<= {gate}",
        }
    if max_metric_drop is not None:
        gate = float(max_metric_drop)
        return {
            "primary_ratio": ratio,
            "metric_drop": metric_drop,
            "ppl_relative_increase": None,
            "passes_primary_metric": metric_drop <= gate,
            "primary_gate": f"{metric}_drop<= {gate}",
        }
    return {
        "primary_ratio": ratio,
        "metric_drop": metric_drop,
        "ppl_relative_increase": None,
        "passes_primary_metric": ratio >= float(retention_threshold),
        "primary_gate": f"{metric}_retention>= {float(retention_threshold)}",
    }


def candidate_sort_key(row: dict[str, Any]) -> tuple[int, float, float, str]:
    patch_n = int(row.get("patch_n", 10**9))
    if row.get("metric") == "ppl":
        quality = float(row.get("ppl_relative_increase", float("inf")))
        return (patch_n, quality, 0.0, str(row.get("eval_root", "")))
    if row.get("metric_drop") is not None and row.get("primary_gate", "").find("_drop<=") >= 0:
        quality = float(row.get("metric_drop"))
        return (patch_n, quality, 0.0, str(row.get("eval_root", "")))
    retention = float(row.get("retention") or 0.0)
    avg_nll_delta = row.get("avg_nll_delta")
    nll_key = float(avg_nll_delta) if avg_nll_delta is not None else float("inf")
    return (patch_n, -retention, nll_key, str(row.get("eval_root", "")))


def main() -> None:
    args = build_parser().parse_args()
    eval_roots = collect_eval_roots(args)
    if not eval_roots:
        raise SystemExit("no eval roots supplied")

    candidates: list[dict[str, Any]] = []
    for eval_root in eval_roots:
        summary_path = summary_path_for_eval(eval_root)
        route_path = route_path_for_eval(eval_root, args.direction)
        if not summary_path.is_file() or not route_path.is_file():
            candidates.append(
                {
                    "eval_root": str(eval_root),
                    "status": "missing_artifact",
                    "summary": str(summary_path),
                    "route": str(route_path),
                }
            )
            continue
        rows = load_json(summary_path)
        route = load_json(route_path)
        methods = row_by_method(rows, args.direction, args.dataset)
        full = methods.get("full_consumer")
        scd = methods.get("scd_full")
        patch_layers = [int(layer) for layer in route.get("patch_layers", [])]
        reuse_layers = [int(layer) for layer in route.get("reuse_layers", [])]
        if not full or not scd:
            candidates.append(
                {
                    "eval_root": str(eval_root),
                    "status": "missing_method",
                    "patch_n": len(patch_layers),
                    "reuse_n": len(reuse_layers),
                    "route": str(route_path),
                }
            )
            continue
        full_metric = float(full.get(args.metric) or 0.0)
        scd_metric = float(scd.get(args.metric) or 0.0)
        gate = primary_gate(
            metric=args.metric,
            full_metric=full_metric,
            scd_metric=scd_metric,
            retention_threshold=float(args.retention_threshold),
            max_ppl_relative_increase=args.max_ppl_relative_increase,
            max_metric_drop=args.max_metric_drop,
        )
        retention = gate["primary_ratio"] if args.metric in {"f1", "em"} else None
        full_f1 = float(full.get("f1") or 0.0)
        scd_f1 = float(scd.get("f1") or 0.0)
        full_em = float(full.get("em") or 0.0)
        scd_em = float(scd.get("em") or 0.0)
        full_ppl = full.get("ppl")
        scd_ppl = scd.get("ppl")
        ppl_ratio = None
        if full_ppl is not None and scd_ppl is not None and float(full_ppl) > 0:
            ppl_ratio = float(scd_ppl) / float(full_ppl)
        f1_retention = scd_f1 / full_f1 if full_f1 > 0 else None
        em_retention = scd_em / full_em if full_em > 0 else None
        full_nll = full.get("avg_nll")
        scd_nll = scd.get("avg_nll")
        avg_nll_delta = None if full_nll is None or scd_nll is None else float(scd_nll) - float(full_nll)
        scd_ttft_ms = scd.get("ttft_ms")
        avg_nll_within_advisory = args.max_avg_nll_delta is None or (
            avg_nll_delta is not None and float(avg_nll_delta) <= float(args.max_avg_nll_delta)
        )
        passes_nll = (not args.enforce_avg_nll_delta) or avg_nll_within_advisory
        passes_ttft = args.max_scd_ttft_ms is None or (
            scd_ttft_ms is not None and float(scd_ttft_ms) <= float(args.max_scd_ttft_ms)
        )
        candidates.append(
            {
                "eval_root": str(eval_root),
                "status": "ok",
                "direction": args.direction,
                "dataset": args.dataset,
                "metric": args.metric,
                "patch_n": len(patch_layers),
                "reuse_n": len(reuse_layers),
                "patch_layers": patch_layers,
                "reuse_layers": reuse_layers,
                "route": str(route_path),
                "full_consumer": full_metric,
                "scd_full": scd_metric,
                "retention": retention,
                "primary_ratio": gate["primary_ratio"],
                "metric_drop": gate["metric_drop"],
                "primary_gate": gate["primary_gate"],
                "passes_primary_metric": gate["passes_primary_metric"],
                "full_f1": full_f1,
                "scd_f1": scd_f1,
                "f1_retention": f1_retention,
                "full_em": full_em,
                "scd_em": scd_em,
                "em_retention": em_retention,
                "full_ppl": full_ppl,
                "scd_ppl": scd_ppl,
                "ppl_ratio": ppl_ratio,
                "ppl_relative_increase": gate["ppl_relative_increase"],
                "avg_nll_delta": avg_nll_delta,
                "passes": bool(gate["passes_primary_metric"]) and passes_nll and passes_ttft,
                "passes_retention": bool(gate["passes_primary_metric"]) if args.metric in {"f1", "em"} else None,
                "passes_ppl_relative_increase": bool(gate["passes_primary_metric"]) if args.metric == "ppl" else None,
                "passes_avg_nll_delta": passes_nll,
                "avg_nll_delta_within_advisory": avg_nll_within_advisory,
                "avg_nll_delta_advisory_exceeded": not avg_nll_within_advisory,
                "passes_scd_ttft_ms": passes_ttft,
                "scd_ttft_ms": scd_ttft_ms,
                "full_ttft_ms": full.get("ttft_ms"),
            }
        )

    passing = [
        row
        for row in candidates
        if row.get("status") == "ok" and bool(row.get("passes"))
    ]
    passing.sort(key=candidate_sort_key)
    selected = passing[0] if passing else None
    report = {
        "selection_objective": "maximize_reuse_subject_to_quality_gate",
        "direction": args.direction,
        "dataset": args.dataset,
        "metric": args.metric,
        "retention_threshold": float(args.retention_threshold),
        "max_metric_drop": args.max_metric_drop,
        "max_ppl_relative_increase": (
            0.10
            if args.metric == "ppl" and args.max_ppl_relative_increase is None
            else args.max_ppl_relative_increase
        ),
        "max_avg_nll_delta": args.max_avg_nll_delta,
        "max_avg_nll_delta_is_hard_gate": bool(args.enforce_avg_nll_delta),
        "max_scd_ttft_ms": args.max_scd_ttft_ms,
        "selected": selected,
        "candidates": sorted(
            candidates,
            key=candidate_sort_key,
        ),
    }

    out_report = Path(args.out_report)
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if selected and args.out_route:
        out_route = Path(args.out_route)
        out_route.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(selected["route"]), out_route)
        report["copied_selected_route"] = str(out_route)
        out_report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"quality_gate_report={out_report}")
    if selected:
        print(
            "selected_route="
            f"{selected['route']} patch_n={selected['patch_n']} reuse_n={selected['reuse_n']} "
            f"primary_ratio={float(selected['primary_ratio']):.6f}"
        )
    else:
        raise SystemExit("no candidate passed the quality gate")


if __name__ == "__main__":
    main()
