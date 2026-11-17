#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


DEFAULT_FORBID_IMAGES = ["patch_error1.pdf", "layer_selection.pdf", "stack_viz.pdf"]
DEFAULT_REQUIRED_IMAGES = [
    "generated_figures/layerwise_xnorm_replacement.pdf",
    "generated_figures/layer_selection_gain_profile.pdf",
    "generated_figures/patch_budget_curve.pdf",
    "generated_figures/stack_viz.pdf",
]
DEFAULT_REQUIRED_BUNDLE_FILES = [
    "figures/layerwise_xnorm_replacement.pdf",
    "figures/layer_selection_gain_profile.pdf",
    "figures/patch_budget_curve.pdf",
    "figures/stack_viz.pdf",
    "tables/patch_gain_curve.md",
    "tables/patch_anchor_table.md",
    "tables/patch_anchor_table.tex",
    "tables/patch_anchor_table.csv",
    "tables/neighboring_baselines.md",
    "tables/neighboring_baselines.tex",
    "tables/neighboring_baselines.csv",
    "manifests/neighboring_baselines_manifest.json",
]
DEFAULT_REQUIRED_TEXT = [
    r"\label{tab:neighboring_baselines}",
    r"\label{tab:patchonly_anchors}",
    "same-protocol CMRC2018 rerun",
    "HotpotQA-compressed reports the full7405 selected route",
    r"\label{tab:current_cmrc_selected_route}",
    r"\label{tab:current_reuse_boundary}",
    r"\label{tab:current_non_cmrc_diagnostics}",
    r"\label{tab:offline_cost}",
]
DEFAULT_FORBIDDEN_TEXT = [
    "not part of the current verified release evidence",
    "do not use them as a claim-facing comparison",
    "full7405 remains pending",
    "full7405 pending",
    "full7405 is not yet closed",
    "full7405 remains the final open",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Finalize the SCD-ICML paper artifact bundle after HotpotQA-compressed "
            "full7405 result tables have been generated."
        )
    )
    parser.add_argument("--result-tables-dir", required=True)
    parser.add_argument("--profile-figures-dir", required=True)
    parser.add_argument("--stack-viz-dir", required=True)
    parser.add_argument("--offline-cost-dir", required=True)
    parser.add_argument("--neighboring-baselines-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--paper-tex", default="", help="Optional paper .tex to validate after bundling")
    parser.add_argument("--paper-base-dir", default="", help="Base directory for paper figures; defaults to paper dir")
    parser.add_argument(
        "--selection-report",
        default="",
        help=(
            "Optional HotpotQA full7405 selector JSON. Defaults to "
            "<result-tables-dir>/../hotpotqa_full7405_f1_drop003_selection.json when present."
        ),
    )
    parser.add_argument("--allow-missing", action="store_true", help="Pass --allow-missing to build_paper_artifact_bundle.py")
    parser.add_argument("--skip-paper-check", action="store_true", help="Only build the bundle; do not run check_paper_artifacts.py")
    parser.add_argument(
        "--require-result-source-substring",
        action="append",
        default=["full7405"],
        help=(
            "Substring that must appear in at least one mixed_route_boundary_table.csv "
            "source row before finalization; defaults to full7405 and may repeat."
        ),
    )
    parser.add_argument("--out-report", default="", help="Optional finalizer JSON report path; defaults to <out-dir>/finalize_report.json")
    return parser


def script_path(name: str) -> Path:
    return Path(__file__).resolve().parent / name


def run_command(cmd: list[str], cwd: Path) -> dict[str, Any]:
    completed = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True, check=False)
    row = {
        "command": cmd,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }
    if completed.returncode != 0:
        raise RuntimeError(json.dumps(row, indent=2))
    return row


def load_manifest(out_dir: Path) -> dict[str, Any]:
    manifest_path = out_dir / "paper_artifact_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing bundle manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    if manifest.get("status") != "ok":
        raise RuntimeError(f"bundle manifest status is not ok: {manifest.get('status')!r}")
    return manifest


def validate_result_tables(result_tables_dir: Path, required_source_substrings: list[str]) -> dict[str, Any]:
    table_path = result_tables_dir / "mixed_route_boundary_table.csv"
    if not table_path.is_file():
        raise FileNotFoundError(f"missing mixed route table: {table_path}")
    with table_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"mixed route table has no rows: {table_path}")
    sources = [str(row.get("source", "")) for row in rows]
    missing = [
        substring
        for substring in required_source_substrings
        if substring and not any(substring in source for source in sources)
    ]
    if missing:
        raise RuntimeError(
            "mixed route table does not contain required source substring(s): "
            + ", ".join(missing)
            + f" in {table_path}"
        )
    return {
        "table": str(table_path),
        "rows": len(rows),
        "required_source_substrings": required_source_substrings,
    }


def load_selected_route_requirement(args: argparse.Namespace) -> dict[str, Any]:
    explicit = Path(args.selection_report) if args.selection_report else None
    inferred = Path(args.result_tables_dir).resolve().parent / "hotpotqa_full7405_f1_drop003_selection.json"
    report_path = explicit if explicit is not None else inferred
    if not report_path.is_file():
        return {
            "selection_report": str(report_path),
            "status": "not_found",
            "required_text": "",
        }
    report = json.loads(report_path.read_text(encoding="utf-8-sig"))
    selected = report.get("selected") or {}
    reuse_n = selected.get("reuse_n")
    patch_n = selected.get("patch_n")
    if reuse_n is None or patch_n is None:
        raise RuntimeError(f"selection report has no selected reuse_n/patch_n: {report_path}")
    required_text = f"Reuse{int(reuse_n):02d}/Patch{int(patch_n):02d}"
    return {
        "selection_report": str(report_path),
        "status": "ok",
        "required_text": required_text,
        "reuse_n": int(reuse_n),
        "patch_n": int(patch_n),
    }


def build_bundle(args: argparse.Namespace, cwd: Path) -> dict[str, Any]:
    cmd = [
        sys.executable,
        str(script_path("build_paper_artifact_bundle.py")),
        "--result-tables-dir",
        args.result_tables_dir,
        "--profile-figures-dir",
        args.profile_figures_dir,
        "--stack-viz-dir",
        args.stack_viz_dir,
        "--offline-cost-dir",
        args.offline_cost_dir,
        "--neighboring-baselines-dir",
        args.neighboring_baselines_dir,
        "--out-dir",
        args.out_dir,
    ]
    if args.allow_missing:
        cmd.append("--allow-missing")
    return run_command(cmd, cwd)


def check_paper(
    args: argparse.Namespace,
    cwd: Path,
    out_dir: Path,
    selected_route_requirement: dict[str, Any],
) -> dict[str, Any] | None:
    if args.skip_paper_check or not args.paper_tex:
        return None
    report_path = out_dir / "paper_artifact_check_report.json"
    cmd = [
        sys.executable,
        str(script_path("check_paper_artifacts.py")),
        "--paper-tex",
        args.paper_tex,
        "--bundle-dir",
        args.out_dir,
        "--out-json",
        str(report_path),
    ]
    if args.paper_base_dir:
        cmd.extend(["--base-dir", args.paper_base_dir])
    for value in DEFAULT_FORBID_IMAGES:
        cmd.extend(["--forbid-image", value])
    for value in DEFAULT_REQUIRED_IMAGES:
        cmd.extend(["--require-image", value])
    for value in DEFAULT_REQUIRED_BUNDLE_FILES:
        cmd.extend(["--require-bundle-file", value])
    for value in DEFAULT_REQUIRED_TEXT:
        cmd.extend(["--require-text", value])
    if selected_route_requirement.get("required_text"):
        cmd.extend(["--require-text", str(selected_route_requirement["required_text"])])
    for value in DEFAULT_FORBIDDEN_TEXT:
        cmd.extend(["--forbid-text", value])
    return run_command(cmd, cwd)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cwd = Path(__file__).resolve().parents[1]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    commands: list[dict[str, Any]] = []
    result_check = validate_result_tables(Path(args.result_tables_dir), args.require_result_source_substring)
    selected_route_requirement = load_selected_route_requirement(args)
    commands.append(build_bundle(args, cwd))
    manifest = load_manifest(out_dir)
    paper_check = check_paper(args, cwd, out_dir, selected_route_requirement)
    if paper_check is not None:
        commands.append(paper_check)

    report = {
        "status": "ok",
        "bundle_dir": str(out_dir),
        "bundle_status": manifest.get("status"),
        "copied": len(manifest.get("copied", [])),
        "missing": len(manifest.get("missing", [])),
        "result_table_check": result_check,
        "selected_route_paper_requirement": selected_route_requirement,
        "paper_check": not args.skip_paper_check and bool(args.paper_tex),
        "commands": commands,
    }
    report_path = Path(args.out_report) if args.out_report else out_dir / "finalize_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"final_bundle={out_dir}")
    print(f"finalize_report={report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
