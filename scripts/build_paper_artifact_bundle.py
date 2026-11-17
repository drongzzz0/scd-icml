#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


RESULT_TABLE_FILES = [
    "patch_anchor_table.csv",
    "patch_anchor_table.md",
    "patch_anchor_table.tex",
    "mixed_route_boundary_table.csv",
    "mixed_route_boundary_table.md",
    "mixed_route_boundary_table.tex",
    "ttft_breakdown_table.csv",
    "ttft_breakdown_table.md",
    "ttft_breakdown_table.tex",
    "latency_quality_pareto.csv",
    "latency_quality_pareto.md",
    "latency_quality_pareto_cmrc2018.svg",
    "latency_quality_pareto_hotpotqa_compressed.svg",
    "latency_quality_pareto_wikitext2.svg",
    "bandwidth_sensitivity.csv",
    "bandwidth_sensitivity.md",
    "bandwidth_sensitivity.svg",
    "route_curve_cmrc2018.csv",
    "route_curve_cmrc2018.md",
    "route_curve_cmrc2018.svg",
    "route_curve_hotpotqa_compressed.csv",
    "route_curve_hotpotqa_compressed.md",
    "route_curve_hotpotqa_compressed.svg",
    "route_curve_wikitext2.csv",
    "route_curve_wikitext2.md",
    "route_curve_wikitext2.svg",
    "scd_icml_result_tables_manifest.json",
]

PROFILE_FILES = {
    "layer_error_profile.svg": "figures/layerwise_xnorm_replacement.svg",
    "layer_error_profile.pdf": "figures/layerwise_xnorm_replacement.pdf",
    "layer_error_profile.png": "figures/layerwise_xnorm_replacement.png",
    "layer_gain_profile.svg": "figures/layer_selection_gain_profile.svg",
    "layer_gain_profile.pdf": "figures/layer_selection_gain_profile.pdf",
    "layer_gain_profile.png": "figures/layer_selection_gain_profile.png",
    "patch_gain_curve.svg": "figures/patch_budget_curve.svg",
    "patch_gain_curve.pdf": "figures/patch_budget_curve.pdf",
    "patch_gain_curve.png": "figures/patch_budget_curve.png",
    "layer_profile.csv": "tables/layer_profile.csv",
    "patch_gain_curve.csv": "tables/patch_gain_curve.csv",
    "patch_gain_curve.md": "tables/patch_gain_curve.md",
    "analysis_figures_manifest.json": "manifests/analysis_figures_manifest.json",
}

STACK_CANDIDATE_GROUPS = {
    "figures/stack_viz.svg": ["stack_viz_current_tsne.svg", "stack_viz.svg"],
    "figures/stack_viz.pdf": ["stack_viz_current_tsne.pdf", "stack_viz.pdf"],
    "figures/stack_viz.png": ["stack_viz_current_tsne.png", "stack_viz.png"],
}

OFFLINE_COST_FILES = {
    "offline_cost_table.csv": "tables/offline_cost_table.csv",
    "offline_cost_table.md": "tables/offline_cost_table.md",
    "offline_cost_summary.json": "manifests/offline_cost_summary.json",
    "offline_cost_wall_time.svg": "figures/offline_cost_wall_time.svg",
}

NEIGHBORING_BASELINE_FILES = {
    "neighboring_baselines.csv": "tables/neighboring_baselines.csv",
    "neighboring_baselines.md": "tables/neighboring_baselines.md",
    "neighboring_baselines.tex": "tables/neighboring_baselines.tex",
    "neighboring_baselines_manifest.json": "manifests/neighboring_baselines_manifest.json",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Assemble paper-ready SCD-ICML tables and figures from generated "
            "result, profile, stack/t-SNE, and offline-cost artifact directories."
        )
    )
    parser.add_argument("--result-tables-dir", required=True)
    parser.add_argument("--profile-figures-dir", default="")
    parser.add_argument("--stack-viz-dir", default="")
    parser.add_argument("--offline-cost-dir", default="")
    parser.add_argument("--neighboring-baselines-dir", default="")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Write a manifest even when optional or required inputs are missing.",
    )
    return parser


def copy_one(src: Path, dst: Path, copied: list[dict[str, str]]) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    copied.append({"source": str(src), "target": str(dst)})


def maybe_copy_file(
    base: Path,
    rel: str,
    out_dir: Path,
    target_rel: str,
    copied: list[dict[str, str]],
    missing: list[dict[str, str]],
    required: bool,
) -> None:
    src = base / rel
    if src.is_file():
        copy_one(src, out_dir / target_rel, copied)
    else:
        missing.append({"source": str(src), "target": target_rel, "required": str(required).lower()})


def copy_mapping(
    base: Path | None,
    mapping: dict[str, str],
    out_dir: Path,
    copied: list[dict[str, str]],
    missing: list[dict[str, str]],
    required: bool,
) -> None:
    if base is None:
        for rel, target in mapping.items():
            missing.append({"source": rel, "target": target, "required": str(required).lower()})
        return
    for rel, target in mapping.items():
        maybe_copy_file(base, rel, out_dir, target, copied, missing, required=required)


def copy_candidate_groups(
    base: Path | None,
    groups: dict[str, list[str]],
    out_dir: Path,
    copied: list[dict[str, str]],
    missing: list[dict[str, str]],
) -> None:
    for target, candidates in groups.items():
        if base is None:
            missing.append({"source": "|".join(candidates), "target": target, "required": "false"})
            continue
        for rel in candidates:
            src = base / rel
            if src.is_file():
                copy_one(src, out_dir / target, copied)
                break
        else:
            missing.append(
                {
                    "source": "|".join(str(base / rel) for rel in candidates),
                    "target": target,
                    "required": "false",
                }
            )


def write_readme(path: Path, manifest: dict[str, Any]) -> None:
    lines = [
        "# SCD-ICML Paper Artifact Bundle",
        "",
        "This bundle is assembled from generated artifacts. It does not compute or invent metrics.",
        "",
        f"- Status: `{manifest['status']}`",
        f"- Copied files: `{len(manifest['copied'])}`",
        f"- Missing files: `{len(manifest['missing'])}`",
        "",
        "Use `paper_artifact_manifest.json` for source paths and missing-item checks.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out_dir = Path(args.out_dir)
    result_dir = Path(args.result_tables_dir)
    profile_dir = Path(args.profile_figures_dir) if args.profile_figures_dir else None
    stack_dir = Path(args.stack_viz_dir) if args.stack_viz_dir else None
    cost_dir = Path(args.offline_cost_dir) if args.offline_cost_dir else None
    neighboring_dir = Path(args.neighboring_baselines_dir) if args.neighboring_baselines_dir else None

    out_dir.mkdir(parents=True, exist_ok=True)
    copied: list[dict[str, str]] = []
    missing: list[dict[str, str]] = []

    for name in RESULT_TABLE_FILES:
        target = f"tables/{name}" if not name.endswith(".svg") else f"figures/{name}"
        if name.endswith("_manifest.json"):
            target = f"manifests/{name}"
        maybe_copy_file(result_dir, name, out_dir, target, copied, missing, required=True)

    copy_mapping(profile_dir, PROFILE_FILES, out_dir, copied, missing, required=False)
    copy_candidate_groups(stack_dir, STACK_CANDIDATE_GROUPS, out_dir, copied, missing)
    copy_mapping(cost_dir, OFFLINE_COST_FILES, out_dir, copied, missing, required=False)
    copy_mapping(neighboring_dir, NEIGHBORING_BASELINE_FILES, out_dir, copied, missing, required=False)

    required_missing = [row for row in missing if row.get("required") == "true"]
    status = "ok" if not required_missing else "missing_required"
    manifest = {
        "status": status,
        "inputs": {
            "result_tables_dir": str(result_dir),
            "profile_figures_dir": str(profile_dir) if profile_dir else "",
            "stack_viz_dir": str(stack_dir) if stack_dir else "",
            "offline_cost_dir": str(cost_dir) if cost_dir else "",
            "neighboring_baselines_dir": str(neighboring_dir) if neighboring_dir else "",
        },
        "copied": copied,
        "missing": missing,
    }
    (out_dir / "paper_artifact_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    write_readme(out_dir / "README.md", manifest)
    print(f"paper_artifact_bundle={out_dir}")
    if required_missing and not args.allow_missing:
        print(f"missing_required={len(required_missing)}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
