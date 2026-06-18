from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT_FILES = [
    "LICENSE",
    "README.md",
    "REPRODUCING_SCD_ICML.md",
    "DATA_PREPARATION.md",
    "OPEN_SOURCE_RELEASE_MANIFEST.md",
    "docs/scd_vllm_plugin_mvp.md",
    "reports/scd_vllm_repo_audit.md",
    "reports/scd_vllm_status.md",
    "pyproject.toml",
]

CONFIG_FILES = [
    "configs/data_ser.yaml",
    "configs/eval_scd_icml_no_fallback_no_recompute.json",
    "configs/eval_scd_icml_full_scd_no_fallback_no_recompute.json",
    "configs/eval_baseline_reuse_only_fulldev3219_20260405.json",
    "configs/eval_patch_only_fulldev3219_20260405.json",
    "configs/models_mistrallite_to_mistral7_template.json",
    "configs/models_ser_mistrallite_to_mistral7_mql.json",
    "configs/models_ser_mistrallite_to_mistral7_mql_single_gpu.json",
    "configs/models_ser_mistral7_to_mistrallite_mql.json",
    "configs/scd_vllm_mvp.yaml",
    "configs/train_reuse_mistral_self_layerwise_20260404.json",
    "configs/mistral_self_patch_xnorm_nativekv_shared_residual_mlp_paper_k6_e1_h2048_r512_b2_20260404.json",
    "configs/mistral_self_patch_xnorm_nativekv_shared_residual_mlp_alllayers_e1_h2048_r512_b2_20260404.json",
    "configs/mistral_self_patch_xnorm_nativekv_residual_mlp_parallel_e1_h2048_b2_20260404.json",
    "configs/mistral_self_patch_xnorm_nativekv_linear_parallel_20260404.json",
    "configs/route_selection.yaml",
    "configs/route_reuse_patchonly_all32_20260404.json",
    "configs/routes/scd_icml_mistrallite_to_mistral7_reuse19_patch13_cmrc200.json",
    "configs/routes/scd_icml_mistrallite_to_mistral7_patch20_reuse12_cmrc200.json",
    "configs/routes/scd_icml_mistrallite_to_mistral7_patch26_reuse6_cmrc200.json",
]

SCRIPT_FILES = [
    "scripts/run_ser_scd_icml_mistral_repro.sh",
    "scripts/run_ser_scd_icml_patchonly_anchor.sh",
    "scripts/run_ser_scd_icml_parallel_patch_shards.sh",
    "scripts/run_ser_neighboring_baselines_cmrc.sh",
    "scripts/make_paper_scd_route.py",
    "scripts/make_paper_scd_candidate_routes.py",
    "scripts/merge_patch_ckpts.py",
    "scripts/run_ser_scd_icml_eval_existing_route.sh",
    "scripts/run_ser_scd_icml_conditional_route_screen.sh",
    "scripts/select_quality_gated_scd_route.py",
    "scripts/generate_conditional_reuse_routes.py",
    "scripts/collect_route_eval_metrics.py",
    "scripts/filter_strict_eval_by_qids.py",
    "scripts/build_scd_icml_result_tables.py",
    "scripts/build_scd_icml_analysis_figures.py",
    "scripts/extract_stack_viz_inputs.py",
    "scripts/build_stack_viz.py",
    "scripts/build_offline_cost_table.py",
    "scripts/build_paper_artifact_bundle.py",
    "scripts/finalize_scd_icml_full7405_bundle.py",
    "scripts/run_svdllm_style_kv_cmrc.py",
    "scripts/run_slicegpt_style_kv_cmrc.py",
    "scripts/run_tensorllm_style_kv_cmrc.py",
    "scripts/build_neighboring_baseline_table.py",
    "scripts/wait_postprocess_scd_icml_full7405_tables.sh",
    "scripts/wait_finalize_scd_icml_full7405_bundle.sh",
    "scripts/check_paper_artifacts.py",
    "scripts/report_pairwise_interactions.py",
    "scripts/estimate_layer_mapping.py",
    "scripts/profile_routes.py",
    "scripts/select_patch_reuse_route.py",
    "scripts/evaluate_selected_route.py",
    "scripts/validate_scd_route.py",
    "scripts/check_release_readiness.py",
    "scripts/check_release_smoke_status.py",
    "scripts/build_release_tree.py",
    "scripts/scd_vllm/run_00_repo_audit.sh",
    "scripts/scd_vllm/run_01_env_probe.sh",
    "scripts/scd_vllm/run_02_hf_export.sh",
    "scripts/scd_vllm/run_03_vllm_baseline.sh",
    "scripts/scd_vllm/run_04_truekv_oracle.sh",
    "scripts/scd_vllm/run_05_reconstructed_export.sh",
    "scripts/scd_vllm/run_05_scd_injection.sh",
    "scripts/scd_vllm/run_06_report.sh",
]

EXAMPLE_FILES = [
    "examples/scd_vllm_official_launch.py",
]

COPY_DIRS = [
    "src/scd_repro",
    "src/scd_vllm",
    "src/scd_vllm_plugin",
    "tests",
]

IGNORE_NAMES = {
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "search_blockwise_hybrid.py",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a non-destructive SCD-ICML runtime release preview tree")
    parser.add_argument("--root", default="", help="runtime_repo root; defaults to parent of this script")
    parser.add_argument("--out", default="", help="output directory for release preview")
    parser.add_argument("--force", action="store_true", help="replace the output directory if it exists")
    parser.add_argument("--skip-check", action="store_true", help="do not run check_release_readiness.py on the output")
    return parser.parse_args()


def ignore_generated(_dir: str, names: list[str]) -> set[str]:
    return {name for name in names if name in IGNORE_NAMES or name.endswith(".pyc") or name.endswith(".pyo")}


def copy_file(root: Path, out: Path, rel: str) -> None:
    src = root / rel
    if not src.is_file():
        raise FileNotFoundError(f"required file missing: {rel}")
    dst = out / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def copy_dir(root: Path, out: Path, rel: str) -> None:
    src = root / rel
    if not src.is_dir():
        raise FileNotFoundError(f"required directory missing: {rel}")
    dst = out / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst, ignore=ignore_generated, dirs_exist_ok=True)


def collect_existing(root: Path, pattern: str) -> set[str]:
    return {str(path.relative_to(root)).replace("\\", "/") for path in root.glob(pattern) if path.is_file()}


def write_report(root: Path, out: Path, included: list[str]) -> None:
    included_set = set(included)
    all_root_files = collect_existing(root, "*")
    all_configs = collect_existing(root, "configs/*")
    all_scripts = collect_existing(root, "scripts/*")
    excluded_root = sorted(path for path in all_root_files if path not in included_set)
    excluded_configs = sorted(path for path in all_configs if path not in included_set)
    excluded_scripts = sorted(path for path in all_scripts if path not in included_set)

    lines = [
        "# Release Preview Build Report",
        "",
        f"- Source root: `{root}`",
        f"- Output root: `{out}`",
        f"- Included files/directories: {len(included)}",
        "",
        "## Included",
        "",
    ]
    for rel in sorted(included):
        lines.append(f"- `{rel}`")

    lines.extend(
        [
            "",
            "## Not Included From Runtime Root",
            "",
            "These files remain in the working tree for traceability but are not part of the default SCD-ICML Mistral release preview.",
            "",
            "### Root Files",
            "",
        ]
    )
    lines.extend(f"- `{rel}`" for rel in excluded_root)
    lines.extend(["", "### Configs", ""])
    lines.extend(f"- `{rel}`" for rel in excluded_configs)
    lines.extend(["", "### Scripts", ""])
    lines.extend(f"- `{rel}`" for rel in excluded_scripts)
    lines.append("")
    (out / "RELEASE_BUILD_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def run_check(out: Path) -> None:
    cmd = [sys.executable, str(out / "scripts" / "check_release_readiness.py"), "--root", str(out)]
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    subprocess.run(cmd, check=True, env=env)


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve() if args.root else Path(__file__).resolve().parents[1]
    out = Path(args.out).resolve() if args.out else root / "release_preview" / "scd_icml_runtime_release"

    if not root.exists():
        raise SystemExit(f"runtime root not found: {root}")
    if out.exists():
        if not args.force:
            raise SystemExit(f"output exists; pass --force to replace: {out}")
        shutil.rmtree(out)
    out.mkdir(parents=True)

    included: list[str] = []
    for rel in ROOT_FILES + CONFIG_FILES + SCRIPT_FILES + EXAMPLE_FILES:
        copy_file(root, out, rel)
        included.append(rel)
    for rel in COPY_DIRS:
        copy_dir(root, out, rel)
        included.append(rel + "/")

    write_report(root=root, out=out, included=included)
    if not args.skip_check:
        run_check(out)

    print(f"release_preview={out}")
    print(f"report={out / 'RELEASE_BUILD_REPORT.md'}")


if __name__ == "__main__":
    main()
