from __future__ import annotations

import argparse
import ast
import importlib
import json
import py_compile
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path


REQUIRED_FILES = [
    "LICENSE",
    "README.md",
    "REPRODUCING_SCD_ICML.md",
    "DATA_PREPARATION.md",
    "OPEN_SOURCE_RELEASE_MANIFEST.md",
    "docs/scd_vllm_plugin_mvp.md",
    "reports/scd_vllm_repo_audit.md",
    "reports/scd_vllm_status.md",
    "pyproject.toml",
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
    "examples/scd_vllm_official_launch.py",
    "tests/test_cmrc_split.py",
    "tests/test_layer_mapping.py",
    "tests/test_metrics.py",
    "tests/test_quality_gated_route_script.py",
    "tests/test_analysis_figures_script.py",
    "tests/test_stack_viz_input_extractor.py",
    "tests/test_stack_viz_script.py",
    "tests/test_offline_cost_table_script.py",
    "tests/test_paper_artifact_bundle_script.py",
    "tests/test_finalize_full7405_bundle_script.py",
    "tests/test_paper_artifact_check_script.py",
    "tests/test_filter_strict_eval_by_qids_script.py",
    "tests/test_result_tables_script.py",
    "tests/test_neighboring_baseline_table_script.py",
    "tests/test_route_serialization.py",
    "tests/test_selector.py",
    "tests/test_text_benchmarks.py",
    "tests/test_scd_vllm_bundle.py",
    "tests/test_scd_vllm_batch_benchmark.py",
    "tests/test_scd_vllm_doctor.py",
    "tests/test_scd_vllm_hf_export.py",
    "tests/test_scd_vllm_injection.py",
    "tests/test_scd_vllm_kv_spec.py",
    "tests/test_scd_vllm_metrics.py",
    "tests/test_scd_vllm_plugin.py",
    "tests/test_scd_vllm_reconstructed_export.py",
    "tests/test_scd_vllm_report.py",
    "tests/test_scd_vllm_truekv_oracle.py",
    "tests/test_scd_vllm_vllm_baseline.py",
    "tests/test_scd_vllm_official_launch_example.py",
    "tests/test_release_tree_script.py",
    "src/scd_repro/text_benchmarks.py",
    "src/scd_repro/adapters.py",
    "src/scd_repro/direct_kv_residual_patch.py",
    "src/scd_repro/strict_runtime.py",
    "src/scd_repro/cli/capture_pair.py",
    "src/scd_repro/cli/eval_text_strict.py",
    "src/scd_repro/cli/summarize_mistral_repro.py",
    "src/scd_vllm/__init__.py",
    "src/scd_vllm/batch_benchmark.py",
    "src/scd_vllm/bundle.py",
    "src/scd_vllm/env_probe.py",
    "src/scd_vllm/hf_export.py",
    "src/scd_vllm/injection.py",
    "src/scd_vllm/kv_spec.py",
    "src/scd_vllm/metrics.py",
    "src/scd_vllm/preloaded_connector.py",
    "src/scd_vllm/repo_audit.py",
    "src/scd_vllm/report.py",
    "src/scd_vllm/reconstructed_export.py",
    "src/scd_vllm/truekv_oracle.py",
    "src/scd_vllm/vllm_baseline.py",
    "src/scd_vllm_plugin/__init__.py",
    "src/scd_vllm_plugin/cli.py",
    "src/scd_vllm_plugin/doctor.py",
    "src/scd_vllm_plugin/engine.py",
]

JSON_FILES = [
    "configs/data_ser.yaml",
    "configs/eval_scd_icml_no_fallback_no_recompute.json",
    "configs/eval_scd_icml_full_scd_no_fallback_no_recompute.json",
    "configs/eval_baseline_reuse_only_fulldev3219_20260405.json",
    "configs/eval_patch_only_fulldev3219_20260405.json",
    "configs/models_mistrallite_to_mistral7_template.json",
    "configs/models_ser_mistrallite_to_mistral7_mql.json",
    "configs/models_ser_mistrallite_to_mistral7_mql_single_gpu.json",
    "configs/models_ser_mistral7_to_mistrallite_mql.json",
    "configs/train_reuse_mistral_self_layerwise_20260404.json",
    "configs/mistral_self_patch_xnorm_nativekv_shared_residual_mlp_paper_k6_e1_h2048_r512_b2_20260404.json",
    "configs/mistral_self_patch_xnorm_nativekv_shared_residual_mlp_alllayers_e1_h2048_r512_b2_20260404.json",
    "configs/mistral_self_patch_xnorm_nativekv_residual_mlp_parallel_e1_h2048_b2_20260404.json",
    "configs/mistral_self_patch_xnorm_nativekv_linear_parallel_20260404.json",
    "configs/route_reuse_patchonly_all32_20260404.json",
    "configs/routes/scd_icml_mistrallite_to_mistral7_reuse19_patch13_cmrc200.json",
    "configs/routes/scd_icml_mistrallite_to_mistral7_patch20_reuse12_cmrc200.json",
    "configs/routes/scd_icml_mistrallite_to_mistral7_patch26_reuse6_cmrc200.json",
]

PY_COMPILE_FILES = [
    "src/scd_repro/text_benchmarks.py",
    "src/scd_repro/adapters.py",
    "src/scd_repro/direct_kv_residual_patch.py",
    "src/scd_repro/strict_runtime.py",
    "src/scd_repro/cli/capture_pair.py",
    "src/scd_repro/cli/eval_text_strict.py",
    "src/scd_repro/cli/summarize_mistral_repro.py",
    "src/scd_vllm/__init__.py",
    "src/scd_vllm/batch_benchmark.py",
    "src/scd_vllm/bundle.py",
    "src/scd_vllm/env_probe.py",
    "src/scd_vllm/hf_export.py",
    "src/scd_vllm/injection.py",
    "src/scd_vllm/kv_spec.py",
    "src/scd_vllm/metrics.py",
    "src/scd_vllm/preloaded_connector.py",
    "src/scd_vllm/repo_audit.py",
    "src/scd_vllm/report.py",
    "src/scd_vllm/reconstructed_export.py",
    "src/scd_vllm/truekv_oracle.py",
    "src/scd_vllm/vllm_baseline.py",
    "src/scd_vllm_plugin/__init__.py",
    "src/scd_vllm_plugin/cli.py",
    "src/scd_vllm_plugin/doctor.py",
    "src/scd_vllm_plugin/engine.py",
    "scripts/make_paper_scd_route.py",
    "scripts/make_paper_scd_candidate_routes.py",
    "scripts/merge_patch_ckpts.py",
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
    "scripts/check_paper_artifacts.py",
    "scripts/report_pairwise_interactions.py",
    "scripts/select_quality_gated_scd_route.py",
    "scripts/estimate_layer_mapping.py",
    "scripts/profile_routes.py",
    "scripts/select_patch_reuse_route.py",
    "scripts/evaluate_selected_route.py",
    "scripts/validate_scd_route.py",
    "scripts/check_release_readiness.py",
    "scripts/check_release_smoke_status.py",
    "scripts/build_release_tree.py",
    "examples/scd_vllm_official_launch.py",
    "tests/test_cmrc_split.py",
    "tests/test_layer_mapping.py",
    "tests/test_metrics.py",
    "tests/test_quality_gated_route_script.py",
    "tests/test_analysis_figures_script.py",
    "tests/test_stack_viz_input_extractor.py",
    "tests/test_stack_viz_script.py",
    "tests/test_offline_cost_table_script.py",
    "tests/test_paper_artifact_bundle_script.py",
    "tests/test_finalize_full7405_bundle_script.py",
    "tests/test_paper_artifact_check_script.py",
    "tests/test_filter_strict_eval_by_qids_script.py",
    "tests/test_result_tables_script.py",
    "tests/test_neighboring_baseline_table_script.py",
    "tests/test_route_serialization.py",
    "tests/test_selector.py",
    "tests/test_text_benchmarks.py",
    "tests/test_scd_vllm_bundle.py",
    "tests/test_scd_vllm_batch_benchmark.py",
    "tests/test_scd_vllm_doctor.py",
    "tests/test_scd_vllm_hf_export.py",
    "tests/test_scd_vllm_injection.py",
    "tests/test_scd_vllm_kv_spec.py",
    "tests/test_scd_vllm_metrics.py",
    "tests/test_scd_vllm_plugin.py",
    "tests/test_scd_vllm_reconstructed_export.py",
    "tests/test_scd_vllm_report.py",
    "tests/test_scd_vllm_truekv_oracle.py",
    "tests/test_scd_vllm_vllm_baseline.py",
    "tests/test_scd_vllm_official_launch_example.py",
    "tests/test_release_tree_script.py",
]

EXPECTED_METHODS = {
    "full_consumer",
    "raw_kv",
    "quantized_4bit",
    "reuse_only",
    "scd_full",
}

EXPECTED_DATASETS = {"cmrc2018", "hotpotqa", "wikitext2"}

SHELL_FILES = [
    "scripts/run_ser_scd_icml_mistral_repro.sh",
    "scripts/run_ser_scd_icml_patchonly_anchor.sh",
    "scripts/run_ser_scd_icml_parallel_patch_shards.sh",
    "scripts/run_ser_neighboring_baselines_cmrc.sh",
    "scripts/run_ser_scd_icml_eval_existing_route.sh",
    "scripts/run_ser_scd_icml_conditional_route_screen.sh",
    "scripts/wait_postprocess_scd_icml_full7405_tables.sh",
    "scripts/wait_finalize_scd_icml_full7405_bundle.sh",
    "scripts/scd_vllm/run_00_repo_audit.sh",
    "scripts/scd_vllm/run_01_env_probe.sh",
    "scripts/scd_vllm/run_02_hf_export.sh",
    "scripts/scd_vllm/run_03_vllm_baseline.sh",
    "scripts/scd_vllm/run_04_truekv_oracle.sh",
    "scripts/scd_vllm/run_05_reconstructed_export.sh",
    "scripts/scd_vllm/run_05_scd_injection.sh",
    "scripts/scd_vllm/run_06_report.sh",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check release readiness for the SCD-ICML Mistral reproduction path")
    parser.add_argument("--root", default="", help="runtime_repo root; defaults to parent of this script")
    parser.add_argument("--run-root", default="", help="optional verified run root to validate aggregate summary rows")
    return parser.parse_args()


def fail(errors: list[str], message: str) -> None:
    errors.append(message)


def check_required_files(root: Path, errors: list[str]) -> None:
    for rel in REQUIRED_FILES:
        path = root / rel
        if not path.is_file():
            fail(errors, f"missing required file: {rel}")


def check_json(root: Path, errors: list[str]) -> None:
    for rel in JSON_FILES:
        path = root / rel
        if not path.is_file():
            continue
        try:
            json.loads(path.read_text(encoding="utf-8-sig"))
        except Exception as exc:
            fail(errors, f"invalid JSON {rel}: {exc}")


def check_py_compile(root: Path, errors: list[str]) -> None:
    with tempfile.TemporaryDirectory(prefix="scd_release_pycompile_") as tmp_dir:
        tmp = Path(tmp_dir)
        for rel in PY_COMPILE_FILES:
            path = root / rel
            if not path.is_file():
                continue
            cfile = tmp / (rel.replace("/", "__").replace("\\", "__") + ".pyc")
            try:
                py_compile.compile(str(path), cfile=str(cfile), doraise=True)
            except Exception as exc:
                fail(errors, f"py_compile failed for {rel}: {exc}")


def check_shell_scripts(root: Path, errors: list[str]) -> None:
    bash = shutil.which("bash")
    for rel in SHELL_FILES:
        path = root / rel
        if not path.is_file():
            continue
        data = path.read_bytes()
        if b"\r\n" in data:
            fail(errors, f"shell script must use LF line endings: {rel}")
        if not data.startswith(b"#!/usr/bin/env bash\n"):
            fail(errors, f"shell script missing bash shebang: {rel}")
        text = data.decode("utf-8", errors="replace")
        if "set -euo pipefail" not in text:
            fail(errors, f"shell script missing strict mode: {rel}")
        if bash:
            try:
                subprocess.run(
                    [bash, "-n", str(path)],
                    check=True,
                    capture_output=True,
                    text=True,
                    cwd=str(root),
                )
            except subprocess.CalledProcessError as exc:
                fail(errors, f"bash -n failed for {rel}: {exc.stderr or exc.stdout}")


def check_python_source_hygiene(root: Path, errors: list[str]) -> None:
    for base_rel in ("src", "scripts", "examples"):
        base = root / base_rel
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*.py")):
            if path.read_bytes().startswith(b"\xef\xbb\xbf"):
                fail(errors, f"Python source has UTF-8 BOM: {path.relative_to(root).as_posix()}")


def normalize_dependency_name(spec: str) -> str:
    name = spec.split(";", 1)[0].strip()
    name = re.split(r"\s*(?:\[|==|>=|<=|~=|!=|>|<)", name, maxsplit=1)[0]
    return name.lower().replace("-", "_")


def check_declared_dependencies(root: Path, errors: list[str]) -> None:
    pyproject = root / "pyproject.toml"
    src_root = root / "src"
    if not pyproject.is_file() or not src_root.is_dir():
        return
    try:
        cfg = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    except Exception as exc:
        fail(errors, f"cannot read pyproject for dependency check: {exc}")
        return
    declared = {
        normalize_dependency_name(dep)
        for dep in cfg.get("project", {}).get("dependencies", [])
    }
    stdlib = set(getattr(sys, "stdlib_module_names", ()))
    stdlib.update({"__future__"})
    local_roots = {
        path.name
        for path in src_root.iterdir()
        if path.is_dir() and (path / "__init__.py").is_file()
    }
    direct_imports: dict[str, set[str]] = {}
    for path in sorted(src_root.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except Exception as exc:
            fail(errors, f"cannot parse Python source for dependency check {path.relative_to(root).as_posix()}: {exc}")
            continue
        rel = path.relative_to(root).as_posix()
        for node in ast.walk(tree):
            modules: list[str] = []
            if isinstance(node, ast.Import):
                modules = [alias.name.split(".", 1)[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                modules = [node.module.split(".", 1)[0]]
            for module in modules:
                if module in stdlib or module in local_roots:
                    continue
                direct_imports.setdefault(module, set()).add(rel)
    missing = sorted(module for module in direct_imports if normalize_dependency_name(module) not in declared)
    for module in missing:
        files = ", ".join(sorted(direct_imports[module])[:5])
        fail(errors, f"direct dependency import {module!r} is not declared in pyproject.toml; examples: {files}")


def check_pyproject(root: Path, errors: list[str]) -> None:
    path = root / "pyproject.toml"
    if not path.is_file():
        return
    try:
        cfg = tomllib.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        fail(errors, f"invalid pyproject.toml: {exc}")
        return
    scripts = cfg.get("project", {}).get("scripts", {})
    expected = {
        "eval-text-strict": "scd_repro.cli.eval_text_strict:main",
        "summarize-mistral-repro": "scd_repro.cli.summarize_mistral_repro:main",
        "scd-vllm": "scd_vllm_plugin.cli:main",
    }
    for name, target in expected.items():
        if scripts.get(name) != target:
            fail(errors, f"missing or wrong script entry {name}: expected {target!r}, got {scripts.get(name)!r}")
    if "search-blockwise-hybrid" in scripts:
        fail(errors, "paper-facing release must not expose the GitHub-style search-blockwise-hybrid console script")


def check_import_smoke(root: Path, errors: list[str]) -> None:
    pyproject = root / "pyproject.toml"
    if not pyproject.is_file():
        return
    try:
        cfg = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    except Exception as exc:
        fail(errors, f"cannot read pyproject for import smoke: {exc}")
        return
    scripts = cfg.get("project", {}).get("scripts", {})
    src_path = str((root / "src").resolve())
    old_path = list(sys.path)
    sys.path.insert(0, src_path)
    try:
        for name, target in sorted(scripts.items()):
            module_name = str(target).split(":", 1)[0]
            try:
                importlib.import_module(module_name)
            except Exception as exc:
                fail(errors, f"console script {name} cannot import {module_name}: {exc}")
    finally:
        sys.path[:] = old_path


def check_wrapper(root: Path, errors: list[str]) -> None:
    path = root / "scripts" / "run_ser_scd_icml_mistral_repro.sh"
    if not path.is_file():
        return
    text = path.read_text(encoding="utf-8")
    required_snippets = [
        'DIRECTIONS="${DIRECTIONS:-lite_to_7b}"',
        'DATASETS="${DATASETS:-cmrc2018}"',
        'PATCH_TRAIN_CONFIG="${PATCH_TRAIN_CONFIG:-configs/mistral_self_patch_xnorm_nativekv_shared_residual_mlp_alllayers_e1_h2048_r512_b2_20260404.json}"',
        'EVAL_CONFIG_TEMPLATE="${EVAL_CONFIG_TEMPLATE:-configs/eval_scd_icml_full_scd_no_fallback_no_recompute.json}"',
        'ROUTE_MODE="${ROUTE_MODE:-template}"',
        'ROUTE_TEMPLATE="${ROUTE_TEMPLATE:-configs/routes/scd_icml_mistrallite_to_mistral7_reuse19_patch13_cmrc200.json}"',
        'ROUTE_PATCH_BUDGET="${ROUTE_PATCH_BUDGET:-6}"',
        'WAIT_CAPTURE_FREE_MB="${WAIT_CAPTURE_FREE_MB:-18000}"',
        'WAIT_TRAIN_FREE_MB="${WAIT_TRAIN_FREE_MB:-6000}"',
        'WAIT_EVAL_FREE_MB="${WAIT_EVAL_FREE_MB:-18000}"',
        "pick_free_gpus()",
        "waiting to select ${AUTO_GPU_COUNT} GPU(s) with free memory",
        "cuda_logical_indices_from_devices()",
        "selected CUDA_VISIBLE_DEVICES=",
        "AUTO_SELECTED_CUDA_VISIBLE_DEVICES=1",
        "reselected CUDA_VISIBLE_DEVICES=",
        "ROUTE_PATCH_LAYERS",
        "route_patch_layers.strip()",
        "make_paper_scd_route.py",
        "MODELS_CONFIG_LITE_TO_7B",
        "python -m scd_repro.cli.summarize_mistral_repro",
        "--expected-directions",
        "--expected-datasets",
    ]
    for snippet in required_snippets:
        if snippet not in text:
            fail(errors, f"wrapper missing snippet: {snippet}")
    if 'python - "${RUN_ROOT}"' in text:
        fail(errors, "wrapper still contains old inline summary Python block")


def check_neighboring_wrapper(root: Path, errors: list[str]) -> None:
    path = root / "scripts" / "run_ser_neighboring_baselines_cmrc.sh"
    if not path.is_file():
        return
    text = path.read_text(encoding="utf-8")
    required_snippets = [
        'SCD_SUMMARY="${SCD_SUMMARY:-}"',
        'neighboring_baselines_table=pending_scd_summary',
        "The adapted neighboring baselines completed, but the consolidated table was",
        "not generated because the SCD row must use the same protocol",
        "build_neighboring_baseline_table.py",
        "--require-compatible-protocol",
    ]
    for snippet in required_snippets:
        if snippet not in text:
            fail(errors, f"neighboring wrapper missing protocol-safety snippet: {snippet}")
    stale_snippet = "SCD_SUMMARY=\"${SCD_SUMMARY:-/disk_n/"
    if stale_snippet in text:
        fail(errors, "neighboring wrapper must not default SCD_SUMMARY to a fixed prior run")


def check_eval_config(root: Path, errors: list[str]) -> None:
    path = root / "configs" / "eval_scd_icml_no_fallback_no_recompute.json"
    if not path.is_file():
        return
    cfg = json.loads(path.read_text(encoding="utf-8-sig"))
    if cfg.get("runtime", {}).get("cpu_fallback") is not False:
        fail(errors, "no-fallback eval config must disable runtime.cpu_fallback")
    runtime_eval = cfg.get("runtime_eval", {})
    expected_flags = {
        "run_full_consumer_baseline",
        "run_raw_kv",
        "run_quantized_4bit",
        "run_reuse_only",
        "run_scd_full",
    }
    for flag in expected_flags:
        if runtime_eval.get(flag) is not True:
            fail(errors, f"eval config does not enable {flag}")
    disabled_flags = {
        "run_selective_recompute",
        "run_reuse_fallback",
        "run_patch_only",
        "run_reuse_patch",
        "run_reuse_patch_recompute",
    }
    for flag in disabled_flags:
        if runtime_eval.get(flag) is not False:
            fail(errors, f"no-fallback eval config must disable {flag}")
    route_cfg = cfg.get("route", {})
    if bool(route_cfg.get("enable_short_recompute", False)):
        fail(errors, "no-fallback eval config must disable short recompute")
    if runtime_eval.get("default_recompute_segment_length") != 0:
        fail(errors, "no-fallback eval config default_recompute_segment_length must be 0")
    if runtime_eval.get("max_eval_samples") != 200:
        fail(errors, "eval config max_eval_samples should be 200 for the paper-facing run")
    if int(route_cfg.get("patch_budget", 0) or 0) != 6:
        fail(errors, "paper-facing eval config route.patch_budget must be 6")

    selected_path = root / "configs" / "eval_scd_icml_full_scd_no_fallback_no_recompute.json"
    if selected_path.is_file():
        selected_cfg = json.loads(selected_path.read_text(encoding="utf-8-sig"))
        if selected_cfg.get("runtime", {}).get("cpu_fallback") is not False:
            fail(errors, "selected-route eval config must disable runtime.cpu_fallback")
        selected_runtime_eval = selected_cfg.get("runtime_eval", {})
        selected_enabled = {"run_full_consumer_baseline", "run_scd_full"}
        for flag in selected_enabled:
            if selected_runtime_eval.get(flag) is not True:
                fail(errors, f"selected-route eval config does not enable {flag}")
        for flag, value in selected_runtime_eval.items():
            if flag.startswith("run_") and flag not in selected_enabled and value is not False:
                fail(errors, f"selected-route eval config must disable {flag}")
        if selected_runtime_eval.get("default_recompute_segment_length") != 0:
            fail(errors, "selected-route eval config default_recompute_segment_length must be 0")


def check_paper_scd_route_semantics(root: Path, errors: list[str]) -> None:
    script = root / "scripts" / "make_paper_scd_route.py"
    validator = root / "scripts" / "validate_scd_route.py"
    if not script.is_file() or not validator.is_file():
        return
    with tempfile.TemporaryDirectory(prefix="scd_paper_route_") as tmp_dir:
        tmp = Path(tmp_dir)
        patch_dir = tmp / "patch"
        model_b = tmp / "model_b"
        patch_dir.mkdir()
        model_b.mkdir()
        (patch_dir / "meta.json").write_text(
            json.dumps({"selected_layers": [0, 1, 2, 3], "num_layers": 4}),
            encoding="utf-8",
        )
        (model_b / "config.json").write_text(
            json.dumps({"num_hidden_layers": 4}),
            encoding="utf-8",
        )
        models_config = tmp / "models.json"
        models_config.write_text(
            json.dumps({"model_b": {"path": str(model_b)}}),
            encoding="utf-8",
        )
        profile = tmp / "profile.json"
        profile.write_text(
            json.dumps(
                {
                    "num_layers": 4,
                    "recommended_patch_layers": [3, 1, 2, 0],
                    "layers": [
                        {"layer": 0, "expected_gain": 0.1},
                        {"layer": 1, "expected_gain": 0.8},
                        {"layer": 2, "expected_gain": 0.2},
                        {"layer": 3, "expected_gain": 0.9},
                    ],
                }
            ),
            encoding="utf-8",
        )
        eval_config = tmp / "eval.json"
        eval_config.write_text(json.dumps({"route": {"patch_budget": 2}}), encoding="utf-8")
        out = tmp / "route.json"
        try:
            subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--profile",
                    str(profile),
                    "--patch-dir",
                    str(patch_dir),
                    "--models-config",
                    str(models_config),
                    "--eval-config",
                    str(eval_config),
                    "--out",
                    str(out),
                ],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            route = json.loads(out.read_text(encoding="utf-8"))
            validation = subprocess.run(
                [sys.executable, str(validator), "--route", str(out), "--require-mixed"],
                cwd=root,
                text=True,
                capture_output=True,
            )
        except Exception as exc:
            fail(errors, f"make_paper_scd_route semantic smoke failed: {exc}")
            return
    if route.get("route_semantics") != "paper_scd_sparse_patch_reuse_all_other_layers":
        fail(errors, f"paper SCD route has wrong semantics: {route.get('route_semantics')!r}")
    if route.get("patch_layers") != [1, 3]:
        fail(errors, f"paper SCD route should select top-k profile patch layers [1, 3], got {route.get('patch_layers')}")
    if route.get("reuse_layers") != [0, 2]:
        fail(errors, f"paper SCD route should reuse non-patch layers [0, 2], got {route.get('reuse_layers')}")
    if route.get("fallback_layers") != []:
        fail(errors, f"paper SCD route fallback_layers must be empty, got {route.get('fallback_layers')}")
    if route.get("selected_recompute_layers") != [] or route.get("recompute_segment_length_by_layer") != {}:
        fail(errors, "paper SCD route must not use recompute")
    if validation.returncode != 0:
        fail(errors, f"validate_scd_route.py rejected a valid paper mixed route: {validation.stderr or validation.stdout}")


def check_selected_route(root: Path, errors: list[str]) -> None:
    path = root / "configs" / "routes" / "scd_icml_mistrallite_to_mistral7_reuse19_patch13_cmrc200.json"
    if not path.is_file():
        return
    route = json.loads(path.read_text(encoding="utf-8-sig"))
    patch_layers = [int(layer) for layer in route.get("patch_layers", [])]
    reuse_layers = [int(layer) for layer in route.get("reuse_layers", [])]
    num_layers = int(route.get("num_layers", 0))
    if route.get("route_semantics") != "paper_scd_sparse_patch_reuse_all_other_layers":
        fail(errors, f"selected Reuse19 route has wrong semantics: {route.get('route_semantics')!r}")
    if len(reuse_layers) != 19 or len(patch_layers) != 13:
        fail(errors, f"selected Reuse19 route should be Reuse19/Patch13, got reuse={len(reuse_layers)} patch={len(patch_layers)}")
    if route.get("fallback_layers") != []:
        fail(errors, f"selected Reuse19 route fallback_layers must be empty, got {route.get('fallback_layers')}")
    if route.get("selected_recompute_layers") != [] or route.get("recompute_segment_length_by_layer") != {}:
        fail(errors, "selected Reuse19 route must not use recompute")
    if set(patch_layers) & set(reuse_layers):
        fail(errors, f"selected Reuse19 route has patch/reuse overlap: {sorted(set(patch_layers) & set(reuse_layers))}")
    if num_layers and set(patch_layers) | set(reuse_layers) != set(range(num_layers)):
        fail(errors, "selected Reuse19 route patch+reuse layers must cover all layers")
    ref = route.get("metadata", {}).get("reference_result", {})
    if float(ref.get("f1_retention", 0.0)) < 0.95:
        fail(errors, "selected Reuse19 route reference F1 retention must be at least 0.95")
    if float(ref.get("em_retention", 0.0)) < 0.95:
        fail(errors, "selected Reuse19 route reference EM retention must be at least 0.95")


def check_vllm_demo_route(root: Path, errors: list[str]) -> None:
    path = root / "configs" / "routes" / "scd_icml_mistrallite_to_mistral7_patch20_reuse12_cmrc200.json"
    if not path.is_file():
        return
    route = json.loads(path.read_text(encoding="utf-8-sig"))
    patch_layers = [int(layer) for layer in route.get("patch_layers", [])]
    reuse_layers = [int(layer) for layer in route.get("reuse_layers", [])]
    num_layers = int(route.get("num_layers", 0))
    if route.get("route_semantics") != "release_demo_sparse_patch_reuse_all_other_layers":
        fail(errors, f"vLLM demo route has wrong semantics: {route.get('route_semantics')!r}")
    if len(reuse_layers) != 12 or len(patch_layers) != 20:
        fail(errors, f"vLLM demo route should be Reuse12/Patch20, got reuse={len(reuse_layers)} patch={len(patch_layers)}")
    if route.get("fallback_layers") != []:
        fail(errors, f"vLLM demo route fallback_layers must be empty, got {route.get('fallback_layers')}")
    if route.get("selected_recompute_layers") != [] or route.get("recompute_segment_length_by_layer") != {}:
        fail(errors, "vLLM demo route must not use recompute")
    if set(patch_layers) & set(reuse_layers):
        fail(errors, f"vLLM demo route has patch/reuse overlap: {sorted(set(patch_layers) & set(reuse_layers))}")
    if num_layers and set(patch_layers) | set(reuse_layers) != set(range(num_layers)):
        fail(errors, "vLLM demo route patch+reuse layers must cover all layers")
    ref = route.get("metadata", {}).get("reference_result", {})
    full_f1 = float(ref.get("full_consumer_f1", 0.0))
    full_em = float(ref.get("full_consumer_em", 0.0))
    scd_f1 = float(ref.get("scd_f1", 0.0))
    scd_em = float(ref.get("scd_em", 0.0))
    if full_f1 <= 0.0 or scd_f1 < 0.95 * full_f1:
        fail(errors, "vLLM demo route reference F1 retention must be at least 0.95")
    if full_em <= 0.0 or scd_em < 0.95 * full_em:
        fail(errors, "vLLM demo route reference EM retention must be at least 0.95")


def check_quality_reference_route(root: Path, errors: list[str]) -> None:
    path = root / "configs" / "routes" / "scd_icml_mistrallite_to_mistral7_patch26_reuse6_cmrc200.json"
    if not path.is_file():
        return
    route = json.loads(path.read_text(encoding="utf-8-sig"))
    patch_layers = [int(layer) for layer in route.get("patch_layers", [])]
    reuse_layers = [int(layer) for layer in route.get("reuse_layers", [])]
    num_layers = int(route.get("num_layers", 0))
    if route.get("route_semantics") != "paper_scd_sparse_patch_reuse_all_other_layers":
        fail(errors, f"quality reference route has wrong semantics: {route.get('route_semantics')!r}")
    if len(reuse_layers) != 6 or len(patch_layers) != 26:
        fail(errors, f"quality reference route should be Reuse6/Patch26, got reuse={len(reuse_layers)} patch={len(patch_layers)}")
    if route.get("fallback_layers") != []:
        fail(errors, f"quality reference route fallback_layers must be empty, got {route.get('fallback_layers')}")
    if route.get("selected_recompute_layers") != [] or route.get("recompute_segment_length_by_layer") != {}:
        fail(errors, "quality reference route must not use recompute")
    if set(patch_layers) & set(reuse_layers):
        fail(errors, f"quality reference route has patch/reuse overlap: {sorted(set(patch_layers) & set(reuse_layers))}")
    if num_layers and set(patch_layers) | set(reuse_layers) != set(range(num_layers)):
        fail(errors, "quality reference route patch+reuse layers must cover all layers")
    ref = route.get("metadata", {}).get("reference_result", {})
    cmrc = ref.get("cmrc2018", {})
    full_f1 = float(cmrc.get("full_consumer_f1", 0.0))
    full_em = float(cmrc.get("full_consumer_em", 0.0))
    scd_f1 = float(cmrc.get("scd_f1", 0.0))
    scd_em = float(cmrc.get("scd_em", 0.0))
    if full_f1 <= 0.0 or scd_f1 < 0.95 * full_f1:
        fail(errors, "quality reference route CMRC F1 retention must be at least 0.95")
    if full_em <= 0.0 or scd_em < 0.95 * full_em:
        fail(errors, "quality reference route CMRC EM retention must be at least 0.95")
    wikitext = ref.get("wikitext2", {})
    full_ppl = float(wikitext.get("full_consumer_ppl", 0.0))
    scd_ppl = float(wikitext.get("scd_ppl", 0.0))
    if full_ppl <= 0.0 or scd_ppl <= 0.0:
        fail(errors, "quality reference route WikiText-2 PPL evidence is missing")


def load_evidence_json(path: Path, errors: list[str], label: str) -> dict:
    if not path.is_file():
        fail(errors, f"missing {label} evidence artifact: {path}")
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception as exc:
        fail(errors, f"invalid {label} evidence JSON {path}: {exc}")
        return {}
    if not isinstance(data, dict):
        fail(errors, f"{label} evidence must be a JSON object: {path}")
        return {}
    return data


def check_quality_pair(
    *,
    rows: dict,
    label: str,
    sample_count: int,
    errors: list[str],
    scd_method: str = "reuse_patch",
    require_qa_improvement: bool = False,
    require_ttft_improvement: bool = False,
    primary_metric: str = "qa",
) -> None:
    full = rows.get("full_consumer") or {}
    scd = rows.get(scd_method) or {}
    if not full or not scd:
        fail(errors, f"{label} quality evidence must contain full_consumer and {scd_method} rows")
        return
    if int(full.get("num_samples", 0) or 0) != sample_count:
        fail(errors, f"{label} full_consumer sample_count should be {sample_count}, got {full.get('num_samples')!r}")
    if int(scd.get("num_samples", 0) or 0) != sample_count:
        fail(errors, f"{label} {scd_method} sample_count should be {sample_count}, got {scd.get('num_samples')!r}")
    if primary_metric == "qa":
        for key in ("f1", "em"):
            full_value = float(full.get(key, 0.0) or 0.0)
            scd_value = float(scd.get(key, 0.0) or 0.0)
            if full_value <= 0.0 or scd_value <= 0.0:
                fail(errors, f"{label} QA metric {key} must be positive for both full and SCD")
            if require_qa_improvement and scd_value < full_value:
                fail(errors, f"{label} SCD {key} should be at least full_consumer {key}")
    else:
        full_nll = float(full.get("avg_nll", 0.0) or 0.0)
        scd_nll = float(scd.get("avg_nll", 0.0) or 0.0)
        if full_nll <= 0.0 or scd_nll <= 0.0:
            fail(errors, f"{label} WikiText-2 avg_nll must be positive for both full and SCD")
    if require_ttft_improvement:
        full_ttft = float(full.get("ttft_ms", 0.0) or 0.0)
        scd_ttft = float(scd.get("ttft_ms", 0.0) or 0.0)
        if full_ttft <= 0.0 or scd_ttft <= 0.0 or scd_ttft >= full_ttft:
            fail(errors, f"{label} SCD TTFT should be lower than full_consumer TTFT")


def check_multidataset_quality_evidence(root: Path, errors: list[str]) -> None:
    results_root = root.parent.parent / "results"
    if not results_root.is_dir():
        return

    mistral_root = results_root / "scd_vllm_multidataset_full_strict_20260617"
    mistral_qa_specs = [
        ("cmrc_patch20_reuse12_per_mode_summary.json", "Mistral CMRC2018 patch20/reuse12", 3219),
        ("cmrc_patch26_reuse6_per_mode_summary.json", "Mistral CMRC2018 patch26/reuse6", 3219),
        ("hotpotqa_patch20_reuse12_per_mode_summary.json", "Mistral HotpotQA patch20/reuse12", 7405),
        ("hotpotqa_patch26_reuse6_per_mode_summary.json", "Mistral HotpotQA patch26/reuse6", 7405),
    ]
    for filename, label, sample_count in mistral_qa_specs:
        rows = load_evidence_json(mistral_root / filename, errors, label)
        if rows:
            check_quality_pair(
                rows=rows,
                label=label,
                sample_count=sample_count,
                errors=errors,
                require_qa_improvement=True,
                require_ttft_improvement=True,
            )

    wikitext = load_evidence_json(
        results_root / "wikitext2_patch20_patch26_full_current_20260618_034812" / "wikitext2_summary.json",
        errors,
        "Mistral WikiText-2",
    )
    if wikitext:
        rows = wikitext.get("rows", [])
        if not isinstance(rows, list):
            fail(errors, "Mistral WikiText-2 evidence rows must be a list")
        else:
            for route_name in ("patch20_reuse12", "patch26_reuse6"):
                by_method = {
                    str(row.get("method")): row
                    for row in rows
                    if str(row.get("route_name")) == route_name
                }
                check_quality_pair(
                    rows=by_method,
                    label=f"Mistral WikiText-2 {route_name}",
                    sample_count=763,
                    errors=errors,
                    scd_method="scd_full",
                    require_ttft_improvement=True,
                    primary_metric="nll",
                )

    qwen = load_evidence_json(
        results_root / "qwen17_gray_full_strict_microsharded_20260617" / "aggregate_summary.json",
        errors,
        "Qwen3-1.7B gray multidataset",
    )
    if qwen:
        datasets = qwen.get("datasets", {})
        expected = {
            "cmrc2018": ("qa", 3219),
            "hotpotqa": ("qa", 7405),
            "wikitext2": ("nll", 763),
        }
        for dataset, (primary_metric, sample_count) in expected.items():
            payload = datasets.get(dataset, {})
            rows = payload.get("rows", []) if isinstance(payload, dict) else []
            by_method = {
                str(row.get("method")): row
                for row in rows
                if isinstance(row, dict)
            }
            check_quality_pair(
                rows=by_method,
                label=f"Qwen3-1.7B gray {dataset}",
                sample_count=sample_count,
                errors=errors,
                primary_metric=primary_metric,
                require_qa_improvement=primary_metric == "qa",
            )


def check_qwen_vllm_evidence(root: Path, errors: list[str]) -> None:
    results_root = root.parent.parent / "results"
    if not results_root.is_dir():
        return
    doctor_path = results_root / "a800_qwen17_scd_vllm_doctor_20260618_cache" / "doctor.json"
    plugin_path = results_root / "a800_qwen17_scd_vllm_plugin_smoke_20260618_cache" / "plugin_metrics.json"

    def load_result_json(path: Path) -> dict:
        if not path.is_file():
            fail(errors, f"missing Qwen vLLM evidence artifact: {path}")
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8-sig"))
        except Exception as exc:
            fail(errors, f"invalid Qwen vLLM evidence JSON {path}: {exc}")
            return {}

    doctor = load_result_json(doctor_path)
    if doctor:
        if doctor.get("pass") is not True:
            fail(errors, "Qwen SCD-vLLM doctor must pass")
        if int(doctor.get("failure_count", -1)) != 0:
            fail(errors, "Qwen SCD-vLLM doctor failure_count must be 0")
        route_checks = [
            check
            for check in doctor.get("checks", [])
            if check.get("name") == "route_strictness"
        ]
        if not route_checks or route_checks[0].get("status") != "pass":
            fail(errors, "Qwen SCD-vLLM route strictness check must pass")
        else:
            details = route_checks[0].get("details", {})
            if int(details.get("num_layers", 0)) != 28:
                fail(errors, "Qwen SCD-vLLM route must report 28 layers")
            if int(details.get("patch_count", -1)) != 6 or int(details.get("reuse_count", -1)) != 22:
                fail(errors, "Qwen SCD-vLLM route must be patch6/reuse22")
            if int(details.get("fallback_count", -1)) != 0 or int(details.get("recompute_count", -1)) != 0:
                fail(errors, "Qwen SCD-vLLM route must not use fallback or recompute")

    plugin = load_result_json(plugin_path)
    if plugin:
        smoke = plugin.get("smoke", {})
        export = plugin.get("export", {})
        if plugin.get("pass") is not True:
            fail(errors, "Qwen SCD-vLLM plugin smoke must pass")
        for key in ("adapter_smoke_pass", "scd_injection_pass", "bundle_token_match", "same_token_ids", "loaded_generation_ok"):
            if smoke.get(key) is not True:
                fail(errors, f"Qwen SCD-vLLM smoke gate {key} must be true")
        if export.get("route_name") != "paper_scd_reuse_patch_k6":
            fail(errors, f"Qwen SCD-vLLM route_name should be paper_scd_reuse_patch_k6, got {export.get('route_name')!r}")
        if int(export.get("num_layers", 0)) != 28:
            fail(errors, "Qwen SCD-vLLM export must contain 28 layers")
        if int(smoke.get("aligned_tokens", 0)) <= 0:
            fail(errors, "Qwen SCD-vLLM smoke must load at least one aligned token block")
        if int(export.get("payload_bytes", 0)) <= 0 or int(smoke.get("payload_bytes", 0)) <= 0:
            fail(errors, "Qwen SCD-vLLM payload bytes must be positive")


def check_batch_benchmark_metrics(
    *,
    root: Path,
    errors: list[str],
    artifact_dir: str,
    label: str,
    expected_route_id: str,
    require_preloaded_stats: bool = False,
) -> None:
    results_root = root.parent.parent / "results"
    if not results_root.is_dir():
        return
    path = results_root / artifact_dir / "metrics.json"
    if not path.is_file():
        fail(errors, f"missing {label} SCD-vLLM batch benchmark artifact: {path}")
        return
    try:
        metrics = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception as exc:
        fail(errors, f"invalid {label} SCD-vLLM batch benchmark JSON {path}: {exc}")
        return
    if metrics.get("benchmark_pass") is not True:
        fail(errors, f"{label} SCD-vLLM batch benchmark must pass")
    if "repeated-prefix trusted microbenchmark" not in str(metrics.get("claim_boundary", "")):
        fail(errors, f"{label} SCD-vLLM batch benchmark must declare its narrow claim boundary")
    if metrics.get("bundle_type") != "reuse_patch":
        fail(errors, f"{label} SCD-vLLM batch benchmark bundle_type should be reuse_patch, got {metrics.get('bundle_type')!r}")
    if metrics.get("route_id") != expected_route_id:
        fail(errors, f"{label} SCD-vLLM batch benchmark route_id should be {expected_route_id}, got {metrics.get('route_id')!r}")
    if metrics.get("kv_connector") != "SCDPreloadedConnector":
        fail(errors, f"{label} SCD-vLLM batch benchmark should use SCDPreloadedConnector, got {metrics.get('kv_connector')!r}")
    if metrics.get("kv_connector_module_path") != "scd_vllm.preloaded_connector":
        fail(errors, f"{label} SCD-vLLM batch benchmark should use scd_vllm.preloaded_connector")
    if int(metrics.get("batch_size", 0)) < 1 or int(metrics.get("expected_requests", 0)) < 1:
        fail(errors, f"{label} SCD-vLLM batch benchmark must report positive batch/request counts")
    for key in ("load_generation_ok", "bundle_token_match", "storage_ready"):
        if metrics.get(key) is not True:
            fail(errors, f"{label} SCD-vLLM batch benchmark gate {key} must be true")
    if metrics.get("prefix_cache_generation_ok") is not True:
        fail(errors, f"{label} SCD-vLLM batch benchmark prefix-cache baseline must pass")
    for key in (
        "load_tokens_per_second",
        "load_requests_per_second",
        "baseline_tokens_per_second",
        "baseline_requests_per_second",
        "prefix_cache_tokens_per_second",
        "prefix_cache_requests_per_second",
    ):
        if float(metrics.get(key, 0.0)) <= 0.0:
            fail(errors, f"{label} SCD-vLLM batch benchmark metric {key} must be positive")
    if require_preloaded_stats:
        stats = metrics.get("preloaded_connector_stats") or {}
        if stats.get("exists") is not True:
            fail(errors, f"{label} SCD-vLLM preloaded connector stats must exist")
        if int(stats.get("invalid_files", 0) or 0) != 0:
            fail(errors, f"{label} SCD-vLLM preloaded connector stats must not contain invalid files")
        for key in ("worker_count", "cache_hits", "cache_misses", "cache_events", "cached_files", "cached_bytes"):
            if int(stats.get(key, 0) or 0) <= 0:
                fail(errors, f"{label} SCD-vLLM preloaded connector stats {key} must be positive")
        if float(stats.get("cache_hit_rate", 0.0) or 0.0) <= 0.0:
            fail(errors, f"{label} SCD-vLLM preloaded connector cache_hit_rate must be positive")


def check_mistral_batch_benchmark_evidence(root: Path, errors: list[str]) -> None:
    check_batch_benchmark_metrics(
        root=root,
        errors=errors,
        artifact_dir="a800_mistral_patch20_scd_vllm_preloadedstats_20260618_cache",
        label="Mistral patch20/reuse12",
        expected_route_id="scd_icml_mistrallite_to_mistral7_patch20_reuse12_cmrc200",
        require_preloaded_stats=True,
    )
    check_batch_benchmark_metrics(
        root=root,
        errors=errors,
        artifact_dir="a800_mistral_patch26_scd_vllm_preloadedstats_20260618_cache",
        label="Mistral patch26/reuse6",
        expected_route_id="scd_icml_mistrallite_to_mistral7_patch26_reuse6_cmrc200",
        require_preloaded_stats=True,
    )


def check_qwen_batch_benchmark_evidence(root: Path, errors: list[str]) -> None:
    check_batch_benchmark_metrics(
        root=root,
        errors=errors,
        artifact_dir="a800_qwen17_scd_vllm_preloadedstats_20260618_cache",
        label="Qwen",
        expected_route_id="paper_scd_reuse_patch_k6",
        require_preloaded_stats=True,
    )


def check_launch_smoke_metrics(
    *,
    root: Path,
    errors: list[str],
    artifact_dir: str,
    label: str,
    min_request_count: int,
    expected_prompt_token_count: int | None = None,
) -> None:
    results_root = root.parent.parent / "results"
    if not results_root.is_dir():
        return
    path = results_root / artifact_dir / "launch_smoke.json"
    if not path.is_file():
        fail(errors, f"missing {label} SCD-vLLM launch-smoke artifact: {path}")
        return
    try:
        metrics = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception as exc:
        fail(errors, f"invalid {label} SCD-vLLM launch-smoke JSON {path}: {exc}")
        return
    if metrics.get("launch_smoke_pass") is not True:
        fail(errors, f"{label} launch-smoke must pass")
    if "official vLLM LLM" not in str(metrics.get("claim_boundary", "")):
        fail(errors, f"{label} launch-smoke must declare official-vLLM claim boundary")
    for key in ("generation_ok", "bundle_token_match"):
        if metrics.get(key) is not True:
            fail(errors, f"{label} launch-smoke gate {key} must be true")
    if metrics.get("kv_connector") != "SCDPreloadedConnector":
        fail(errors, f"{label} launch-smoke should use SCDPreloadedConnector")
    if metrics.get("kv_connector_module_path") != "scd_vllm.preloaded_connector":
        fail(errors, f"{label} launch-smoke should use scd_vllm.preloaded_connector")
    if metrics.get("disable_hybrid_kv_cache_manager") is not True:
        fail(errors, f"{label} launch-smoke must disable hybrid KV cache manager")
    if int(metrics.get("request_count", 0) or 0) < min_request_count:
        fail(errors, f"{label} launch-smoke request_count below expected minimum")
    if int(metrics.get("output_tokens", 0) or 0) <= 0:
        fail(errors, f"{label} launch-smoke output_tokens must be positive")
    if float(metrics.get("tokens_per_second", 0.0) or 0.0) <= 0.0:
        fail(errors, f"{label} launch-smoke tokens_per_second must be positive")
    if expected_prompt_token_count is not None and int(metrics.get("prompt_token_count", -1) or -1) != expected_prompt_token_count:
        fail(errors, f"{label} launch-smoke prompt_token_count should be {expected_prompt_token_count}")
    storage = metrics.get("storage") or {}
    if storage.get("exists") is not True or int(storage.get("safetensor_files", 0) or 0) <= 0:
        fail(errors, f"{label} launch-smoke storage must contain safetensors")
    if metrics.get("preloaded_connector_final_stats_collected_after_child_exit") is not True:
        fail(errors, f"{label} launch-smoke must collect final stats after child exit")
    stats = metrics.get("preloaded_connector_stats") or {}
    if stats.get("exists") is not True:
        fail(errors, f"{label} launch-smoke preloaded connector stats must exist")
    if int(stats.get("invalid_files", 0) or 0) != 0:
        fail(errors, f"{label} launch-smoke preloaded connector stats must not contain invalid files")
    for key in ("worker_count", "cache_hits", "cache_misses", "cache_events", "cached_files", "cached_bytes", "final_workers"):
        if int(stats.get(key, 0) or 0) <= 0:
            fail(errors, f"{label} launch-smoke preloaded connector stats {key} must be positive")
    if float(stats.get("cache_hit_rate", 0.0) or 0.0) <= 0.0:
        fail(errors, f"{label} launch-smoke cache_hit_rate must be positive")


def check_launch_smoke_evidence(root: Path, errors: list[str]) -> None:
    check_launch_smoke_metrics(
        root=root,
        errors=errors,
        artifact_dir="a800_mistral_patch26_scd_vllm_launchsmoke_20260618_cache",
        label="Mistral patch26/reuse6",
        min_request_count=8,
        expected_prompt_token_count=229,
    )
    check_launch_smoke_metrics(
        root=root,
        errors=errors,
        artifact_dir="a800_qwen17_scd_vllm_launchsmoke_20260618_cache",
        label="Qwen",
        min_request_count=128,
        expected_prompt_token_count=27,
    )


def check_model_template(root: Path, errors: list[str]) -> None:
    path = root / "configs" / "models_mistrallite_to_mistral7_template.json"
    if not path.is_file():
        return
    cfg = json.loads(path.read_text(encoding="utf-8-sig"))
    if cfg.get("model_a", {}).get("name") != "MistralLite":
        fail(errors, "model template model_a.name should be MistralLite")
    if "Mistral-7B" not in str(cfg.get("model_b", {}).get("name", "")):
        fail(errors, "model template model_b.name should be Mistral-7B")
    if cfg.get("architecture", {}).get("num_layers") != 32:
        fail(errors, "model template architecture.num_layers should be 32")


def check_patch_train_config(root: Path, errors: list[str]) -> None:
    path = root / "configs" / "mistral_self_patch_xnorm_nativekv_shared_residual_mlp_paper_k6_e1_h2048_r512_b2_20260404.json"
    if not path.is_file():
        return
    cfg = json.loads(path.read_text(encoding="utf-8-sig"))
    if cfg.get("runtime", {}).get("cpu_fallback") is not False:
        fail(errors, "paper k=6 patch train config must disable runtime.cpu_fallback")
    patch_cfg = cfg.get("patch", {})
    if patch_cfg.get("patch_budget") != 6:
        fail(errors, "paper k=6 patch train config must use patch_budget=6")
    if patch_cfg.get("force_residual_layers_as_selected") is not False:
        fail(errors, "paper k=6 patch train config must let the offline profile select sparse Patch layers")


def check_reuse_train_config(root: Path, errors: list[str]) -> None:
    path = root / "configs" / "train_reuse_mistral_self_layerwise_20260404.json"
    if not path.is_file():
        return
    cfg = json.loads(path.read_text(encoding="utf-8-sig"))
    if cfg.get("runtime", {}).get("cpu_fallback") is not False:
        fail(errors, "Mistral reuse train config must disable runtime.cpu_fallback")


def check_selected_patch_train_config(root: Path, errors: list[str]) -> None:
    path = root / "configs" / "mistral_self_patch_xnorm_nativekv_shared_residual_mlp_alllayers_e1_h2048_r512_b2_20260404.json"
    if not path.is_file():
        return
    cfg = json.loads(path.read_text(encoding="utf-8-sig"))
    if cfg.get("runtime", {}).get("cpu_fallback") is not False:
        fail(errors, "selected-route all-layer patch train config must disable runtime.cpu_fallback")
    patch_cfg = cfg.get("patch", {})
    if patch_cfg.get("patch_budget") != 32:
        fail(errors, "selected-route all-layer patch train config must use patch_budget=32")
    if patch_cfg.get("force_residual_layers_as_selected") is not True:
        fail(errors, "selected-route all-layer patch train config must force all residual layers as selected")


def check_docs(root: Path, errors: list[str]) -> None:
    docs = {
        "README.md": ["REPRODUCING_SCD_ICML.md", "DATA_PREPARATION.md", "OPEN_SOURCE_RELEASE_MANIFEST.md"],
        "REPRODUCING_SCD_ICML.md": [
            "no-fallback/no-recompute",
            "paper_scd_sparse_patch_reuse_all_other_layers",
            "reuse_layers=<all non-patch layers>",
            "teacher-aligned conditional route search",
            "fallback_layers=[]",
            "Qwen-32B",
            "bandwidth_sensitivity",
            "reuse04_patch28",
            "reuse08_patch24",
        ],
        "DATA_PREPARATION.md": ["cmrc2018_train.json", "hotpot_dev_distractor_v1.json", "wiki.test.txt"],
        "OPEN_SOURCE_RELEASE_MANIFEST.md": ["check_release_readiness.py", "check_release_smoke_status.py", "make_paper_scd_route.py", "make_paper_scd_candidate_routes.py", "select_quality_gated_scd_route.py", "select_patch_reuse_route.py", "collect_route_eval_metrics.py", "filter_strict_eval_by_qids.py", "build_scd_icml_result_tables.py", "build_scd_icml_analysis_figures.py", "extract_stack_viz_inputs.py", "build_stack_viz.py", "build_offline_cost_table.py", "build_paper_artifact_bundle.py", "finalize_scd_icml_full7405_bundle.py", "run_ser_neighboring_baselines_cmrc.sh", "run_svdllm_style_kv_cmrc.py", "run_slicegpt_style_kv_cmrc.py", "run_tensorllm_style_kv_cmrc.py", "build_neighboring_baseline_table.py", "wait_postprocess_scd_icml_full7405_tables.sh", "wait_finalize_scd_icml_full7405_bundle.sh", "check_paper_artifacts.py", "report_pairwise_interactions.py", "validate_scd_route.py", "build_release_tree.py", "examples/scd_vllm_official_launch.py", "scd-vllm doctor", "scd-vllm launch-spec", "scd-vllm launch-smoke", "scd-vllm batch-benchmark", "src/scd_vllm_plugin/", "configs/scd_vllm_mvp.yaml", "scripts/scd_vllm/run_06_report.sh", "tests/", "15 rows", "lite_to_7b", "bandwidth_sensitivity", "patch20_reuse12", "higher-Reuse", "patch26_reuse6", "quality-first reference route", "a800_scd_vllm_raw_bank_patch26_run_20260618_044521", "a800_mistral_patch20_scd_vllm_preloadedstats_20260618_cache", "a800_mistral_patch26_scd_vllm_preloadedstats_20260618_cache", "a800_mistral_patch26_scd_vllm_launchsmoke_20260618_cache", "cache_hits", "a800_qwen17_scd_vllm_preloadedstats_20260618_cache", "a800_qwen17_scd_vllm_launchsmoke_20260618_cache", "SCDPreloadedConnector", "kv-connector-module-path", "prefix-cache", "repeated-prefix trusted microbenchmark", "NLL/PPL"],
        "docs/scd_vllm_plugin_mvp.md": ["scd-vllm doctor", "strict no-fallback/no-recompute", "scd-vllm run", "scd-vllm launch-spec", "scd-vllm launch-smoke", "scd-vllm batch-benchmark", "examples/scd_vllm_official_launch.py", "ExampleConnector", "SCDPreloadedConnector", "kv-connector-module-path", "vllm.LLM", "patch26_reuse6"],
        "reports/scd_vllm_repo_audit.md": ["Existing entry points", "Proposed mapping to scd_vllm modules", "Implementation/release risks", "dataset-specific routes", "src/scd_vllm/injection.py"],
        "reports/scd_vllm_status.md": ["conda activate cache", "vLLM: `0.18.0`", "patch20_reuse12", "patch26_reuse6", "greedy_ntok", "Payload Bytes", "a800_mistral_patch20_scd_vllm_preloadedstats_20260618_cache", "a800_mistral_patch26_scd_vllm_preloadedstats_20260618_cache", "a800_mistral_patch26_scd_vllm_launchsmoke_20260618_cache", "Cache Hits", "Qwen3-1.7B", "a800_qwen17_vllm_baseline_20260618_0530", "a800_qwen17_scd_vllm_plugin_smoke_20260618_cache", "a800_qwen17_scd_vllm_preloadedstats_20260618_cache", "a800_qwen17_scd_vllm_launchsmoke_20260618_cache", "plain-vLLM compatibility", "Qwen raw-bank-to-vLLM", "SCDPreloadedConnector", "launch-spec", "launch-smoke", "prefix-cache", "repeated-prefix trusted microbenchmark", "Dataset Quality Evidence", "NLL/PPL", "not a production-grade vLLM connector"],
        "LICENSE": ["MIT License", "SCD reproduction team"],
    }
    for rel, snippets in docs.items():
        path = root / rel
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        for snippet in snippets:
            if snippet not in text:
                fail(errors, f"{rel} missing snippet: {snippet}")
        stale_doc_phrases = [
            "HotpotQA main row is not closed",
            "must finish before the HotpotQA main result row is closed",
            "pending compressed-protocol full7405 route table",
            "being closed with full7405 evaluation",
            "must be rechecked with the current train/eval split before closing the main row",
        ]
        for phrase in stale_doc_phrases:
            if phrase in text:
                fail(errors, f"{rel} contains stale full7405 documentation phrase: {phrase}")
    readme = root / "README.md"
    if readme.is_file():
        text = readme.read_text(encoding="utf-8")
        stale_quickstart_paths = [
            "configs/models.yaml",
            "configs/data.yaml",
            "configs/train.yaml",
        ]
        for stale_path in stale_quickstart_paths:
            if stale_path in text and not (root / stale_path).is_file():
                fail(errors, f"README.md references missing quick-start config: {stale_path}")


def check_run_root(run_root: Path, errors: list[str]) -> None:
    summary = run_root / "scd_icml_mistral_summary.json"
    if not summary.is_file():
        fail(errors, f"run root summary missing: {summary}")
        return
    rows = json.loads(summary.read_text(encoding="utf-8-sig"))
    present = {(str(row.get("direction")), str(row.get("dataset")), str(row.get("method"))) for row in rows}
    missing = []
    for dataset in EXPECTED_DATASETS:
        for method in EXPECTED_METHODS:
            key = ("lite_to_7b", dataset, method)
            if key not in present:
                missing.append("/".join(key))
    if missing:
        fail(errors, "run root summary missing expected rows: " + ", ".join(sorted(missing)))
    if len(rows) != len(EXPECTED_DATASETS) * len(EXPECTED_METHODS):
        fail(errors, f"expected {len(EXPECTED_DATASETS) * len(EXPECTED_METHODS)} aggregate rows, got {len(rows)}")
    route_path = run_root / "lite_to_7b" / "route" / "route_cmrc2018_seed42.json"
    if not route_path.is_file():
        fail(errors, f"run root route missing: {route_path}")
        return
    route = json.loads(route_path.read_text(encoding="utf-8-sig"))
    num_layers = int(route.get("num_layers", 0))
    patch_layers = [int(layer) for layer in route.get("patch_layers", [])]
    reuse_layers = [int(layer) for layer in route.get("reuse_layers", [])]
    fallback_layers = [int(layer) for layer in route.get("fallback_layers", [])]
    recompute_layers = [int(layer) for layer in route.get("selected_recompute_layers", [])]
    recompute_map = route.get("recompute_segment_length_by_layer", {})
    if fallback_layers:
        fail(errors, f"run root route must not use fallback_layers, got {fallback_layers}")
    if recompute_layers or recompute_map:
        fail(errors, f"run root route must not use recompute, got layers={recompute_layers} map={recompute_map}")
    if num_layers <= 0:
        fail(errors, f"run root route has invalid num_layers={num_layers}")
    else:
        all_layers = set(range(num_layers))
        patch_set = set(patch_layers)
        reuse_set = set(reuse_layers)
        if patch_set & reuse_set:
            fail(errors, f"run root route patch/reuse overlap: {sorted(patch_set & reuse_set)}")
        if patch_set | reuse_set != all_layers:
            fail(errors, f"run root route patch+reuse must cover all layers 0..{num_layers - 1}")
        if not patch_set or not reuse_set:
            fail(errors, "paper run root route must be mixed: both patch_layers and reuse_layers must be non-empty")
        if len(patch_set) >= num_layers:
            fail(errors, f"paper run root route must maximize Reuse under a quality gate, got all-patch route with {len(patch_set)} patch layers")
        if route.get("route_semantics") != "paper_scd_sparse_patch_reuse_all_other_layers":
            fail(errors, f"paper run root route has wrong semantics: {route.get('route_semantics')!r}")


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve() if args.root else Path(__file__).resolve().parents[1]
    errors: list[str] = []
    if not root.exists():
        fail(errors, f"root does not exist: {root}")
    else:
        check_required_files(root, errors)
        check_json(root, errors)
        check_python_source_hygiene(root, errors)
        check_py_compile(root, errors)
        check_shell_scripts(root, errors)
        check_pyproject(root, errors)
        check_declared_dependencies(root, errors)
        check_import_smoke(root, errors)
        check_wrapper(root, errors)
        check_neighboring_wrapper(root, errors)
        check_eval_config(root, errors)
        check_paper_scd_route_semantics(root, errors)
        check_selected_route(root, errors)
        check_vllm_demo_route(root, errors)
        check_quality_reference_route(root, errors)
        check_multidataset_quality_evidence(root, errors)
        check_qwen_vllm_evidence(root, errors)
        check_mistral_batch_benchmark_evidence(root, errors)
        check_qwen_batch_benchmark_evidence(root, errors)
        check_launch_smoke_evidence(root, errors)
        check_model_template(root, errors)
        check_reuse_train_config(root, errors)
        check_patch_train_config(root, errors)
        check_selected_patch_train_config(root, errors)
        check_docs(root, errors)
    if args.run_root:
        check_run_root(Path(args.run_root), errors)

    if errors:
        print("release readiness check failed:", file=sys.stderr)
        for err in errors:
            print(f"- {err}", file=sys.stderr)
        raise SystemExit(1)
    print(f"release readiness check passed: {root}")


if __name__ == "__main__":
    main()
