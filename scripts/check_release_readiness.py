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
    "configs/train_reuse_mistral_self_layerwise_20260404.json",
    "configs/mistral_self_patch_xnorm_nativekv_shared_residual_mlp_paper_k6_e1_h2048_r512_b2_20260404.json",
    "configs/mistral_self_patch_xnorm_nativekv_shared_residual_mlp_alllayers_e1_h2048_r512_b2_20260404.json",
    "configs/mistral_self_patch_xnorm_nativekv_residual_mlp_parallel_e1_h2048_b2_20260404.json",
    "configs/mistral_self_patch_xnorm_nativekv_linear_parallel_20260404.json",
    "configs/route_selection.yaml",
    "configs/route_reuse_patchonly_all32_20260404.json",
    "configs/routes/scd_icml_mistrallite_to_mistral7_reuse19_patch13_cmrc200.json",
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
    "src/scd_repro/text_benchmarks.py",
    "src/scd_repro/adapters.py",
    "src/scd_repro/direct_kv_residual_patch.py",
    "src/scd_repro/strict_runtime.py",
    "src/scd_repro/cli/capture_pair.py",
    "src/scd_repro/cli/eval_text_strict.py",
    "src/scd_repro/cli/summarize_mistral_repro.py",
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
]

PY_COMPILE_FILES = [
    "src/scd_repro/text_benchmarks.py",
    "src/scd_repro/adapters.py",
    "src/scd_repro/direct_kv_residual_patch.py",
    "src/scd_repro/strict_runtime.py",
    "src/scd_repro/cli/capture_pair.py",
    "src/scd_repro/cli/eval_text_strict.py",
    "src/scd_repro/cli/summarize_mistral_repro.py",
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
    for base_rel in ("src", "scripts"):
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
    local_roots = {"scd_repro", "scd_vllm", "scd_vllm_plugin"}
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
        "OPEN_SOURCE_RELEASE_MANIFEST.md": ["check_release_readiness.py", "check_release_smoke_status.py", "make_paper_scd_route.py", "make_paper_scd_candidate_routes.py", "select_quality_gated_scd_route.py", "select_patch_reuse_route.py", "collect_route_eval_metrics.py", "filter_strict_eval_by_qids.py", "build_scd_icml_result_tables.py", "build_scd_icml_analysis_figures.py", "extract_stack_viz_inputs.py", "build_stack_viz.py", "build_offline_cost_table.py", "build_paper_artifact_bundle.py", "finalize_scd_icml_full7405_bundle.py", "run_ser_neighboring_baselines_cmrc.sh", "run_svdllm_style_kv_cmrc.py", "run_slicegpt_style_kv_cmrc.py", "run_tensorllm_style_kv_cmrc.py", "build_neighboring_baseline_table.py", "wait_postprocess_scd_icml_full7405_tables.sh", "wait_finalize_scd_icml_full7405_bundle.sh", "check_paper_artifacts.py", "report_pairwise_interactions.py", "validate_scd_route.py", "build_release_tree.py", "tests/", "15 rows", "lite_to_7b", "bandwidth_sensitivity"],
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
