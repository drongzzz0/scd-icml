from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REQUIRED_RESULT_FILES = [
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

NEIGHBORING_BASELINE_FILES = [
    "neighboring_baselines.csv",
    "neighboring_baselines.md",
    "neighboring_baselines.tex",
    "neighboring_baselines_manifest.json",
]


def write_neighboring_baseline_files(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for name in NEIGHBORING_BASELINE_FILES:
        (path / name).write_text("artifact\n", encoding="utf-8")


class FinalizeFull7405BundleScriptTest(unittest.TestCase):
    def test_cli_builds_bundle_and_runs_paper_check(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            results = root / "results"
            profile = root / "profile"
            stack = root / "stack"
            cost = root / "cost"
            neighboring = root / "neighboring"
            paper = root / "paper"
            out = root / "bundle"
            for directory in [results, profile, stack, cost, neighboring, paper / "generated_figures"]:
                directory.mkdir(parents=True)

            for name in REQUIRED_RESULT_FILES:
                (results / name).write_text("artifact\n", encoding="utf-8")
            (results / "mixed_route_boundary_table.csv").write_text(
                "dataset,route_name,source\n"
                "hotpotqa_compressed,conditional_reuse_reuse08_patch24,"
                "/runs/scd_icml_full7405_postprocess_20260611_001/full7405_route_metrics.json\n",
                encoding="utf-8",
            )
            (root / "hotpotqa_full7405_f1_drop003_selection.json").write_text(
                json.dumps({"selected": {"reuse_n": 8, "patch_n": 24}}),
                encoding="utf-8",
            )
            for name in [
                "layer_error_profile.svg",
                "layer_error_profile.pdf",
                "layer_error_profile.png",
                "layer_gain_profile.svg",
                "layer_gain_profile.pdf",
                "layer_gain_profile.png",
                "patch_gain_curve.svg",
                "patch_gain_curve.pdf",
                "patch_gain_curve.png",
                "layer_profile.csv",
                "patch_gain_curve.csv",
                "patch_gain_curve.md",
                "analysis_figures_manifest.json",
            ]:
                (profile / name).write_text("artifact\n", encoding="utf-8")
            for name in ["stack_viz_current_tsne.svg", "stack_viz_current_tsne.pdf", "stack_viz_current_tsne.png"]:
                (stack / name).write_text("artifact\n", encoding="utf-8")
            for name in ["offline_cost_table.csv", "offline_cost_table.md", "offline_cost_summary.json", "offline_cost_wall_time.svg"]:
                (cost / name).write_text("artifact\n", encoding="utf-8")
            write_neighboring_baseline_files(neighboring)
            for name in [
                "layerwise_xnorm_replacement.pdf",
                "layer_selection_gain_profile.pdf",
                "patch_budget_curve.pdf",
                "stack_viz.pdf",
            ]:
                (paper / "generated_figures" / name).write_text("artifact\n", encoding="utf-8")
            tex = paper / "paper.tex"
            tex.write_text(
                r"""
\includegraphics{generated_figures/layerwise_xnorm_replacement.pdf}
\includegraphics{generated_figures/layer_selection_gain_profile.pdf}
\includegraphics{generated_figures/patch_budget_curve.pdf}
\includegraphics{generated_figures/stack_viz.pdf}
\label{tab:current_cmrc_selected_route}
\label{tab:current_reuse_boundary}
\label{tab:current_non_cmrc_diagnostics}
\label{tab:offline_cost}
\label{tab:neighboring_baselines}
\label{tab:patchonly_anchors}
The same-protocol CMRC2018 rerun restores the adapted neighboring table.
HotpotQA-compressed reports the full7405 selected route.
SCD Reuse08/Patch24 is the selected HotpotQA full7405 route.
""",
                encoding="utf-8",
            )

            completed = subprocess.run(
                [
                    sys.executable,
                    "scripts/finalize_scd_icml_full7405_bundle.py",
                    "--result-tables-dir",
                    str(results),
                    "--profile-figures-dir",
                    str(profile),
                    "--stack-viz-dir",
                    str(stack),
                    "--offline-cost-dir",
                    str(cost),
                    "--neighboring-baselines-dir",
                    str(neighboring),
                    "--out-dir",
                    str(out),
                    "--paper-tex",
                    str(tex),
                    "--paper-base-dir",
                    str(paper),
                ],
                cwd=Path(__file__).resolve().parents[1],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr + completed.stdout)
            self.assertTrue((out / "finalize_report.json").is_file())
            self.assertTrue((out / "paper_artifact_check_report.json").is_file())
            self.assertTrue((out / "tables" / "patch_gain_curve.md").is_file())
            self.assertTrue((out / "tables" / "neighboring_baselines.md").is_file())
            report = json.loads((out / "finalize_report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "ok")
            self.assertEqual(report["bundle_status"], "ok")
            self.assertTrue(report["paper_check"])
            self.assertEqual(report["result_table_check"]["rows"], 1)

    def test_cli_rejects_stale_full7405_pending_paper_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            results = root / "results"
            profile = root / "profile"
            stack = root / "stack"
            cost = root / "cost"
            neighboring = root / "neighboring"
            paper = root / "paper"
            out = root / "bundle"
            for directory in [results, profile, stack, cost, neighboring, paper / "generated_figures"]:
                directory.mkdir(parents=True)

            for name in REQUIRED_RESULT_FILES:
                (results / name).write_text("artifact\n", encoding="utf-8")
            (results / "mixed_route_boundary_table.csv").write_text(
                "dataset,route_name,source\n"
                "hotpotqa_compressed,conditional_reuse_reuse04_patch28,"
                "/runs/scd_icml_full7405_postprocess_20260611_001/full7405_route_metrics.json\n",
                encoding="utf-8",
            )
            (root / "hotpotqa_full7405_f1_drop003_selection.json").write_text(
                json.dumps({"selected": {"reuse_n": 4, "patch_n": 28}}),
                encoding="utf-8",
            )
            for name in [
                "layer_error_profile.svg",
                "layer_error_profile.pdf",
                "layer_error_profile.png",
                "layer_gain_profile.svg",
                "layer_gain_profile.pdf",
                "layer_gain_profile.png",
                "patch_gain_curve.svg",
                "patch_gain_curve.pdf",
                "patch_gain_curve.png",
                "layer_profile.csv",
                "patch_gain_curve.csv",
                "patch_gain_curve.md",
                "analysis_figures_manifest.json",
            ]:
                (profile / name).write_text("artifact\n", encoding="utf-8")
            for name in ["stack_viz_current_tsne.svg", "stack_viz_current_tsne.pdf", "stack_viz_current_tsne.png"]:
                (stack / name).write_text("artifact\n", encoding="utf-8")
            for name in ["offline_cost_table.csv", "offline_cost_table.md", "offline_cost_summary.json", "offline_cost_wall_time.svg"]:
                (cost / name).write_text("artifact\n", encoding="utf-8")
            write_neighboring_baseline_files(neighboring)
            for name in [
                "layerwise_xnorm_replacement.pdf",
                "layer_selection_gain_profile.pdf",
                "patch_budget_curve.pdf",
                "stack_viz.pdf",
            ]:
                (paper / "generated_figures" / name).write_text("artifact\n", encoding="utf-8")
            tex = paper / "paper.tex"
            tex.write_text(
                r"""
\includegraphics{generated_figures/layerwise_xnorm_replacement.pdf}
\includegraphics{generated_figures/layer_selection_gain_profile.pdf}
\includegraphics{generated_figures/patch_budget_curve.pdf}
\includegraphics{generated_figures/stack_viz.pdf}
\label{tab:current_cmrc_selected_route}
\label{tab:current_reuse_boundary}
\label{tab:current_non_cmrc_diagnostics}
\label{tab:offline_cost}
\label{tab:neighboring_baselines}
\label{tab:patchonly_anchors}
The same-protocol CMRC2018 rerun restores the adapted neighboring table.
HotpotQA-compressed reports the full7405 selected route.
SCD Reuse04/Patch28 is the selected HotpotQA full7405 route.
The full7405 remains pending.
""",
                encoding="utf-8",
            )

            completed = subprocess.run(
                [
                    sys.executable,
                    "scripts/finalize_scd_icml_full7405_bundle.py",
                    "--result-tables-dir",
                    str(results),
                    "--profile-figures-dir",
                    str(profile),
                    "--stack-viz-dir",
                    str(stack),
                    "--offline-cost-dir",
                    str(cost),
                    "--neighboring-baselines-dir",
                    str(neighboring),
                    "--out-dir",
                    str(out),
                    "--paper-tex",
                    str(tex),
                    "--paper-base-dir",
                    str(paper),
                ],
                cwd=Path(__file__).resolve().parents[1],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("forbidden text present", completed.stderr + completed.stdout)

    def test_cli_rejects_paper_route_text_mismatch_when_selector_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            results = root / "results"
            profile = root / "profile"
            stack = root / "stack"
            cost = root / "cost"
            neighboring = root / "neighboring"
            paper = root / "paper"
            out = root / "bundle"
            for directory in [results, profile, stack, cost, neighboring, paper / "generated_figures"]:
                directory.mkdir(parents=True)

            for name in REQUIRED_RESULT_FILES:
                (results / name).write_text("artifact\n", encoding="utf-8")
            (results / "mixed_route_boundary_table.csv").write_text(
                "dataset,route_name,source\n"
                "hotpotqa_compressed,conditional_reuse_reuse05_patch27,"
                "/runs/scd_icml_full7405_boundary_postprocess_20260611_001/full7405_route_metrics.json\n",
                encoding="utf-8",
            )
            (root / "hotpotqa_full7405_f1_drop003_selection.json").write_text(
                json.dumps({"selected": {"reuse_n": 5, "patch_n": 27}}),
                encoding="utf-8",
            )
            for name in [
                "layer_error_profile.svg",
                "layer_error_profile.pdf",
                "layer_error_profile.png",
                "layer_gain_profile.svg",
                "layer_gain_profile.pdf",
                "layer_gain_profile.png",
                "patch_gain_curve.svg",
                "patch_gain_curve.pdf",
                "patch_gain_curve.png",
                "layer_profile.csv",
                "patch_gain_curve.csv",
                "patch_gain_curve.md",
                "analysis_figures_manifest.json",
            ]:
                (profile / name).write_text("artifact\n", encoding="utf-8")
            for name in ["stack_viz_current_tsne.svg", "stack_viz_current_tsne.pdf", "stack_viz_current_tsne.png"]:
                (stack / name).write_text("artifact\n", encoding="utf-8")
            for name in ["offline_cost_table.csv", "offline_cost_table.md", "offline_cost_summary.json", "offline_cost_wall_time.svg"]:
                (cost / name).write_text("artifact\n", encoding="utf-8")
            write_neighboring_baseline_files(neighboring)
            for name in [
                "layerwise_xnorm_replacement.pdf",
                "layer_selection_gain_profile.pdf",
                "patch_budget_curve.pdf",
                "stack_viz.pdf",
            ]:
                (paper / "generated_figures" / name).write_text("artifact\n", encoding="utf-8")
            tex = paper / "paper.tex"
            tex.write_text(
                r"""
\includegraphics{generated_figures/layerwise_xnorm_replacement.pdf}
\includegraphics{generated_figures/layer_selection_gain_profile.pdf}
\includegraphics{generated_figures/patch_budget_curve.pdf}
\includegraphics{generated_figures/stack_viz.pdf}
\label{tab:current_cmrc_selected_route}
\label{tab:current_reuse_boundary}
\label{tab:current_non_cmrc_diagnostics}
\label{tab:offline_cost}
\label{tab:neighboring_baselines}
\label{tab:patchonly_anchors}
The same-protocol CMRC2018 rerun restores the adapted neighboring table.
HotpotQA-compressed reports the full7405 selected route.
SCD Reuse04/Patch28 is stale if the selector chooses a higher route.
""",
                encoding="utf-8",
            )

            completed = subprocess.run(
                [
                    sys.executable,
                    "scripts/finalize_scd_icml_full7405_bundle.py",
                    "--result-tables-dir",
                    str(results),
                    "--profile-figures-dir",
                    str(profile),
                    "--stack-viz-dir",
                    str(stack),
                    "--offline-cost-dir",
                    str(cost),
                    "--neighboring-baselines-dir",
                    str(neighboring),
                    "--out-dir",
                    str(out),
                    "--paper-tex",
                    str(tex),
                    "--paper-base-dir",
                    str(paper),
                ],
                cwd=Path(__file__).resolve().parents[1],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("required text missing: Reuse05/Patch27", completed.stderr + completed.stdout)

    def test_cli_rejects_pre_full7405_result_table_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            results = root / "results"
            profile = root / "profile"
            stack = root / "stack"
            cost = root / "cost"
            neighboring = root / "neighboring"
            out = root / "bundle"
            for directory in [results, profile, stack, cost, neighboring]:
                directory.mkdir(parents=True)
            for name in REQUIRED_RESULT_FILES:
                (results / name).write_text("artifact\n", encoding="utf-8")
            (results / "mixed_route_boundary_table.csv").write_text(
                "dataset,route_name,source\n"
                "hotpotqa_compressed,conditional_reuse_reuse08_patch24,"
                "/runs/scd_icml_boundary_hotpotqa_compressed_mid200_20260611_001/route_metrics.json\n",
                encoding="utf-8",
            )

            completed = subprocess.run(
                [
                    sys.executable,
                    "scripts/finalize_scd_icml_full7405_bundle.py",
                    "--result-tables-dir",
                    str(results),
                    "--profile-figures-dir",
                    str(profile),
                    "--stack-viz-dir",
                    str(stack),
                    "--offline-cost-dir",
                    str(cost),
                    "--neighboring-baselines-dir",
                    str(neighboring),
                    "--out-dir",
                    str(out),
                    "--skip-paper-check",
                    "--allow-missing",
                ],
                cwd=Path(__file__).resolve().parents[1],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("full7405", completed.stderr + completed.stdout)


if __name__ == "__main__":
    unittest.main()
