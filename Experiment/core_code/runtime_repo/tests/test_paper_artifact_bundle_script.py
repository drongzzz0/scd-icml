from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class PaperArtifactBundleScriptTest(unittest.TestCase):
    def test_cli_copies_artifacts_and_writes_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result_dir = root / "results"
            profile_dir = root / "profile"
            stack_dir = root / "stack"
            cost_dir = root / "cost"
            neighboring_dir = root / "neighboring"
            out_dir = root / "bundle"
            for directory in [result_dir, profile_dir, stack_dir, cost_dir, neighboring_dir]:
                directory.mkdir(parents=True)

            for name in [
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
            ]:
                (result_dir / name).write_text("artifact\n", encoding="utf-8")
            (profile_dir / "layer_error_profile.svg").write_text("<svg />\n", encoding="utf-8")
            (profile_dir / "layer_error_profile.pdf").write_text("%PDF\n", encoding="utf-8")
            (profile_dir / "layer_error_profile.png").write_text("png\n", encoding="utf-8")
            (profile_dir / "layer_gain_profile.svg").write_text("<svg />\n", encoding="utf-8")
            (profile_dir / "layer_gain_profile.pdf").write_text("%PDF\n", encoding="utf-8")
            (profile_dir / "layer_gain_profile.png").write_text("png\n", encoding="utf-8")
            (profile_dir / "patch_gain_curve.svg").write_text("<svg />\n", encoding="utf-8")
            (profile_dir / "patch_gain_curve.pdf").write_text("%PDF\n", encoding="utf-8")
            (profile_dir / "patch_gain_curve.png").write_text("png\n", encoding="utf-8")
            (profile_dir / "patch_gain_curve.csv").write_text("patch_budget,cumulative_gain\n", encoding="utf-8")
            (profile_dir / "patch_gain_curve.md").write_text("# Patch Gain Curve\n", encoding="utf-8")
            (stack_dir / "stack_viz_current_tsne.svg").write_text("<svg />\n", encoding="utf-8")
            (cost_dir / "offline_cost_table.csv").write_text("label,seconds\n", encoding="utf-8")
            for name in [
                "neighboring_baselines.csv",
                "neighboring_baselines.md",
                "neighboring_baselines.tex",
                "neighboring_baselines_manifest.json",
            ]:
                (neighboring_dir / name).write_text("artifact\n", encoding="utf-8")

            subprocess.run(
                [
                    sys.executable,
                    "scripts/build_paper_artifact_bundle.py",
                    "--result-tables-dir",
                    str(result_dir),
                    "--profile-figures-dir",
                    str(profile_dir),
                    "--stack-viz-dir",
                    str(stack_dir),
                    "--offline-cost-dir",
                    str(cost_dir),
                    "--neighboring-baselines-dir",
                    str(neighboring_dir),
                    "--out-dir",
                    str(out_dir),
                    "--allow-missing",
                ],
                cwd=Path(__file__).resolve().parents[1],
                check=True,
            )
            self.assertTrue((out_dir / "tables" / "patch_anchor_table.tex").is_file())
            self.assertTrue((out_dir / "figures" / "layerwise_xnorm_replacement.svg").is_file())
            self.assertTrue((out_dir / "figures" / "layerwise_xnorm_replacement.pdf").is_file())
            self.assertTrue((out_dir / "figures" / "stack_viz.svg").is_file())
            self.assertTrue((out_dir / "tables" / "patch_gain_curve.md").is_file())
            self.assertTrue((out_dir / "tables" / "neighboring_baselines.md").is_file())
            self.assertTrue((out_dir / "manifests" / "neighboring_baselines_manifest.json").is_file())
            manifest = json.loads((out_dir / "paper_artifact_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "ok")
            self.assertGreaterEqual(len(manifest["copied"]), 20)

    def test_cli_fails_when_required_result_table_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result_dir = root / "results"
            out_dir = root / "bundle"
            result_dir.mkdir()
            completed = subprocess.run(
                [
                    sys.executable,
                    "scripts/build_paper_artifact_bundle.py",
                    "--result-tables-dir",
                    str(result_dir),
                    "--out-dir",
                    str(out_dir),
                ],
                cwd=Path(__file__).resolve().parents[1],
                check=False,
            )
            self.assertEqual(completed.returncode, 2)
            manifest = json.loads((out_dir / "paper_artifact_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "missing_required")


if __name__ == "__main__":
    unittest.main()
