from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")

from scripts import build_scd_icml_analysis_figures


class AnalysisFiguresScriptTest(unittest.TestCase):
    def test_builds_profile_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            profile = root / "profile.json"
            out_dir = root / "out"
            profile.write_text(
                json.dumps(
                    {
                        "num_layers": 3,
                        "recommended_patch_layers": [1],
                        "layers": [
                            {"layer": 0, "reuse_error": 0.5, "patch_error": 0.4, "patch_kv_error": 0.4},
                            {"layer": 1, "reuse_error": 0.6, "patch_error": 0.2, "patch_kv_error": 0.2},
                            {"layer": 2, "reuse_error": 0.3, "patch_error": 0.25, "patch_kv_error": 0.25},
                        ],
                        "patch_gain_curve": [0.4, 0.5],
                    }
                ),
                encoding="utf-8",
            )
            rc = build_scd_icml_analysis_figures.main(
                ["--profile", str(profile), "--out-dir", str(out_dir), "--title-prefix", "test"]
            )
            self.assertEqual(rc, 0)
            for name in [
                "layer_profile.csv",
                "layer_profile.md",
                "layer_error_profile.svg",
                "layer_error_profile.pdf",
                "layer_error_profile.png",
                "layer_gain_profile.svg",
                "layer_gain_profile.pdf",
                "layer_gain_profile.png",
                "patch_gain_curve.csv",
                "patch_gain_curve.md",
                "patch_gain_curve.svg",
                "patch_gain_curve.pdf",
                "patch_gain_curve.png",
                "analysis_figures_manifest.json",
            ]:
                self.assertTrue((out_dir / name).is_file(), name)
            self.assertIn("0.400000", (out_dir / "layer_profile.csv").read_text(encoding="utf-8"))
            self.assertIn("<svg", (out_dir / "layer_gain_profile.svg").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
