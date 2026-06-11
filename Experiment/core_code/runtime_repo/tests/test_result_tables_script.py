from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class ResultTablesScriptTest(unittest.TestCase):
    def test_cli_writes_latex_tables(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            patch_json = root / "patch.json"
            route_json = root / "routes.json"
            out_dir = root / "out"
            patch_json.write_text(
                json.dumps(
                    [
                        {
                            "dataset": "cmrc2018",
                            "method": "patch_only",
                            "f1": 0.7,
                            "em": 0.4,
                            "avg_nll": 0.5,
                            "ppl": 1.6,
                            "ttft_ms": 42.0,
                            "payload_mb": 12.0,
                        },
                        {
                            "dataset": "cmrc2018",
                            "method": "reuse_only",
                            "f1": 0.66,
                            "em": 0.36,
                            "avg_nll": 0.7,
                            "ppl": 2.0,
                            "ttft_ms": 30.0,
                            "payload_mb": 6.0,
                        }
                    ]
                ),
                encoding="utf-8",
            )
            route_json.write_text(
                json.dumps(
                    {
                        "rows": [
                            {
                                "dataset": "cmrc2018",
                                "route_name": "reuse_01_patch_31",
                                "status": "ok",
                                "patch_layers": list(range(31)),
                                "reuse_layers": [31],
                                "full_f1": 0.7,
                                "f1": 0.68,
                                "f1_retention": 0.97,
                                "full_em": 0.4,
                                "em": 0.39,
                                "ppl": 1.8,
                                "ppl_delta": 0.2,
                                "full_avg_nll": 0.5,
                                "avg_nll": 0.6,
                                "avg_nll_delta": 0.1,
                                "full_ttft_ms": 100.0,
                                "ttft_ms": 40.0,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            subprocess.run(
                [
                    sys.executable,
                    "scripts/build_scd_icml_result_tables.py",
                    "--patch-anchor-json",
                    str(patch_json),
                    "--route-metrics-json",
                    str(route_json),
                    "--out-dir",
                    str(out_dir),
                ],
                cwd=Path(__file__).resolve().parents[1],
                check=True,
            )
            patch_tex = (out_dir / "patch_anchor_table.tex").read_text(encoding="utf-8")
            route_tex = (out_dir / "mixed_route_boundary_table.tex").read_text(encoding="utf-8")
            ttft_tex = (out_dir / "ttft_breakdown_table.tex").read_text(encoding="utf-8")
            self.assertIn(r"\begin{tabular}", patch_tex)
            self.assertIn(r"patch\_only", patch_tex)
            self.assertIn(r"reuse\_01\_patch\_31", route_tex)
            self.assertIn(r"reuse\_01\_patch\_31", ttft_tex)
            self.assertTrue((out_dir / "latency_quality_pareto.csv").is_file())
            self.assertTrue((out_dir / "latency_quality_pareto_cmrc2018.svg").is_file())
            latency_csv = (out_dir / "latency_quality_pareto.csv").read_text(encoding="utf-8")
            self.assertIn("payload_bytes", latency_csv)
            self.assertIn("estimated_from_patch_reuse_anchor_per_layer", latency_csv)
            bandwidth_csv = (out_dir / "bandwidth_sensitivity.csv").read_text(encoding="utf-8")
            self.assertIn("10.000", bandwidth_csv)
            self.assertIn("payload_bytes", bandwidth_csv)
            self.assertIn("reuse_01_patch_31", bandwidth_csv)
            self.assertTrue((out_dir / "bandwidth_sensitivity.svg").is_file())
            manifest = json.loads((out_dir / "scd_icml_result_tables_manifest.json").read_text(encoding="utf-8"))
            self.assertIn("mixed_route_boundary_table.tex", manifest["outputs"])
            self.assertIn("ttft_breakdown_table.tex", manifest["outputs"])
            self.assertIn("latency_quality_pareto_cmrc2018.svg", manifest["outputs"])
            self.assertIn("bandwidth_sensitivity.svg", manifest["outputs"])


if __name__ == "__main__":
    unittest.main()
