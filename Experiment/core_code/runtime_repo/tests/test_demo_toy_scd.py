from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ToySCDDemoTest(unittest.TestCase):
    def test_toy_demo_writes_route_metrics_and_svg(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "toy_scd"
            subprocess.run(
                [
                    sys.executable,
                    "examples/demo_toy_scd.py",
                    "--out",
                    str(out),
                    "--samples",
                    "160",
                    "--layers",
                    "8",
                    "--dim",
                    "10",
                    "--patch-budget",
                    "3",
                ],
                cwd=ROOT,
                check=True,
                text=True,
                capture_output=True,
            )
            route = json.loads((out / "route.json").read_text(encoding="utf-8"))
            metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))

            self.assertEqual(route["num_layers"], 8)
            self.assertGreater(len(route["patch_layers"]), 0)
            self.assertEqual(
                sorted(route["reuse_layers"] + route["patch_layers"]),
                list(range(8)),
            )
            self.assertLess(metrics["scd_route"]["kv_mse"], metrics["reuse_only"]["kv_mse"])
            self.assertLess(metrics["scd_route"]["simulated_ttft_ms"], metrics["patch_only"]["simulated_ttft_ms"])
            self.assertTrue((out / "layer_errors.csv").is_file())
            self.assertTrue((out / "route_layers.svg").is_file())


if __name__ == "__main__":
    unittest.main()
