from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


def write_summary(path: Path, method: str, f1: float, ttft_ms: float, meta: dict | None = None) -> None:
    path.write_text(
        json.dumps(
            {
                "meta": meta or {},
                "summary": [
                    {
                        "method": "full_consumer",
                        "method_label": "Full Consumer",
                        "f1": 0.8,
                        "em": 0.5,
                        "ttft_ms": 200.0,
                        "avg_nll": 0.1,
                        "bytes_transferred": 0,
                        "compression_ratio": None,
                    },
                    {
                        "method": method,
                        "method_label": method,
                        "f1": f1,
                        "em": 0.4,
                        "ttft_ms": ttft_ms,
                        "avg_nll": 0.2,
                        "bytes_transferred": 123,
                        "compression_ratio": 0.5,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )


class NeighboringBaselineTableScriptTest(unittest.TestCase):
    def test_cli_writes_neighboring_baseline_table(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            svd = root / "svd.json"
            slicegpt = root / "slicegpt.json"
            tensorllm = root / "tensorllm.json"
            scd = root / "scd.json"
            out_dir = root / "out"
            write_summary(svd, "svdllm_style_kv", 0.56, 240.0)
            write_summary(slicegpt, "slicegpt_style_kv", 0.68, 210.0)
            write_summary(tensorllm, "tensorllm_style_kv", 0.0, 1250.0)
            write_summary(scd, "scd_full", 0.76, 100.0)

            subprocess.run(
                [
                    sys.executable,
                    "scripts/build_neighboring_baseline_table.py",
                    "--svdllm-summary",
                    str(svd),
                    "--slicegpt-summary",
                    str(slicegpt),
                    "--tensorllm-summary",
                    str(tensorllm),
                    "--scd-summary",
                    str(scd),
                    "--out-dir",
                    str(out_dir),
                ],
                cwd=Path(__file__).resolve().parents[1],
                check=True,
            )

            md = (out_dir / "neighboring_baselines.md").read_text(encoding="utf-8")
            tex = (out_dir / "neighboring_baselines.tex").read_text(encoding="utf-8")
            csv_text = (out_dir / "neighboring_baselines.csv").read_text(encoding="utf-8")
            manifest = json.loads((out_dir / "neighboring_baselines_manifest.json").read_text(encoding="utf-8"))

            self.assertIn("SVD-LLM-style adaptation", md)
            self.assertIn("70.0%", md)
            self.assertIn("0.83x", md)
            self.assertIn("SliceGPT-style adapted KV transfer", csv_text)
            self.assertIn(r"\begin{tabular}", tex)
            self.assertEqual(manifest["status"], "ok")
            self.assertEqual(len(manifest["rows"]), 4)

    def test_cli_marks_protocol_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            svd = root / "svd.json"
            slicegpt = root / "slicegpt.json"
            tensorllm = root / "tensorllm.json"
            scd = root / "scd.json"
            out_dir = root / "out"
            common_meta = {"limit": 200, "max_length": 256, "max_new_tokens": 32}
            write_summary(svd, "svdllm_style_kv", 0.56, 240.0, common_meta)
            write_summary(slicegpt, "slicegpt_style_kv", 0.68, 210.0, common_meta)
            write_summary(tensorllm, "tensorllm_style_kv", 0.0, 1250.0, common_meta)
            write_summary(scd, "scd_full", 0.76, 100.0, {"limit": 200, "max_length": 1024, "max_new_tokens": 32})

            subprocess.run(
                [
                    sys.executable,
                    "scripts/build_neighboring_baseline_table.py",
                    "--svdllm-summary",
                    str(svd),
                    "--slicegpt-summary",
                    str(slicegpt),
                    "--tensorllm-summary",
                    str(tensorllm),
                    "--scd-summary",
                    str(scd),
                    "--out-dir",
                    str(out_dir),
                ],
                cwd=Path(__file__).resolve().parents[1],
                check=True,
            )

            md = (out_dir / "neighboring_baselines.md").read_text(encoding="utf-8")
            manifest = json.loads((out_dir / "neighboring_baselines_manifest.json").read_text(encoding="utf-8"))

            self.assertIn("Protocol warning", md)
            self.assertTrue(any("max_length" in warning for warning in manifest["warnings"]))
            self.assertEqual(manifest["protocols"]["scd"]["max_length"], 1024)

    def test_cli_can_require_protocol_compatibility(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            svd = root / "svd.json"
            slicegpt = root / "slicegpt.json"
            tensorllm = root / "tensorllm.json"
            scd = root / "scd.json"
            out_dir = root / "out"
            common_meta = {"limit": 200, "max_length": 256, "max_new_tokens": 32}
            write_summary(svd, "svdllm_style_kv", 0.56, 240.0, common_meta)
            write_summary(slicegpt, "slicegpt_style_kv", 0.68, 210.0, common_meta)
            write_summary(tensorllm, "tensorllm_style_kv", 0.0, 1250.0, common_meta)
            write_summary(scd, "scd_full", 0.76, 100.0, {"limit": 200, "max_length": 1024, "max_new_tokens": 32})

            proc = subprocess.run(
                [
                    sys.executable,
                    "scripts/build_neighboring_baseline_table.py",
                    "--svdllm-summary",
                    str(svd),
                    "--slicegpt-summary",
                    str(slicegpt),
                    "--tensorllm-summary",
                    str(tensorllm),
                    "--scd-summary",
                    str(scd),
                    "--require-compatible-protocol",
                    "--out-dir",
                    str(out_dir),
                ],
                cwd=Path(__file__).resolve().parents[1],
                text=True,
                capture_output=True,
            )

            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("incompatible neighboring-baseline protocols", proc.stderr + proc.stdout)
            self.assertFalse((out_dir / "neighboring_baselines.md").exists())


if __name__ == "__main__":
    unittest.main()
