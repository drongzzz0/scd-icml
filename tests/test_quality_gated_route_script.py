from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "select_quality_gated_scd_route.py"


def _write_eval_root(base: Path, name: str, reuse_n: int, scd_ppl: float) -> Path:
    root = base / name
    route_dir = root / "lite_to_7b" / "route"
    route_dir.mkdir(parents=True)
    patch_n = 32 - reuse_n
    route = {
        "patch_layers": list(range(patch_n)),
        "reuse_layers": list(range(patch_n, 32)),
        "route_semantics": "paper_scd_sparse_patch_reuse_all_other_layers",
    }
    (route_dir / "route_cmrc2018_seed42.json").write_text(
        json.dumps(route),
        encoding="utf-8",
    )
    summary = [
        {
            "direction": "lite_to_7b",
            "dataset": "wikitext2",
            "method": "full_consumer",
            "ppl": 7.584029,
            "avg_nll": 2.026045,
            "ttft_ms": 62.0,
        },
        {
            "direction": "lite_to_7b",
            "dataset": "wikitext2",
            "method": "scd_full",
            "ppl": scd_ppl,
            "avg_nll": 2.12,
            "ttft_ms": 31.0,
        },
    ]
    (root / "scd_icml_mistral_summary.json").write_text(
        json.dumps(summary),
        encoding="utf-8",
    )
    return root


def _write_qa_eval_root(base: Path, name: str, reuse_n: int, scd_f1: float, scd_em: float) -> Path:
    root = base / name
    route_dir = root / "lite_to_7b" / "route"
    route_dir.mkdir(parents=True)
    patch_n = 32 - reuse_n
    route = {
        "patch_layers": list(range(patch_n)),
        "reuse_layers": list(range(patch_n, 32)),
        "route_semantics": "paper_scd_sparse_patch_reuse_all_other_layers",
    }
    (route_dir / "route_cmrc2018_seed42.json").write_text(
        json.dumps(route),
        encoding="utf-8",
    )
    summary = [
        {
            "direction": "lite_to_7b",
            "dataset": "hotpotqa_compressed",
            "method": "full_consumer",
            "f1": 0.330551810560885,
            "em": 0.25,
            "avg_nll": 1.615902,
            "ttft_ms": 80.0,
        },
        {
            "direction": "lite_to_7b",
            "dataset": "hotpotqa_compressed",
            "method": "scd_full",
            "f1": scd_f1,
            "em": scd_em,
            "avg_nll": 2.0,
            "ttft_ms": 40.0,
        },
    ]
    (root / "scd_icml_mistral_summary.json").write_text(
        json.dumps(summary),
        encoding="utf-8",
    )
    return root


class QualityGatedRouteScriptTest(unittest.TestCase):
    def test_ppl_gate_selects_max_reuse_within_relative_increase(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            _write_eval_root(tmp_dir, "reuse20", reuse_n=20, scd_ppl=8.329555)
            _write_eval_root(tmp_dir, "reuse21", reuse_n=21, scd_ppl=8.700000)
            out_report = tmp_dir / "report.json"
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--eval-glob",
                    str(tmp_dir / "reuse*"),
                    "--dataset",
                    "wikitext2",
                    "--metric",
                    "ppl",
                    "--max-ppl-relative-increase",
                    "0.10",
                    "--out-report",
                    str(out_report),
                ],
                cwd=ROOT,
                check=True,
                text=True,
                capture_output=True,
            )
            self.assertIn("selected_route=", result.stdout)
            report = json.loads(out_report.read_text(encoding="utf-8"))
            self.assertEqual(report["selected"]["reuse_n"], 20)
            candidates = {Path(row["eval_root"]).name: row for row in report["candidates"]}
            self.assertTrue(candidates["reuse20"]["passes_ppl_relative_increase"])
            self.assertFalse(candidates["reuse21"]["passes_ppl_relative_increase"])

    def test_absolute_metric_drop_gate_handles_low_f1_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            _write_qa_eval_root(tmp_dir, "reuse08", reuse_n=8, scd_f1=0.3054495736199803, scd_em=0.235)
            _write_qa_eval_root(tmp_dir, "reuse09", reuse_n=9, scd_f1=0.2493376031830099, scd_em=0.190)
            out_report = tmp_dir / "report.json"
            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--eval-glob",
                    str(tmp_dir / "reuse*"),
                    "--dataset",
                    "hotpotqa_compressed",
                    "--metric",
                    "f1",
                    "--max-metric-drop",
                    "0.03",
                    "--out-report",
                    str(out_report),
                ],
                cwd=ROOT,
                check=True,
                text=True,
                capture_output=True,
            )
            report = json.loads(out_report.read_text(encoding="utf-8"))
            self.assertEqual(report["selected"]["reuse_n"], 8)
            candidates = {Path(row["eval_root"]).name: row for row in report["candidates"]}
            self.assertTrue(candidates["reuse08"]["passes_primary_metric"])
            self.assertFalse(candidates["reuse09"]["passes_primary_metric"])


if __name__ == "__main__":
    unittest.main()
