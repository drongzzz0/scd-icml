from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _sample(qid: str, method: str, f1: float, nll: float, tokens: int, ttft: float) -> dict[str, object]:
    return {
        "qid": qid,
        "method": method,
        "pred_text": "x",
        "gold_answers": ["x"],
        "f1": f1,
        "em": f1,
        "avg_logp": -nll,
        "avg_nll": nll,
        "teacher_token_count": tokens,
        "payload_mb": 1.0 if method != "full_consumer" else 0.0,
        "prompt_tokens": 10,
        "prefix_tokens": 9,
        "first_token_id": 1,
        "bytes_transferred": 1024 if method != "full_consumer" else 0,
        "raw_cache_bytes": 4096,
        "compression_ratio": 4.0 if method != "full_consumer" else None,
        "timing_ms": {
            "ttft": ttft,
            "total": ttft + 5.0,
        },
    }


class FilterStrictEvalByQidsScriptTest(unittest.TestCase):
    def test_cli_excludes_qids_and_writes_route_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            eval_dir = root / "eval"
            eval_dir.mkdir()
            rows = [
                _sample("keep", "full_consumer", 1.0, 1.0, 2, 100.0),
                _sample("drop", "full_consumer", 0.0, 5.0, 1, 200.0),
                _sample("keep", "scd_full", 0.5, 2.0, 2, 40.0),
                _sample("drop", "scd_full", 0.0, 6.0, 1, 60.0),
            ]
            with (eval_dir / "per_sample_metrics.jsonl").open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row) + "\n")
            exclude = root / "exclude.txt"
            exclude.write_text("drop\n", encoding="utf-8")
            out = root / "filtered"
            route_summary = root / "scd_icml_mistral_summary.json"

            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/filter_strict_eval_by_qids.py"),
                    "--strict-eval-dir",
                    str(eval_dir),
                    "--out-dir",
                    str(out),
                    "--exclude-qid-file",
                    str(exclude),
                    "--summary-list-out",
                    str(route_summary),
                    "--dataset",
                    "hotpotqa_compressed",
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr + completed.stdout)
            summary = json.loads((out / "per_mode_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["full_consumer"]["num_samples"], 1)
            self.assertEqual(summary["scd_full"]["num_samples"], 1)
            self.assertAlmostEqual(summary["full_consumer"]["f1"], 1.0)
            self.assertAlmostEqual(summary["scd_full"]["f1"], 0.5)
            self.assertAlmostEqual(summary["full_consumer"]["avg_nll"], 1.0)
            self.assertAlmostEqual(summary["scd_full"]["avg_nll"], 2.0)
            route_rows = json.loads(route_summary.read_text(encoding="utf-8"))
            scd = [row for row in route_rows if row["method"] == "scd_full"][0]
            self.assertEqual(scd["dataset"], "hotpotqa_compressed")
            self.assertAlmostEqual(scd["ttft_ms"], 40.0)


if __name__ == "__main__":
    unittest.main()
