from __future__ import annotations

import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.build_offline_cost_table import parse_elapsed_seconds, parse_time_file


class OfflineCostTableScriptTest(unittest.TestCase):
    def test_parse_elapsed_seconds(self) -> None:
        self.assertAlmostEqual(parse_elapsed_seconds("12:35.66") or 0.0, 755.66)
        self.assertAlmostEqual(parse_elapsed_seconds("1:02:03") or 0.0, 3723.0)
        self.assertIsNone(parse_elapsed_seconds("bad"))

    def test_parse_time_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "time.txt"
            path.write_text(
                "\n".join(
                    [
                        "User time (seconds): 10.50",
                        "System time (seconds): 2.25",
                        "Elapsed (wall clock) time (h:mm:ss or m:ss): 1:02.50",
                        "Maximum resident set size (kbytes): 1048576",
                        "Exit status: 0",
                    ]
                ),
                encoding="utf-8",
            )
            row = parse_time_file(path)
            self.assertEqual(row["status"], "ok")
            self.assertAlmostEqual(row["elapsed_seconds"], 62.5)
            self.assertAlmostEqual(row["max_rss_gib"], 1.0)

    def test_cli_writes_outputs_and_marks_bad_log(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            good = root / "good.txt"
            bad = root / "bad.txt"
            out_dir = root / "out"
            good.write_text(
                "\n".join(
                    [
                        "User time (seconds): 3.00",
                        "System time (seconds): 1.00",
                        "Elapsed (wall clock) time (h:mm:ss or m:ss): 0:04.00",
                        "Maximum resident set size (kbytes): 2097152",
                        "Exit status: 0",
                    ]
                ),
                encoding="utf-8",
            )
            bad.write_text("not a time log\n", encoding="utf-8")
            subprocess.run(
                [
                    sys.executable,
                    "scripts/build_offline_cost_table.py",
                    "--entry",
                    f"good={good}",
                    "--entry",
                    f"bad={bad}",
                    "--out-dir",
                    str(out_dir),
                ],
                cwd=Path(__file__).resolve().parents[1],
                check=True,
            )
            with (out_dir / "offline_cost_table.csv").open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["status"], "ok")
            self.assertEqual(rows[1]["status"], "no_time_fields")
            self.assertTrue((out_dir / "offline_cost_wall_time.svg").exists())


if __name__ == "__main__":
    unittest.main()
