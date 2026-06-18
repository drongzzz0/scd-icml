import json
import tempfile
import unittest
from pathlib import Path

from scd_repro.cmrc import load_cmrc2018_samples
from scd_repro.cli.eval_cmrc_strict import _resolve_load_limit as resolve_modern_cmrc_load_limit
from scd_repro.cli_eval_cmrc_strict import _resolve_load_limit as resolve_legacy_cmrc_load_limit
from scd_repro.text_benchmarks import load_text_benchmark_samples


def _write_cmrc(path: Path, qid: str, context: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "data": [
            {
                "paragraphs": [
                    {
                        "context": context,
                        "qas": [
                            {
                                "id": qid,
                                "question": "question",
                                "answers": [{"text": "answer"}],
                            }
                        ],
                    }
                ]
            }
        ]
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


class CMRCSplitTests(unittest.TestCase):
    def test_default_split_is_dev(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_cmrc(root / "cmrc2018" / "cmrc2018_dev.json", "dev_qid", "dev context")
            _write_cmrc(root / "cmrc2018" / "cmrc2018_train.json", "train_qid", "train context")

            rows = load_cmrc2018_samples(root, limit=1)

            self.assertEqual(rows[0].qid, "dev_qid")

    def test_train_split_is_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_cmrc(root / "cmrc2018" / "cmrc2018_dev.json", "dev_qid", "dev context")
            _write_cmrc(root / "cmrc2018" / "cmrc2018_train.json", "train_qid", "train context")

            rows = load_text_benchmark_samples("cmrc2018", root, limit=1, cmrc_split="train")

            self.assertEqual(rows[0].qid, "train_qid")

    def test_missing_dev_does_not_fallback_to_train(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_cmrc(root / "cmrc2018" / "cmrc2018_train.json", "train_qid", "train context")

            with self.assertRaises(FileNotFoundError):
                load_cmrc2018_samples(root, limit=1)

    def test_cmrc_strict_eval_limit_zero_means_full_dataset(self) -> None:
        for resolve in (resolve_modern_cmrc_load_limit, resolve_legacy_cmrc_load_limit):
            self.assertIsNone(resolve(0))
            self.assertIsNone(resolve(-1))
            self.assertEqual(resolve(1), 1)
            self.assertEqual(resolve(200), 200)


if __name__ == "__main__":
    unittest.main()
