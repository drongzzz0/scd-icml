from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from scd_repro.wikitext2 import load_wikitext2_samples


def _long_line(tag: str) -> str:
    words = [f"{tag}{idx}" for idx in range(24)]
    return " ".join(words) + " " + ("extra " * 20)


class WikiText2SplitTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.mkdtemp(prefix="wikitext2_split_")
        root = Path(self._tmp) / "wikitext-2"
        root.mkdir(parents=True, exist_ok=True)
        (root / "wiki.train.txt").write_text(_long_line("train"), encoding="utf-8")
        (root / "wiki.valid.txt").write_text(_long_line("valid"), encoding="utf-8")
        (root / "wiki.test.txt").write_text(_long_line("test"), encoding="utf-8")
        self.data_root = Path(self._tmp)

    def tearDown(self) -> None:
        shutil.rmtree(self._tmp)

    def test_train_split_prefers_train_file(self) -> None:
        samples = load_wikitext2_samples(self.data_root, limit=10, split="train")
        self.assertEqual(len(samples), 1)
        self.assertTrue(samples[0].qid.startswith("wiki.train_"))
        self.assertIn("train", samples[0].source_text)

    def test_valid_split_alias(self) -> None:
        samples = load_wikitext2_samples(self.data_root, limit=10, split="validation")
        self.assertEqual(len(samples), 1)
        self.assertTrue(samples[0].qid.startswith("wiki.valid_"))
        self.assertIn("valid", samples[0].source_text)

    def test_test_split_default(self) -> None:
        samples = load_wikitext2_samples(self.data_root, limit=10)
        self.assertEqual(len(samples), 1)
        self.assertTrue(samples[0].qid.startswith("wiki.test_"))
        self.assertIn("test", samples[0].source_text)

    def test_all_split_collects_everything(self) -> None:
        samples = load_wikitext2_samples(self.data_root, limit=10, split="all")
        self.assertEqual(len(samples), 3)
        prefixes = {sample.qid.split("_")[0] for sample in samples}
        self.assertEqual(prefixes, {"wiki.train", "wiki.valid", "wiki.test"})


if __name__ == "__main__":
    unittest.main()
