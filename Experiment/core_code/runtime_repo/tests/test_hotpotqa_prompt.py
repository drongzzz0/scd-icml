from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scd_repro.hotpotqa import load_hotpotqa_samples, postprocess_hotpotqa_prediction


class HotpotQAPromptTest(unittest.TestCase):
    def test_load_samples_puts_question_first_and_covers_named_entities(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            data_dir = root / "HotpotQA" / "raw"
            data_dir.mkdir(parents=True, exist_ok=True)
            payload = [
                {
                    "_id": "q1",
                    "question": "Which case was brought to court first Miller v. California or Gates v. Collier?",
                    "answer": "Miller v. California",
                    "context": [
                        ["Trusty system (prison)", ["The trusty system was a penitentiary system enforced in parts of the United States until the 1980s."] * 8],
                        ["Irrelevant case A", ["A different case about unrelated prison policy."] * 8],
                        ["Miller v. California", ["Miller v. California was argued on January 17 and 18, 1972, and decided on June 21, 1973."] * 4],
                        ["Irrelevant case B", ["This case concerns a separate civil dispute with no date overlap."] * 8],
                        ["Gates v. Collier", ["Gates v. Collier was first argued on October 5, 1972."] * 4],
                        ["Irrelevant case C", ["This paragraph contains no useful support for the question."] * 8],
                    ],
                }
            ]
            path = data_dir / "hotpot_dev_distractor_v1.json"
            path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

            sample = load_hotpotqa_samples(root, limit=1)[0]

            self.assertIn("Question: Which case was brought to court first Miller v. California or Gates v. Collier?", sample.prompt)
            self.assertLess(sample.prompt.index("Question:"), sample.prompt.index("Evidence:"))
            self.assertIn("Miller v. California", sample.prompt)
            self.assertIn("Gates v. Collier", sample.prompt)
            self.assertLess(len(sample.prompt), 1700)

    def test_postprocess_normalizes_yes_no_and_strips_question_prefix(self) -> None:
        self.assertEqual(postprocess_hotpotqa_prediction("Yes"), "yes")
        self.assertEqual(postprocess_hotpotqa_prediction("No, they are not"), "no")
        self.assertEqual(postprocess_hotpotqa_prediction("Question: What is the arena capacity?"), "What is the arena capacity?")
        self.assertEqual(postprocess_hotpotqa_prediction("Answer: David Weissman"), "David Weissman")


if __name__ == "__main__":
    unittest.main()
