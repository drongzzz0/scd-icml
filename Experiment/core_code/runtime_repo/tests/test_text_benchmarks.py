from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scd_repro.text_benchmarks import (
    build_hotpotqa_compressed_prompt,
    build_hotpotqa_prompt,
    load_hotpotqa_compressed_samples,
    load_hotpotqa_samples,
)


class HotpotQAPromptTest(unittest.TestCase):
    def test_hotpotqa_prompt_keeps_full_context_and_answer_tail(self) -> None:
        context = []
        for idx in range(8):
            context.append(
                [
                    f"Irrelevant {idx}",
                    [
                        "This distractor paragraph is intentionally long. " * 30,
                    ],
                ]
            )
        context.extend(
            [
                ["Scott Derrickson", ["Scott Derrickson is an American film director."]],
                ["Ed Wood", ["Ed Wood was an American filmmaker and actor."]],
            ]
        )
        prompt, evidence = build_hotpotqa_prompt(
            context,
            "Were Scott Derrickson and Ed Wood of the same nationality?",
        )

        self.assertIn("[Scott Derrickson]", evidence)
        self.assertIn("[Ed Wood]", evidence)
        self.assertIn("[Irrelevant 0]", evidence)
        self.assertIn("[Irrelevant 7]", evidence)
        self.assertTrue(prompt.endswith("Answer:"))
        self.assertIn("Question: Were Scott Derrickson and Ed Wood", prompt)

    def test_hotpotqa_loader_uses_full_context(self) -> None:
        row = {
            "_id": "case1",
            "question": "Were Scott Derrickson and Ed Wood of the same nationality?",
            "answer": "yes",
            "supporting_facts": [["Scott Derrickson", 0], ["Ed Wood", 0]],
            "context": [
                ["Noise", ["Noise sentence. " * 200]],
                ["Scott Derrickson", ["Scott Derrickson is an American film director."]],
                ["Ed Wood", ["Ed Wood was an American filmmaker and actor."]],
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "hotpotqa"
            data_dir.mkdir()
            (data_dir / "hotpot_dev_distractor_v1.json").write_text(
                json.dumps([row]),
                encoding="utf-8",
            )

            sample = load_hotpotqa_samples(root, limit=1)[0]

        self.assertEqual(sample.qid, "case1")
        self.assertEqual(sample.answers, ["yes"])
        self.assertIn("[Scott Derrickson]", sample.context)
        self.assertIn("[Noise]", sample.context)
        self.assertTrue(sample.prompt.endswith("Answer:"))

    def test_hotpotqa_compressed_loader_uses_question_aware_evidence(self) -> None:
        row = {
            "_id": "case1",
            "question": "Were Scott Derrickson and Ed Wood of the same nationality?",
            "answer": "yes",
            "supporting_facts": [["Scott Derrickson", 0], ["Ed Wood", 0]],
            "context": [
                ["Noise", ["Noise sentence. " * 500]],
                ["Scott Derrickson", ["Scott Derrickson is an American film director."]],
                ["Ed Wood", ["Ed Wood was an American filmmaker and actor."]],
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "hotpotqa"
            data_dir.mkdir()
            (data_dir / "hotpot_dev_distractor_v1.json").write_text(
                json.dumps([row]),
                encoding="utf-8",
            )

            sample = load_hotpotqa_compressed_samples(root, limit=1)[0]

        self.assertEqual(sample.qid, "case1")
        self.assertEqual(sample.answers, ["yes"])
        self.assertLessEqual(len(sample.context), 900)
        self.assertIn("[Scott Derrickson]", sample.context)
        self.assertIn("[Ed Wood]", sample.context)
        self.assertLess(sample.context.count("Noise sentence."), 500)
        self.assertTrue(sample.prompt.startswith("Read the question and answer using only the evidence."))
        self.assertTrue(sample.prompt.endswith("Answer:"))

    def test_hotpotqa_compressed_prompt_template_matches_historical_protocol(self) -> None:
        prompt = build_hotpotqa_compressed_prompt("[A] evidence", "Question?")

        self.assertIn("Output only the shortest answer.", prompt)
        self.assertIn("If the question is yes/no, answer only yes or no.", prompt)
        self.assertIn("Question: Question?", prompt)
        self.assertIn("Evidence:\n[A] evidence", prompt)
        self.assertTrue(prompt.endswith("Answer:"))


if __name__ == "__main__":
    unittest.main()
