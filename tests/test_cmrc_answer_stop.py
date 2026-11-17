import unittest
from unittest.mock import patch

from scripts.cmrc_answer_stop import CMRCAnswerStop, TimedAnswerStop, measure_answer_request


class FixtureTokenizer:
    tokens = {13: "\n", 14: " \r\n", 5: "甲", 6: "乙", 7: "\n续", 8: "\n答案"}

    def decode(self, ids, *, skip_special_tokens):
        return "".join(self.tokens[i] for i in ids)


def extract_answer(text, context=None):
    return text.strip().split("\n", 1)[0]


class CMRCAnswerStopTests(unittest.TestCase):
    def make_observer(self, limit=32):
        return CMRCAnswerStop(
            FixtureTokenizer(), extract_answer, "甲乙", ("\n", "\r"), {2}, limit
        )

    def test_leading_newlines_do_not_stop_or_cancel_later_answer(self):
        observe = self.make_observer()
        self.assertIsNone(observe((13,)))
        self.assertIsNone(observe((13, 14)))
        self.assertIsNone(observe((13, 14, 5)))
        self.assertIsNone(observe((13, 14, 5, 6)))
        self.assertEqual(observe((13, 14, 5, 6, 7)), "text_boundary")
        self.assertEqual(observe.prediction, "甲乙")

    def test_same_token_leading_newline_and_answer_then_eos(self):
        observe = self.make_observer()
        self.assertIsNone(observe((8,)))
        self.assertEqual(observe((8, 2)), "stop_token")
        self.assertEqual(observe.prediction, "答案")

    def test_whitespace_only_cannot_extend_generation_budget(self):
        observe = self.make_observer(limit=2)
        self.assertIsNone(observe((13,)))
        self.assertEqual(observe((13, 14)), "max_new_tokens")
        self.assertEqual(observe.prediction, "")

    def test_request_clock_includes_work_before_first_callback(self):
        with patch("scripts.cmrc_answer_stop.time.perf_counter", side_effect=[1.0, 1.08, 1.14, 1.15]):
            timed = TimedAnswerStop(self.make_observer())
            self.assertIsNone(timed((8,)))
            self.assertEqual(timed((8, 2)), "stop_token")
        self.assertAlmostEqual(timed.first_token_ms, 80.0)
        self.assertAlmostEqual(timed.completion_ms, 150.0)

    def test_request_segments_use_same_target_submit_and_restore_engine(self):
        class Engine:
            def add_request(self, request_id):
                return request_id

        engine = Engine()
        add_request = engine.add_request

        def request(observe):
            self.assertEqual(engine.add_request("request"), "request")
            self.assertIsNone(observe((8,)))
            self.assertEqual(observe((8, 2)), "stop_token")
            return "result"

        with patch("scripts.cmrc_answer_stop.time.perf_counter", side_effect=[1, 1.02, 1.1, 1.19, 1.2, 1.23]):
            result, measurements = measure_answer_request(engine, request, self.make_observer())
        self.assertEqual(result, "result")
        self.assertEqual(engine.add_request, add_request)
        for key, value in {
            "pre_target_submit_ms": 20, "target_to_first_token_ms": 80,
            "decode_after_first_ms": 100, "stop_to_return_ms": 30, "request_wall_ms": 230,
        }.items():
            self.assertAlmostEqual(measurements[key], value)
