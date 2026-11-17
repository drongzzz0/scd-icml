from __future__ import annotations

import unittest

import numpy as np

from scd_vllm.metrics import aggregate_numeric, evaluate_scd_gate, logit_rel_l2, payload_ratio, safe_float


class SCDVLLMMetricsTest(unittest.TestCase):
    def test_safe_float_and_aggregate_skip_na(self) -> None:
        self.assertIsNone(safe_float("N/A"))
        summary = aggregate_numeric([{"x": 1.0}, {"x": "N/A"}, {"x": 3.0}], "x")
        self.assertEqual(summary["count"], 2)
        self.assertEqual(summary["mean"], 2.0)

    def test_payload_ratio(self) -> None:
        self.assertEqual(payload_ratio(25, 100), 0.25)
        self.assertIsNone(payload_ratio(25, 0))

    def test_logit_rel_l2(self) -> None:
        self.assertEqual(logit_rel_l2(np.ones((2, 2)), np.ones((2, 2))), 0.0)
        self.assertGreater(logit_rel_l2(np.zeros((2, 2)), np.ones((2, 2))), 0.0)

    def test_gate(self) -> None:
        result = evaluate_scd_gate(
            {
                "payload_bytes": 40,
                "raw_consumer_kv_bytes": 100,
                "logit_rel_l2_vs_full_consumer": 0.01,
                "logit_kl_vs_full_consumer": 0.01,
                "ppl_delta_vs_full_consumer": 0.02,
                "task_metric_delta_vs_full_consumer": -0.01,
            },
            {
                "require_payload_smaller_than_raw_kv": True,
                "max_logit_rel_l2": 0.05,
                "max_logit_kl": 0.05,
                "max_ppl_delta": 0.1,
                "max_task_metric_drop_abs": 0.02,
            },
        )
        self.assertTrue(result.passed)
        failed = evaluate_scd_gate(
            {"payload_bytes": 200, "raw_consumer_kv_bytes": 100},
            {"require_payload_smaller_than_raw_kv": True},
        )
        self.assertFalse(failed.passed)


if __name__ == "__main__":
    unittest.main()
