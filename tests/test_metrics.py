from __future__ import annotations

import unittest

import numpy as np

from scd_repro.route_selection.metrics import hidden_drift, kv_drift, logit_kl, teacher_objective


class MetricsTest(unittest.TestCase):
    def test_drift_masks_padding_and_handles_zero_teacher(self) -> None:
        teacher = np.zeros((1, 2, 2), dtype=np.float64)
        route = np.ones((1, 2, 2), dtype=np.float64)
        mask = np.array([[1, 0]], dtype=np.float64)
        value = hidden_drift(route, teacher, mask=mask)
        self.assertTrue(np.isfinite(value))
        self.assertGreater(value, 0)

    def test_kv_drift_and_logit_kl_are_finite(self) -> None:
        teacher = np.ones((1, 2, 2), dtype=np.float64)
        route = teacher * 1.5
        self.assertGreater(kv_drift(route_k=route, teacher_k=teacher, route_v=route, teacher_v=teacher), 0)
        kl = logit_kl(
            teacher_logits=np.array([[[2.0, 0.0], [0.0, 2.0]]]),
            route_logits=np.array([[[1.0, 0.0], [0.0, 1.0]]]),
            mask=np.array([[1, 0]]),
        )
        self.assertTrue(np.isfinite(kl))
        self.assertGreaterEqual(kl, 0)

    def test_teacher_objective_weights(self) -> None:
        value = teacher_objective(
            {"kv_drift_mean": 1.0, "hidden_drift_mean": 2.0, "logit_kl": 3.0, "ppl_delta": 4.0},
            {"kv": 1.0, "hidden": 0.5, "kl": 0.0, "ppl": 2.0, "cost": 0.0},
        )
        self.assertEqual(value, 10.0)

    def test_teacher_objective_uses_nll_delta_when_ppl_missing(self) -> None:
        value = teacher_objective(
            {"avg_nll_delta": 0.25},
            {"kv": 0.0, "hidden": 0.0, "kl": 0.0, "ppl": 2.0, "cost": 0.0},
        )
        self.assertEqual(value, 0.5)


if __name__ == "__main__":
    unittest.main()
