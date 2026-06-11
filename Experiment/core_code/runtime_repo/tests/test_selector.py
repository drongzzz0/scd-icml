from __future__ import annotations

import unittest

from scd_repro.route_selection.schema import RoutePlan
from scd_repro.route_selection.selectors import ConditionalReusePruner, SelectorConstraints, pairwise_interactions


def _metrics_for(route: RoutePlan) -> dict[str, float]:
    bad_reuse = {1}
    drift = 0.0
    cost = 0.0
    for layer in route.layers:
        if layer.action == "patch":
            cost += 10.0
        elif layer.consumer_layer in bad_reuse:
            drift += 10.0
        else:
            drift += 0.1
    return {
        "kv_drift_mean": drift,
        "hidden_drift_mean": 0.0,
        "logit_kl": 0.0,
        "ppl_delta": 0.0,
        "payload_bytes": cost,
    }


class SelectorTest(unittest.TestCase):
    def test_conditional_pruner_keeps_bad_layer_patch(self) -> None:
        initial = RoutePlan.all_patch(5)
        pruner = ConditionalReusePruner(
            _metrics_for,
            constraints=SelectorConstraints(max_delta_per_step=1.0, max_total_drift=1.0),
        )
        result = pruner.run(initial)
        selected = RoutePlan.from_dict(result["selected_route"])
        self.assertEqual(result["status"], "complete")
        self.assertIn(1, selected.patch_layers)
        self.assertEqual(set(selected.reuse_layers), {0, 2, 3, 4})

    def test_pairwise_interaction_detects_positive_conflict(self) -> None:
        base = RoutePlan.all_patch(3)

        def evaluator(route: RoutePlan) -> dict[str, float]:
            reuse = set(route.reuse_layers)
            drift = 0.1 * len(reuse)
            if {0, 1}.issubset(reuse):
                drift += 5.0
            return {"kv_drift_mean": drift, "payload_bytes": 10.0 * len(route.patch_layers)}

        rows = pairwise_interactions(base, [0, 1], evaluator)
        self.assertGreater(rows[0]["interaction"], 4.0)

    def test_strict_eval_proxy_constraints_gate_routes(self) -> None:
        initial = RoutePlan.all_patch(2)

        def evaluator(route: RoutePlan) -> dict[str, float]:
            if 0 in route.reuse_layers:
                return {
                    "avg_nll_delta": 0.5,
                    "f1_retention": 0.8,
                    "payload_bytes": 10.0,
                }
            return {
                "avg_nll_delta": 0.0,
                "f1_retention": 1.0,
                "payload_bytes": 20.0,
            }

        pruner = ConditionalReusePruner(
            evaluator,
            constraints=SelectorConstraints(min_f1_retention=0.95, max_avg_nll_delta=0.1),
        )
        result = pruner.run(initial, max_steps=1)
        selected = RoutePlan.from_dict(result["selected_route"])
        self.assertNotIn(0, selected.reuse_layers)


if __name__ == "__main__":
    unittest.main()
