from __future__ import annotations

import unittest

from scd_repro.cli.build_route import _legacy_build_route
from scd_repro.routing import RouteSearchConfig, TaskLayerProfile, greedy_search_task_route
from scd_repro.strict_runtime import _normalize_runtime_route


class RouteGuardrailTest(unittest.TestCase):
    def test_legacy_route_rejects_incompatible_patch_kv_scale(self) -> None:
        profile = {
            "patch_metric": "patch_kv_error",
            "layers": [
                {"layer": 0, "reuse_error": 0.2, "patch_error": 100000.0, "patch_kv_error": 100000.0},
                {"layer": 1, "reuse_error": 0.4, "patch_error": 200000.0, "patch_kv_error": 200000.0},
                {"layer": 2, "reuse_error": 0.6, "patch_error": 300000.0, "patch_kv_error": 300000.0},
            ],
        }
        eval_cfg = {"route": {"threshold_mode": "adaptive_p20_p80", "patch_budget": 1}}
        with self.assertRaisesRegex(ValueError, "incompatible scales"):
            _legacy_build_route(profile, eval_cfg)

    def test_task_level_route_fallbacks_unsearched_layers(self) -> None:
        profiles = {
            0: TaskLayerProfile(0, 1, -10.0, -1.0, 0.0, 10),
            1: TaskLayerProfile(1, 1, -10.0, -9.0, 0.0, 10),
            2: TaskLayerProfile(2, 1, -10.0, None, 0.0, 0),
            3: TaskLayerProfile(3, 1, -10.0, None, 0.0, 0),
        }
        route = greedy_search_task_route(
            profiles,
            candidate_layers=[0, 1],
            config=RouteSearchConfig(patch_budget=1, enable_short_recompute=False),
        )
        self.assertEqual(route["patch_layers"], [0])
        self.assertEqual(route["fallback_layers"], [2, 3])
        self.assertEqual(route["reuse_layers"], [1])

    def test_strict_runtime_rejects_incomplete_reuse_patch_policy(self) -> None:
        route = {"reuse_layers": [1], "patch_layers": [0], "fallback_layers": []}
        with self.assertRaisesRegex(ValueError, "does not cover layers"):
            _normalize_runtime_route(
                route,
                mode="reuse_patch",
                num_layers=3,
                fail_on_missing_route_fields=True,
                fail_on_missing_patch_fields=True,
            )

    def test_strict_runtime_rejects_overlapping_reuse_patch_policy(self) -> None:
        route = {"reuse_layers": [0, 1], "patch_layers": [0], "fallback_layers": [2]}
        with self.assertRaisesRegex(ValueError, "overlapping assignments"):
            _normalize_runtime_route(
                route,
                mode="reuse_patch",
                num_layers=3,
                fail_on_missing_route_fields=True,
                fail_on_missing_patch_fields=True,
            )


if __name__ == "__main__":
    unittest.main()
