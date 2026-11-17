from __future__ import annotations

import unittest

from scd_repro.route_selection.schema import RoutePlan, route_fingerprint


class RouteSerializationTest(unittest.TestCase):
    def test_route_round_trip_preserves_mapping(self) -> None:
        route = RoutePlan.all_patch(4, source_mapping=[0, 1, 1, 2])
        route = route.with_action(2, "reuse")
        payload = route.to_dict()
        loaded = RoutePlan.from_dict(payload)
        self.assertEqual(loaded.patch_layers, [0, 1, 3])
        self.assertEqual(loaded.reuse_layers, [2])
        self.assertEqual(loaded.source_layer_mapping, [0, 1, 1, 2])
        self.assertEqual(route_fingerprint(route), route_fingerprint(loaded))

    def test_legacy_route_conversion(self) -> None:
        legacy = {"num_layers": 4, "patch_layers": [1, 3], "reuse_layers": [0, 2]}
        route = RoutePlan.from_legacy_route(legacy, producer_layers=2)
        self.assertEqual(route.patch_layers, [1, 3])
        self.assertEqual(route.reuse_layers, [0, 2])
        self.assertEqual(route.source_layer_mapping, [0, 0, 1, 1])


if __name__ == "__main__":
    unittest.main()
