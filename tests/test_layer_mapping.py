from __future__ import annotations

import unittest

from scd_repro.route_selection.layer_mapping import (
    estimate_layer_mapping,
    proportional_mapping,
    same_index_mapping,
    validate_monotonic_mapping,
)


class LayerMappingTest(unittest.TestCase):
    def test_same_index_mapping(self) -> None:
        self.assertEqual(same_index_mapping(4, 4), [0, 1, 2, 3])

    def test_proportional_mapping_is_monotonic(self) -> None:
        mapping = proportional_mapping(3, 5)
        self.assertEqual(mapping, [0, 0, 1, 2, 2])
        validate_monotonic_mapping(mapping, producer_layers=3)

    def test_same_index_rejects_mismatched_depth(self) -> None:
        with self.assertRaises(ValueError):
            estimate_layer_mapping(strategy="same_index", producer_layers=2, consumer_layers=4)


if __name__ == "__main__":
    unittest.main()
