from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

import numpy as np

from scripts.extract_stack_viz_inputs import (
    build_distance_rows,
    build_tsne_rows,
    choose_layer,
    load_vectors_npz,
    sample_ids,
    save_vectors_npz,
)


class StackVizInputExtractorTest(unittest.TestCase):
    def test_choose_layer_prefers_explicit_layer(self) -> None:
        self.assertEqual(choose_layer(32, 7, None), 7)
        with self.assertRaises(ValueError):
            choose_layer(4, 9, None)

    def test_sample_ids_is_deterministic_and_sorted(self) -> None:
        first = sample_ids(20, 5, 42)
        second = sample_ids(20, 5, 42)
        self.assertEqual(first, second)
        self.assertEqual(first, sorted(first))
        self.assertEqual(len(first), 5)

    def test_build_distance_rows_reports_improvement(self) -> None:
        point_ids = [0, 1, 2]
        target = np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32)
        source = np.asarray([[2.0, 0.0], [0.0, 2.0], [2.0, 2.0]], dtype=np.float32)
        translated = np.asarray([[1.1, 0.0], [0.0, 1.1], [1.1, 1.1]], dtype=np.float32)
        rows, meta = build_distance_rows(3, point_ids, source, target, translated)
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0]["layer"], 3)
        self.assertGreater(rows[0]["rel_l2_improvement"], 0.0)
        self.assertGreater(meta["median_rel_l2_reduction"], 0.0)

    def test_build_tsne_rows_has_three_series(self) -> None:
        point_ids = [0, 1, 2, 3]
        source = np.asarray([[0.0, 0.0], [0.1, 0.0], [0.0, 0.1], [0.1, 0.1]], dtype=np.float32)
        target = source + 2.0
        translated = source + 1.8
        rows, meta = build_tsne_rows(5, point_ids, source, target, translated, perplexity=5, seed=42)
        self.assertEqual(len(rows), 12)
        self.assertEqual({row["series"] for row in rows}, {"source", "target", "translated_reuse"})
        self.assertEqual(meta["points_per_series"], 4)
        self.assertIn(meta["reducer"], {"tsne", "pca_fallback"})

    def test_vector_npz_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "vectors.npz"
            point_ids = [2, 4]
            source = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
            target = source + 0.5
            translated = source + 0.25
            save_vectors_npz(path, 9, point_ids, source, target, translated)
            layer, loaded_ids, loaded_source, loaded_target, loaded_translated = load_vectors_npz(path)
            self.assertEqual(layer, 9)
            self.assertEqual(loaded_ids, point_ids)
            np.testing.assert_allclose(loaded_source, source)
            np.testing.assert_allclose(loaded_target, target)
            np.testing.assert_allclose(loaded_translated, translated)


if __name__ == "__main__":
    unittest.main()
