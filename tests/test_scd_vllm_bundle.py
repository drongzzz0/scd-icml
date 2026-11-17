from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from scd_vllm.bundle import (
    BundleSchemaError,
    array_sha256,
    make_manifest,
    normalize_kv_tensor,
    normalize_kv_tensor_from_heads_first,
    payload_nbytes,
    read_manifest,
    stable_json_hash,
    validate_manifest,
    write_manifest,
)
from scd_vllm.kv_spec import KVSpec


class BundleTest(unittest.TestCase):
    def _spec(self) -> KVSpec:
        return KVSpec(
            model_id="consumer",
            tokenizer_id="tok",
            num_layers=2,
            num_attention_heads=4,
            num_kv_heads=2,
            head_dim=8,
            hidden_size=32,
            dtype="bf16",
        )

    def test_manifest_roundtrip(self) -> None:
        token_ids = np.array([[1, 2, 3]], dtype=np.int64)
        manifest = make_manifest(
            consumer_model="consumer",
            tokenizer_id="tok",
            token_ids=token_ids,
            bundle_type="full_consumer",
            dtype="bf16",
            kv_spec=self._spec(),
            route={"patch_layers": [1], "reuse_layers": [0]},
            route_id="r1",
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            write_manifest(path, manifest)
            loaded = read_manifest(path)
        self.assertEqual(loaded.token_hash, array_sha256(token_ids))
        self.assertEqual(loaded.route_hash, stable_json_hash({"patch_layers": [1], "reuse_layers": [0]}))

    def test_manifest_validation_rejects_missing_kv_spec_field(self) -> None:
        data = make_manifest(
            consumer_model="consumer",
            tokenizer_id="tok",
            token_ids=np.array([1, 2], dtype=np.int64),
            bundle_type="full_consumer",
            dtype="bf16",
            kv_spec=self._spec(),
        ).to_dict()
        del data["kv_spec"]["head_dim"]
        with self.assertRaises(BundleSchemaError):
            validate_manifest(data)

    def test_payload_nbytes_handles_nested_values(self) -> None:
        arr = np.zeros((2, 3), dtype=np.float32)
        self.assertGreaterEqual(payload_nbytes({"arr": arr, "label": "x"}), arr.nbytes + 1)

    def test_normalize_kv_tensor(self) -> None:
        hf = np.zeros((1, 2, 3, 4), dtype=np.float32)
        out = normalize_kv_tensor(hf)
        self.assertEqual(out.shape, (3, 2, 4))
        heads_first = np.zeros((2, 3, 4), dtype=np.float32)
        out2 = normalize_kv_tensor_from_heads_first(heads_first)
        self.assertEqual(out2.shape, (3, 2, 4))


if __name__ == "__main__":
    unittest.main()
