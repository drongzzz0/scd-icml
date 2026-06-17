from __future__ import annotations

import unittest

import numpy as np

from scd_vllm.kv_spec import KVSpec, KVSpecError, assert_consumer_kv_compatible, assert_token_ids_identical


class KVSpecTest(unittest.TestCase):
    def _spec(self, **overrides) -> KVSpec:
        data = {
            "model_id": "producer",
            "tokenizer_id": "tok",
            "num_layers": 32,
            "num_attention_heads": 32,
            "num_kv_heads": 8,
            "head_dim": 128,
            "hidden_size": 4096,
            "dtype": "bfloat16",
            "rope_theta": 1000000.0,
        }
        data.update(overrides)
        return KVSpec(**data)

    def test_compatible_specs_pass(self) -> None:
        assert_consumer_kv_compatible(self._spec(), self._spec(model_id="consumer"))

    def test_incompatible_shape_fails(self) -> None:
        with self.assertRaises(KVSpecError):
            assert_consumer_kv_compatible(self._spec(), self._spec(num_layers=28))

    def test_dtype_aliases_are_compatible(self) -> None:
        assert_consumer_kv_compatible(self._spec(dtype="torch.bfloat16"), self._spec(dtype="bf16"))

    def test_token_ids_identical(self) -> None:
        ids = np.array([[1, 2, 3]], dtype=np.int64)
        assert_token_ids_identical(ids, ids.copy())
        with self.assertRaises(KVSpecError):
            assert_token_ids_identical(ids, np.array([[1, 2, 4]], dtype=np.int64))


if __name__ == "__main__":
    unittest.main()
