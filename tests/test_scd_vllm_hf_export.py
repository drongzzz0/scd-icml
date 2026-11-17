from __future__ import annotations

import numpy as np

from scd_vllm.hf_export import build_position_ids, iter_cache_layers, load_prompt, normalize_cache_tensor


class _Layer:
    def __init__(self, key, value):
        self.keys = key
        self.values = value


class _CacheWithLayers:
    def __init__(self, layers):
        self.layers = layers


def test_build_position_ids_from_attention_mask() -> None:
    mask = np.array([[1, 1, 1], [1, 0, 1]], dtype=np.int64)

    assert np.array_equal(build_position_ids(mask), np.array([[0, 1, 2], [0, 0, 1]], dtype=np.int64))


def test_iter_cache_layers_supports_tuple_cache() -> None:
    k0 = np.zeros((1, 2, 3, 4))
    v0 = np.ones((1, 2, 3, 4))
    k1 = np.zeros((1, 2, 3, 4)) + 2
    v1 = np.ones((1, 2, 3, 4)) + 3

    layers = list(iter_cache_layers(((k0, v0), (k1, v1))))

    assert [(idx, key.shape, value.shape) for idx, key, value in layers] == [
        (0, (1, 2, 3, 4), (1, 2, 3, 4)),
        (1, (1, 2, 3, 4), (1, 2, 3, 4)),
    ]


def test_iter_cache_layers_supports_layer_objects() -> None:
    cache = _CacheWithLayers([_Layer(np.zeros((1, 2, 3, 4)), np.ones((1, 2, 3, 4)))])

    layers = list(iter_cache_layers(cache))

    assert len(layers) == 1
    assert layers[0][0] == 0


def test_normalize_cache_tensor_accepts_hf_shape() -> None:
    arr = np.arange(1 * 2 * 3 * 4).reshape(1, 2, 3, 4)

    normalized = normalize_cache_tensor(arr)

    assert normalized.shape == (3, 2, 4)
    assert np.array_equal(normalized[:, 0, :], arr[0, 0, :, :])


def test_load_prompt_rejects_empty_prompt() -> None:
    try:
        load_prompt("", "")
    except RuntimeError as exc:
        assert "empty" in str(exc)
    else:
        raise AssertionError("empty prompt should fail")
