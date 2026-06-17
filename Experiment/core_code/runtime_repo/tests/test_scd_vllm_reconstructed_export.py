from __future__ import annotations

import pytest

from scd_vllm.reconstructed_export import ReconstructedExportError, _normalize_cache_tensor


def test_normalize_cache_tensor_accepts_hf_cache_layout() -> None:
    torch = pytest.importorskip("torch")
    tensor = torch.arange(1 * 2 * 5 * 4, dtype=torch.float32).reshape(1, 2, 5, 4)

    out = _normalize_cache_tensor(tensor, cache_tokens=4, torch=torch)

    assert tuple(out.shape) == (4, 2, 4)
    assert float(out[0, 0, 0]) == float(tensor[0, 0, 0, 0])
    assert float(out[0, 1, 0]) == float(tensor[0, 1, 0, 0])


def test_normalize_cache_tensor_rejects_short_cache() -> None:
    torch = pytest.importorskip("torch")
    tensor = torch.zeros((1, 2, 3, 4), dtype=torch.float32)

    with pytest.raises(ReconstructedExportError):
        _normalize_cache_tensor(tensor, cache_tokens=4, torch=torch)
