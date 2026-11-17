from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .io_utils import save_json
from .torch_utils import require_torch


@dataclass
class CacheShapeSpec:
    batch_size: int
    num_layers: int
    num_key_value_heads: int
    prefix_tokens: int
    head_dim: int
    dtype: str = "float32"
    device: str = "cpu"


class CacheWriter:
    """KV 缓存唯一写入口，并执行严格形状与类型断言。"""

    def __init__(self, spec: CacheShapeSpec) -> None:
        self.spec = spec
        self.k_cache: dict[int, Any] = {}
        self.v_cache: dict[int, Any] = {}
        self.audit_rows: list[dict[str, object]] = []

    @staticmethod
    def _shape(arr: Any) -> tuple[int, ...]:
        t = require_torch()
        if t.is_tensor(arr):
            return tuple(arr.shape)
        return tuple(np.shape(arr))

    @staticmethod
    def _dtype_name(arr: Any) -> str:
        t = require_torch()
        if t.is_tensor(arr):
            return str(arr.dtype).replace("torch.", "")
        return str(getattr(arr, "dtype", type(arr)))

    @staticmethod
    def _device_name(arr: Any) -> str:
        t = require_torch()
        if t.is_tensor(arr):
            return str(arr.device)
        return "cpu"

    def _assert_tensor(self, layer_idx: int, name: str, arr: Any) -> None:
        shape = self._shape(arr)
        if len(shape) != 4:
            raise ValueError(f"{name}[layer={layer_idx}] 必须是 4D [B,KVH,T,HD]，实际为 {shape}")
        expected = (
            self.spec.batch_size,
            self.spec.num_key_value_heads,
            self.spec.prefix_tokens,
            self.spec.head_dim,
        )
        if shape != expected:
            raise ValueError(f"{name}[layer={layer_idx}] 形状不匹配，期望={expected}，实际={shape}")

    def update(self, layer_idx: int, k: Any, v: Any, cache_kwargs: dict[str, object] | None = None) -> None:
        t = require_torch()
        if not (0 <= layer_idx < self.spec.num_layers):
            raise ValueError(f"layer_idx 越界: {layer_idx}")
        self._assert_tensor(layer_idx, "k", k)
        self._assert_tensor(layer_idx, "v", v)

        if t.is_tensor(k):
            k_store = k.detach().contiguous()
        else:
            k_store = np.ascontiguousarray(k)
        if t.is_tensor(v):
            v_store = v.detach().contiguous()
        else:
            v_store = np.ascontiguousarray(v)

        self.k_cache[layer_idx] = k_store
        self.v_cache[layer_idx] = v_store
        self.audit_rows.append(
            {
                "layer_idx": layer_idx,
                "k_shape": list(self._shape(k_store)),
                "v_shape": list(self._shape(v_store)),
                "dtype": self._dtype_name(k_store),
                "device": self._device_name(k_store),
                "cache_kwargs": cache_kwargs or {},
            }
        )

    def export_audit(self, out_path: str | Path) -> None:
        payload = {
            "num_layers_expected": self.spec.num_layers,
            "num_layers_written": len(self.k_cache),
            "rows": self.audit_rows,
        }
        save_json(out_path, payload)

    def is_complete(self) -> bool:
        return len(self.k_cache) == self.spec.num_layers and len(self.v_cache) == self.spec.num_layers
