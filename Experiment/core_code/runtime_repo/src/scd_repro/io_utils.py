from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .torch_utils import ensure_parent, require_torch, tensor_to_cpu, tensor_to_device, tensor_to_numpy


@dataclass
class TensorBundle:
    arrays: dict[str, Any]
    meta: dict[str, Any]


def save_bundle(path: str | Path, arrays: dict[str, Any], meta: dict[str, Any] | None = None, artifact_dtype: str | None = None) -> None:
    t = require_torch()
    p = ensure_parent(path)
    payload = {
        "arrays": {key: tensor_to_cpu(value, dtype=artifact_dtype) for key, value in arrays.items()},
        "meta": meta or {},
    }
    t.save(payload, p)


def load_bundle(
    path: str | Path,
    *,
    device: str | None = "cpu",
    dtype: str | None = None,
    as_numpy: bool = False,
    cpu_fallback: bool = True,
) -> TensorBundle:
    t = require_torch()
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)

    payload = t.load(p, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and "arrays" in payload:
        arrays = payload.get("arrays", {})
        meta = payload.get("meta", {})
    else:
        arrays = payload
        meta = {}

    if as_numpy:
        return TensorBundle(arrays={key: tensor_to_numpy(value) for key, value in arrays.items()}, meta=meta)

    return TensorBundle(
        arrays={key: tensor_to_device(value, device=device, dtype=dtype, cpu_fallback=cpu_fallback) for key, value in arrays.items()},
        meta=meta,
    )


def save_json(path: str | Path, payload: Any) -> None:
    p = ensure_parent(path)
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_json(path: str | Path) -> Any:
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8"))
