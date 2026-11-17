"""SCD cache bundle 的 manifest 和完整性校验。

bundle 是 vLLM 集成路径的交换格式：它绑定 token 序列、KV shape、路线
hash、模型标识和质量门限。后续转换为 ExampleConnector storage 之前，
必须先通过这里的 schema 和 hash 校验。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .kv_spec import KVSpec


SCHEMA_VERSION = "scd-cache-bundle-v1"


class BundleSchemaError(ValueError):
    """Raised when an SCD cache bundle manifest is invalid."""


@dataclass
class SCDCacheBundleManifest:
    producer_model: str | None
    consumer_model: str
    tokenizer_id: str
    token_hash: str
    bundle_type: str
    dtype: str
    num_tokens: int
    kv_spec: dict[str, Any]
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    schema_version: str = SCHEMA_VERSION
    prefix_text_hash: str | None = None
    route_id: str | None = None
    route_hash: str | None = None
    device_origin: str = "cpu"
    quality_gate: dict[str, Any] = field(default_factory=lambda: {"gate_name": "", "thresholds": {}})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SCDCacheBundleManifest":
        validate_manifest(data)
        return cls(
            producer_model=data.get("producer_model"),
            consumer_model=str(data["consumer_model"]),
            tokenizer_id=str(data["tokenizer_id"]),
            token_hash=str(data["token_hash"]),
            bundle_type=str(data["bundle_type"]),
            dtype=str(data["dtype"]),
            num_tokens=int(data["num_tokens"]),
            kv_spec=dict(data["kv_spec"]),
            created_at=str(data.get("created_at") or datetime.now(timezone.utc).isoformat()),
            schema_version=str(data.get("schema_version") or SCHEMA_VERSION),
            prefix_text_hash=data.get("prefix_text_hash"),
            route_id=data.get("route_id"),
            route_hash=data.get("route_hash"),
            device_origin=str(data.get("device_origin") or "cpu"),
            quality_gate=dict(data.get("quality_gate") or {"gate_name": "", "thresholds": {}}),
        )


def make_manifest(
    *,
    consumer_model: str,
    tokenizer_id: str,
    token_ids: np.ndarray,
    bundle_type: str,
    dtype: str,
    kv_spec: KVSpec | dict[str, Any],
    producer_model: str | None = None,
    prefix_text: str | None = None,
    route: dict[str, Any] | None = None,
    route_id: str | None = None,
    device_origin: str = "cpu",
    quality_gate: dict[str, Any] | None = None,
) -> SCDCacheBundleManifest:
    token_array = np.asarray(token_ids)
    route_hash = stable_json_hash(route) if route is not None else None
    spec_dict = kv_spec.to_dict() if isinstance(kv_spec, KVSpec) else dict(kv_spec)
    return SCDCacheBundleManifest(
        producer_model=producer_model,
        consumer_model=consumer_model,
        tokenizer_id=tokenizer_id,
        token_hash=array_sha256(token_array),
        prefix_text_hash=text_sha256(prefix_text) if prefix_text is not None else None,
        bundle_type=bundle_type,
        route_id=route_id,
        route_hash=route_hash,
        dtype=dtype,
        device_origin=device_origin,
        num_tokens=int(token_array.size if token_array.ndim == 1 else token_array.shape[-1]),
        kv_spec=spec_dict,
        quality_gate=quality_gate or {"gate_name": "", "thresholds": {}},
    )


def validate_manifest(data: dict[str, Any]) -> None:
    required = [
        "schema_version",
        "created_at",
        "consumer_model",
        "tokenizer_id",
        "token_hash",
        "bundle_type",
        "dtype",
        "device_origin",
        "num_tokens",
        "kv_spec",
        "quality_gate",
    ]
    missing = [key for key in required if key not in data]
    if missing:
        raise BundleSchemaError("manifest missing keys: " + ", ".join(missing))
    if data["schema_version"] != SCHEMA_VERSION:
        raise BundleSchemaError(f"unsupported schema_version: {data['schema_version']}")
    if not isinstance(data["kv_spec"], dict):
        raise BundleSchemaError("kv_spec must be an object")
    for key in ("num_layers", "num_kv_heads", "head_dim"):
        if key not in data["kv_spec"]:
            raise BundleSchemaError(f"kv_spec missing {key}")
    if int(data["num_tokens"]) < 0:
        raise BundleSchemaError("num_tokens must be non-negative")


def write_manifest(path: str | Path, manifest: SCDCacheBundleManifest | dict[str, Any]) -> None:
    data = manifest.to_dict() if isinstance(manifest, SCDCacheBundleManifest) else dict(manifest)
    validate_manifest(data)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_manifest(path: str | Path) -> SCDCacheBundleManifest:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return SCDCacheBundleManifest.from_dict(data)


def array_sha256(array: np.ndarray) -> str:
    arr = np.ascontiguousarray(array)
    h = hashlib.sha256()
    h.update(str(arr.dtype).encode("utf-8"))
    h.update(json.dumps(list(arr.shape), separators=(",", ":")).encode("utf-8"))
    h.update(arr.tobytes(order="C"))
    return h.hexdigest()


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_json_hash(data: Any) -> str:
    encoded = json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def payload_nbytes(*items: Any) -> int:
    total = 0
    for item in items:
        if item is None:
            continue
        if isinstance(item, np.ndarray):
            total += int(item.nbytes)
        elif isinstance(item, dict):
            total += payload_nbytes(*item.values())
        elif isinstance(item, (list, tuple)):
            total += payload_nbytes(*item)
        elif hasattr(item, "numel") and hasattr(item, "element_size"):
            total += int(item.numel() * item.element_size())
        elif isinstance(item, (bytes, bytearray)):
            total += len(item)
        else:
            total += len(str(item).encode("utf-8"))
    return total


def normalize_kv_tensor(array: np.ndarray) -> np.ndarray:
    """Normalize common KV tensor shapes to [T, H_kv, D].

    Accepted inputs:
    - [T, H, D]
    - [1, H, T, D]
    - [H, T, D] when the first axis is clearly heads and the second is tokens
    """

    arr = np.asarray(array)
    if arr.ndim == 3:
        # Keep [T, H, D] as-is.  For [H, T, D], callers can use the explicit
        # normalize_kv_tensor_from_heads_first helper to avoid ambiguous guesses.
        return np.ascontiguousarray(arr)
    if arr.ndim == 4 and arr.shape[0] == 1:
        return np.ascontiguousarray(np.transpose(arr[0], (1, 0, 2)))
    raise BundleSchemaError(f"unsupported KV tensor shape for normalization: {arr.shape}")


def normalize_kv_tensor_from_heads_first(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array)
    if arr.ndim != 3:
        raise BundleSchemaError(f"expected [H,T,D], got {arr.shape}")
    return np.ascontiguousarray(np.transpose(arr, (1, 0, 2)))
