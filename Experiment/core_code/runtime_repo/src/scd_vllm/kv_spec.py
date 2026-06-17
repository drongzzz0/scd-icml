from __future__ import annotations

import importlib
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


class KVSpecError(ValueError):
    """Raised when a bundle/runtime KV spec is not safe to consume."""


@dataclass(frozen=True)
class KVSpec:
    model_id: str
    tokenizer_id: str
    num_layers: int
    num_attention_heads: int
    num_kv_heads: int
    head_dim: int
    hidden_size: int
    dtype: str
    max_position_embeddings: int | None = None
    rope_theta: float | None = None
    rope_scaling: dict[str, Any] | None = None
    block_size: int | None = None
    vllm_version: str | None = None
    attention_backend: str | None = None
    kv_layout: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "KVSpec":
        return cls(
            model_id=str(data.get("model_id") or ""),
            tokenizer_id=str(data.get("tokenizer_id") or data.get("model_id") or ""),
            num_layers=int(data.get("num_layers") or 0),
            num_attention_heads=int(data.get("num_attention_heads") or 0),
            num_kv_heads=int(data.get("num_kv_heads") or data.get("num_key_value_heads") or 0),
            head_dim=int(data.get("head_dim") or 0),
            hidden_size=int(data.get("hidden_size") or 0),
            dtype=str(data.get("dtype") or ""),
            max_position_embeddings=_optional_int(data.get("max_position_embeddings")),
            rope_theta=_optional_float(data.get("rope_theta")),
            rope_scaling=data.get("rope_scaling") if isinstance(data.get("rope_scaling"), dict) else None,
            block_size=_optional_int(data.get("block_size")),
            vllm_version=_optional_str(data.get("vllm_version")),
            attention_backend=_optional_str(data.get("attention_backend")),
            kv_layout=_optional_str(data.get("kv_layout")),
        )


def _optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def _optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _optional_str(value: Any) -> str | None:
    if value in (None, ""):
        return None
    return str(value)


def _get_attr_any(obj: Any, names: tuple[str, ...], default: Any = None) -> Any:
    for name in names:
        if isinstance(obj, dict) and name in obj:
            return obj[name]
        if hasattr(obj, name):
            return getattr(obj, name)
    return default


def _infer_num_layers(cfg: Any) -> int:
    return int(_get_attr_any(cfg, ("num_hidden_layers", "n_layer", "num_layers"), 0) or 0)


def _infer_num_attention_heads(cfg: Any) -> int:
    return int(_get_attr_any(cfg, ("num_attention_heads", "n_head", "n_heads"), 0) or 0)


def _infer_hidden_size(cfg: Any) -> int:
    return int(_get_attr_any(cfg, ("hidden_size", "n_embd", "d_model"), 0) or 0)


def _infer_num_kv_heads(cfg: Any, num_attention_heads: int) -> int:
    value = _get_attr_any(cfg, ("num_key_value_heads", "n_kv_head", "num_kv_heads"), None)
    if value in (None, ""):
        return int(num_attention_heads)
    return int(value)


def _infer_head_dim(cfg: Any, hidden_size: int, num_attention_heads: int) -> int:
    value = _get_attr_any(cfg, ("head_dim", "attention_head_size"), None)
    if value not in (None, ""):
        return int(value)
    if hidden_size and num_attention_heads:
        return int(hidden_size) // int(num_attention_heads)
    return 0


def kv_spec_from_hf_config(model_id_or_path: str, dtype: str) -> KVSpec:
    """Read an HF config without loading model weights."""

    try:
        from transformers import AutoConfig
    except Exception as exc:  # pragma: no cover - depends on optional install state
        raise RuntimeError("transformers is required to read Hugging Face configs") from exc

    cfg = AutoConfig.from_pretrained(model_id_or_path, trust_remote_code=True)
    num_layers = _infer_num_layers(cfg)
    num_attention_heads = _infer_num_attention_heads(cfg)
    hidden_size = _infer_hidden_size(cfg)
    num_kv_heads = _infer_num_kv_heads(cfg, num_attention_heads)
    head_dim = _infer_head_dim(cfg, hidden_size, num_attention_heads)
    tokenizer_id = str(_get_attr_any(cfg, ("tokenizer_class",), None) or model_id_or_path)
    return KVSpec(
        model_id=str(model_id_or_path),
        tokenizer_id=tokenizer_id,
        num_layers=num_layers,
        num_attention_heads=num_attention_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        hidden_size=hidden_size,
        dtype=str(dtype),
        max_position_embeddings=_optional_int(_get_attr_any(cfg, ("max_position_embeddings",), None)),
        rope_theta=_optional_float(_get_attr_any(cfg, ("rope_theta",), None)),
        rope_scaling=_get_attr_any(cfg, ("rope_scaling",), None)
        if isinstance(_get_attr_any(cfg, ("rope_scaling",), None), dict)
        else None,
        kv_layout="hf_contiguous",
    )


def kv_spec_from_vllm_runtime(llm_or_engine: Any) -> KVSpec:
    """Best-effort KV spec extraction from a live vLLM object.

    vLLM's public/private object graph changes across versions.  This helper is
    deliberately conservative: it records what can be discovered and lets
    compatibility checks fail if required fields are absent.
    """

    vllm_version = None
    try:  # pragma: no cover - vLLM is not present in the local Windows env
        vllm = importlib.import_module("vllm")
        vllm_version = getattr(vllm, "__version__", None)
    except Exception:
        pass

    cfg = _find_vllm_model_config(llm_or_engine)
    hf_cfg = _get_attr_any(cfg, ("hf_config", "model_config", "config"), cfg)
    num_layers = _infer_num_layers(hf_cfg)
    num_attention_heads = _infer_num_attention_heads(hf_cfg)
    hidden_size = _infer_hidden_size(hf_cfg)
    num_kv_heads = _infer_num_kv_heads(hf_cfg, num_attention_heads)
    head_dim = _infer_head_dim(hf_cfg, hidden_size, num_attention_heads)
    model_id = str(_get_attr_any(cfg, ("model", "model_id", "model_path"), "") or "")
    dtype = str(_get_attr_any(cfg, ("dtype",), "") or "")
    block_size = _optional_int(_get_attr_any(cfg, ("block_size",), None))
    return KVSpec(
        model_id=model_id,
        tokenizer_id=str(_get_attr_any(cfg, ("tokenizer",), model_id) or model_id),
        num_layers=num_layers,
        num_attention_heads=num_attention_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        hidden_size=hidden_size,
        dtype=dtype,
        max_position_embeddings=_optional_int(_get_attr_any(hf_cfg, ("max_position_embeddings",), None)),
        rope_theta=_optional_float(_get_attr_any(hf_cfg, ("rope_theta",), None)),
        rope_scaling=_get_attr_any(hf_cfg, ("rope_scaling",), None)
        if isinstance(_get_attr_any(hf_cfg, ("rope_scaling",), None), dict)
        else None,
        block_size=block_size,
        vllm_version=vllm_version,
        attention_backend=_optional_str(_get_attr_any(cfg, ("attention_backend",), None)),
        kv_layout="vllm_paged",
    )


def _find_vllm_model_config(obj: Any) -> Any:
    candidates = [
        obj,
        _get_attr_any(obj, ("llm_engine",), None),
        _get_attr_any(obj, ("engine",), None),
        _get_attr_any(obj, ("model_executor",), None),
    ]
    for candidate in list(candidates):
        if candidate is None:
            continue
        for name in ("model_config", "vllm_config", "config"):
            value = _get_attr_any(candidate, (name,), None)
            if value is not None:
                candidates.append(value)
                if name == "model_config":
                    return value
    return obj


def assert_consumer_kv_compatible(bundle_spec: KVSpec, runtime_spec: KVSpec) -> None:
    mismatches: list[str] = []
    for field in (
        "num_layers",
        "num_attention_heads",
        "num_kv_heads",
        "head_dim",
        "hidden_size",
    ):
        left = getattr(bundle_spec, field)
        right = getattr(runtime_spec, field)
        if left and right and left != right:
            mismatches.append(f"{field}: bundle={left} runtime={right}")
        elif not left or not right:
            mismatches.append(f"{field}: missing bundle={left} runtime={right}")

    if bundle_spec.dtype and runtime_spec.dtype and _normalize_dtype(bundle_spec.dtype) != _normalize_dtype(runtime_spec.dtype):
        mismatches.append(f"dtype: bundle={bundle_spec.dtype} runtime={runtime_spec.dtype}")
    if bundle_spec.rope_theta and runtime_spec.rope_theta and float(bundle_spec.rope_theta) != float(runtime_spec.rope_theta):
        mismatches.append(f"rope_theta: bundle={bundle_spec.rope_theta} runtime={runtime_spec.rope_theta}")
    if mismatches:
        raise KVSpecError("KV specs are not compatible: " + "; ".join(mismatches))


def _normalize_dtype(value: str) -> str:
    value = str(value).lower().replace("torch.", "")
    aliases = {"bfloat16": "bf16", "float16": "fp16", "float32": "fp32"}
    return aliases.get(value, value)


def assert_token_ids_identical(a: np.ndarray, b: np.ndarray) -> None:
    arr_a = np.asarray(a)
    arr_b = np.asarray(b)
    if arr_a.shape != arr_b.shape:
        raise KVSpecError(f"token ids shape mismatch: {arr_a.shape} != {arr_b.shape}")
    if not np.array_equal(arr_a, arr_b):
        diff = np.argwhere(arr_a != arr_b)
        first = tuple(int(x) for x in diff[0]) if diff.size else ()
        raise KVSpecError(f"token ids differ at {first}")
