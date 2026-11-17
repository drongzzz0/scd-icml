from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .torch_utils import require_torch


@dataclass(frozen=True)
class RopeSpec:
    rope_theta: float
    head_dim: int
    rotary_dim: int
    scaling_type: str = "none"
    scaling_factor: float = 1.0
    max_position_embeddings: int = 0
    original_max_position_embeddings: int = 0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RopeSpec":
        return cls(
            rope_theta=float(payload.get("rope_theta", 10000.0)),
            head_dim=int(payload["head_dim"]),
            rotary_dim=int(payload.get("rotary_dim", payload["head_dim"])),
            scaling_type=str(payload.get("scaling_type", "none") or "none"),
            scaling_factor=float(payload.get("scaling_factor", 1.0) or 1.0),
            max_position_embeddings=int(payload.get("max_position_embeddings", 0) or 0),
            original_max_position_embeddings=int(payload.get("original_max_position_embeddings", 0) or 0),
        )


def _normalize_rotary_dim(head_dim: int, rotary_dim: int | None = None, partial_rotary_factor: float | None = None) -> int:
    if head_dim <= 0:
        raise ValueError(f"head_dim 必须为正整数，实际为 {head_dim}")
    if rotary_dim is None:
        factor = float(partial_rotary_factor if partial_rotary_factor is not None else 1.0)
        rotary_dim = int(round(head_dim * factor))
    rotary_dim = max(2, min(int(rotary_dim), int(head_dim)))
    if rotary_dim % 2 != 0:
        rotary_dim -= 1
    if rotary_dim <= 0:
        raise ValueError(f"rotary_dim 必须为正偶数，实际为 {rotary_dim}")
    return rotary_dim


def rope_spec_from_config_dict(config: dict[str, Any]) -> RopeSpec:
    hidden_size = int(config.get("hidden_size") or config.get("n_embd") or 0)
    num_attention_heads = int(config.get("num_attention_heads") or config.get("n_head") or 0)
    head_dim = int(config.get("head_dim") or 0)
    if head_dim <= 0:
        if hidden_size <= 0 or num_attention_heads <= 0:
            raise ValueError("无法从模型配置推断 head_dim，请检查 hidden_size / num_attention_heads")
        head_dim = hidden_size // num_attention_heads

    rope_scaling = config.get("rope_scaling")
    scaling_type = "none"
    scaling_factor = 1.0
    original_max_position_embeddings = int(config.get("max_position_embeddings", 0) or 0)
    if isinstance(rope_scaling, dict):
        scaling_type = str(rope_scaling.get("rope_type") or rope_scaling.get("type") or "linear")
        factor = rope_scaling.get("factor")
        if factor is None:
            factor = rope_scaling.get("short_factor")
        if isinstance(factor, (int, float)):
            scaling_factor = float(factor)
        original_max_position_embeddings = int(
            rope_scaling.get("original_max_position_embeddings")
            or rope_scaling.get("max_position_embeddings")
            or original_max_position_embeddings
            or 0
        )

    return RopeSpec(
        rope_theta=float(config.get("rope_theta") or config.get("rotary_emb_base") or 10000.0),
        head_dim=head_dim,
        rotary_dim=_normalize_rotary_dim(
            head_dim,
            rotary_dim=config.get("rotary_dim"),
            partial_rotary_factor=config.get("partial_rotary_factor"),
        ),
        scaling_type=scaling_type,
        scaling_factor=float(scaling_factor or 1.0),
        max_position_embeddings=int(config.get("max_position_embeddings", 0) or 0),
        original_max_position_embeddings=original_max_position_embeddings,
    )


def rope_spec_from_hf_config(config: Any) -> RopeSpec:
    if hasattr(config, "to_dict"):
        return rope_spec_from_config_dict(config.to_dict())
    if isinstance(config, dict):
        return rope_spec_from_config_dict(config)
    raise TypeError(f"不支持的模型配置类型: {type(config)!r}")


def load_rope_spec_from_model_path(model_path: str | Path) -> RopeSpec:
    config_path = Path(model_path) / "config.json"
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    return rope_spec_from_config_dict(payload)


def _rotate_half_numpy(x: np.ndarray) -> np.ndarray:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    out = np.empty_like(x)
    out[..., ::2] = -x2
    out[..., 1::2] = x1
    return out


def _rotate_half_torch(x):
    t = require_torch()
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    out = t.empty_like(x)
    out[..., ::2] = -x2
    out[..., 1::2] = x1
    return out


def _scaled_inv_freq_numpy(rotary_dim: int, base: float, scaling_type: str, scaling_factor: float, seq_len: int, original_max_position_embeddings: int) -> np.ndarray:
    idx = np.arange(0, rotary_dim, 2, dtype=np.float64)
    if scaling_type in {"dynamic", "dynamic_ntk"} and original_max_position_embeddings > 0 and seq_len > original_max_position_embeddings and rotary_dim > 2:
        ratio = (scaling_factor * seq_len / original_max_position_embeddings) - (scaling_factor - 1.0)
        base = float(base) * (ratio ** (rotary_dim / (rotary_dim - 2)))
    inv_freq = 1.0 / (float(base) ** (idx / rotary_dim))
    if scaling_type in {"linear", "yarn", "longrope", "llama3"} and scaling_factor and scaling_factor != 1.0:
        inv_freq = inv_freq / float(scaling_factor)
    return inv_freq


def _scaled_inv_freq_torch(rotary_dim: int, base: float, scaling_type: str, scaling_factor: float, seq_len: int, original_max_position_embeddings: int, device) -> Any:
    t = require_torch()
    idx = t.arange(0, rotary_dim, 2, dtype=t.float32, device=device)
    if scaling_type in {"dynamic", "dynamic_ntk"} and original_max_position_embeddings > 0 and seq_len > original_max_position_embeddings and rotary_dim > 2:
        ratio = (scaling_factor * seq_len / original_max_position_embeddings) - (scaling_factor - 1.0)
        base = float(base) * (ratio ** (rotary_dim / (rotary_dim - 2)))
    inv_freq = 1.0 / (float(base) ** (idx / rotary_dim))
    if scaling_type in {"linear", "yarn", "longrope", "llama3"} and scaling_factor and scaling_factor != 1.0:
        inv_freq = inv_freq / float(scaling_factor)
    return inv_freq


def build_rope_cache(
    seq_len: int,
    head_dim: int,
    base: float = 10000.0,
    *,
    rotary_dim: int | None = None,
    scaling_type: str = "none",
    scaling_factor: float = 1.0,
    original_max_position_embeddings: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    rotary_dim = _normalize_rotary_dim(head_dim, rotary_dim=rotary_dim)
    inv_freq = _scaled_inv_freq_numpy(
        rotary_dim=rotary_dim,
        base=base,
        scaling_type=scaling_type,
        scaling_factor=scaling_factor,
        seq_len=seq_len,
        original_max_position_embeddings=original_max_position_embeddings,
    )
    pos = np.arange(seq_len, dtype=np.float64)
    freqs = np.outer(pos, inv_freq)
    cos = np.repeat(np.cos(freqs), 2, axis=1).astype(np.float32)
    sin = np.repeat(np.sin(freqs), 2, axis=1).astype(np.float32)
    return cos, sin


def build_rope_cache_torch(seq_len: int, spec: RopeSpec, device=None, dtype=None) -> tuple[Any, Any]:
    t = require_torch()
    target_dtype = dtype or t.float32
    inv_freq = _scaled_inv_freq_torch(
        rotary_dim=spec.rotary_dim,
        base=spec.rope_theta,
        scaling_type=spec.scaling_type,
        scaling_factor=spec.scaling_factor,
        seq_len=seq_len,
        original_max_position_embeddings=spec.original_max_position_embeddings,
        device=device,
    )
    pos = t.arange(seq_len, device=device, dtype=t.float32)
    freqs = t.outer(pos, inv_freq)
    cos = t.repeat_interleave(t.cos(freqs), repeats=2, dim=1).to(dtype=target_dtype)
    sin = t.repeat_interleave(t.sin(freqs), repeats=2, dim=1).to(dtype=target_dtype)
    return cos, sin


def _apply_rope_numpy(x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    if x.ndim != 4:
        raise ValueError(f"期望形状为 [B,KVH,T,HD]，实际为 {x.shape}")
    if cos.shape != sin.shape:
        raise ValueError("cos 和 sin 的形状必须一致")
    rotary_dim = int(cos.shape[1])
    if x.shape[-2] != cos.shape[0]:
        raise ValueError("RoPE 缓存的序列长度与输入张量不一致")
    if rotary_dim > x.shape[-1]:
        raise ValueError("RoPE 缓存的旋转维度不能大于输入 head_dim")
    rot = x[..., :rotary_dim]
    tail = x[..., rotary_dim:]
    cos_e = cos[np.newaxis, np.newaxis, :, :]
    sin_e = sin[np.newaxis, np.newaxis, :, :]
    rot_out = (rot * cos_e) + (_rotate_half_numpy(rot) * sin_e)
    if tail.size == 0:
        return rot_out
    return np.concatenate([rot_out, tail], axis=-1)


def apply_rope(k_raw: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    return _apply_rope_numpy(k_raw, cos, sin)


def invert_rope(k_rope: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    return _apply_rope_numpy(k_rope, cos, -sin)


def _apply_rope_torch(x, cos, sin):
    t = require_torch()
    if x.ndim != 4:
        raise ValueError(f"期望形状为 [B,KVH,T,HD]，实际为 {tuple(x.shape)}")
    if tuple(cos.shape) != tuple(sin.shape):
        raise ValueError("cos 和 sin 的形状必须一致")
    rotary_dim = int(cos.shape[1])
    if int(x.shape[-2]) != int(cos.shape[0]):
        raise ValueError("RoPE 缓存的序列长度与输入张量不一致")
    if rotary_dim > int(x.shape[-1]):
        raise ValueError("RoPE 缓存的旋转维度不能大于输入 head_dim")
    rot = x[..., :rotary_dim]
    tail = x[..., rotary_dim:]
    cos_e = cos.unsqueeze(0).unsqueeze(0)
    sin_e = sin.unsqueeze(0).unsqueeze(0)
    rot_out = (rot * cos_e) + (_rotate_half_torch(rot) * sin_e)
    if int(tail.shape[-1]) == 0:
        return rot_out
    return t.cat([rot_out, tail], dim=-1)


def apply_rope_torch(x, cos, sin):
    return _apply_rope_torch(x, cos, sin)


def invert_rope_torch(x, cos, sin):
    return _apply_rope_torch(x, cos, -sin)


def de_rope_k_torch(k_rope, spec: RopeSpec):
    cos, sin = build_rope_cache_torch(
        seq_len=int(k_rope.shape[-2]),
        spec=spec,
        device=k_rope.device,
        dtype=k_rope.dtype,
    )
    return invert_rope_torch(k_rope, cos, sin)


def re_rope_k_torch(k_raw, spec: RopeSpec):
    cos, sin = build_rope_cache_torch(
        seq_len=int(k_raw.shape[-2]),
        spec=spec,
        device=k_raw.device,
        dtype=k_raw.dtype,
    )
    return apply_rope_torch(k_raw, cos, sin)


def remap_rope_k_torch(k_src, src_spec: RopeSpec, dst_spec: RopeSpec):
    raw = de_rope_k_torch(k_src, src_spec)
    return re_rope_k_torch(raw, dst_spec)

def load_rope_specs_from_models_config(models_config_path: str | Path) -> tuple[RopeSpec, RopeSpec]:
    payload = json.loads(Path(models_config_path).read_text(encoding="utf-8-sig"))
    model_a_path = payload["model_a"]["path"]
    model_b_path = payload["model_b"]["path"]
    return load_rope_spec_from_model_path(model_a_path), load_rope_spec_from_model_path(model_b_path)


def resolve_rope_specs(meta: dict[str, Any], models_config_path: str | Path | None = None) -> tuple[RopeSpec, RopeSpec]:
    rope_meta = meta.get("rope")
    if isinstance(rope_meta, dict) and isinstance(rope_meta.get("model_a"), dict) and isinstance(rope_meta.get("model_b"), dict):
        return RopeSpec.from_dict(rope_meta["model_a"]), RopeSpec.from_dict(rope_meta["model_b"])
    if models_config_path:
        return load_rope_specs_from_models_config(models_config_path)
    raise ValueError("pairs manifest 中缺少 rope 元数据，且未提供 models-config，无法解析 A/B 的 RoPE 参数")
