from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .torch_utils import require_torch, tensor_to_device, tensor_to_numpy


REUSE_MODE_XKV_GROUP = "xkv_group"
REUSE_MODE_LEGACY_LAYERWISE = "legacy_layerwise"
TRANSLATOR_PATH_SHARED_Z = "shared_z"
SVD_BACKEND_MOMENTS_EIGH = "moments_eigh"
SHARED_Z_ACTIVATION_IDENTITY = "identity"


@dataclass
class ReuseGroupWeights:
    enc_k: Any
    dec_k_b: Any
    enc_v: Any
    dec_v_b: Any
    mean_a_k: Any | None = None
    mean_b_k: Any | None = None
    mean_a_v: Any | None = None
    mean_b_v: Any | None = None
    rank_k: int = 0
    rank_v: int = 0

    def __post_init__(self) -> None:
        if self.rank_k <= 0:
            self.rank_k = int(self.enc_k.shape[1])
        if self.rank_v <= 0:
            self.rank_v = int(self.enc_v.shape[1])


@dataclass
class ReuseWeights:
    group_size: int
    rank: int
    num_layers: int
    num_groups: int
    weights: dict[int, ReuseGroupWeights]
    mode: str = REUSE_MODE_XKV_GROUP
    default_path: str = TRANSLATOR_PATH_SHARED_Z
    rank_k: int = 0
    rank_v: int = 0
    group_layers: dict[int, list[int]] | None = None
    center_before_svd: bool = False
    svd_backend: str = SVD_BACKEND_MOMENTS_EIGH
    max_groups: int = 0
    k_de_rope: bool = True
    k_re_rope: bool = True
    translator_hidden_dim: int = 0
    translator_activation: str = SHARED_Z_ACTIVATION_IDENTITY

    def __post_init__(self) -> None:
        if self.rank_k <= 0:
            self.rank_k = int(self.rank)
        if self.rank_v <= 0:
            self.rank_v = int(self.rank)
        if self.rank <= 0:
            self.rank = max(int(self.rank_k), int(self.rank_v))
        if self.group_layers is None:
            self.group_layers = group_layers_from_size(self.num_layers, self.group_size, max_groups=self.max_groups)
        else:
            normalized: dict[int, list[int]] = {}
            for gid, layers in self.group_layers.items():
                normalized[int(gid)] = [int(layer) for layer in layers]
            self.group_layers = normalized
        self.num_groups = len(self.group_layers)


def resolve_xkv_group_config(cfg: dict[str, Any]) -> dict[str, Any]:
    xkv_cfg = cfg.get("xkv_group", {}) if isinstance(cfg, dict) else {}
    reuse_cfg = cfg.get("reuse", {}) if isinstance(cfg, dict) else {}
    translator_cfg = cfg.get("translator", {}) if isinstance(cfg, dict) else {}
    shared_z_cfg = translator_cfg.get("shared_z", {}) if isinstance(translator_cfg, dict) else {}

    mode = str(cfg.get("mode") or xkv_cfg.get("mode") or reuse_cfg.get("mode") or REUSE_MODE_XKV_GROUP)
    default_path = str(translator_cfg.get("default_path") or TRANSLATOR_PATH_SHARED_Z)
    group_size = int(xkv_cfg.get("group_size") or reuse_cfg.get("group_size") or 1)
    legacy_rank = int(xkv_cfg.get("rank") or reuse_cfg.get("rank") or 128)
    rank_k = int(xkv_cfg.get("rank_k") or reuse_cfg.get("rank_k") or legacy_rank)
    rank_v = int(xkv_cfg.get("rank_v") or reuse_cfg.get("rank_v") or legacy_rank)
    ridge_lambda = float(xkv_cfg.get("ridge_lambda") or reuse_cfg.get("ridge_lambda") or 1e-4)
    center_before_svd = bool(xkv_cfg.get("center_before_svd", False))
    svd_backend = str(xkv_cfg.get("svd_backend") or SVD_BACKEND_MOMENTS_EIGH)
    max_groups = int(xkv_cfg.get("max_groups") or 0)
    return {
        "mode": mode,
        "default_path": default_path,
        "group_size": max(1, group_size),
        "rank": max(rank_k, rank_v),
        "rank_k": max(1, rank_k),
        "rank_v": max(1, rank_v),
        "ridge_lambda": float(ridge_lambda),
        "center_before_svd": center_before_svd,
        "svd_backend": svd_backend,
        "max_groups": max_groups,
        "translator_hidden_dim": int(shared_z_cfg.get("hidden_dim") or 0),
        "translator_activation": str(shared_z_cfg.get("activation") or SHARED_Z_ACTIVATION_IDENTITY),
    }


def group_layers_from_size(num_layers: int, group_size: int, *, max_groups: int = 0) -> dict[int, list[int]]:
    if num_layers <= 0:
        raise ValueError(f"num_layers 必须为正整数，实际为 {num_layers}")
    if group_size <= 0:
        raise ValueError(f"group_size 必须为正整数，实际为 {group_size}")
    out: dict[int, list[int]] = {}
    num_groups = (num_layers + group_size - 1) // group_size
    if max_groups > 0 and num_groups > max_groups:
        raise ValueError(f"xkv_group 需要 {num_groups} 个组，超过 max_groups={max_groups}")
    for gid in range(num_groups):
        start = gid * group_size
        stop = min(start + group_size, num_layers)
        out[gid] = list(range(start, stop))
    return out


def group_layers_for_gid(weights: ReuseWeights, gid: int) -> list[int]:
    if weights.group_layers is None:
        return group_layers_from_size(weights.num_layers, weights.group_size, max_groups=weights.max_groups)[gid]
    return list(weights.group_layers[int(gid)])


def build_group_specs(group_layers: dict[int, list[int]], head_dim: int) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for gid in sorted(group_layers):
        layers = [int(layer) for layer in group_layers[gid]]
        slices = []
        for offset, layer in enumerate(layers):
            start = int(offset * head_dim)
            end = int(start + head_dim)
            slices.append({"layer": int(layer), "start": start, "end": end})
        specs.append(
            {
                "group_id": int(gid),
                "layers": layers,
                "packed_feature_dim": int(len(layers) * head_dim),
                "layer_slices": slices,
            }
        )
    return specs


def flatten_kv(x):
    shape = tuple(x.shape)
    if len(shape) != 4:
        raise ValueError(f"期望 4D 张量，实际为 {shape}")
    b, kvh, t_steps, hd = shape
    return x.reshape(b * kvh * t_steps, hd)


def unflatten_kv(x, b: int, kvh: int, t_steps: int, hd: int):
    shape = tuple(x.shape)
    if len(shape) != 2:
        raise ValueError(f"期望 2D 矩阵，实际为 {shape}")
    return x.reshape(b, kvh, t_steps, hd)


def pack_group(layer_tensors: list[Any]):
    mats = [flatten_kv(tensor) for tensor in layer_tensors]
    first = mats[0]
    t = require_torch()
    if t.is_tensor(first):
        return t.cat(mats, dim=1)
    return np.concatenate(mats, axis=1)


def unpack_group(group_mat, group_size: int, hd: int, b: int, kvh: int, t_steps: int) -> list[Any]:
    out: list[Any] = []
    for idx in range(group_size):
        sl = group_mat[:, idx * hd : (idx + 1) * hd]
        out.append(unflatten_kv(sl, b, kvh, t_steps, hd))
    return out


def split_group_decoded_kv_to_layers(group_mat, *, num_layers: int, head_dim: int, batch_size: int, num_key_value_heads: int, prefix_tokens: int) -> list[Any]:
    return unpack_group(group_mat, num_layers, head_dim, batch_size, num_key_value_heads, prefix_tokens)


def _ridge_solve(x: np.ndarray, y: np.ndarray, ridge_lambda: float) -> np.ndarray:
    x_t_x = x.T @ x
    reg = ridge_lambda * np.eye(x_t_x.shape[0], dtype=x_t_x.dtype)
    inv = np.linalg.pinv(x_t_x + reg)
    return inv @ x.T @ y


def _joint_subspace(a: np.ndarray, b: np.ndarray, rank: int) -> np.ndarray:
    x = np.concatenate([a, b], axis=1)
    u, s, _vt = np.linalg.svd(x, full_matrices=False)
    r = min(rank, u.shape[1])
    z = u[:, :r] * s[:r]
    return z


def _center_rows_numpy(x: np.ndarray, mean: np.ndarray | None, *, subtract: bool) -> np.ndarray:
    if mean is None:
        return x
    if subtract:
        return x - mean.astype(x.dtype, copy=False)
    return x + mean.astype(x.dtype, copy=False)


def _center_rows_torch(x, mean, *, subtract: bool):
    t = require_torch()
    if mean is None:
        return x
    if not t.is_tensor(mean):
        mean = t.tensor(mean, device=x.device, dtype=x.dtype)
    else:
        mean = mean.to(device=x.device, dtype=x.dtype)
    if subtract:
        return x - mean
    return x + mean


def _maybe_center_rows(x, mean, *, subtract: bool):
    t = require_torch()
    if t.is_tensor(x):
        return _center_rows_torch(x, mean, subtract=subtract)
    return _center_rows_numpy(x, mean, subtract=subtract)


def _center_moments_torch(aa, ab, bb, sum_a, sum_b, count: float):
    t = require_torch()
    if count <= 0:
        return aa, ab, bb, None, None
    mean_a = (sum_a / count).to(dtype=t.float32)
    mean_b = (sum_b / count).to(dtype=t.float32)
    aa = aa - count * t.outer(mean_a, mean_a)
    ab = ab - count * t.outer(mean_a, mean_b)
    bb = bb - count * t.outer(mean_b, mean_b)
    return aa, ab, bb, mean_a, mean_b


def _group_shared_z_from_centered_moments_torch(aa, ab, bb, rank: int, ridge_lambda: float):
    t = require_torch()
    gram = t.cat([t.cat([aa, ab], dim=1), t.cat([ab.T, bb], dim=1)], dim=0)
    eigvals, eigvecs = t.linalg.eigh(gram)
    order = t.argsort(eigvals, descending=True)
    r = min(rank, eigvecs.shape[1])
    basis = eigvecs[:, order[:r]].to(dtype=t.float32)

    reg_in = ridge_lambda * t.eye(aa.shape[0], device=aa.device, dtype=aa.dtype)
    lhs_enc = t.linalg.pinv(aa + reg_in)
    xy = t.cat([aa, ab], dim=1)
    enc = lhs_enc @ xy @ basis

    ztz = basis.T @ gram @ basis
    reg_latent = ridge_lambda * t.eye(ztz.shape[0], device=ztz.device, dtype=ztz.dtype)
    rhs = basis.T @ t.cat([ab, bb], dim=0)
    dec = t.linalg.pinv(ztz + reg_latent) @ rhs
    return enc.to(dtype=t.float32), dec.to(dtype=t.float32)


def train_reuse_group_from_moments_torch(
    aa,
    ab,
    bb,
    *,
    rank: int,
    ridge_lambda: float,
    sum_a=None,
    sum_b=None,
    count: float = 0.0,
    center_before_svd: bool = False,
):
    if center_before_svd:
        aa, ab, bb, mean_a, mean_b = _center_moments_torch(aa, ab, bb, sum_a, sum_b, count)
    else:
        mean_a = None
        mean_b = None
    enc, dec = _group_shared_z_from_centered_moments_torch(aa, ab, bb, rank=rank, ridge_lambda=ridge_lambda)
    return enc, dec, mean_a, mean_b


def train_reuse_group(
    a_k: np.ndarray,
    b_k: np.ndarray,
    a_v: np.ndarray,
    b_v: np.ndarray,
    *,
    rank: int | None = None,
    ridge_lambda: float,
    rank_k: int | None = None,
    rank_v: int | None = None,
    center_before_svd: bool = False,
) -> ReuseGroupWeights:
    rank_k = int(rank_k or rank or 0)
    rank_v = int(rank_v or rank or 0)
    if rank_k <= 0 or rank_v <= 0:
        raise ValueError("rank_k 和 rank_v 必须为正整数")

    mean_a_k = a_k.mean(axis=0, dtype=np.float32) if center_before_svd else None
    mean_b_k = b_k.mean(axis=0, dtype=np.float32) if center_before_svd else None
    mean_a_v = a_v.mean(axis=0, dtype=np.float32) if center_before_svd else None
    mean_b_v = b_v.mean(axis=0, dtype=np.float32) if center_before_svd else None

    a_k_centered = _center_rows_numpy(a_k, mean_a_k, subtract=True)
    b_k_centered = _center_rows_numpy(b_k, mean_b_k, subtract=True)
    a_v_centered = _center_rows_numpy(a_v, mean_a_v, subtract=True)
    b_v_centered = _center_rows_numpy(b_v, mean_b_v, subtract=True)

    z_k = _joint_subspace(a_k_centered, b_k_centered, rank=rank_k)
    z_v = _joint_subspace(a_v_centered, b_v_centered, rank=rank_v)

    enc_k = _ridge_solve(a_k_centered, z_k, ridge_lambda)
    enc_v = _ridge_solve(a_v_centered, z_v, ridge_lambda)
    dec_k_b = _ridge_solve(z_k, b_k_centered, ridge_lambda)
    dec_v_b = _ridge_solve(z_v, b_v_centered, ridge_lambda)

    return ReuseGroupWeights(
        enc_k=enc_k,
        dec_k_b=dec_k_b,
        enc_v=enc_v,
        dec_v_b=dec_v_b,
        mean_a_k=mean_a_k,
        mean_b_k=mean_b_k,
        mean_a_v=mean_a_v,
        mean_b_v=mean_b_v,
        rank_k=int(enc_k.shape[1]),
        rank_v=int(enc_v.shape[1]),
    )


def train_reuse_group_torch(
    a_k,
    b_k,
    a_v,
    b_v,
    *,
    rank: int | None = None,
    ridge_lambda: float,
    rank_k: int | None = None,
    rank_v: int | None = None,
    center_before_svd: bool = False,
) -> ReuseGroupWeights:
    t = require_torch()
    device = str(a_k.device)
    a_k = tensor_to_device(a_k, device=device, dtype="float32")
    b_k = tensor_to_device(b_k, device=device, dtype="float32")
    a_v = tensor_to_device(a_v, device=device, dtype="float32")
    b_v = tensor_to_device(b_v, device=device, dtype="float32")
    rank_k = int(rank_k or rank or 0)
    rank_v = int(rank_v or rank or 0)
    if rank_k <= 0 or rank_v <= 0:
        raise ValueError("rank_k 和 rank_v 必须为正整数")

    mean_a_k = t.mean(a_k, dim=0) if center_before_svd else None
    mean_b_k = t.mean(b_k, dim=0) if center_before_svd else None
    mean_a_v = t.mean(a_v, dim=0) if center_before_svd else None
    mean_b_v = t.mean(b_v, dim=0) if center_before_svd else None

    a_k_centered = _center_rows_torch(a_k, mean_a_k, subtract=True)
    b_k_centered = _center_rows_torch(b_k, mean_b_k, subtract=True)
    a_v_centered = _center_rows_torch(a_v, mean_a_v, subtract=True)
    b_v_centered = _center_rows_torch(b_v, mean_b_v, subtract=True)

    gram_aa_k = a_k_centered.T @ a_k_centered
    gram_ab_k = a_k_centered.T @ b_k_centered
    gram_bb_k = b_k_centered.T @ b_k_centered
    gram_aa_v = a_v_centered.T @ a_v_centered
    gram_ab_v = a_v_centered.T @ b_v_centered
    gram_bb_v = b_v_centered.T @ b_v_centered

    enc_k, dec_k_b = _group_shared_z_from_centered_moments_torch(gram_aa_k, gram_ab_k, gram_bb_k, rank=rank_k, ridge_lambda=ridge_lambda)
    enc_v, dec_v_b = _group_shared_z_from_centered_moments_torch(gram_aa_v, gram_ab_v, gram_bb_v, rank=rank_v, ridge_lambda=ridge_lambda)
    return ReuseGroupWeights(
        enc_k=enc_k,
        dec_k_b=dec_k_b,
        enc_v=enc_v,
        dec_v_b=dec_v_b,
        mean_a_k=None if mean_a_k is None else mean_a_k.to(dtype=t.float32),
        mean_b_k=None if mean_b_k is None else mean_b_k.to(dtype=t.float32),
        mean_a_v=None if mean_a_v is None else mean_a_v.to(dtype=t.float32),
        mean_b_v=None if mean_b_v is None else mean_b_v.to(dtype=t.float32),
        rank_k=int(enc_k.shape[1]),
        rank_v=int(enc_v.shape[1]),
    )


def encode_group_k(a_k, w: ReuseGroupWeights):
    centered = _maybe_center_rows(a_k, w.mean_a_k, subtract=True)
    return centered @ w.enc_k


def encode_group_v(a_v, w: ReuseGroupWeights):
    centered = _maybe_center_rows(a_v, w.mean_a_v, subtract=True)
    return centered @ w.enc_v


def decode_group_k(z_k, w: ReuseGroupWeights):
    rec = z_k @ w.dec_k_b
    return _maybe_center_rows(rec, w.mean_b_k, subtract=False)


def decode_group_v(z_v, w: ReuseGroupWeights):
    rec = z_v @ w.dec_v_b
    return _maybe_center_rows(rec, w.mean_b_v, subtract=False)


def encode_group_shared_z(a_k, a_v, w: ReuseGroupWeights):
    return encode_group_k(a_k, w), encode_group_v(a_v, w)


def decode_group_shared_z(z_k, z_v, w: ReuseGroupWeights):
    return decode_group_k(z_k, w), decode_group_v(z_v, w)


def encode_group(a_k, a_v, w: ReuseGroupWeights):
    return encode_group_shared_z(a_k, a_v, w)


def decode_group(z_k, z_v, w: ReuseGroupWeights):
    return decode_group_shared_z(z_k, z_v, w)


def move_reuse_weights(weights: ReuseWeights, device: str, dtype: str = "float32", cpu_fallback: bool = True) -> ReuseWeights:
    moved: dict[int, ReuseGroupWeights] = {}
    for gid, row in weights.weights.items():
        moved[gid] = ReuseGroupWeights(
            enc_k=tensor_to_device(row.enc_k, device=device, dtype=dtype, cpu_fallback=cpu_fallback),
            dec_k_b=tensor_to_device(row.dec_k_b, device=device, dtype=dtype, cpu_fallback=cpu_fallback),
            enc_v=tensor_to_device(row.enc_v, device=device, dtype=dtype, cpu_fallback=cpu_fallback),
            dec_v_b=tensor_to_device(row.dec_v_b, device=device, dtype=dtype, cpu_fallback=cpu_fallback),
            mean_a_k=None if row.mean_a_k is None else tensor_to_device(row.mean_a_k, device=device, dtype=dtype, cpu_fallback=cpu_fallback),
            mean_b_k=None if row.mean_b_k is None else tensor_to_device(row.mean_b_k, device=device, dtype=dtype, cpu_fallback=cpu_fallback),
            mean_a_v=None if row.mean_a_v is None else tensor_to_device(row.mean_a_v, device=device, dtype=dtype, cpu_fallback=cpu_fallback),
            mean_b_v=None if row.mean_b_v is None else tensor_to_device(row.mean_b_v, device=device, dtype=dtype, cpu_fallback=cpu_fallback),
            rank_k=row.rank_k,
            rank_v=row.rank_v,
        )
    return ReuseWeights(
        group_size=weights.group_size,
        rank=weights.rank,
        rank_k=weights.rank_k,
        rank_v=weights.rank_v,
        num_layers=weights.num_layers,
        num_groups=weights.num_groups,
        weights=moved,
        mode=weights.mode,
        default_path=weights.default_path,
        group_layers=weights.group_layers,
        center_before_svd=weights.center_before_svd,
        svd_backend=weights.svd_backend,
        max_groups=weights.max_groups,
        k_de_rope=weights.k_de_rope,
        k_re_rope=weights.k_re_rope,
        translator_hidden_dim=weights.translator_hidden_dim,
        translator_activation=weights.translator_activation,
    )


def reuse_weights_to_numpy(weights: ReuseWeights) -> ReuseWeights:
    moved: dict[int, ReuseGroupWeights] = {}
    for gid, row in weights.weights.items():
        moved[gid] = ReuseGroupWeights(
            enc_k=tensor_to_numpy(row.enc_k),
            dec_k_b=tensor_to_numpy(row.dec_k_b),
            enc_v=tensor_to_numpy(row.enc_v),
            dec_v_b=tensor_to_numpy(row.dec_v_b),
            mean_a_k=None if row.mean_a_k is None else tensor_to_numpy(row.mean_a_k),
            mean_b_k=None if row.mean_b_k is None else tensor_to_numpy(row.mean_b_k),
            mean_a_v=None if row.mean_a_v is None else tensor_to_numpy(row.mean_a_v),
            mean_b_v=None if row.mean_b_v is None else tensor_to_numpy(row.mean_b_v),
            rank_k=row.rank_k,
            rank_v=row.rank_v,
        )
    return ReuseWeights(
        group_size=weights.group_size,
        rank=weights.rank,
        rank_k=weights.rank_k,
        rank_v=weights.rank_v,
        num_layers=weights.num_layers,
        num_groups=weights.num_groups,
        weights=moved,
        mode=weights.mode,
        default_path=weights.default_path,
        group_layers=weights.group_layers,
        center_before_svd=weights.center_before_svd,
        svd_backend=weights.svd_backend,
        max_groups=weights.max_groups,
        k_de_rope=weights.k_de_rope,
        k_re_rope=weights.k_re_rope,
        translator_hidden_dim=weights.translator_hidden_dim,
        translator_activation=weights.translator_activation,
    )
