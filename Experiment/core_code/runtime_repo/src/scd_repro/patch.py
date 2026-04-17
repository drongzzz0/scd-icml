from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .attention import masked_mse
from .torch_utils import require_torch, tensor_to_device, tensor_to_numpy

PATCH_FIT_MODE_ALIGNED_RUNTIME_COMPOSED_V1 = "aligned_runtime_composed_v1"
PATCH_FIT_MODE_DIRECT_KV_V1 = "direct_kv_v1"
PATCH_FIT_MODE_DIRECT_KV_KNORM_V1 = "direct_kv_knorm_v1"
PATCH_FIT_MODE_XNORM_SEMANTIC_V1 = "xnorm_semantic_v1"

_TORCH = require_torch()
_NN = _TORCH.nn
_F = _TORCH.nn.functional


@dataclass
class PatchLayerWeights:
    w: Any
    b: Any


@dataclass
class PatchWeights:
    code_dim: int
    hidden_dim: int
    selected_layers: list[int]
    layers: dict[int, PatchLayerWeights]


@dataclass
class RuntimeAlignedPatchFit:
    align: PatchLayerWeights
    wk: Any
    wv: Any
    patch_xnorm_error: float
    patch_k_error: float
    patch_v_error: float
    patch_kv_error: float


@dataclass
class DirectKVPatchFit:
    wk: Any
    wv: Any
    patch_k_error: float
    patch_v_error: float
    patch_kv_error: float


@dataclass
class SemanticPatchLossBreakdown:
    hidden_mse: float
    k_mse: float
    v_mse: float
    sem_kl: float
    total_loss: float
    topk_match: float


class PatchMLP(_NN.Module):
    def __init__(self, code_dim: int, hidden_dim: int, mlp_hidden_dim: int) -> None:
        super().__init__()
        if code_dim <= 0 or hidden_dim <= 0 or mlp_hidden_dim <= 0:
            raise ValueError("PatchMLP 维度必须为正整数")
        self.fc1 = _NN.Linear(code_dim, mlp_hidden_dim)
        self.act = _NN.GELU()
        self.fc2 = _NN.Linear(mlp_hidden_dim, hidden_dim)

    def forward(self, z_h):
        return self.fc2(self.act(self.fc1(z_h)))


class SemanticProbe(_NN.Module):
    def __init__(self, hidden_dim: int, probe_dim: int, num_classes: int) -> None:
        super().__init__()
        if hidden_dim <= 0 or probe_dim <= 0 or num_classes <= 0:
            raise ValueError("SemanticProbe 维度必须为正整数")
        self.fc1 = _NN.Linear(hidden_dim, probe_dim)
        self.act = _NN.GELU()
        self.fc2 = _NN.Linear(probe_dim, num_classes)

    def forward(self, hidden_states):
        return self.fc2(self.act(self.fc1(hidden_states)))


class XNormSemanticPatchLayer(_NN.Module):
    def __init__(
        self,
        hidden_dim: int,
        code_dim: int,
        mlp_hidden_dim: int,
        *,
        enc_h_is_identity: bool,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0 or code_dim <= 0:
            raise ValueError("XNormSemanticPatchLayer 维度必须为正整数")
        if enc_h_is_identity and code_dim != hidden_dim:
            raise ValueError("enc_h_is_identity=True 时 code_dim 必须等于 hidden_dim")
        self.hidden_dim = int(hidden_dim)
        self.code_dim = int(code_dim)
        self.enc_h_is_identity = bool(enc_h_is_identity)
        if self.enc_h_is_identity:
            self.enc_h = None
        else:
            self.enc_h = _NN.Parameter(_TORCH.empty((hidden_dim, code_dim), dtype=_TORCH.float32))
            _NN.init.xavier_uniform_(self.enc_h)
        self.patch_mlp = PatchMLP(code_dim=code_dim, hidden_dim=hidden_dim, mlp_hidden_dim=mlp_hidden_dim)

    def encode(self, xnorm_a):
        if self.enc_h is None:
            return xnorm_a
        return xnorm_a @ self.enc_h

    def forward(self, xnorm_a):
        z_h = self.encode(xnorm_a)
        return self.patch_mlp(z_h)


def _augment_with_bias_torch(x):
    t = require_torch()
    return t.cat([x, t.ones((x.shape[0], 1), device=x.device, dtype=x.dtype)], dim=1)


def create_patch_align_stats_torch(hidden_dim: int, device: str | Any) -> dict[str, Any]:
    t = require_torch()
    return {
        "ztz": t.zeros((hidden_dim + 1, hidden_dim + 1), device=device, dtype=t.float32),
        "zty": t.zeros((hidden_dim + 1, hidden_dim), device=device, dtype=t.float32),
        "yty": t.zeros((hidden_dim, hidden_dim), device=device, dtype=t.float32),
        "rows": t.tensor(0.0, device=device, dtype=t.float64),
    }


def create_patch_kv_stats_torch(hidden_dim: int, kv_dim: int, device: str | Any) -> dict[str, Any]:
    t = require_torch()
    return {
        "xhat_txhat": t.zeros((hidden_dim, hidden_dim), device=device, dtype=t.float32),
        "xhat_tk": t.zeros((hidden_dim, kv_dim), device=device, dtype=t.float32),
        "xhat_tv": t.zeros((hidden_dim, kv_dim), device=device, dtype=t.float32),
        "ktk": t.zeros((kv_dim, kv_dim), device=device, dtype=t.float32),
        "vtv": t.zeros((kv_dim, kv_dim), device=device, dtype=t.float32),
        "rows": t.tensor(0.0, device=device, dtype=t.float64),
    }


def create_direct_kv_stats_torch(hidden_dim: int, kv_dim: int, device: str | Any) -> dict[str, Any]:
    t = require_torch()
    feature_dim = hidden_dim + 1
    return {
        "ztz": t.zeros((feature_dim, feature_dim), device=device, dtype=t.float32),
        "ztk": t.zeros((feature_dim, kv_dim), device=device, dtype=t.float32),
        "ztv": t.zeros((feature_dim, kv_dim), device=device, dtype=t.float32),
        "ktk": t.zeros((kv_dim, kv_dim), device=device, dtype=t.float32),
        "vtv": t.zeros((kv_dim, kv_dim), device=device, dtype=t.float32),
        "rows": t.tensor(0.0, device=device, dtype=t.float64),
    }


def create_patch_error_stats_torch(device: str | Any) -> dict[str, Any]:
    t = require_torch()
    return {
        "xnorm_sse": t.tensor(0.0, device=device, dtype=t.float64),
        "k_sse": t.tensor(0.0, device=device, dtype=t.float64),
        "v_sse": t.tensor(0.0, device=device, dtype=t.float64),
        "rows": t.tensor(0.0, device=device, dtype=t.float64),
    }


def create_direct_kv_error_stats_torch(device: str | Any) -> dict[str, Any]:
    t = require_torch()
    return {
        "k_sse": t.tensor(0.0, device=device, dtype=t.float64),
        "v_sse": t.tensor(0.0, device=device, dtype=t.float64),
        "rows": t.tensor(0.0, device=device, dtype=t.float64),
    }


def _frobenius_inner_torch(lhs, rhs) -> Any:
    t = require_torch()
    return t.sum(lhs * rhs, dtype=t.float64)


def _mean_square_from_stats_torch(sse: Any, rows: float, width: int) -> float:
    return float(sse.item() / max(1.0, rows * float(width)))


def accumulate_patch_align_stats_torch(xa, xb, stats: dict[str, Any]) -> None:
    t = require_torch()
    device = stats["ztz"].device
    xa = tensor_to_device(xa, device=str(device), dtype="float32")
    xb = tensor_to_device(xb, device=str(device), dtype="float32")
    if xa.ndim != 2 or xb.ndim != 2:
        raise ValueError("xa 和 xb 必须是 [N,D] 矩阵")
    z_aug = _augment_with_bias_torch(xa)
    stats["ztz"] += z_aug.T @ z_aug
    stats["zty"] += z_aug.T @ xb
    stats["yty"] += xb.T @ xb
    stats["rows"] += float(xa.shape[0])


def solve_patch_align_from_stats_torch(stats: dict[str, Any], ridge_lambda: float = 1e-4) -> tuple[PatchLayerWeights, float]:
    t = require_torch()
    ztz = stats["ztz"]
    zty = stats["zty"]
    yty = stats["yty"]
    rows = float(stats["rows"].item())
    hidden_dim = int(yty.shape[0])
    reg = ridge_lambda * t.eye(int(ztz.shape[0]), device=ztz.device, dtype=ztz.dtype)
    try:
        w_aug = t.linalg.solve(ztz + reg, zty)
        sse = _frobenius_inner_torch(ztz @ w_aug, w_aug) - (2.0 * _frobenius_inner_torch(w_aug, zty)) + t.trace(yty).to(dtype=t.float64)
    except RuntimeError:
        ztz_cpu = (ztz + reg).cpu()
        zty_cpu = zty.cpu()
        yty_cpu = yty.cpu()
        w_aug = t.linalg.pinv(ztz_cpu) @ zty_cpu
        sse = _frobenius_inner_torch(ztz_cpu @ w_aug, w_aug) - (2.0 * _frobenius_inner_torch(w_aug, zty_cpu)) + t.trace(yty_cpu).to(dtype=t.float64)
        w_aug = w_aug.to(device=ztz.device, dtype=ztz.dtype)
    mse = _mean_square_from_stats_torch(sse, rows, hidden_dim)
    return PatchLayerWeights(w=w_aug[:-1, :], b=w_aug[-1, :]), mse


def accumulate_patch_kv_stats_torch(xa, kb_raw, vb, align: PatchLayerWeights, stats: dict[str, Any]) -> None:
    device = stats["xhat_txhat"].device
    xa = tensor_to_device(xa, device=str(device), dtype="float32")
    kb_raw = tensor_to_device(kb_raw, device=str(device), dtype="float32")
    vb = tensor_to_device(vb, device=str(device), dtype="float32")
    if xa.ndim != 2 or kb_raw.ndim != 2 or vb.ndim != 2:
        raise ValueError("xa、kb_raw、vb 必须是 [N,D] 矩阵")
    align_dev = PatchLayerWeights(
        w=tensor_to_device(align.w, device=str(device), dtype="float32"),
        b=tensor_to_device(align.b, device=str(device), dtype="float32"),
    )
    xb_hat = infer_patch_layer_torch(xa, align_dev)
    stats["xhat_txhat"] += xb_hat.T @ xb_hat
    stats["xhat_tk"] += xb_hat.T @ kb_raw
    stats["xhat_tv"] += xb_hat.T @ vb
    stats["ktk"] += kb_raw.T @ kb_raw
    stats["vtv"] += vb.T @ vb
    stats["rows"] += float(xa.shape[0])


def accumulate_direct_kv_stats_torch(xa, kb_raw, vb, stats: dict[str, Any]) -> None:
    device = stats["ztz"].device
    xa = tensor_to_device(xa, device=str(device), dtype="float32")
    kb_raw = tensor_to_device(kb_raw, device=str(device), dtype="float32")
    vb = tensor_to_device(vb, device=str(device), dtype="float32")
    if xa.ndim != 2 or kb_raw.ndim != 2 or vb.ndim != 2:
        raise ValueError("xa、kb_raw、vb 必须是 [N,D] 矩阵")
    z_aug = _augment_with_bias_torch(xa)
    stats["ztz"] += z_aug.T @ z_aug
    stats["ztk"] += z_aug.T @ kb_raw
    stats["ztv"] += z_aug.T @ vb
    stats["ktk"] += kb_raw.T @ kb_raw
    stats["vtv"] += vb.T @ vb
    stats["rows"] += float(xa.shape[0])


def accumulate_runtime_aligned_patch_error_stats_torch(
    xa,
    xb,
    kb_raw,
    vb,
    align: PatchLayerWeights,
    wk,
    wv,
    stats: dict[str, Any],
) -> None:
    device = stats["xnorm_sse"].device
    xa = tensor_to_device(xa, device=str(device), dtype="float32")
    xb = tensor_to_device(xb, device=str(device), dtype="float32")
    kb_raw = tensor_to_device(kb_raw, device=str(device), dtype="float32")
    vb = tensor_to_device(vb, device=str(device), dtype="float32")
    align_dev = PatchLayerWeights(
        w=tensor_to_device(align.w, device=str(device), dtype="float32"),
        b=tensor_to_device(align.b, device=str(device), dtype="float32"),
    )
    wk = tensor_to_device(wk, device=str(device), dtype="float32")
    wv = tensor_to_device(wv, device=str(device), dtype="float32")
    xb_hat = infer_patch_layer_torch(xa, align_dev)
    kb_hat = xb_hat @ wk
    vb_hat = xb_hat @ wv
    t = require_torch()
    stats["xnorm_sse"] += t.sum((xb_hat - xb) ** 2, dtype=t.float64)
    stats["k_sse"] += t.sum((kb_hat - kb_raw) ** 2, dtype=t.float64)
    stats["v_sse"] += t.sum((vb_hat - vb) ** 2, dtype=t.float64)
    stats["rows"] += float(xa.shape[0])


def accumulate_direct_kv_error_stats_torch(xa, kb_raw, vb, wk, wv, stats: dict[str, Any]) -> None:
    device = stats["k_sse"].device
    xa = tensor_to_device(xa, device=str(device), dtype="float32")
    kb_raw = tensor_to_device(kb_raw, device=str(device), dtype="float32")
    vb = tensor_to_device(vb, device=str(device), dtype="float32")
    wk = tensor_to_device(wk, device=str(device), dtype="float32")
    wv = tensor_to_device(wv, device=str(device), dtype="float32")
    kb_hat, vb_hat = infer_direct_kv_patch_layer_torch(xa, wk, wv)
    t = require_torch()
    stats["k_sse"] += t.sum((kb_hat - kb_raw) ** 2, dtype=t.float64)
    stats["v_sse"] += t.sum((vb_hat - vb) ** 2, dtype=t.float64)
    stats["rows"] += float(xa.shape[0])


def finalize_runtime_aligned_patch_error_stats_torch(stats: dict[str, Any], hidden_dim: int, kv_dim: int) -> tuple[float, float, float, float]:
    rows = float(stats["rows"].item())
    patch_xnorm_error = _mean_square_from_stats_torch(stats["xnorm_sse"], rows, hidden_dim)
    patch_k_error = _mean_square_from_stats_torch(stats["k_sse"], rows, kv_dim)
    patch_v_error = _mean_square_from_stats_torch(stats["v_sse"], rows, kv_dim)
    patch_kv_error = 0.5 * (patch_k_error + patch_v_error)
    return patch_xnorm_error, patch_k_error, patch_v_error, patch_kv_error


def finalize_direct_kv_error_stats_torch(stats: dict[str, Any], kv_dim: int) -> tuple[float, float, float]:
    rows = float(stats["rows"].item())
    patch_k_error = _mean_square_from_stats_torch(stats["k_sse"], rows, kv_dim)
    patch_v_error = _mean_square_from_stats_torch(stats["v_sse"], rows, kv_dim)
    patch_kv_error = 0.5 * (patch_k_error + patch_v_error)
    return patch_k_error, patch_v_error, patch_kv_error


def solve_patch_kv_from_stats_torch(stats: dict[str, Any], ridge_lambda: float = 1e-4) -> tuple[Any, Any, float, float, float]:
    t = require_torch()
    xhat_txhat = stats["xhat_txhat"]
    xhat_tk = stats["xhat_tk"]
    xhat_tv = stats["xhat_tv"]
    ktk = stats["ktk"]
    vtv = stats["vtv"]
    rows = float(stats["rows"].item())
    hidden_dim = int(xhat_txhat.shape[0])
    kv_dim = int(ktk.shape[0])
    reg = ridge_lambda * t.eye(hidden_dim, device=xhat_txhat.device, dtype=xhat_txhat.dtype)
    try:
        wk = t.linalg.solve(xhat_txhat + reg, xhat_tk)
        wv = t.linalg.solve(xhat_txhat + reg, xhat_tv)
        k_sse = _frobenius_inner_torch(xhat_txhat @ wk, wk) - (2.0 * _frobenius_inner_torch(wk, xhat_tk)) + t.trace(ktk).to(dtype=t.float64)
        v_sse = _frobenius_inner_torch(xhat_txhat @ wv, wv) - (2.0 * _frobenius_inner_torch(wv, xhat_tv)) + t.trace(vtv).to(dtype=t.float64)
    except RuntimeError:
        xhat_txhat_cpu = (xhat_txhat + reg).cpu()
        xhat_tk_cpu = xhat_tk.cpu()
        xhat_tv_cpu = xhat_tv.cpu()
        ktk_cpu = ktk.cpu()
        vtv_cpu = vtv.cpu()
        wk = t.linalg.pinv(xhat_txhat_cpu) @ xhat_tk_cpu
        wv = t.linalg.pinv(xhat_txhat_cpu) @ xhat_tv_cpu
        k_sse = _frobenius_inner_torch(xhat_txhat_cpu @ wk, wk) - (2.0 * _frobenius_inner_torch(wk, xhat_tk_cpu)) + t.trace(ktk_cpu).to(dtype=t.float64)
        v_sse = _frobenius_inner_torch(xhat_txhat_cpu @ wv, wv) - (2.0 * _frobenius_inner_torch(wv, xhat_tv_cpu)) + t.trace(vtv_cpu).to(dtype=t.float64)
        wk = wk.to(device=xhat_txhat.device, dtype=xhat_txhat.dtype)
        wv = wv.to(device=xhat_txhat.device, dtype=xhat_txhat.dtype)
    patch_k_error = _mean_square_from_stats_torch(k_sse, rows, kv_dim)
    patch_v_error = _mean_square_from_stats_torch(v_sse, rows, kv_dim)
    patch_kv_error = 0.5 * (patch_k_error + patch_v_error)
    return wk, wv, patch_k_error, patch_v_error, patch_kv_error


def solve_direct_kv_from_stats_torch(stats: dict[str, Any], ridge_lambda: float = 1e-4) -> tuple[Any, Any, float, float, float]:
    t = require_torch()
    ztz = stats["ztz"]
    ztk = stats["ztk"]
    ztv = stats["ztv"]
    ktk = stats["ktk"]
    vtv = stats["vtv"]
    rows = float(stats["rows"].item())
    feature_dim = int(ztz.shape[0])
    kv_dim = int(ktk.shape[0])
    reg = ridge_lambda * t.eye(feature_dim, device=ztz.device, dtype=ztz.dtype)
    try:
        wk = t.linalg.solve(ztz + reg, ztk)
        wv = t.linalg.solve(ztz + reg, ztv)
        k_sse = _frobenius_inner_torch(ztz @ wk, wk) - (2.0 * _frobenius_inner_torch(wk, ztk)) + t.trace(ktk).to(dtype=t.float64)
        v_sse = _frobenius_inner_torch(ztz @ wv, wv) - (2.0 * _frobenius_inner_torch(wv, ztv)) + t.trace(vtv).to(dtype=t.float64)
    except RuntimeError:
        ztz_cpu = (ztz + reg).cpu()
        ztk_cpu = ztk.cpu()
        ztv_cpu = ztv.cpu()
        ktk_cpu = ktk.cpu()
        vtv_cpu = vtv.cpu()
        wk = t.linalg.pinv(ztz_cpu) @ ztk_cpu
        wv = t.linalg.pinv(ztz_cpu) @ ztv_cpu
        k_sse = _frobenius_inner_torch(ztz_cpu @ wk, wk) - (2.0 * _frobenius_inner_torch(wk, ztk_cpu)) + t.trace(ktk_cpu).to(dtype=t.float64)
        v_sse = _frobenius_inner_torch(ztz_cpu @ wv, wv) - (2.0 * _frobenius_inner_torch(wv, ztv_cpu)) + t.trace(vtv_cpu).to(dtype=t.float64)
        wk = wk.to(device=ztz.device, dtype=ztz.dtype)
        wv = wv.to(device=ztz.device, dtype=ztz.dtype)
    patch_k_error = _mean_square_from_stats_torch(k_sse, rows, kv_dim)
    patch_v_error = _mean_square_from_stats_torch(v_sse, rows, kv_dim)
    patch_kv_error = 0.5 * (patch_k_error + patch_v_error)
    return wk, wv, patch_k_error, patch_v_error, patch_kv_error


def fit_runtime_aligned_patch_layer_torch(xa, xb, kb_raw, vb, ridge_lambda: float = 1e-4) -> RuntimeAlignedPatchFit:
    t = require_torch()
    base_device = xa.device if t.is_tensor(xa) else "cpu"
    hidden_dim = int(xb.shape[-1])
    kv_dim = int(kb_raw.shape[-1])

    align_stats = create_patch_align_stats_torch(hidden_dim, base_device)
    accumulate_patch_align_stats_torch(xa, xb, align_stats)
    align, _patch_xnorm_formula = solve_patch_align_from_stats_torch(align_stats, ridge_lambda=ridge_lambda)

    kv_stats = create_patch_kv_stats_torch(hidden_dim, kv_dim, base_device)
    accumulate_patch_kv_stats_torch(xa, kb_raw, vb, align, kv_stats)
    wk, wv, _patch_k_formula, _patch_v_formula, _patch_kv_formula = solve_patch_kv_from_stats_torch(kv_stats, ridge_lambda=ridge_lambda)
    error_stats = create_patch_error_stats_torch(base_device)
    accumulate_runtime_aligned_patch_error_stats_torch(xa, xb, kb_raw, vb, align, wk, wv, error_stats)
    patch_xnorm_error, patch_k_error, patch_v_error, patch_kv_error = finalize_runtime_aligned_patch_error_stats_torch(
        error_stats,
        hidden_dim=hidden_dim,
        kv_dim=kv_dim,
    )

    return RuntimeAlignedPatchFit(
        align=align,
        wk=wk,
        wv=wv,
        patch_xnorm_error=patch_xnorm_error,
        patch_k_error=patch_k_error,
        patch_v_error=patch_v_error,
        patch_kv_error=patch_kv_error,
    )


def fit_direct_kv_patch_layer_torch(xa, kb_raw, vb, ridge_lambda: float = 1e-4) -> DirectKVPatchFit:
    t = require_torch()
    base_device = xa.device if t.is_tensor(xa) else "cpu"
    hidden_dim = int(xa.shape[-1])
    kv_dim = int(kb_raw.shape[-1])

    direct_stats = create_direct_kv_stats_torch(hidden_dim, kv_dim, base_device)
    accumulate_direct_kv_stats_torch(xa, kb_raw, vb, direct_stats)
    wk, wv, _patch_k_formula, _patch_v_formula, _patch_kv_formula = solve_direct_kv_from_stats_torch(direct_stats, ridge_lambda=ridge_lambda)

    error_stats = create_direct_kv_error_stats_torch(base_device)
    accumulate_direct_kv_error_stats_torch(xa, kb_raw, vb, wk, wv, error_stats)
    patch_k_error, patch_v_error, patch_kv_error = finalize_direct_kv_error_stats_torch(error_stats, kv_dim=kv_dim)

    return DirectKVPatchFit(
        wk=wk,
        wv=wv,
        patch_k_error=patch_k_error,
        patch_v_error=patch_v_error,
        patch_kv_error=patch_kv_error,
    )


def train_patch_layer(z_h: np.ndarray, xnorm_b: np.ndarray, ridge_lambda: float = 1e-4) -> PatchLayerWeights:
    """训练仿射对齐器：x_hat = z @ W + b。"""
    if z_h.ndim != 2 or xnorm_b.ndim != 2:
        raise ValueError("z_h 和 xnorm_b 必须是 [N,D] 矩阵")
    z_aug = np.concatenate([z_h, np.ones((z_h.shape[0], 1), dtype=z_h.dtype)], axis=1)
    reg = ridge_lambda * np.eye(z_aug.shape[1], dtype=z_aug.dtype)
    w_aug = np.linalg.pinv(z_aug.T @ z_aug + reg) @ z_aug.T @ xnorm_b
    w = w_aug[:-1, :]
    b = w_aug[-1, :]
    return PatchLayerWeights(w=w, b=b)


def train_patch_layer_torch(z_h, xnorm_b, ridge_lambda: float = 1e-4) -> PatchLayerWeights:
    t = require_torch()
    z_h = tensor_to_device(z_h, device=str(z_h.device) if t.is_tensor(z_h) else "cpu", dtype="float32")
    xnorm_b = tensor_to_device(xnorm_b, device=str(z_h.device), dtype="float32")
    z_aug = t.cat([z_h, t.ones((z_h.shape[0], 1), device=z_h.device, dtype=z_h.dtype)], dim=1)
    reg = ridge_lambda * t.eye(z_aug.shape[1], device=z_aug.device, dtype=z_aug.dtype)
    w_aug = t.linalg.pinv(z_aug.T @ z_aug + reg) @ z_aug.T @ xnorm_b
    return PatchLayerWeights(w=w_aug[:-1, :], b=w_aug[-1, :])


def infer_patch_layer(z_h: np.ndarray, weights: PatchLayerWeights) -> np.ndarray:
    return z_h @ weights.w + weights.b


def infer_patch_layer_torch(z_h, weights: PatchLayerWeights):
    return z_h @ weights.w + weights.b


def infer_direct_kv_patch_layer_torch(xa, wk, wv):
    xa_aug = _augment_with_bias_torch(xa)
    return xa_aug @ wk, xa_aug @ wv


def hidden_loss(xnorm_b_hat, xnorm_b):
    if tuple(xnorm_b_hat.shape) != tuple(xnorm_b.shape):
        raise ValueError(f"hidden_loss 形状不匹配: {tuple(xnorm_b_hat.shape)} vs {tuple(xnorm_b.shape)}")
    return _F.mse_loss(xnorm_b_hat, xnorm_b)


def aux_k_loss(k_raw_hat, k_raw_b):
    if tuple(k_raw_hat.shape) != tuple(k_raw_b.shape):
        raise ValueError(f"aux_k_loss 形状不匹配: {tuple(k_raw_hat.shape)} vs {tuple(k_raw_b.shape)}")
    return _F.mse_loss(k_raw_hat, k_raw_b)


def aux_v_loss(v_hat, v_b):
    if tuple(v_hat.shape) != tuple(v_b.shape):
        raise ValueError(f"aux_v_loss 形状不匹配: {tuple(v_hat.shape)} vs {tuple(v_b.shape)}")
    return _F.mse_loss(v_hat, v_b)


def semantic_kl_loss(student_logits, teacher_logits, tau: float = 1.0, direction: str = "teacher_to_student"):
    if tuple(student_logits.shape) != tuple(teacher_logits.shape):
        raise ValueError(f"semantic_kl_loss 形状不匹配: {tuple(student_logits.shape)} vs {tuple(teacher_logits.shape)}")
    if tau <= 0.0:
        raise ValueError("tau 必须大于 0")
    scaled_student = student_logits / float(tau)
    scaled_teacher = teacher_logits.detach() / float(tau)
    if str(direction or "teacher_to_student") != "teacher_to_student":
        raise ValueError(f"暂不支持的 semantic_loss_direction: {direction}")
    teacher_probs = _F.softmax(scaled_teacher, dim=-1)
    student_log_probs = _F.log_softmax(scaled_student, dim=-1)
    return _F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (float(tau) ** 2)


def topk_match_at_k(student_logits, teacher_logits, k: int = 3) -> float:
    if tuple(student_logits.shape) != tuple(teacher_logits.shape):
        raise ValueError(f"topk_match_at_k 形状不匹配: {tuple(student_logits.shape)} vs {tuple(teacher_logits.shape)}")
    if student_logits.ndim != 2:
        raise ValueError("topk_match_at_k 仅支持 [N,C] logits")
    width = int(student_logits.shape[-1])
    topk = max(1, min(int(k), width))
    student_topk = _TORCH.topk(student_logits, k=topk, dim=-1).indices
    teacher_topk = _TORCH.topk(teacher_logits, k=topk, dim=-1).indices
    match = (student_topk == teacher_topk).all(dim=-1).to(dtype=_TORCH.float32)
    return float(match.mean().item())


def freeze_module(module: _NN.Module) -> _NN.Module:
    for param in module.parameters():
        param.requires_grad_(False)
    module.eval()
    return module


def pack_module_state(prefix: str, module: _NN.Module) -> dict[str, Any]:
    arrays: dict[str, Any] = {}
    for key, value in module.state_dict().items():
        arrays[f"{prefix}.{key}"] = value.detach().to(device="cpu", dtype=_TORCH.float32).contiguous()
    return arrays


def unpack_module_state(module: _NN.Module, arrays: dict[str, Any], prefix: str) -> _NN.Module:
    state_dict = {}
    needle = f"{prefix}."
    for key, value in arrays.items():
        if key.startswith(needle):
            state_dict[key[len(needle) :]] = tensor_to_device(value, device="cpu", dtype="float32")
    if not state_dict:
        raise KeyError(f"缺少模块前缀 {prefix}")
    module.load_state_dict(state_dict)
    return module


def pack_xnorm_semantic_patch_layer(layer: int, module: XNormSemanticPatchLayer) -> dict[str, Any]:
    arrays = pack_module_state(f"patch_mlp.{layer}", module.patch_mlp)
    if module.enc_h is not None:
        arrays[f"enc_h.{layer}"] = module.enc_h.detach().to(device="cpu", dtype=_TORCH.float32).contiguous()
    return arrays


def load_semantic_patch_layer_from_ckpt(
    arrays: dict[str, Any],
    meta: dict[str, Any],
    *,
    layer: int,
    device: str | Any = "cpu",
    dtype: str = "float32",
) -> XNormSemanticPatchLayer:
    fit_mode = str(meta.get("fit_mode") or "")
    if fit_mode != PATCH_FIT_MODE_XNORM_SEMANTIC_V1:
        raise ValueError(f"仅支持 {PATCH_FIT_MODE_XNORM_SEMANTIC_V1}，实际为 {fit_mode or '<empty>'}")
    hidden_dim = int(meta["hidden_dim"])
    code_dim = int(meta.get("code_dim") or hidden_dim)
    mlp_hidden_dim = int(meta.get("mlp_hidden_dim") or hidden_dim)
    enc_h_is_identity = bool(meta.get("enc_h_is_identity", False))
    module = XNormSemanticPatchLayer(
        hidden_dim=hidden_dim,
        code_dim=code_dim,
        mlp_hidden_dim=mlp_hidden_dim,
        enc_h_is_identity=enc_h_is_identity,
    )
    if not enc_h_is_identity:
        key = f"enc_h.{int(layer)}"
        if key not in arrays:
            raise KeyError(f"缺少键 {key}")
        module.enc_h.data.copy_(tensor_to_device(arrays[key], device="cpu", dtype="float32"))
    unpack_module_state(module.patch_mlp, arrays, f"patch_mlp.{int(layer)}")
    return module.to(device=device, dtype=tensor_to_device(_TORCH.zeros((), dtype=_TORCH.float32), device=device, dtype=dtype).dtype).eval()


def load_semantic_probe_from_ckpt(
    arrays: dict[str, Any],
    meta: dict[str, Any],
    *,
    device: str | Any = "cpu",
    dtype: str = "float32",
    layer: int | None = None,
) -> SemanticProbe:
    fit_mode = str(meta.get("fit_mode") or "")
    if fit_mode != PATCH_FIT_MODE_XNORM_SEMANTIC_V1:
        raise ValueError(f"仅支持 {PATCH_FIT_MODE_XNORM_SEMANTIC_V1}，实际为 {fit_mode or '<empty>'}")
    if str(meta.get("semantic_source") or "probe") != "probe":
        raise ValueError("当前仅支持 semantic_source=probe 的 checkpoint")
    hidden_dim = int(meta["hidden_dim"])
    probe_dim = int(meta.get("semantic_probe_dim") or hidden_dim)
    num_classes = int(meta.get("semantic_probe_num_classes") or hidden_dim)
    probe = SemanticProbe(hidden_dim=hidden_dim, probe_dim=probe_dim, num_classes=num_classes)
    shared = bool(meta.get("semantic_probe_shared", True))
    prefix = "semantic_probe" if shared else f"semantic_probe.{int(layer)}"
    unpack_module_state(probe, arrays, prefix)
    return probe.to(device=device, dtype=tensor_to_device(_TORCH.zeros((), dtype=_TORCH.float32), device=device, dtype=dtype).dtype).eval()


def load_semantic_vocab_head_from_ckpt(
    arrays: dict[str, Any],
    meta: dict[str, Any],
    *,
    device: str | Any = "cpu",
    dtype: str = "float32",
) -> _NN.Module:
    fit_mode = str(meta.get("fit_mode") or "")
    if fit_mode != PATCH_FIT_MODE_XNORM_SEMANTIC_V1:
        raise ValueError(f"仅支持 {PATCH_FIT_MODE_XNORM_SEMANTIC_V1}，实际为 {fit_mode or '<empty>'}")
    hidden_dim = int(meta.get("semantic_probe_num_classes") or meta["hidden_dim"])
    vocab_size = int(meta.get("vocab_size") or 0)
    if vocab_size <= 0:
        raise ValueError("semantic vocab head 需要 meta.vocab_size")
    head = _NN.Linear(hidden_dim, vocab_size, bias=True)
    unpack_module_state(head, arrays, "semantic_vocab_head")
    return head.to(device=device, dtype=tensor_to_device(_TORCH.zeros((), dtype=_TORCH.float32), device=device, dtype=dtype).dtype).eval()


def patch_weights_to_numpy(weights: PatchWeights) -> PatchWeights:
    layers: dict[int, PatchLayerWeights] = {}
    for layer, row in weights.layers.items():
        layers[layer] = PatchLayerWeights(w=tensor_to_numpy(row.w), b=tensor_to_numpy(row.b))
    return PatchWeights(
        code_dim=weights.code_dim,
        hidden_dim=weights.hidden_dim,
        selected_layers=list(weights.selected_layers),
        layers=layers,
    )


def attention_alignment_loss(
    q_logits_teacher: np.ndarray,
    q_logits_student: np.ndarray,
    mask: np.ndarray,
    alpha: float,
    rep_mse: float,
) -> float:
    attn_mse = masked_mse(q_logits_teacher, q_logits_student, mask)
    return float(alpha * rep_mse + (1.0 - alpha) * attn_mse)
