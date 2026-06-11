from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class RouteObjectiveWeights:
    kv: float = 1.0
    hidden: float = 1.0
    kl: float = 1.0
    ppl: float = 1.0
    cost: float = 0.0


def _as_array(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=np.float64)


def _expand_mask(mask: Any, target_ndim: int) -> np.ndarray:
    arr = np.asarray(mask, dtype=np.float64)
    while arr.ndim < target_ndim:
        arr = np.expand_dims(arr, axis=-1)
    return arr


def masked_relative_l2(route_value: Any, teacher_value: Any, mask: Any | None = None, eps: float = 1e-9) -> float:
    route = _as_array(route_value)
    teacher = _as_array(teacher_value)
    if route.shape != teacher.shape:
        raise ValueError(f"shape mismatch: route={route.shape} teacher={teacher.shape}")
    diff = route - teacher
    denom = teacher
    if mask is not None:
        expanded = _expand_mask(mask, route.ndim)
        diff = diff * expanded
        denom = denom * expanded
    numerator = float(np.sqrt(np.sum(diff * diff)))
    denominator = float(np.sqrt(np.sum(denom * denom)))
    return numerator / max(denominator, float(eps))


def kv_drift(
    *,
    route_k: Any,
    teacher_k: Any,
    route_v: Any,
    teacher_v: Any,
    mask: Any | None = None,
    eps: float = 1e-9,
) -> float:
    return float(
        masked_relative_l2(route_k, teacher_k, mask=mask, eps=eps)
        + masked_relative_l2(route_v, teacher_v, mask=mask, eps=eps)
    )


def hidden_drift(route_hidden: Any, teacher_hidden: Any, mask: Any | None = None, eps: float = 1e-9) -> float:
    return masked_relative_l2(route_hidden, teacher_hidden, mask=mask, eps=eps)


def _softmax(logits: np.ndarray) -> np.ndarray:
    x = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(x)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def logit_kl(teacher_logits: Any, route_logits: Any, mask: Any | None = None, eps: float = 1e-9) -> float:
    teacher = _softmax(_as_array(teacher_logits))
    route = np.clip(_softmax(_as_array(route_logits)), eps, 1.0)
    teacher = np.clip(teacher, eps, 1.0)
    kl = np.sum(teacher * (np.log(teacher) - np.log(route)), axis=-1)
    if mask is not None:
        mask_arr = np.asarray(mask, dtype=np.float64)
        if mask_arr.shape != kl.shape:
            raise ValueError(f"mask shape {mask_arr.shape} does not match KL shape {kl.shape}")
        denom = max(float(mask_arr.sum()), float(eps))
        return float((kl * mask_arr).sum() / denom)
    return float(np.mean(kl))


def ppl_delta(route_ppl: float, teacher_ppl: float, *, relative: bool = False, eps: float = 1e-9) -> float:
    if relative:
        return float(float(route_ppl) / max(float(teacher_ppl), float(eps)) - 1.0)
    return float(float(route_ppl) - float(teacher_ppl))


def cost_value(metrics: dict[str, Any]) -> float:
    if metrics.get("cost") is not None:
        return float(metrics["cost"])
    keys = ["payload_bytes", "estimated_ttft_ms", "patch_compute_ms", "auxiliary_params"]
    return float(sum(float(metrics.get(key, 0.0) or 0.0) for key in keys))


def teacher_objective(metrics: dict[str, Any], weights: RouteObjectiveWeights | dict[str, float] | None = None) -> float:
    if weights is None:
        weights_obj = RouteObjectiveWeights()
    elif isinstance(weights, dict):
        weights_obj = RouteObjectiveWeights(
            kv=float(weights.get("kv", 1.0)),
            hidden=float(weights.get("hidden", 1.0)),
            kl=float(weights.get("kl", 1.0)),
            ppl=float(weights.get("ppl", 1.0)),
            cost=float(weights.get("cost", 0.0)),
        )
    else:
        weights_obj = weights
    total = 0.0
    total += weights_obj.kv * float(metrics.get("kv_drift_mean", metrics.get("kv_drift", 0.0)) or 0.0)
    total += weights_obj.hidden * float(metrics.get("hidden_drift_mean", metrics.get("hidden_drift", 0.0)) or 0.0)
    total += weights_obj.kl * float(metrics.get("logit_kl", 0.0) or 0.0)
    total += weights_obj.ppl * float(
        metrics.get("ppl_delta", metrics.get("avg_nll_delta", metrics.get("relative_ppl", 0.0))) or 0.0
    )
    total += weights_obj.cost * cost_value(metrics)
    if not math.isfinite(total):
        raise ValueError(f"non-finite teacher objective: {total}")
    return float(total)
