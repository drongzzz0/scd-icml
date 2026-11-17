from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


NA_VALUES = {None, "", "N/A", "NA", "nan"}


@dataclass(frozen=True)
class GateResult:
    passed: bool
    failures: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {"passed": self.passed, "failures": list(self.failures)}


def safe_float(value: Any) -> float | None:
    if value in NA_VALUES:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out):
        return None
    return out


def payload_ratio(payload_bytes: int | float | None, raw_kv_bytes: int | float | None) -> float | None:
    payload = safe_float(payload_bytes)
    raw = safe_float(raw_kv_bytes)
    if payload is None or raw is None or raw <= 0:
        return None
    return payload / raw


def logit_rel_l2(route_logits: np.ndarray, full_logits: np.ndarray) -> float:
    route = np.asarray(route_logits, dtype=np.float64)
    full = np.asarray(full_logits, dtype=np.float64)
    if route.shape != full.shape:
        raise ValueError(f"logit shape mismatch: {route.shape} != {full.shape}")
    denom = np.linalg.norm(full.reshape(-1))
    if denom == 0:
        denom = 1.0
    return float(np.linalg.norm((route - full).reshape(-1)) / denom)


def aggregate_numeric(rows: list[dict[str, Any]], key: str) -> dict[str, float | int | None]:
    values = [safe_float(row.get(key)) for row in rows]
    numeric = [v for v in values if v is not None]
    if not numeric:
        return {"count": 0, "mean": None, "p50": None, "p95": None}
    arr = np.asarray(numeric, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
    }


def evaluate_scd_gate(metrics: dict[str, Any], gate: dict[str, Any]) -> GateResult:
    failures: list[str] = []
    if bool(gate.get("require_payload_smaller_than_raw_kv", False)):
        ratio = payload_ratio(metrics.get("payload_bytes"), metrics.get("raw_consumer_kv_bytes"))
        if ratio is None:
            failures.append("payload ratio unavailable")
        elif ratio >= 1.0:
            failures.append(f"payload/raw KV ratio {ratio:.6g} >= 1")

    checks = [
        ("logit_rel_l2_vs_full_consumer", "max_logit_rel_l2"),
        ("logit_kl_vs_full_consumer", "max_logit_kl"),
        ("ppl_delta_vs_full_consumer", "max_ppl_delta"),
        ("task_metric_delta_vs_full_consumer", "max_task_metric_drop_abs"),
    ]
    for metric_key, gate_key in checks:
        threshold = safe_float(gate.get(gate_key))
        value = safe_float(metrics.get(metric_key))
        if threshold is None:
            continue
        if value is None:
            failures.append(f"{metric_key} unavailable")
        elif abs(value) > threshold:
            failures.append(f"{metric_key}={value:.6g} exceeds {gate_key}={threshold:.6g}")

    if bool(gate.get("first_token_top1_match_required", False)) and not bool(
        metrics.get("first_token_top1_match_vs_full_consumer")
    ):
        failures.append("first token top1 does not match")

    return GateResult(passed=not failures, failures=failures)
