from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class LayerProfile:
    layer: int
    reuse_error: float
    patch_error: float


@dataclass
class TaskLayerProfile:
    layer: int
    token_count: int
    reuse_logp_sum: float
    patch_logp_sum: float | None
    recompute_logp_sum: float
    patch_bytes: int


@dataclass(frozen=True)
class CandidateRouteDecision:
    layer: int
    mode: str
    segment_length: int = 0


@dataclass
class RouteSearchConfig:
    patch_budget: int
    top_m_candidates: int = 0
    quality_metric: str = "delta_logp"
    cost_metric: str = "delta_ttft_ms"
    route_objective: str = "gain_per_cost"
    latency_budget_ms: float | None = None
    min_positive_gain: float = 0.0
    allow_patch_only: bool = True
    allow_patch_plus_recompute: bool = True
    enable_short_recompute: bool = False
    recompute_segment_lengths: list[int] = field(default_factory=list)
    recompute_eval_mode: str = "best_of_lengths"
    bandwidth_gbps: float = 200.0
    cache_search_results: bool = True
    max_evals: int = 0
    recompute_cost_per_layer_ms: float = 1.0
    patch_compute_cost_ms_per_layer: float = 0.0
    weighted_cost_ttft_coeff: float = 1.0
    weighted_cost_bytes_coeff: float = 0.0
    score_eps: float = 1e-9


def compute_layer_profiles(reuse_errors: list[float], patch_errors: list[float]) -> list[LayerProfile]:
    if len(reuse_errors) != len(patch_errors):
        raise ValueError("reuse_error 和 patch_error 的长度必须一致")
    return [LayerProfile(layer=i, reuse_error=float(reuse_errors[i]), patch_error=float(patch_errors[i])) for i in range(len(reuse_errors))]


def compute_adaptive_thresholds(
    reuse_errors: list[float],
    *,
    reuse_percentile: float = 20.0,
    fallback_percentile: float = 80.0,
    min_gap: float = 1e-6,
) -> tuple[float, float]:
    if not reuse_errors:
        raise ValueError("reuse_error 不能为空，无法计算自适应阈值")
    arr = np.asarray(reuse_errors, dtype=np.float64)
    reuse_threshold = float(np.percentile(arr, reuse_percentile))
    fallback_threshold = float(np.percentile(arr, fallback_percentile))
    if fallback_threshold <= reuse_threshold:
        fallback_threshold = reuse_threshold + float(min_gap)
    return reuse_threshold, fallback_threshold


def build_route(
    profiles: list[LayerProfile],
    reuse_threshold: float,
    fallback_threshold: float,
    patch_budget: int,
) -> dict[str, list[int]]:
    reuse_layers: list[int] = []
    patch_layers: list[int] = []

    gains = {row.layer: (row.reuse_error - row.patch_error) for row in profiles}
    candidate_layers: list[int] = []

    for row in profiles:
        if row.reuse_error <= reuse_threshold:
            reuse_layers.append(row.layer)
        else:
            candidate_layers.append(row.layer)

    positive_candidates = [layer for layer in candidate_layers if gains[layer] > 0.0]
    if patch_budget > 0 and positive_candidates:
        selected = sorted(
            positive_candidates,
            key=lambda layer: (gains[layer], next(row.reuse_error for row in profiles if row.layer == layer)),
            reverse=True,
        )[:patch_budget]
        patch_layers = sorted(selected)

    fallback_layers = sorted(set(candidate_layers) - set(patch_layers))

    return {
        "reuse_layers": sorted(reuse_layers),
        "patch_layers": sorted(patch_layers),
        "fallback_layers": sorted(fallback_layers),
    }


def expected_patch_gain_curve(profiles: list[LayerProfile], max_k: int) -> list[float]:
    gains = sorted([(p.reuse_error - p.patch_error) for p in profiles], reverse=True)
    out: list[float] = []
    running = 0.0
    for i in range(max_k):
        if i < len(gains):
            running += max(0.0, gains[i])
        out.append(float(running))
    return out


def _sorted_layers(task_profiles: dict[int, TaskLayerProfile], candidate_layers: list[int] | None = None) -> list[int]:
    if candidate_layers:
        return sorted({int(layer) for layer in candidate_layers})
    return sorted(int(layer) for layer in task_profiles.keys())


def _recompute_covered_layers(num_layers: int, recompute_segment_length_by_layer: dict[int, int]) -> list[int]:
    covered: set[int] = set()
    for layer, seg_len in recompute_segment_length_by_layer.items():
        if int(seg_len) <= 0:
            continue
        start = int(layer) + 1
        stop = min(int(num_layers), start + int(seg_len))
        covered.update(range(start, stop))
    return sorted(covered)


def _selection_score(quality_gain: float, cost_value: float, config: RouteSearchConfig) -> float:
    if config.route_objective == "gain_per_cost":
        if quality_gain <= 0.0:
            return float(quality_gain)
        return float(quality_gain / max(float(cost_value), float(config.score_eps)))
    if config.route_objective in {"quality_gain", "delta_logp", "neg_delta_nll"}:
        return float(quality_gain)
    raise ValueError(f"未知的 route_objective: {config.route_objective}")


def _quality_value(metrics: dict[str, float], config: RouteSearchConfig) -> float:
    key = str(config.quality_metric or "delta_logp")
    if key == "delta_logp":
        return float(metrics.get("delta_logp", 0.0))
    if key == "neg_delta_nll":
        return float(-metrics.get("delta_nll", 0.0))
    if key == "delta_em":
        return float(metrics.get("delta_em", 0.0))
    if key == "delta_f1":
        return float(metrics.get("delta_f1", 0.0))
    raise ValueError(f"未知的 quality_metric: {key}")


def _cost_value(metrics: dict[str, float], config: RouteSearchConfig) -> float:
    key = str(config.cost_metric or "delta_ttft_ms")
    if key == "delta_ttft_ms":
        return float(max(0.0, metrics.get("delta_ttft_ms", 0.0)))
    if key == "transfer_bytes":
        return float(max(0.0, metrics.get("delta_transfer_bytes", 0.0)))
    if key == "weighted_cost":
        return float(
            max(0.0, metrics.get("delta_ttft_ms", 0.0)) * float(config.weighted_cost_ttft_coeff)
            + max(0.0, metrics.get("delta_transfer_bytes", 0.0)) * float(config.weighted_cost_bytes_coeff)
        )
    raise ValueError(f"未知的 cost_metric: {key}")


def evaluate_task_level_route(
    task_profiles: dict[int, TaskLayerProfile],
    *,
    candidate_layers: list[int] | None,
    selected_patch_layers: list[int],
    recompute_segment_length_by_layer: dict[int, int],
    config: RouteSearchConfig,
    baseline_metrics: dict[str, float] | None = None,
) -> dict[str, float | int | list[int] | dict[str, int]]:
    eval_layers = _sorted_layers(task_profiles, candidate_layers)
    if not eval_layers:
        raise ValueError("task_profiles 不能为空")

    patch_set = {int(layer) for layer in selected_patch_layers}
    recompute_map = {int(layer): int(seg_len) for layer, seg_len in recompute_segment_length_by_layer.items() if int(seg_len) > 0}
    covered_layers = set(_recompute_covered_layers(max(eval_layers) + 1, recompute_map))

    total_logp_sum = 0.0
    total_tokens = 0
    for layer in eval_layers:
        row = task_profiles[int(layer)]
        mode = "reuse"
        if int(layer) in patch_set:
            mode = "patch"
        if int(layer) in covered_layers:
            mode = "recompute"
        if mode == "patch" and row.patch_logp_sum is None:
            raise ValueError(f"layer={layer} 缺少 patch 语义指标，无法评估 route")
        if mode == "reuse":
            total_logp_sum += float(row.reuse_logp_sum)
        elif mode == "patch":
            total_logp_sum += float(row.patch_logp_sum)
        elif mode == "recompute":
            total_logp_sum += float(row.recompute_logp_sum)
        total_tokens += int(row.token_count)

    if total_tokens <= 0:
        raise ValueError("task-level route 评估缺少 token_count")

    bytes_transferred = int(sum(int(task_profiles[layer].patch_bytes) for layer in patch_set if layer in task_profiles))
    transfer_ms = (float(bytes_transferred) * 8.0 / (float(config.bandwidth_gbps) * 1e9)) * 1000.0 if bytes_transferred > 0 else 0.0
    recompute_layers = sorted(layer for layer in covered_layers if layer in task_profiles)
    recompute_ms = float(len(recompute_layers)) * float(config.recompute_cost_per_layer_ms)
    patch_compute_ms = float(len(patch_set)) * float(config.patch_compute_cost_ms_per_layer)
    ttft_ms = float(transfer_ms + recompute_ms + patch_compute_ms)
    avg_logp = float(total_logp_sum / float(total_tokens))
    avg_nll = float(-avg_logp)

    baseline_avg_logp = float((baseline_metrics or {}).get("avg_logp", avg_logp))
    baseline_avg_nll = float((baseline_metrics or {}).get("avg_nll", avg_nll))
    baseline_ttft_ms = float((baseline_metrics or {}).get("ttft_ms", 0.0))
    baseline_transfer_bytes = float((baseline_metrics or {}).get("bytes_transferred", 0.0))
    delta_logp = float(avg_logp - baseline_avg_logp)
    delta_nll = float(avg_nll - baseline_avg_nll)
    delta_ttft_ms = float(ttft_ms - baseline_ttft_ms)
    delta_transfer_bytes = float(bytes_transferred - baseline_transfer_bytes)

    metrics = {
        "avg_logp": avg_logp,
        "avg_nll": avg_nll,
        "ttft_ms": float(ttft_ms),
        "bytes_transferred": float(bytes_transferred),
        "delta_logp": delta_logp,
        "delta_nll": delta_nll,
        "delta_ttft_ms": delta_ttft_ms,
        "delta_transfer_bytes": delta_transfer_bytes,
        "selected_patch_layers": sorted(patch_set),
        "selected_recompute_layers": sorted(recompute_map.keys()),
        "recompute_segment_length_by_layer": {str(layer): int(seg_len) for layer, seg_len in sorted(recompute_map.items())},
        "recompute_covered_layers": recompute_layers,
        "num_eval_layers": len(eval_layers),
        "num_eval_tokens": total_tokens,
    }
    quality_gain = _quality_value(metrics, config)
    cost = _cost_value(metrics, config)
    metrics["gain_per_cost"] = float(_selection_score(quality_gain, cost, RouteSearchConfig(**{**config.__dict__, "route_objective": "gain_per_cost"})))
    metrics["selection_score"] = float(_selection_score(quality_gain, cost, config))
    return metrics


def _candidate_variants(layer: int, config: RouteSearchConfig) -> list[CandidateRouteDecision]:
    variants: list[CandidateRouteDecision] = []
    if config.allow_patch_only:
        variants.append(CandidateRouteDecision(layer=int(layer), mode="patch_only", segment_length=0))
    if config.enable_short_recompute and config.allow_patch_plus_recompute:
        lengths = [int(x) for x in config.recompute_segment_lengths if int(x) > 0]
        if lengths:
            if str(config.recompute_eval_mode or "best_of_lengths") == "fixed_length":
                variants.append(CandidateRouteDecision(layer=int(layer), mode="patch_plus_recompute", segment_length=int(lengths[0])))
            else:
                for seg_len in lengths:
                    variants.append(CandidateRouteDecision(layer=int(layer), mode="patch_plus_recompute", segment_length=int(seg_len)))
    return variants


def _apply_candidate_decision(
    selected_patch_layers: list[int],
    recompute_segment_length_by_layer: dict[int, int],
    decision: CandidateRouteDecision,
) -> tuple[list[int], dict[int, int]]:
    patch_layers = sorted(set(int(layer) for layer in selected_patch_layers) | {int(decision.layer)})
    recompute_map = {int(layer): int(seg_len) for layer, seg_len in recompute_segment_length_by_layer.items()}
    if decision.mode == "patch_only":
        recompute_map.pop(int(decision.layer), None)
    elif decision.mode == "patch_plus_recompute":
        recompute_map[int(decision.layer)] = int(decision.segment_length)
    else:
        raise ValueError(f"未知的 candidate mode: {decision.mode}")
    return patch_layers, recompute_map


def _select_best_variant_for_candidate(
    task_profiles: dict[int, TaskLayerProfile],
    candidate_layers: list[int],
    current_patch_layers: list[int],
    current_recompute_map: dict[int, int],
    candidate_layer: int,
    config: RouteSearchConfig,
    baseline_metrics: dict[str, float],
    current_metrics: dict[str, float],
) -> list[dict[str, object]]:
    proposals: list[dict[str, object]] = []
    for decision in _candidate_variants(candidate_layer, config):
        patch_layers, recompute_map = _apply_candidate_decision(current_patch_layers, current_recompute_map, decision)
        metrics = evaluate_task_level_route(
            task_profiles,
            candidate_layers=candidate_layers,
            selected_patch_layers=patch_layers,
            recompute_segment_length_by_layer=recompute_map,
            config=config,
            baseline_metrics=baseline_metrics,
        )
        inc_quality = _quality_value(metrics, config) - _quality_value(current_metrics, config)
        inc_cost = _cost_value(metrics, config) - _cost_value(current_metrics, config)
        proposals.append(
            {
                "layer": int(candidate_layer),
                "mode": decision.mode,
                "segment_length": int(decision.segment_length),
                "metrics": metrics,
                "incremental_quality_gain": float(inc_quality),
                "incremental_cost": float(inc_cost),
                "selection_score": float(_selection_score(float(inc_quality), float(inc_cost), config)),
            }
        )
    return proposals


def greedy_search_task_route(
    task_profiles: dict[int, TaskLayerProfile],
    *,
    candidate_layers: list[int],
    config: RouteSearchConfig,
) -> dict[str, object]:
    eval_layers = _sorted_layers(task_profiles, candidate_layers)
    baseline_metrics = evaluate_task_level_route(
        task_profiles,
        candidate_layers=eval_layers,
        selected_patch_layers=[],
        recompute_segment_length_by_layer={},
        config=config,
        baseline_metrics=None,
    )

    restore_one_records: list[dict[str, object]] = []
    for layer in candidate_layers:
        restore_one_records.extend(
            _select_best_variant_for_candidate(
                task_profiles,
                eval_layers,
                current_patch_layers=[],
                current_recompute_map={},
                candidate_layer=int(layer),
                config=config,
                baseline_metrics=baseline_metrics,
                current_metrics=baseline_metrics,
            )
        )

    best_restore_by_layer: dict[int, dict[str, object]] = {}
    for row in restore_one_records:
        layer = int(row["layer"])
        existing = best_restore_by_layer.get(layer)
        if existing is None or float(row["selection_score"]) > float(existing["selection_score"]):
            best_restore_by_layer[layer] = row

    screened = [best_restore_by_layer[layer] for layer in sorted(best_restore_by_layer)]
    screened = [row for row in screened if float(row["incremental_quality_gain"]) > float(config.min_positive_gain)]
    screened.sort(key=lambda row: (float(row["selection_score"]), float(row["incremental_quality_gain"])), reverse=True)
    if int(config.top_m_candidates) > 0:
        screened = screened[: int(config.top_m_candidates)]
    screened_layers = [int(row["layer"]) for row in screened]

    current_patch_layers: list[int] = []
    current_recompute_map: dict[int, int] = {}
    current_metrics = baseline_metrics
    remaining_layers = list(screened_layers)
    search_trace: list[dict[str, object]] = []
    eval_count = len(restore_one_records)
    stop_reason = "no_candidates"

    while remaining_layers and len(current_patch_layers) < int(config.patch_budget):
        if int(config.max_evals) > 0 and eval_count >= int(config.max_evals):
            stop_reason = "max_evals"
            break
        proposals: list[dict[str, object]] = []
        for layer in remaining_layers:
            proposals.extend(
                _select_best_variant_for_candidate(
                    task_profiles,
                    eval_layers,
                    current_patch_layers=current_patch_layers,
                    current_recompute_map=current_recompute_map,
                    candidate_layer=int(layer),
                    config=config,
                    baseline_metrics=baseline_metrics,
                    current_metrics=current_metrics,
                )
            )
        eval_count += len(proposals)
        if not proposals:
            stop_reason = "no_remaining_variants"
            break
        proposals.sort(
            key=lambda row: (float(row["selection_score"]), float(row["incremental_quality_gain"]), -float(row["incremental_cost"])),
            reverse=True,
        )
        best = proposals[0]
        if float(best["incremental_quality_gain"]) <= float(config.min_positive_gain):
            stop_reason = "no_positive_gain"
            break
        best_metrics = dict(best["metrics"])
        if config.latency_budget_ms is not None and math.isfinite(float(config.latency_budget_ms)) and float(best_metrics["ttft_ms"]) > float(config.latency_budget_ms):
            stop_reason = "latency_budget_exceeded"
            break
        decision = CandidateRouteDecision(
            layer=int(best["layer"]),
            mode=str(best["mode"]),
            segment_length=int(best["segment_length"]),
        )
        current_patch_layers, current_recompute_map = _apply_candidate_decision(current_patch_layers, current_recompute_map, decision)
        current_metrics = best_metrics
        remaining_layers = [layer for layer in remaining_layers if int(layer) != int(best["layer"])]
        search_trace.append(
            {
                "step": len(search_trace) + 1,
                "accepted_layer": int(best["layer"]),
                "mode": str(best["mode"]),
                "segment_length": int(best["segment_length"]),
                "selection_score": float(best["selection_score"]),
                "incremental_quality_gain": float(best["incremental_quality_gain"]),
                "incremental_cost": float(best["incremental_cost"]),
                "route_metrics": best_metrics,
            }
        )
        stop_reason = "patch_budget_reached" if len(current_patch_layers) >= int(config.patch_budget) else "no_positive_gain"

    recompute_covered_layers = _recompute_covered_layers(max(eval_layers) + 1, current_recompute_map)
    fallback_layers: list[int] = []
    reuse_layers = sorted(layer for layer in eval_layers if layer not in set(current_patch_layers) and layer not in set(recompute_covered_layers))
    final_metrics = evaluate_task_level_route(
        task_profiles,
        candidate_layers=eval_layers,
        selected_patch_layers=current_patch_layers,
        recompute_segment_length_by_layer=current_recompute_map,
        config=config,
        baseline_metrics=baseline_metrics,
    )

    return {
        "route_version": "task_level_step_b_v1",
        "baseline_metrics": baseline_metrics,
        "candidate_metrics": restore_one_records,
        "screened_candidates": screened,
        "selected_patch_layers": sorted(current_patch_layers),
        "selected_recompute_layers": sorted(current_recompute_map.keys()),
        "recompute_segment_length_by_layer": {str(layer): int(seg_len) for layer, seg_len in sorted(current_recompute_map.items())},
        "recompute_covered_layers": recompute_covered_layers,
        "search_trace": search_trace,
        "final_route_metrics": final_metrics,
        "final_gain_per_cost": float(final_metrics["gain_per_cost"]),
        "stop_reason": stop_reason,
        "num_evaluated_routes": eval_count,
        "patch_layers": sorted(current_patch_layers),
        "fallback_layers": fallback_layers,
        "reuse_layers": reuse_layers,
        "expected_gain": float(final_metrics["delta_logp"]),
    }
