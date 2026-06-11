from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .metrics import RouteObjectiveWeights, cost_value, teacher_objective
from .schema import RoutePlan, route_fingerprint


RouteEvaluator = Callable[[RoutePlan], dict[str, Any]]


class MissingRouteMetrics(KeyError):
    def __init__(self, route: RoutePlan):
        self.route = route
        super().__init__(f"missing metrics for route {route_fingerprint(route)}")


@dataclass(frozen=True)
class SelectorConstraints:
    max_total_drift: float | None = None
    max_delta_per_step: float | None = None
    max_logit_kl: float | None = None
    max_ppl_delta: float | None = None
    max_avg_nll_delta: float | None = None
    max_kv_drift: float | None = None
    max_hidden_drift: float | None = None
    min_f1_retention: float | None = None
    max_estimated_ttft_ms: float | None = None


def metrics_pass_constraints(
    metrics: dict[str, Any],
    *,
    objective: float,
    delta: float,
    constraints: SelectorConstraints,
) -> bool:
    checks = [
        (constraints.max_total_drift, objective),
        (constraints.max_delta_per_step, delta),
        (constraints.max_logit_kl, metrics.get("logit_kl")),
        (constraints.max_ppl_delta, metrics.get("ppl_delta")),
        (constraints.max_avg_nll_delta, metrics.get("avg_nll_delta")),
        (constraints.max_kv_drift, metrics.get("kv_drift_mean", metrics.get("kv_drift"))),
        (constraints.max_hidden_drift, metrics.get("hidden_drift_mean", metrics.get("hidden_drift"))),
        (constraints.max_estimated_ttft_ms, metrics.get("estimated_ttft_ms")),
    ]
    for limit, value in checks:
        if limit is None or value is None:
            continue
        if float(value) > float(limit):
            return False
    if constraints.min_f1_retention is not None and metrics.get("f1_retention") is not None:
        if float(metrics["f1_retention"]) < float(constraints.min_f1_retention):
            return False
    return True


class JsonlMetricEvaluator:
    """Lookup evaluator for externally evaluated routes.

    Rows may include either `route_fingerprint`, `route`, or legacy `patch_layers`.
    Missing trial routes raise MissingRouteMetrics; the selector then writes pending
    route artifacts instead of inventing metrics.
    """

    def __init__(self, rows: list[dict[str, Any]]):
        self.by_fingerprint: dict[str, dict[str, Any]] = {}
        self.by_patch_key: dict[str, dict[str, Any]] = {}
        for row in rows:
            if row.get("route_fingerprint"):
                self.by_fingerprint[str(row["route_fingerprint"])] = row
            route_payload = row.get("route")
            if isinstance(route_payload, dict):
                self.by_fingerprint[route_fingerprint(route_payload)] = row
            if row.get("patch_layers") is not None:
                patch_key = ",".join(str(int(layer)) for layer in sorted(row.get("patch_layers") or []))
                self.by_patch_key[patch_key] = row

    def __call__(self, route: RoutePlan) -> dict[str, Any]:
        fp = route_fingerprint(route)
        row = self.by_fingerprint.get(fp)
        if row is None:
            patch_key = ",".join(str(layer) for layer in route.patch_layers)
            row = self.by_patch_key.get(patch_key)
        if row is None:
            raise MissingRouteMetrics(route)
        metrics = row.get("metrics", row)
        if not isinstance(metrics, dict):
            raise ValueError(f"metrics row is not a dict for route {fp}")
        return metrics


def load_metric_rows(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    metric_path = Path(path)
    if metric_path.suffix.lower() == ".json":
        payload = json.loads(metric_path.read_text(encoding="utf-8-sig"))
        if isinstance(payload, dict):
            payload = payload.get("routes", payload.get("candidates", payload.get("rows", [])))
        if not isinstance(payload, list):
            raise ValueError(f"metric JSON must contain a list of rows: {path}")
        return [dict(row) for row in payload]
    for line in metric_path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


class ConditionalReusePruner:
    def __init__(
        self,
        evaluator: RouteEvaluator,
        *,
        weights: RouteObjectiveWeights | dict[str, float] | None = None,
        constraints: SelectorConstraints | None = None,
        eps: float = 1e-9,
        pair_conflicts: set[tuple[int, int]] | None = None,
    ) -> None:
        self.evaluator = evaluator
        self.weights = weights
        self.constraints = constraints or SelectorConstraints()
        self.eps = float(eps)
        self.pair_conflicts = {tuple(sorted(pair)) for pair in (pair_conflicts or set())}

    def _has_conflict(self, route: RoutePlan, candidate_reuse_layer: int) -> bool:
        reuse_set = set(route.reuse_layers) | {int(candidate_reuse_layer)}
        for left, right in self.pair_conflicts:
            if left in reuse_set and right in reuse_set:
                return True
        return False

    def run(self, initial_route: RoutePlan, *, max_steps: int | None = None) -> dict[str, Any]:
        current = initial_route.copy()
        pending: list[RoutePlan] = []
        trace: list[dict[str, Any]] = []
        try:
            current_metrics = self.evaluator(current)
        except MissingRouteMetrics as exc:
            return {
                "status": "needs_evaluation",
                "reason": "initial_route_missing_metrics",
                "pending_routes": [exc.route.to_dict()],
                "selected_route": current.to_dict(),
                "trace": trace,
            }
        current_objective = teacher_objective(current_metrics, self.weights)
        max_steps = current.num_layers if max_steps is None else int(max_steps)

        for step in range(1, max_steps + 1):
            proposals: list[dict[str, Any]] = []
            for layer in list(current.patch_layers):
                if self._has_conflict(current, layer):
                    continue
                trial = current.with_action(layer, "reuse")
                try:
                    trial_metrics = self.evaluator(trial)
                except MissingRouteMetrics as exc:
                    pending.append(exc.route)
                    continue
                trial_objective = teacher_objective(trial_metrics, self.weights)
                delta = float(trial_objective - current_objective)
                cost_saved = float(cost_value(current_metrics) - cost_value(trial_metrics))
                score = delta / max(cost_saved, self.eps)
                acceptable = metrics_pass_constraints(
                    trial_metrics,
                    objective=trial_objective,
                    delta=delta,
                    constraints=self.constraints,
                )
                proposals.append(
                    {
                        "layer": int(layer),
                        "score": float(score),
                        "delta_objective": delta,
                        "cost_saved": cost_saved,
                        "acceptable": bool(acceptable),
                        "metrics": trial_metrics,
                        "objective": trial_objective,
                        "route": trial,
                    }
                )
            accepted = [row for row in proposals if row["acceptable"]]
            if not accepted:
                if pending:
                    return {
                        "status": "needs_evaluation",
                        "reason": "trial_route_missing_metrics",
                        "selected_route": current.to_dict(),
                        "current_metrics": current_metrics,
                        "current_objective": current_objective,
                        "pending_routes": [route.to_dict() for route in pending],
                        "trace": trace,
                    }
                return {
                    "status": "complete",
                    "stop_reason": "no_acceptable_reuse_move",
                    "selected_route": current.to_dict(),
                    "selected_metrics": current_metrics,
                    "selected_objective": current_objective,
                    "trace": trace,
                }
            accepted.sort(key=lambda row: (float(row["score"]), float(row["delta_objective"]), -float(row["cost_saved"])))
            best = accepted[0]
            current = best["route"]
            current_metrics = best["metrics"]
            current_objective = float(best["objective"])
            trace.append(
                {
                    "step": step,
                    "accepted_reuse_layer": int(best["layer"]),
                    "score": float(best["score"]),
                    "delta_objective": float(best["delta_objective"]),
                    "cost_saved": float(best["cost_saved"]),
                    "objective": current_objective,
                    "patch_layers": current.patch_layers,
                    "reuse_layers": current.reuse_layers,
                    "route": current.to_dict(),
                    "metrics": current_metrics,
                }
            )
            if not current.patch_layers:
                break
        return {
            "status": "complete",
            "stop_reason": "max_steps_or_all_reuse",
            "selected_route": current.to_dict(),
            "selected_metrics": current_metrics,
            "selected_objective": current_objective,
            "trace": trace,
        }


def pairwise_interactions(
    base_route: RoutePlan,
    reuse_layers: list[int],
    evaluator: RouteEvaluator,
    *,
    weights: RouteObjectiveWeights | dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    base_obj = teacher_objective(evaluator(base_route), weights)
    rows: list[dict[str, Any]] = []
    for i, left in enumerate(reuse_layers):
        route_i = base_route.with_action(int(left), "reuse")
        obj_i = teacher_objective(evaluator(route_i), weights)
        for right in reuse_layers[i + 1 :]:
            route_j = base_route.with_action(int(right), "reuse")
            route_ij = route_i.with_action(int(right), "reuse")
            obj_j = teacher_objective(evaluator(route_j), weights)
            obj_ij = teacher_objective(evaluator(route_ij), weights)
            rows.append(
                {
                    "layer_i": int(left),
                    "layer_j": int(right),
                    "interaction": float(obj_ij - obj_i - obj_j + base_obj),
                    "base_objective": base_obj,
                    "objective_i": obj_i,
                    "objective_j": obj_j,
                    "objective_ij": obj_ij,
                }
            )
    return rows
