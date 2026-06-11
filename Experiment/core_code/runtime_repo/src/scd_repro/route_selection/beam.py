from __future__ import annotations

from typing import Any

from .schema import RouteLayer, RoutePlan


def stability_aware_beam_search(
    *,
    num_layers: int,
    source_mapping: list[int],
    local_rows: list[dict[str, Any]],
    beam_size: int = 8,
    top_k_routes: int = 5,
    lambda_cost: float = 0.0,
) -> list[dict[str, Any]]:
    """Approximate Patch/Reuse beam selector from per-layer proxy rows.

    This is intentionally a proxy selector. Returned routes still need
    route-level teacher evaluation before use in a paper-facing claim.
    """

    by_layer = {int(row["layer"]): row for row in local_rows}
    beam: list[tuple[float, float, list[str]]] = [(0.0, 0.0, [])]
    for layer in range(int(num_layers)):
        row = by_layer.get(layer, {})
        eps_patch = float(row.get("epsilon_patch", row.get("patch_error", 0.0)) or 0.0)
        eps_reuse = float(row.get("epsilon_reuse", row.get("reuse_error", 0.0)) or 0.0)
        alpha = float(row.get("alpha", 1.0) or 1.0)
        beta = float(row.get("beta", 1.0) or 1.0)
        patch_cost = float(row.get("patch_cost", row.get("payload_bytes", 0.0)) or 0.0)
        reuse_cost = float(row.get("reuse_cost", 0.0) or 0.0)
        expanded: list[tuple[float, float, list[str]]] = []
        for err, cost, actions in beam:
            patch_err = eps_patch
            patch_total_cost = cost + patch_cost
            expanded.append((patch_err + lambda_cost * patch_total_cost, patch_total_cost, actions + ["patch"]))
            reuse_err = alpha * err + beta * eps_reuse
            reuse_total_cost = cost + reuse_cost
            expanded.append((reuse_err + lambda_cost * reuse_total_cost, reuse_total_cost, actions + ["reuse"]))
        expanded.sort(key=lambda item: (item[0], item[1]))
        beam = expanded[: int(beam_size)]

    out: list[dict[str, Any]] = []
    for score, cost, actions in beam[: int(top_k_routes)]:
        layers = [
            RouteLayer(
                consumer_layer=layer,
                action=action,
                source_layer=int(source_mapping[layer]),
            )
            for layer, action in enumerate(actions)
        ]
        route = RoutePlan(
            layers=layers,
            name="stability_aware_beam_candidate",
            metadata={"selector": "stability_aware_beam_proxy"},
        )
        out.append({"score": float(score), "cost": float(cost), "route": route.to_dict()})
    return out
