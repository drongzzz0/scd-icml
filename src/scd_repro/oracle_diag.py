from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class OracleDiagRun:
    name: str
    kind: str
    layer: int
    recompute_length: int
    limit: int
    route: dict[str, Any]
    runtime_eval: dict[str, Any]
    second_patch_layer: int | None = None

    def to_manifest_row(self) -> dict[str, Any]:
        return asdict(self)


def parse_csv_ints(text: str) -> list[int]:
    values: list[int] = []
    for raw in str(text or "").replace(";", ",").split(","):
        item = raw.strip()
        if not item:
            continue
        values.append(int(item))
    return values


def parse_anchor_pairs(text: str) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for raw in str(text or "").replace(";", ",").split(","):
        item = raw.strip()
        if not item:
            continue
        if "+" in item:
            lhs, rhs = item.split("+", 1)
        elif ":" in item:
            lhs, rhs = item.split(":", 1)
        else:
            raise ValueError(f"非法 anchor pair: {item}")
        first = int(lhs.strip())
        second = int(rhs.strip())
        if first >= second:
            raise ValueError(f"anchor pair 需要 first < second: {item}")
        pairs.append((first, second))
    return pairs


def build_oracle_route(*, num_layers: int, patch_layer: int, recompute_length: int) -> dict[str, Any]:
    num_layers = int(num_layers)
    patch_layer = int(patch_layer)
    recompute_length = int(recompute_length)
    if num_layers <= 0:
        raise ValueError("num_layers 必须大于 0")
    if patch_layer < 0 or patch_layer >= num_layers:
        raise ValueError(f"patch_layer 越界: {patch_layer}")
    if recompute_length < 0:
        raise ValueError("recompute_length 不能为负数")
    route = {
        "reuse_layers": [layer for layer in range(num_layers) if layer != patch_layer],
        "patch_layers": [patch_layer],
        "fallback_layers": [],
        "selected_patch_layers": [patch_layer],
        "selected_recompute_layers": [],
        "recompute_segment_length_by_layer": {},
    }
    if recompute_length > 0:
        route["selected_recompute_layers"] = [patch_layer]
        route["recompute_segment_length_by_layer"] = {str(patch_layer): recompute_length}
    return route


def build_full_suffix_oracle_route(*, num_layers: int, patch_layer: int) -> dict[str, Any]:
    num_layers = int(num_layers)
    patch_layer = int(patch_layer)
    recompute_length = max(0, num_layers - (patch_layer + 1))
    return build_oracle_route(num_layers=num_layers, patch_layer=patch_layer, recompute_length=recompute_length)


def build_dual_anchor_oracle_route(
    *,
    num_layers: int,
    first_patch_layer: int,
    second_patch_layer: int,
    first_recompute_length: int | None = None,
    second_recompute_length: int | None = None,
) -> dict[str, Any]:
    num_layers = int(num_layers)
    first_patch_layer = int(first_patch_layer)
    second_patch_layer = int(second_patch_layer)
    if num_layers <= 0:
        raise ValueError("num_layers 必须大于 0")
    if first_patch_layer < 0 or second_patch_layer >= num_layers:
        raise ValueError("dual-anchor patch layer 越界")
    if first_patch_layer >= second_patch_layer:
        raise ValueError("dual-anchor 需要 first_patch_layer < second_patch_layer")
    seg1 = int(first_recompute_length) if first_recompute_length is not None else max(0, second_patch_layer - first_patch_layer - 1)
    seg2 = int(second_recompute_length) if second_recompute_length is not None else max(0, num_layers - (second_patch_layer + 1))
    if seg1 < 0 or seg2 < 0:
        raise ValueError("dual-anchor recompute length 不能为负数")
    recompute_map: dict[str, int] = {}
    if seg1 > 0:
        recompute_map[str(first_patch_layer)] = seg1
    if seg2 > 0:
        recompute_map[str(second_patch_layer)] = seg2
    return {
        "reuse_layers": [layer for layer in range(num_layers) if layer not in {first_patch_layer, second_patch_layer}],
        "patch_layers": [first_patch_layer, second_patch_layer],
        "fallback_layers": [],
        "selected_patch_layers": [first_patch_layer, second_patch_layer],
        "selected_recompute_layers": [int(layer) for layer in sorted(int(x) for x in recompute_map.keys())],
        "recompute_segment_length_by_layer": recompute_map,
    }


def build_runtime_eval_overrides(*, kind: str, limit: int) -> dict[str, Any]:
    limit = int(limit)
    if limit <= 0:
        raise ValueError("limit 必须大于 0")
    if kind == "patchonly":
        return {
            "run_full_consumer_baseline": True,
            "run_reuse_only": True,
            "run_reuse_fallback": False,
            "run_reuse_patch": False,
            "run_reuse_patch_recompute": False,
            "run_oracle_xnorm_suffix": False,
            "run_oracle_state_suffix": False,
            "run_oracle_kv_patchonly": True,
            "max_eval_samples": limit,
        }
    if kind in {"suffix", "full_suffix", "dual_anchor"}:
        return {
            "run_full_consumer_baseline": True,
            "run_reuse_only": True,
            "run_reuse_fallback": False,
            "run_reuse_patch": False,
            "run_reuse_patch_recompute": False,
            "run_oracle_xnorm_suffix": True,
            "run_oracle_state_suffix": True,
            "run_oracle_kv_patchonly": False,
            "max_eval_samples": limit,
        }
    raise ValueError(f"未知 oracle diag kind: {kind}")


def build_eval_config(base_cfg: dict[str, Any], *, runtime_eval_overrides: dict[str, Any]) -> dict[str, Any]:
    payload = dict(base_cfg or {})
    runtime_eval = dict(payload.get("runtime_eval", {})) if isinstance(payload.get("runtime_eval", {}), dict) else {}
    runtime_eval.update(runtime_eval_overrides)
    payload["runtime_eval"] = runtime_eval
    return payload


def plan_oracle_diag_runs(
    *,
    num_layers: int,
    layers: list[int],
    recompute_lengths: list[int],
    limit: int,
    plan_kind: str = "single_anchor",
    anchor_pairs: list[tuple[int, int]] | None = None,
) -> list[OracleDiagRun]:
    plan_kind = str(plan_kind or "single_anchor")
    unique_layers = sorted({int(layer) for layer in layers})
    unique_recompute = sorted({int(length) for length in recompute_lengths if int(length) >= 0})
    if plan_kind in {"single_anchor", "full_suffix"} and not unique_layers:
        raise ValueError("layers 不能为空")
    if plan_kind == "single_anchor" and not unique_recompute:
        raise ValueError("recompute_lengths 不能为空")
    if plan_kind == "dual_anchor" and not anchor_pairs:
        raise ValueError("anchor_pairs 不能为空")

    runs: list[OracleDiagRun] = []
    if plan_kind == "single_anchor":
        for layer in unique_layers:
            patchonly_route = build_oracle_route(num_layers=num_layers, patch_layer=layer, recompute_length=0)
            runs.append(
                OracleDiagRun(
                    name=f"l{layer:02d}_patchonly",
                    kind="patchonly",
                    layer=layer,
                    recompute_length=0,
                    limit=int(limit),
                    route=patchonly_route,
                    runtime_eval=build_runtime_eval_overrides(kind="patchonly", limit=limit),
                )
            )
            for recompute_length in unique_recompute:
                runs.append(
                    OracleDiagRun(
                        name=f"l{layer:02d}_k{recompute_length:02d}",
                        kind="suffix",
                        layer=layer,
                        recompute_length=recompute_length,
                        limit=int(limit),
                        route=build_oracle_route(
                            num_layers=num_layers,
                            patch_layer=layer,
                            recompute_length=recompute_length,
                        ),
                        runtime_eval=build_runtime_eval_overrides(kind="suffix", limit=limit),
                    )
                )
        return runs
    if plan_kind == "full_suffix":
        for layer in unique_layers:
            runs.append(
                OracleDiagRun(
                    name=f"l{layer:02d}_full",
                    kind="full_suffix",
                    layer=layer,
                    recompute_length=max(0, int(num_layers) - (int(layer) + 1)),
                    limit=int(limit),
                    route=build_full_suffix_oracle_route(num_layers=num_layers, patch_layer=layer),
                    runtime_eval=build_runtime_eval_overrides(kind="full_suffix", limit=limit),
                )
            )
        return runs
    if plan_kind == "dual_anchor":
        for first_patch_layer, second_patch_layer in anchor_pairs or []:
            route = build_dual_anchor_oracle_route(
                num_layers=num_layers,
                first_patch_layer=first_patch_layer,
                second_patch_layer=second_patch_layer,
            )
            runs.append(
                OracleDiagRun(
                    name=f"l{int(first_patch_layer):02d}_l{int(second_patch_layer):02d}_dual",
                    kind="dual_anchor",
                    layer=int(first_patch_layer),
                    second_patch_layer=int(second_patch_layer),
                    recompute_length=max(int(route["recompute_segment_length_by_layer"].get(str(second_patch_layer), 0)), int(route["recompute_segment_length_by_layer"].get(str(first_patch_layer), 0))),
                    limit=int(limit),
                    route=route,
                    runtime_eval=build_runtime_eval_overrides(kind="dual_anchor", limit=limit),
                )
            )
        return runs
    raise ValueError(f"未知 oracle diag plan_kind: {plan_kind}")
