from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .layer_mapping import proportional_mapping


VALID_ACTIONS = {"patch", "reuse"}


@dataclass(frozen=True)
class RouteLayer:
    consumer_layer: int
    action: str
    source_layer: int
    reuse_rank: int | None = None
    patch_module: str | None = None

    def __post_init__(self) -> None:
        if self.action not in VALID_ACTIONS:
            raise ValueError(f"invalid route action: {self.action}")
        if int(self.consumer_layer) < 0:
            raise ValueError("consumer_layer must be non-negative")
        if int(self.source_layer) < 0:
            raise ValueError("source_layer must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "consumer_layer": int(self.consumer_layer),
            "action": str(self.action),
            "source_layer": int(self.source_layer),
        }
        if self.reuse_rank is not None:
            out["reuse_rank"] = int(self.reuse_rank)
        if self.patch_module:
            out["patch_module"] = str(self.patch_module)
        return out

    @classmethod
    def from_dict(cls, payload: dict[str, Any], fallback_consumer_layer: int | None = None) -> "RouteLayer":
        consumer_layer = int(payload.get("consumer_layer", fallback_consumer_layer))
        return cls(
            consumer_layer=consumer_layer,
            action=str(payload["action"]),
            source_layer=int(payload.get("source_layer", consumer_layer)),
            reuse_rank=None if payload.get("reuse_rank") is None else int(payload["reuse_rank"]),
            patch_module=None if payload.get("patch_module") in {"", None} else str(payload["patch_module"]),
        )


@dataclass
class RoutePlan:
    layers: list[RouteLayer]
    name: str = "patch_reuse_route"
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.layers:
            raise ValueError("route must contain at least one layer")
        expected = list(range(len(self.layers)))
        actual = [int(layer.consumer_layer) for layer in self.layers]
        if actual != expected:
            raise ValueError(f"route layers must be contiguous 0..L-1, got {actual}")

    @property
    def num_layers(self) -> int:
        return len(self.layers)

    @property
    def patch_layers(self) -> list[int]:
        return [layer.consumer_layer for layer in self.layers if layer.action == "patch"]

    @property
    def reuse_layers(self) -> list[int]:
        return [layer.consumer_layer for layer in self.layers if layer.action == "reuse"]

    @property
    def source_layer_mapping(self) -> list[int]:
        return [int(layer.source_layer) for layer in self.layers]

    def copy(self) -> "RoutePlan":
        return RoutePlan(
            layers=[RouteLayer.from_dict(layer.to_dict()) for layer in self.layers],
            name=self.name,
            metadata=copy.deepcopy(self.metadata or {}),
        )

    def with_action(self, consumer_layer: int, action: str) -> "RoutePlan":
        if action not in VALID_ACTIONS:
            raise ValueError(f"invalid route action: {action}")
        route = self.copy()
        idx = int(consumer_layer)
        old = route.layers[idx]
        patch_module = old.patch_module if action == "patch" else old.patch_module
        route.layers[idx] = RouteLayer(
            consumer_layer=old.consumer_layer,
            action=action,
            source_layer=old.source_layer,
            reuse_rank=old.reuse_rank,
            patch_module=patch_module,
        )
        return route

    def to_dict(self, *, include_legacy_fields: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "route_version": "teacher_aligned_patch_reuse_v1",
            "name": self.name,
            "num_layers": self.num_layers,
            "layers": [layer.to_dict() for layer in self.layers],
            "source_layer_mapping": self.source_layer_mapping,
            "metadata": copy.deepcopy(self.metadata or {}),
        }
        if include_legacy_fields:
            payload.update(self.to_legacy_route())
        return payload

    def to_legacy_route(self) -> dict[str, Any]:
        return {
            "route_semantics": "paper_scd_sparse_patch_reuse_all_other_layers",
            "num_layers": self.num_layers,
            "reuse_layers": self.reuse_layers,
            "patch_layers": self.patch_layers,
            "selected_patch_layers": self.patch_layers,
            "fallback_layers": [],
            "selected_recompute_layers": [],
            "recompute_segment_length_by_layer": {},
            "paper_constraints": {
                "online_route_search": False,
                "t_fill": 0,
                "patch_and_reuse_cover_all_layers": True,
                "selection_source": "teacher_aligned_conditional_route_search",
            },
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RoutePlan":
        if "layers" in payload:
            layers = [RouteLayer.from_dict(layer, idx) for idx, layer in enumerate(payload["layers"])]
            return cls(layers=layers, name=str(payload.get("name") or "patch_reuse_route"), metadata=dict(payload.get("metadata") or {}))
        return cls.from_legacy_route(payload)

    @classmethod
    def from_legacy_route(
        cls,
        route: dict[str, Any],
        *,
        producer_layers: int | None = None,
        consumer_layers: int | None = None,
        source_mapping: list[int] | None = None,
        patch_module_by_layer: dict[int, str] | None = None,
    ) -> "RoutePlan":
        num_layers = int(consumer_layers or route.get("num_layers") or 0)
        patch_layers = {int(layer) for layer in route.get("selected_patch_layers", route.get("patch_layers", []))}
        reuse_layers = {int(layer) for layer in route.get("reuse_layers", [])}
        if num_layers <= 0:
            max_layer = max(patch_layers | reuse_layers) if (patch_layers or reuse_layers) else -1
            num_layers = max_layer + 1
        if source_mapping is None:
            producer_n = int(producer_layers or num_layers)
            source_mapping = proportional_mapping(producer_n, num_layers)
        if len(source_mapping) != num_layers:
            raise ValueError(f"source mapping length {len(source_mapping)} != num_layers {num_layers}")
        patch_modules = patch_module_by_layer or {}
        layers: list[RouteLayer] = []
        for layer in range(num_layers):
            if layer in patch_layers:
                action = "patch"
            elif reuse_layers and layer not in reuse_layers:
                raise ValueError(f"layer {layer} is neither Patch nor Reuse in legacy route")
            else:
                action = "reuse"
            layers.append(
                RouteLayer(
                    consumer_layer=layer,
                    action=action,
                    source_layer=int(source_mapping[layer]),
                    patch_module=patch_modules.get(layer),
                )
            )
        return cls(
            layers=layers,
            name=str(route.get("name") or "legacy_patch_reuse_route"),
            metadata={"legacy_route": copy.deepcopy(route)},
        )

    @classmethod
    def all_patch(cls, num_layers: int, source_mapping: list[int] | None = None, name: str = "all_patch") -> "RoutePlan":
        if source_mapping is None:
            source_mapping = list(range(int(num_layers)))
        return cls(
            layers=[
                RouteLayer(consumer_layer=layer, action="patch", source_layer=int(source_mapping[layer]))
                for layer in range(int(num_layers))
            ],
            name=name,
        )


def route_fingerprint(route: RoutePlan | dict[str, Any]) -> str:
    plan = route if isinstance(route, RoutePlan) else RoutePlan.from_dict(route)
    key = [
        {"action": layer.action, "source_layer": layer.source_layer, "consumer_layer": layer.consumer_layer}
        for layer in plan.layers
    ]
    return hashlib.sha1(json.dumps(key, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()[:16]


def load_route(path: str | Path) -> RoutePlan:
    return RoutePlan.from_dict(json.loads(Path(path).read_text(encoding="utf-8-sig")))


def save_route(path: str | Path, route: RoutePlan) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(route.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
