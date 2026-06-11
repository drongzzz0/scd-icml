#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a no-fallback/no-recompute SCD route.")
    parser.add_argument("--route", required=True, help="Path to route JSON")
    parser.add_argument("--require-all-patch", action="store_true", help="Require patch_layers to cover every layer")
    parser.add_argument("--require-mixed", action="store_true", help="Require both reuse_layers and patch_layers to be non-empty")
    return parser.parse_args()


def int_list(route: dict[str, Any], key: str) -> list[int]:
    return [int(layer) for layer in route.get(key, [])]


def main() -> None:
    args = parse_args()
    route_path = Path(args.route)
    route = json.loads(route_path.read_text(encoding="utf-8-sig"))
    errors: list[str] = []

    try:
        num_layers = int(route.get("num_layers", 0))
    except Exception:
        num_layers = 0
    patch_layers = int_list(route, "patch_layers")
    reuse_layers = int_list(route, "reuse_layers")
    fallback_layers = int_list(route, "fallback_layers")
    recompute_layers = int_list(route, "selected_recompute_layers")
    recompute_map = route.get("recompute_segment_length_by_layer", {})

    if num_layers <= 0:
        errors.append(f"route.num_layers must be positive, got {route.get('num_layers')!r}")
    if fallback_layers:
        errors.append(f"fallback_layers must be empty, got {fallback_layers}")
    if recompute_layers or recompute_map:
        errors.append(f"recompute must be disabled, got layers={recompute_layers} map={recompute_map}")
    if num_layers > 0:
        all_layers = set(range(num_layers))
        patch_set = set(patch_layers)
        reuse_set = set(reuse_layers)
        overlap = sorted(patch_set & reuse_set)
        if overlap:
            errors.append(f"patch_layers and reuse_layers overlap: {overlap}")
        missing = sorted(all_layers - (patch_set | reuse_set))
        extra = sorted((patch_set | reuse_set) - all_layers)
        if missing:
            errors.append(f"patch_layers plus reuse_layers do not cover layers: {missing}")
        if extra:
            errors.append(f"route contains layers outside 0..{num_layers - 1}: {extra}")
        if args.require_all_patch and patch_set != all_layers:
            errors.append("patch_layers must cover every layer when --require-all-patch is set")
        if args.require_mixed and (not patch_set or not reuse_set):
            errors.append("route must be mixed when --require-mixed is set: both patch_layers and reuse_layers must be non-empty")

    if errors:
        print(f"route validation failed: {route_path}")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)

    print(
        "route validation passed: "
        f"{route_path} num_layers={num_layers} reuse_layers={reuse_layers} patch_layers={patch_layers}"
    )


if __name__ == "__main__":
    main()
