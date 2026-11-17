#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scd_repro.route_selection.schema import RoutePlan, route_fingerprint


def parse_layers(raw: str, default_layers: list[int], *, sort_result: bool = True) -> list[int]:
    if not str(raw or "").strip():
        return list(default_layers)
    layers: list[int] = []
    seen: set[int] = set()
    for item in str(raw).replace(";", ",").split(","):
        part = item.strip()
        if not part:
            continue
        if "-" in part:
            left, right = part.split("-", 1)
            start = int(left.strip())
            end = int(right.strip())
            if end < start:
                raise ValueError(f"invalid descending layer range: {part}")
            for layer in range(start, end + 1):
                if layer not in seen:
                    seen.add(layer)
                    layers.append(layer)
        else:
            layer = int(part)
            if layer not in seen:
                seen.add(layer)
                layers.append(layer)
    return sorted(layers) if sort_result else layers


def parse_budgets(raw: str) -> list[int]:
    budgets: set[int] = set()
    for item in str(raw or "").replace(";", ",").split(","):
        part = item.strip()
        if not part:
            continue
        if "-" in part:
            left, right = part.split("-", 1)
            start = int(left.strip())
            end = int(right.strip())
            if end < start:
                raise ValueError(f"invalid descending budget range: {part}")
            budgets.update(range(start, end + 1))
        else:
            budgets.add(int(part))
    return sorted(budgets)


def parse_reuse_sets(raw: str) -> list[list[int]]:
    sets: list[list[int]] = []
    seen_keys: set[tuple[int, ...]] = set()
    for item in str(raw or "").split(";"):
        part = item.strip()
        if not part:
            continue
        layers = parse_layers(part, [], sort_result=True)
        key = tuple(layers)
        if key and key not in seen_keys:
            seen_keys.add(key)
            sets.append(layers)
    return sets


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate conditional SCD Patch/Reuse candidates by converting one "
            "additional Patch layer to Reuse. This supports paper-style "
            "max-Reuse selection without fallback or recomputation."
        )
    )
    parser.add_argument("--base-route", required=True, help="Current all-Patch or mixed route JSON")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--candidate-layers", default="", help="Comma/range list; defaults to all current Patch layers")
    parser.add_argument(
        "--prefix-budgets",
        default="",
        help=(
            "Optional comma/range reuse counts. When set, generate routes that "
            "convert the first N candidate layers to Reuse instead of one route per layer. "
            "The candidate layer order is preserved."
        ),
    )
    parser.add_argument(
        "--reuse-sets",
        default="",
        help=(
            "Optional semicolon-separated explicit Reuse sets, for example "
            "'1;25;1,8;1,25'. Each set is added to the base route. This is "
            "useful for pairwise/interaction checks and is mutually exclusive "
            "with --prefix-budgets."
        ),
    )
    parser.add_argument("--name-prefix", default="conditional_reuse")
    parser.add_argument("--skip-existing", action="store_true", help="Do not overwrite existing route files")
    return parser


def route_payload(route: RoutePlan, *, base_route: Path, reused_layer: int) -> dict[str, Any]:
    payload = route.to_dict()
    metadata = dict(payload.get("metadata") or {})
    metadata.update(
        {
            "selection_source": "conditional_reuse_pruning",
            "base_route": str(base_route),
            "new_reuse_layer": int(reused_layer),
            "objective": "maximize_reuse_subject_to_quality_gate",
        }
    )
    payload["metadata"] = metadata
    paper_constraints = dict(payload.get("paper_constraints") or {})
    paper_constraints.update(
        {
            "online_route_search": False,
            "t_fill": 0,
            "patch_and_reuse_cover_all_layers": True,
            "fallback_layers_allowed": False,
            "recompute_allowed": False,
        }
    )
    payload["paper_constraints"] = paper_constraints
    return payload


def main() -> None:
    args = build_parser().parse_args()
    base_path = Path(args.base_route)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_payload = json.loads(base_path.read_text(encoding="utf-8-sig"))
    base_route = RoutePlan.from_dict(base_payload)
    prefix_budgets = parse_budgets(args.prefix_budgets)
    explicit_reuse_sets = parse_reuse_sets(args.reuse_sets)
    if prefix_budgets and explicit_reuse_sets:
        raise ValueError("--prefix-budgets and --reuse-sets are mutually exclusive")
    candidate_layers = parse_layers(args.candidate_layers, base_route.patch_layers, sort_result=not prefix_budgets)

    manifest: dict[str, Any] = {
        "route_semantics": "paper_scd_sparse_patch_reuse_all_other_layers",
        "selection_objective": "maximize_reuse_subject_to_quality_gate",
        "generator": "generate_conditional_reuse_routes.py",
        "base_route": str(base_path),
        "base_fingerprint": route_fingerprint(base_route),
        "base_patch_layers": base_route.patch_layers,
        "base_reuse_layers": base_route.reuse_layers,
        "candidate_layers": candidate_layers,
        "candidates": [],
    }

    if explicit_reuse_sets:
        for reuse_set in explicit_reuse_sets:
            trial = base_route.copy()
            for layer in reuse_set:
                if layer not in range(trial.num_layers):
                    raise ValueError(f"explicit reuse layer {layer} is outside 0..{trial.num_layers - 1}")
                trial = trial.with_action(layer, "reuse")
            if not trial.patch_layers or not trial.reuse_layers:
                continue
            payload = route_payload(trial, base_route=base_path, reused_layer=reuse_set[-1])
            payload["metadata"]["explicit_reuse_layers"] = reuse_set
            layer_token = "_".join(f"l{layer:02d}" for layer in trial.reuse_layers)
            route_name = f"{args.name_prefix}_reuse_{layer_token}_patch{len(trial.patch_layers):02d}"
            payload["name"] = route_name
            route_path = out_dir / f"{route_name}.json"
            if route_path.exists() and args.skip_existing:
                pass
            else:
                route_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            manifest["candidates"].append(
                {
                    "route_name": route_name,
                    "route": str(route_path),
                    "route_fingerprint": route_fingerprint(trial),
                    "explicit_reuse_layers": reuse_set,
                    "patch_n": len(trial.patch_layers),
                    "reuse_n": len(trial.reuse_layers),
                    "patch_layers": trial.patch_layers,
                    "reuse_layers": trial.reuse_layers,
                }
            )
        manifest["reuse_sets"] = explicit_reuse_sets
        manifest_path = out_dir / "conditional_reuse_manifest.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"conditional_reuse_manifest={manifest_path} candidates={len(manifest['candidates'])}")
        return

    if prefix_budgets:
        for budget in prefix_budgets:
            if budget <= 0:
                continue
            if budget > len(candidate_layers):
                raise ValueError(f"prefix budget {budget} exceeds candidate layer count {len(candidate_layers)}")
            trial = base_route.copy()
            reuse_prefix = [int(layer) for layer in candidate_layers[:budget]]
            for layer in reuse_prefix:
                if layer not in trial.patch_layers:
                    raise ValueError(f"candidate layer {layer} is not a Patch layer in base route")
                trial = trial.with_action(layer, "reuse")
            if not trial.patch_layers or not trial.reuse_layers:
                continue
            payload = route_payload(trial, base_route=base_path, reused_layer=reuse_prefix[-1])
            route_name = f"{args.name_prefix}_reuse{len(trial.reuse_layers):02d}_patch{len(trial.patch_layers):02d}"
            payload["name"] = route_name
            payload["metadata"]["new_reuse_layers"] = reuse_prefix
            route_path = out_dir / f"{route_name}.json"
            if route_path.exists() and args.skip_existing:
                pass
            else:
                route_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            manifest["candidates"].append(
                {
                    "route_name": route_name,
                    "route": str(route_path),
                    "route_fingerprint": route_fingerprint(trial),
                    "new_reuse_layers": reuse_prefix,
                    "patch_n": len(trial.patch_layers),
                    "reuse_n": len(trial.reuse_layers),
                    "patch_layers": trial.patch_layers,
                    "reuse_layers": trial.reuse_layers,
                }
            )
        manifest["prefix_budgets"] = prefix_budgets
        manifest_path = out_dir / "conditional_reuse_manifest.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"conditional_reuse_manifest={manifest_path} candidates={len(manifest['candidates'])}")
        return

    for layer in candidate_layers:
        layer = int(layer)
        if layer not in base_route.patch_layers:
            raise ValueError(f"candidate layer {layer} is not a Patch layer in base route")
        trial = base_route.with_action(layer, "reuse")
        if not trial.patch_layers or not trial.reuse_layers:
            continue
        payload = route_payload(trial, base_route=base_path, reused_layer=layer)
        route_name = f"{args.name_prefix}_reuse_l{layer:02d}_patch{len(trial.patch_layers):02d}"
        payload["name"] = route_name
        route_path = out_dir / f"{route_name}.json"
        if route_path.exists() and args.skip_existing:
            pass
        else:
            route_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        manifest["candidates"].append(
            {
                "route_name": route_name,
                "route": str(route_path),
                "route_fingerprint": route_fingerprint(trial),
                "new_reuse_layer": layer,
                "patch_n": len(trial.patch_layers),
                "reuse_n": len(trial.reuse_layers),
                "patch_layers": trial.patch_layers,
                "reuse_layers": trial.reuse_layers,
            }
        )

    manifest_path = out_dir / "conditional_reuse_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"conditional_reuse_manifest={manifest_path} candidates={len(manifest['candidates'])}")


if __name__ == "__main__":
    main()
