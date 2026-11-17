#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from make_paper_scd_route import (
    load_json,
    resolve_num_layers,
    score_profile_layers,
    trained_layers_from_meta,
)


DEFAULT_BUDGETS = "1,2,3,4,5,6,8,12,16,24,31"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate fixed paper-SCD candidate routes for quality-gated max-Reuse selection. "
            "Each route uses Patch on the top-k profile-ranked transition layers and Reuse elsewhere."
        )
    )
    parser.add_argument("--profile", required=True, help="Layer profile JSON from profile-layers")
    parser.add_argument("--patch-dir", required=True, help="Merged Patch checkpoint containing meta.json")
    parser.add_argument("--models-config", required=True)
    parser.add_argument("--eval-config", default="", help="Stored in route provenance only")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--budgets", default=DEFAULT_BUDGETS, help="Comma-separated patch budgets or ranges")
    parser.add_argument("--include-trained-max", action="store_true", help="Also emit the largest mixed trained route")
    return parser


def parse_budgets(value: str) -> list[int]:
    budgets: set[int] = set()
    for raw_part in value.replace(";", ",").split(","):
        part = raw_part.strip()
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


def build_route(
    *,
    profile_path: Path,
    patch_dir: Path,
    models_config: Path,
    eval_config: Path | None,
    num_layers: int,
    patch_layers: list[int],
    trained_patch_layers: list[int],
    budget: int,
    ranking: list[int],
) -> dict[str, Any]:
    patch_layers = sorted({int(layer) for layer in patch_layers})
    patch_set = set(patch_layers)
    reuse_layers = [layer for layer in range(num_layers) if layer not in patch_set]
    if not patch_layers or not reuse_layers:
        raise ValueError("candidate route must be mixed")
    return {
        "name": f"paper_scd_candidate_reuse_patch_k{len(patch_layers)}",
        "num_layers": num_layers,
        "route_semantics": "paper_scd_sparse_patch_reuse_all_other_layers",
        "reuse_layers": reuse_layers,
        "patch_layers": patch_layers,
        "selected_patch_layers": patch_layers,
        "fallback_layers": [],
        "selected_recompute_layers": [],
        "recompute_segment_length_by_layer": {},
        "paper_constraints": {
            "online_route_search": False,
            "t_fill": 0,
            "patch_and_reuse_cover_all_layers": True,
            "candidate_budget": int(budget),
            "final_selection_objective": "maximize_reuse_subject_to_quality_gate",
        },
        "source": {
            "profile": str(profile_path),
            "patch_meta": str(patch_dir / "meta.json"),
            "models_config": str(models_config),
            "eval_config": str(eval_config) if eval_config else "",
            "selection_source": "profile_gain_top_k_candidate",
            "patch_budget": int(budget),
            "trained_patch_layers": trained_patch_layers,
            "profile_ranking": ranking,
        },
    }


def main() -> None:
    args = build_parser().parse_args()
    profile_path = Path(args.profile)
    patch_dir = Path(args.patch_dir)
    models_config = Path(args.models_config)
    eval_config = Path(args.eval_config) if args.eval_config else None
    out_dir = Path(args.out_dir)

    profile = load_json(profile_path)
    patch_meta = load_json(patch_dir / "meta.json")
    num_layers = resolve_num_layers(models_config, profile, patch_meta)
    trained_patch_layers = trained_layers_from_meta(patch_meta)
    if not trained_patch_layers:
        raise ValueError(f"no trained patch layers in {patch_dir / 'meta.json'}")

    ranking = score_profile_layers(profile, set(trained_patch_layers))
    if not ranking:
        raise ValueError("profile ranking is empty after intersecting trained layers")
    max_mixed_budget = min(len(ranking), num_layers - 1)
    budgets = [budget for budget in parse_budgets(args.budgets) if 1 <= budget <= max_mixed_budget]
    if args.include_trained_max and max_mixed_budget not in budgets:
        budgets = sorted(set(budgets + [max_mixed_budget]))
    if not budgets:
        raise ValueError(f"no valid mixed budgets; max_mixed_budget={max_mixed_budget}")

    out_dir.mkdir(parents=True, exist_ok=True)
    candidates: list[dict[str, Any]] = []
    for budget in budgets:
        route = build_route(
            profile_path=profile_path,
            patch_dir=patch_dir,
            models_config=models_config,
            eval_config=eval_config,
            num_layers=num_layers,
            patch_layers=ranking[:budget],
            trained_patch_layers=trained_patch_layers,
            budget=budget,
            ranking=ranking,
        )
        route_path = out_dir / f"route_k{len(route['patch_layers']):02d}.json"
        route_path.write_text(json.dumps(route, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        candidates.append(
            {
                "budget": int(budget),
                "patch_n": len(route["patch_layers"]),
                "reuse_n": len(route["reuse_layers"]),
                "patch_layers": route["patch_layers"],
                "reuse_layers": route["reuse_layers"],
                "route": str(route_path),
            }
        )

    manifest = {
        "route_semantics": "paper_scd_sparse_patch_reuse_all_other_layers",
        "selection_objective": "maximize_reuse_subject_to_quality_gate",
        "num_layers": num_layers,
        "profile": str(profile_path),
        "patch_dir": str(patch_dir),
        "models_config": str(models_config),
        "eval_config": str(eval_config) if eval_config else "",
        "ranking": ranking,
        "candidates": candidates,
    }
    manifest_path = out_dir / "candidate_routes_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"candidate_routes_manifest={manifest_path} candidates={len(candidates)} budgets={budgets}")


if __name__ == "__main__":
    main()
