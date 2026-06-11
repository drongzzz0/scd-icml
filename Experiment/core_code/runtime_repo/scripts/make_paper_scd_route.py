#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate the paper SCD sparse Patch + Reuse route.")
    parser.add_argument("--profile", required=True, help="Layer profile JSON from profile-layers")
    parser.add_argument("--patch-dir", required=True, help="Patch checkpoint directory containing meta.json")
    parser.add_argument("--models-config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--eval-config", default="", help="Optional eval config used to read route.patch_budget")
    parser.add_argument("--patch-budget", type=int, default=0, help="Patch budget; defaults to eval route.patch_budget or 6")
    parser.add_argument(
        "--patch-layers",
        default="",
        help="Optional comma-separated layer ids or ranges. When omitted, select from the offline profile.",
    )
    return parser


def parse_layer_spec(value: str) -> list[int]:
    layers: set[int] = set()
    for raw_part in value.replace(";", ",").split(","):
        part = raw_part.strip()
        if not part:
            continue
        if "-" in part:
            left, right = part.split("-", 1)
            start = int(left.strip())
            end = int(right.strip())
            if end < start:
                raise ValueError(f"invalid descending layer range: {part}")
            layers.update(range(start, end + 1))
        else:
            layers.add(int(part))
    return sorted(layers)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def resolve_patch_budget(args: argparse.Namespace) -> int:
    if int(args.patch_budget) > 0:
        return int(args.patch_budget)
    if args.eval_config:
        cfg = load_json(Path(args.eval_config))
        budget = int(cfg.get("route", {}).get("patch_budget", 0) or 0)
        if budget > 0:
            return budget
    return 6


def resolve_num_layers(models_config: Path, profile: dict[str, Any], patch_meta: dict[str, Any]) -> int:
    models = load_json(models_config)
    consumer_cfg_path = Path(models["model_b"]["path"]) / "config.json"
    if consumer_cfg_path.is_file():
        consumer_cfg = load_json(consumer_cfg_path)
        value = consumer_cfg.get("num_hidden_layers") or consumer_cfg.get("n_layer") or consumer_cfg.get("num_layers")
        if value:
            return int(value)
    value = profile.get("num_layers") or patch_meta.get("num_layers")
    if value:
        return int(value)
    layer_ids = [int(row["layer"]) for row in profile.get("layers", []) if "layer" in row]
    if layer_ids:
        return max(layer_ids) + 1
    raise ValueError("cannot resolve num_layers")


def trained_layers_from_meta(patch_meta: dict[str, Any]) -> list[int]:
    layers = patch_meta.get("selected_layers") or patch_meta.get("residual_layers") or []
    return sorted({int(layer) for layer in layers})


def score_profile_layers(profile: dict[str, Any], trained_layers: set[int]) -> list[int]:
    recommended = [int(layer) for layer in profile.get("recommended_patch_layers", []) if int(layer) in trained_layers]
    rows = []
    for row in profile.get("layers", []):
        layer = int(row["layer"])
        if layer not in trained_layers:
            continue
        if "expected_gain" in row:
            score = float(row["expected_gain"])
        else:
            reuse_error = float(row.get("reuse_error", 0.0))
            patch_error = float(row.get("patch_kv_error", row.get("patch_error", 0.0)))
            score = reuse_error - patch_error
        rows.append((score, layer))
    rows.sort(key=lambda item: (item[0], -item[1]), reverse=True)
    ranked = [layer for _score, layer in rows]
    selected: list[int] = []
    seen: set[int] = set()
    for layer in recommended + ranked + sorted(trained_layers):
        if layer in seen:
            continue
        selected.append(layer)
        seen.add(layer)
    return selected


def main() -> None:
    args = build_parser().parse_args()
    profile = load_json(Path(args.profile))
    patch_dir = Path(args.patch_dir)
    patch_meta = load_json(patch_dir / "meta.json")
    patch_budget = resolve_patch_budget(args)
    num_layers = resolve_num_layers(Path(args.models_config), profile, patch_meta)
    trained_patch_layers = trained_layers_from_meta(patch_meta)
    if not trained_patch_layers:
        raise ValueError(f"no trained patch layers in {patch_dir / 'meta.json'}")

    trained_set = set(trained_patch_layers)
    if args.patch_layers:
        patch_layers = parse_layer_spec(args.patch_layers)
        source = "explicit_patch_layers"
    elif len(trained_patch_layers) <= patch_budget:
        patch_layers = list(trained_patch_layers)
        source = "trained_sparse_patch_layers"
    else:
        patch_layers = score_profile_layers(profile, trained_set)[:patch_budget]
        source = "profile_gain_top_k"

    if not patch_layers:
        raise ValueError("paper SCD route resolved to zero patch layers")
    missing_layers = sorted(layer for layer in patch_layers if layer not in trained_set)
    if missing_layers:
        raise ValueError(f"requested patch layers not present in patch checkpoint metadata: {missing_layers}")
    if any(layer < 0 or layer >= num_layers for layer in patch_layers):
        raise ValueError(f"patch layers outside 0..{num_layers - 1}: {patch_layers}")

    patch_layers = sorted({int(layer) for layer in patch_layers})
    patch_set = set(patch_layers)
    reuse_layers = [layer for layer in range(num_layers) if layer not in patch_set]
    if not reuse_layers:
        raise ValueError("paper SCD route must be mixed: at least one layer must use Reuse")

    route = {
        "name": f"paper_scd_reuse_patch_k{len(patch_layers)}",
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
            "patch_budget_default": 6,
            "final_selection_objective": "maximize_reuse_subject_to_quality_gate",
        },
        "source": {
            "profile": str(Path(args.profile)),
            "patch_meta": str(patch_dir / "meta.json"),
            "models_config": str(Path(args.models_config)),
            "eval_config": str(Path(args.eval_config)) if args.eval_config else "",
            "selection_source": source,
            "patch_budget": int(patch_budget),
            "trained_patch_layers": trained_patch_layers,
            "requested_patch_layers": args.patch_layers,
        },
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(route, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"generated_paper_scd_route={out_path} num_layers={num_layers} reuse_layers={reuse_layers} patch_layers={patch_layers}")


if __name__ == "__main__":
    main()
