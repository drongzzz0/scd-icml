#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a strict-runtime route from patch checkpoint metadata.")
    parser.add_argument("--patch-dir", required=True)
    parser.add_argument("--models-config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--patch-layers",
        default="",
        help="Optional comma-separated layer ids or ranges, e.g. '10-13,15,18'. Defaults to all trained patch layers.",
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


def main() -> None:
    args = build_parser().parse_args()
    patch_dir = Path(args.patch_dir)
    models_config = Path(args.models_config)
    out_path = Path(args.out)

    patch_meta = json.loads((patch_dir / "meta.json").read_text(encoding="utf-8"))
    models = json.loads(models_config.read_text(encoding="utf-8-sig"))
    consumer_cfg_path = Path(models["model_b"]["path"]) / "config.json"
    consumer_cfg = json.loads(consumer_cfg_path.read_text(encoding="utf-8"))
    num_layers = int(
        consumer_cfg.get("num_hidden_layers")
        or consumer_cfg.get("n_layer")
        or consumer_cfg.get("num_layers")
        or patch_meta.get("num_layers")
    )
    trained_patch_layers = sorted({int(x) for x in (patch_meta.get("residual_layers") or patch_meta.get("selected_layers") or [])})
    if not trained_patch_layers:
        raise ValueError(f"no patch layers in {patch_dir / 'meta.json'}")
    if args.patch_layers:
        patch_layers = parse_layer_spec(args.patch_layers)
        if not patch_layers:
            raise ValueError("--patch-layers was provided but no layers were parsed")
        trained_set = set(trained_patch_layers)
        missing_layers = sorted(layer for layer in patch_layers if layer not in trained_set)
        if missing_layers:
            raise ValueError(f"requested patch layers not present in patch checkpoint metadata: {missing_layers}")
    else:
        patch_layers = list(trained_patch_layers)

    patch_set = set(patch_layers)
    reuse_layers = [layer for layer in range(num_layers) if layer not in patch_set]
    route = {
        "name": "generated_from_patch_meta",
        "num_layers": num_layers,
        "route_semantics": "reuse_non_patch_layers_plus_selected_patch_layers",
        "reuse_layers": reuse_layers,
        "patch_layers": patch_layers,
        "selected_patch_layers": patch_layers,
        "fallback_layers": [],
        "selected_recompute_layers": [],
        "recompute_segment_length_by_layer": {},
        "source": {
            "patch_meta": str(patch_dir / "meta.json"),
            "models_config": str(models_config),
            "consumer_config": str(consumer_cfg_path),
            "trained_patch_layers": trained_patch_layers,
            "requested_patch_layers": args.patch_layers,
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(route, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"generated_route={out_path} num_layers={num_layers} patch_layers={patch_layers}")


if __name__ == "__main__":
    main()
