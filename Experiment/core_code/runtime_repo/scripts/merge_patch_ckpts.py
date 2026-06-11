#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from scd_repro.io_utils import load_bundle, load_json, save_bundle, save_json


LAYER_KEY_PATTERNS = [
    re.compile(r"^direct_w[KV]\.(\d+)$", re.IGNORECASE),
    re.compile(r"^direct_residual_patch\.(\d+)\."),
    re.compile(r"^xnorm_nativekv_residual_patch\.(\d+)\."),
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge independently trained per-layer patch checkpoints.")
    parser.add_argument("--out", required=True, help="Output patch checkpoint directory")
    parser.add_argument("patch_dirs", nargs="+", help="Input patch checkpoint directories")
    return parser


def layer_from_key(key: str) -> int | None:
    for pattern in LAYER_KEY_PATTERNS:
        match = pattern.match(key)
        if match:
            return int(match.group(1))
    return None


def normalize_metrics_layers(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return {str(k): v for k, v in value.items()}
    if isinstance(value, list):
        out: dict[str, Any] = {}
        for row in value:
            if isinstance(row, dict) and "layer" in row:
                out[str(int(row["layer"]))] = row
        return out
    return {}


def main() -> None:
    args = build_parser().parse_args()
    out_dir = Path(args.out)
    patch_dirs = [Path(path) for path in args.patch_dirs]
    if not patch_dirs:
        raise SystemExit("at least one patch checkpoint is required")

    merged_arrays: dict[str, Any] = {}
    merged_layers: set[int] = set()
    merged_residual_layers: set[int] = set()
    merged_metrics: dict[str, Any] = {}
    base_meta: dict[str, Any] | None = None
    fit_mode: str | None = None

    for patch_dir in patch_dirs:
        meta_path = patch_dir / "meta.json"
        weights_path = patch_dir / "weights.pt"
        metrics_path = patch_dir / "metrics.json"
        if not meta_path.exists() or not weights_path.exists():
            raise FileNotFoundError(f"patch checkpoint is incomplete: {patch_dir}")
        meta = load_json(meta_path)
        current_fit_mode = str(meta.get("fit_mode") or "")
        if fit_mode is None:
            fit_mode = current_fit_mode
            base_meta = dict(meta)
        elif current_fit_mode != fit_mode:
            raise ValueError(f"fit_mode mismatch: {patch_dir} has {current_fit_mode}, expected {fit_mode}")

        selected_layers = {int(layer) for layer in meta.get("selected_layers", [])}
        residual_layers = {int(layer) for layer in meta.get("residual_layers", [])}
        bundle = load_bundle(weights_path, device="cpu", dtype="float32")
        for key, value in bundle.arrays.items():
            layer = layer_from_key(str(key))
            if layer is None:
                raise ValueError(f"cannot safely merge non-layer-local key {key!r} from {patch_dir}")
            if key in merged_arrays:
                raise ValueError(f"duplicate weight key {key!r} while merging {patch_dir}")
            merged_arrays[str(key)] = value
            merged_layers.add(layer)
        merged_layers.update(selected_layers)
        merged_residual_layers.update(residual_layers)

        if metrics_path.exists():
            metrics = load_json(metrics_path)
            if isinstance(metrics, dict):
                for layer, row in normalize_metrics_layers(metrics.get("layers")).items():
                    if layer in merged_metrics:
                        raise ValueError(f"duplicate metrics for layer {layer} while merging {patch_dir}")
                    merged_metrics[layer] = row

    if base_meta is None:
        raise RuntimeError("no metadata loaded")
    selected = sorted(merged_layers)
    residual = sorted(layer for layer in merged_residual_layers if layer in merged_layers)
    if not selected:
        raise ValueError("merged checkpoint has no selected layers")
    base_meta["selected_layers"] = selected
    base_meta["residual_layers"] = residual
    base_meta["num_layers"] = max(int(base_meta.get("num_layers") or 0), max(selected) + 1)
    base_meta["merged_from"] = [str(path) for path in patch_dirs]

    out_dir.mkdir(parents=True, exist_ok=True)
    save_bundle(out_dir / "weights.pt", merged_arrays, artifact_dtype="float32")
    save_json(out_dir / "meta.json", base_meta)
    save_json(out_dir / "metrics.json", {"fit_mode": fit_mode, "layers": merged_metrics})
    print(f"merged_patch_ckpt={out_dir} selected_layers={selected} residual_layers={residual}")


if __name__ == "__main__":
    main()
