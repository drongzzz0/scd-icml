#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from scd_repro.route_selection.config import load_config, write_json
from scd_repro.route_selection.layer_mapping import estimate_layer_mapping, mapping_artifact


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Estimate or materialize source-to-consumer layer mapping for SCD routes.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", default="", help="Override output mapping path")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cfg = load_config(args.config)
    model_cfg = cfg.get("model", {})
    mapping_cfg = cfg.get("mapping", {})
    producer_layers = int(model_cfg.get("producer_num_layers"))
    consumer_layers = int(model_cfg.get("consumer_num_layers"))
    strategy = str(mapping_cfg.get("strategy", "proportional"))
    mapping_path = mapping_cfg.get("mapping_path") or None
    mapping = estimate_layer_mapping(
        strategy=strategy,
        producer_layers=producer_layers,
        consumer_layers=consumer_layers,
        mapping_path=mapping_path,
    )
    out = Path(args.out or mapping_cfg.get("mapping_path") or cfg.get("output", {}).get("dir", "outputs/route_selection") + "/layer_mapping.json")
    artifact = mapping_artifact(
        strategy=strategy,
        producer_layers=producer_layers,
        consumer_layers=consumer_layers,
        mapping=mapping,
    )
    write_json(out, artifact)
    print(f"mapping_path={out}")
    print(f"mapping={mapping}")


if __name__ == "__main__":
    main()
