from __future__ import annotations

import argparse
from pathlib import Path

from scd_validation.common import read_json, write_json
from scd_validation.measurement_builders import build_pair_screening_config, load_pair_metric_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build pair screening config from model inventory and measured pair metrics.")
    parser.add_argument("--inventory", required=True, help="Path to model inventory JSON.")
    parser.add_argument("--pair-metrics", required=True, help="Path to JSONL pair metric rows.")
    parser.add_argument("--output", required=True, help="Output screening config JSON.")
    parser.add_argument("--include-missing-pairs", action="store_true", help="Keep candidate pairs even if metrics are missing.")
    args = parser.parse_args()

    inventory = read_json(Path(args.inventory))
    pair_metric_rows = load_pair_metric_rows(Path(args.pair_metrics))
    config = build_pair_screening_config(
        inventory,
        pair_metric_rows,
        include_missing_pairs=args.include_missing_pairs,
    )
    write_json(Path(args.output), config)
    print(f"wrote pair screening config to {args.output}")
    print(f"candidate_count: {len(config['candidates'])}")


if __name__ == "__main__":
    main()

