from __future__ import annotations

import argparse
from pathlib import Path

from scd_validation.common import read_json, write_json
from scd_validation.measurement_builders import build_original_scd_config, load_prompt_measurement_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build original SCD baseline config from request metadata and prompt measurements.")
    parser.add_argument("--request", required=True, help="Path to baseline request JSON.")
    parser.add_argument("--prompt-measurements", required=True, help="Path to JSONL prompt-level measurement rows.")
    parser.add_argument("--output", required=True, help="Output baseline config JSON.")
    parser.add_argument("--inventory", help="Optional model inventory JSON for producer/consumer validation.")
    args = parser.parse_args()

    request = read_json(Path(args.request))
    prompt_rows = load_prompt_measurement_rows(Path(args.prompt_measurements))
    inventory = read_json(Path(args.inventory)) if args.inventory else None
    config = build_original_scd_config(request, prompt_rows, inventory=inventory)
    write_json(Path(args.output), config)
    print(f"wrote original SCD config to {args.output}")
    print(f"prompt_count: {len(prompt_rows)}")


if __name__ == "__main__":
    main()

