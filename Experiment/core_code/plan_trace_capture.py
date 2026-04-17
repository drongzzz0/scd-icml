from __future__ import annotations

import argparse
from pathlib import Path

from scd_validation.common import read_json, write_json
from scd_validation.trace_plan import build_trace_capture_plan


def main() -> None:
    parser = argparse.ArgumentParser(description="Plan producer/consumer trace capture for a selected similar pair.")
    parser.add_argument("--inventory", required=True, help="Path to model inventory JSON.")
    parser.add_argument("--request", required=True, help="Path to trace capture request JSON.")
    parser.add_argument("--output", required=True, help="Output trace plan JSON.")
    args = parser.parse_args()

    inventory = read_json(Path(args.inventory))
    request = read_json(Path(args.request))
    plan = build_trace_capture_plan(inventory, request)
    write_json(Path(args.output), plan)
    print(f"wrote trace capture plan to {args.output}")
    print(f"operation_count: {len(plan['operations'])}")


if __name__ == "__main__":
    main()

