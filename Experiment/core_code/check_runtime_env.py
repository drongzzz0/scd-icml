from __future__ import annotations

import argparse
from pathlib import Path

from scd_validation.common import write_json
from scd_validation.runtime_env import build_runtime_env_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Check whether the workspace runtime is ready for real Qwen trace capture.")
    parser.add_argument("--output", required=True, help="Output JSON path.")
    args = parser.parse_args()

    report = build_runtime_env_report()
    write_json(Path(args.output), report)
    print(f"runtime_ready_for_qwen_trace: {report['runtime_ready_for_qwen_trace']}")
    missing = [name for name, status in report["required_modules"].items() if not status["available"]]
    print(f"missing_modules: {missing}")


if __name__ == "__main__":
    main()

