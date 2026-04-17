from __future__ import annotations

import argparse
from pathlib import Path

from scd_validation.claim_eval import write_baseline_outputs
from scd_validation.common import read_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate original SCD baseline outputs and claim-gap metrics.")
    parser.add_argument("--config", required=True, help="Path to the original baseline JSON config.")
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parents[1] / "analysis"),
        help="Directory for the Task 3 schema outputs.",
    )
    args = parser.parse_args()

    config = read_json(Path(args.config))
    result = write_baseline_outputs(config, Path(args.output_dir))
    print(f"run_id: {result['run_id']}")
    print(f"patch_layers: {result['patch_layers']}")
    print(f"pass_label: {result['claim_gap_row']['pass_label']}")
    print(f"claim_gap_total: {result['claim_gap_row']['claim_gap_total']:.6f}")


if __name__ == "__main__":
    main()

