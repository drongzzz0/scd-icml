from __future__ import annotations

import argparse
from pathlib import Path

from scd_validation.common import read_json
from scd_validation.pair_screening import write_pair_screen_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Write pair screening summary for SCD validation.")
    parser.add_argument("--config", required=True, help="Path to the pair screening JSON config.")
    parser.add_argument(
        "--output",
        default=str(Path(__file__).resolve().parents[1] / "analysis" / "pair_screen_summary.csv"),
        help="Output CSV path.",
    )
    args = parser.parse_args()

    config = read_json(Path(args.config))
    rows = write_pair_screen_summary(config, Path(args.output))
    close_pairs = [row["pair_id"] for row in rows if row["pair_bucket"] == "close"]
    print(f"wrote {len(rows)} pair rows to {args.output}")
    print(f"close pairs: {close_pairs}")


if __name__ == "__main__":
    main()

