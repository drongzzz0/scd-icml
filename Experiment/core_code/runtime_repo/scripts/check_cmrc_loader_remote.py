#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from scd_repro.cmrc import load_cmrc2018_samples


def main() -> None:
    p = argparse.ArgumentParser(description="Diagnose CMRC loader path visibility")
    p.add_argument("--data-root", required=True)
    p.add_argument("--limit", type=int, default=2)
    args = p.parse_args()

    root = Path(args.data_root)
    dev_path = root / "cmrc2018" / "cmrc2018_dev.json"
    train_path = root / "cmrc2018" / "cmrc2018_train.json"

    print(f"root={root!s}")
    print(f"root_exists={root.exists()} root_is_dir={root.is_dir()}")
    print(f"dev_path={dev_path!s}")
    print(f"dev_exists={dev_path.exists()} dev_is_file={dev_path.is_file()}")
    print(f"train_path={train_path!s}")
    print(f"train_exists={train_path.exists()} train_is_file={train_path.is_file()}")

    rows = load_cmrc2018_samples(root, limit=int(args.limit))
    print(f"rows={len(rows)} first_qid={rows[0].qid if rows else ''}")


if __name__ == "__main__":
    main()
