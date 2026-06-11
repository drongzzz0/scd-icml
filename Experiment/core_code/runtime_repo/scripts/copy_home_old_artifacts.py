from __future__ import annotations

import os
import shutil
from pathlib import Path


BASE = Path("/nas/mql/old/2026-03-16_cleanup/home_student_t_maqianli")

COPIES = [
    (
        Path("/home/student_t/maqianli/scd-repro/.cache_pairs"),
        BASE / "scd-repro" / ".cache_pairs",
    ),
    (
        Path("/home/student_t/maqianli/scd-repro/artifacts"),
        BASE / "scd-repro" / "artifacts",
    ),
    (
        Path("/home/student_t/maqianli/artifacts"),
        BASE / "artifacts",
    ),
    (
        Path("/home/student_t/maqianli/outputs"),
        BASE / "outputs",
    ),
]


def _copy_tree_no_metadata(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for dirpath, dirnames, filenames in os.walk(src):
        rel = Path(dirpath).relative_to(src)
        dst_dir = dst / rel
        dst_dir.mkdir(parents=True, exist_ok=True)
        for dirname in dirnames:
            (dst_dir / dirname).mkdir(parents=True, exist_ok=True)
        for filename in filenames:
            src_file = Path(dirpath) / filename
            dst_file = dst_dir / filename
            try:
                if dst_file.exists() and dst_file.stat().st_size == src_file.stat().st_size:
                    continue
            except Exception:
                pass
            shutil.copyfile(src_file, dst_file)


def main() -> None:
    BASE.mkdir(parents=True, exist_ok=True)
    for src, dst in COPIES:
        if not src.exists():
            print(f"SKIP missing {src}", flush=True)
            continue
        print(f"COPY {src} -> {dst}", flush=True)
        _copy_tree_no_metadata(src, dst)
        print(f"DONE {dst}", flush=True)


if __name__ == "__main__":
    main()
