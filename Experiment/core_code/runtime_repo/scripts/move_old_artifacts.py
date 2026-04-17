from __future__ import annotations

import shutil
from pathlib import Path


BASE = Path("/nas/mql/old/2026-03-16_cleanup")

MOVES = [
    (
        Path("/nas/mql/scd-paper-cmrc-7b2lite-200/pairs/cmrc2018_seed42"),
        BASE / "scd-paper-cmrc-7b2lite-200" / "pairs" / "cmrc2018_seed42",
    ),
    (
        Path("/nas/mql/scd-paper-cmrc-7b2lite-200/pairs/cmrc2018_seed42_smoke16"),
        BASE / "scd-paper-cmrc-7b2lite-200" / "pairs" / "cmrc2018_seed42_smoke16",
    ),
    (
        Path("/nas/mql/scd-paper-cmrc-7b2lite-200/ckpt_patch/cmrc2018_seed42_derope_g1_r128_fix1_p20p80"),
        BASE / "scd-paper-cmrc-7b2lite-200" / "ckpt_patch" / "cmrc2018_seed42_derope_g1_r128_fix1_p20p80",
    ),
    (
        Path("/nas/mql/scd-paper-cmrc-7b2lite-200/ckpt_patch/cmrc2018_seed42_fix3_globalgain_gpu23"),
        BASE / "scd-paper-cmrc-7b2lite-200" / "ckpt_patch" / "cmrc2018_seed42_fix3_globalgain_gpu23",
    ),
    (
        Path("/nas/mql/scd-paper-cmrc-7b2lite-200/ckpt_patch/cmrc2018_seed42_fix4_runtimealigned_gpu67"),
        BASE / "scd-paper-cmrc-7b2lite-200" / "ckpt_patch" / "cmrc2018_seed42_fix4_runtimealigned_gpu67",
    ),
    (
        Path("/nas/mql/scd-paper-cmrc-7b2lite-200/ckpt_patch/cmrc2018_seed42_fix5_directkv_smoke16_gpu5"),
        BASE / "scd-paper-cmrc-7b2lite-200" / "ckpt_patch" / "cmrc2018_seed42_fix5_directkv_smoke16_gpu5",
    ),
    (
        Path("/nas/mql/scd-paper-cmrc-7b2lite-200/ckpt_patch/cmrc2018_seed42_fix5_directkv_routefix4_gpu7"),
        BASE / "scd-paper-cmrc-7b2lite-200" / "ckpt_patch" / "cmrc2018_seed42_fix5_directkv_routefix4_gpu7",
    ),
    (
        Path("/nas/mql/scd-paper-cmrc-7b2lite-200/ckpt_patch/cmrc2018_seed42_fix5_directkv_fullroute_smoke16_gpu4"),
        BASE / "scd-paper-cmrc-7b2lite-200" / "ckpt_patch" / "cmrc2018_seed42_fix5_directkv_fullroute_smoke16_gpu4",
    ),
    (
        Path("/nas/mql/scd-paper-cmrc-7b2lite-200/ckpt_patch/cmrc2018_seed42_fix5_directkv_smoke16_route_gpu4"),
        BASE / "scd-paper-cmrc-7b2lite-200" / "ckpt_patch" / "cmrc2018_seed42_fix5_directkv_smoke16_route_gpu4",
    ),
    (
        Path("/home/student_t/maqianli/scd-repro/.cache_pairs"),
        BASE / "home_student_t_maqianli" / "scd-repro" / ".cache_pairs",
    ),
    (
        Path("/home/student_t/maqianli/scd-repro/artifacts"),
        BASE / "home_student_t_maqianli" / "scd-repro" / "artifacts",
    ),
    (
        Path("/home/student_t/maqianli/artifacts"),
        BASE / "home_student_t_maqianli" / "artifacts",
    ),
    (
        Path("/home/student_t/maqianli/outputs"),
        BASE / "home_student_t_maqianli" / "outputs",
    ),
]


def main() -> None:
    BASE.mkdir(parents=True, exist_ok=True)
    for src, dst in MOVES:
        if not src.exists():
            print(f"SKIP missing {src}", flush=True)
            continue
        if dst.exists():
            print(f"SKIP exists {dst}", flush=True)
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        print(f"MOVE {src} -> {dst}", flush=True)
        shutil.move(str(src), str(dst))
        print(f"DONE {dst}", flush=True)


if __name__ == "__main__":
    main()
