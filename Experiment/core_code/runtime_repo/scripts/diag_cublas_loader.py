from __future__ import annotations

import argparse
import ctypes
import os
import subprocess
import sys
from pathlib import Path


def _find_first(root: Path, suffix: str) -> Path | None:
    matches = sorted(root.rglob(suffix))
    return matches[0] if matches else None


def _readelf_dynamic(lib_path: Path) -> str:
    proc = subprocess.run(
        ["readelf", "-d", str(lib_path)],
        check=False,
        capture_output=True,
        text=True,
    )
    lines = []
    for line in proc.stdout.splitlines():
        if "NEEDED" in line or "RPATH" in line or "RUNPATH" in line:
            lines.append(line.strip())
    return "\n".join(lines)


def _loaded_lib_lines(patterns: tuple[str, ...]) -> list[str]:
    maps_path = Path("/proc/self/maps")
    lines = []
    for line in maps_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if any(pat in line for pat in patterns):
            lines.append(line)
    uniq = []
    seen = set()
    for line in lines:
        path = line.split()[-1]
        if path not in seen:
            seen.add(path)
            uniq.append(line)
    return uniq


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose libcublas/libcublasLt loader paths.")
    parser.add_argument("--load-pip-lib", action="store_true", help="Explicitly dlopen the pip-installed libcublas.so.12")
    args = parser.parse_args()

    conda_prefix = os.environ.get("CONDA_PREFIX", "")
    root = Path(conda_prefix) if conda_prefix else Path(sys.prefix)
    pip_cublas = _find_first(root, "nvidia/cublas/lib/libcublas.so.12")
    pip_cublaslt = _find_first(root, "nvidia/cublas/lib/libcublasLt.so.12")

    print("CONDA_PREFIX", conda_prefix)
    print("LD_LIBRARY_PATH", os.environ.get("LD_LIBRARY_PATH", ""))
    print("pip libcublas", pip_cublas or "MISSING")
    print("pip libcublasLt", pip_cublaslt or "MISSING")
    print("system cuda dirs", " ".join(str(p) for p in sorted(Path("/usr/local").glob("cuda*"))) or "NONE")

    if pip_cublas is not None:
        print("readelf libcublas")
        print(_readelf_dynamic(pip_cublas))

    if args.load_pip_lib and pip_cublas is not None:
        try:
            ctypes.CDLL(str(pip_cublas), mode=ctypes.RTLD_GLOBAL)
            print("dlopen pip libcublas: OK")
        except OSError as exc:
            print("dlopen pip libcublas: FAIL", repr(exc))

    print("loaded cuBLAS libs")
    for line in _loaded_lib_lines(("libcublas.so.12", "libcublasLt.so.12", "libcudart.so", "libcuda.so")):
        print(line)


if __name__ == "__main__":
    main()
