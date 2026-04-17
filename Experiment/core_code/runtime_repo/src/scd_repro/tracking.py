from __future__ import annotations

import hashlib
import inspect
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .io_utils import save_json


def get_git_commit(cwd: str | Path = ".") -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(cwd), stderr=subprocess.DEVNULL)
        return out.decode("utf-8").strip()
    except Exception:
        return "unknown"


def hash_text(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def command_from_argv(argv: list[str]) -> str:
    return " ".join(argv)


def build_manifest(command: str, config_paths: dict[str, str], seed: int) -> dict[str, Any]:
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "seed": seed,
        "config_paths": config_paths,
        "git_commit": get_git_commit(Path.cwd()),
        "hostname": os.environ.get("COMPUTERNAME") or os.environ.get("HOSTNAME") or "unknown",
    }


def write_manifest(out_dir: str | Path, command: str, config_paths: dict[str, str], seed: int) -> None:
    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(command=command, config_paths=config_paths, seed=seed)
    save_json(p / "run_manifest.json", manifest)


def dump_function_signature(func: Any, out_path: str | Path) -> None:
    sig = str(inspect.signature(func))
    save_json(out_path, {"signature": sig})
