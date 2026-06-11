from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from ..config import load_json_yaml
from ..logging_utils import setup_logger


def base_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--seed", type=int, default=42, help="全局随机种子")
    parser.add_argument("--log", type=str, default="", help="可选日志文件路径")
    return parser


def add_accel_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--accel-mode", choices=["off", "balanced", "aggressive"], default="")
    parser.add_argument("--layer-block-size", type=int, default=0)
    parser.add_argument("--group-block-size", type=int, default=0)


def resolve_accel_options(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    compute = cfg.get("compute", {}) if isinstance(cfg, dict) else {}
    accel_mode = str(args.accel_mode or compute.get("accel_mode") or "balanced")
    layer_block_size = int(args.layer_block_size or compute.get("layer_block_size") or 4)
    group_block_size = int(args.group_block_size or compute.get("group_block_size") or 2)
    return {
        "accel_mode": accel_mode,
        "layer_block_size": max(1, layer_block_size),
        "group_block_size": max(1, group_block_size),
    }


def resolve_logger(log_path: str):
    return setup_logger(log_file=Path(log_path) if log_path else None)


def load_cfg(path: str) -> dict:
    return load_json_yaml(path)
