from __future__ import annotations

from pathlib import Path

from .schema import (
    ValidationError,
    validate_pairs,
    validate_patch_ckpt,
    validate_predictions,
    validate_reuse_ckpt,
    validate_route,
)


def validate_path(path: str | Path) -> str:
    p = Path(path)
    if not p.exists():
        raise ValidationError(f"路径不存在: {p}")

    if p.is_file() and p.name.endswith("route"):
        validate_route(p)
        return "route"

    if p.is_file() and p.name == "route.json":
        validate_route(p)
        return "route"

    if p.is_file() and p.name == "predictions.json":
        validate_predictions(p)
        return "predictions"

    if p.is_dir():
        if (p / "manifest.json").exists() and (p / "chunks").exists():
            validate_pairs(p)
            return "pairs"
        if (p / "weights.pt").exists() and (p / "meta.json").exists():
            meta_text = (p / "meta.json").read_text(encoding="utf-8")
            if "selected_layers" in meta_text:
                validate_patch_ckpt(p)
                return "patch_ckpt"
            validate_reuse_ckpt(p)
            return "reuse_ckpt"

    raise ValidationError("不支持的路径类型。期望为 pairs 目录、复用检查点目录、补丁检查点目录、route.json 或 predictions.json")
