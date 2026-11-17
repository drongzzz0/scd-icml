from __future__ import annotations

import os
from pathlib import Path
from typing import Any


MODEL_CACHE_SUBDIRS = {
    "HF_HOME": "hf_home",
    "HF_HUB_CACHE": "hf_hub",
    "HF_ASSETS_CACHE": "hf_assets",
    "TRANSFORMERS_CACHE": "transformers",
    "MODELSCOPE_CACHE": "modelscope",
    "TORCH_HOME": "torch",
    "XDG_CACHE_HOME": "xdg",
    "VLLM_CACHE_ROOT": "vllm",
}


def configure_model_cache(cache_root: str | Path | None, *, override: bool = True) -> dict[str, Any]:
    """Route model downloads and runtime caches to a storage directory."""

    if not cache_root:
        return {"enabled": False, "env": current_model_cache_env(), "download_dir": None}

    root = Path(cache_root).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    env: dict[str, str | None] = {}
    for key, subdir in MODEL_CACHE_SUBDIRS.items():
        path = root / subdir
        path.mkdir(parents=True, exist_ok=True)
        if override or not os.environ.get(key):
            os.environ[key] = str(path)
        env[key] = os.environ.get(key)

    return {
        "enabled": True,
        "cache_root": str(root),
        "env": env,
        "download_dir": env.get("HF_HUB_CACHE"),
    }


def current_model_cache_env() -> dict[str, str | None]:
    return {key: os.environ.get(key) for key in MODEL_CACHE_SUBDIRS}
