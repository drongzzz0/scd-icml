from __future__ import annotations

import importlib.util
import platform
import sys


REQUIRED_RUNTIME_MODULES = [
    "torch",
    "transformers",
    "datasets",
    "accelerate",
    "safetensors",
    "sentencepiece",
]


def _module_status(name: str) -> dict:
    spec = importlib.util.find_spec(name)
    if spec is None:
        return {"available": False, "version": None}
    module = __import__(name)
    return {"available": True, "version": getattr(module, "__version__", "unknown")}


def build_runtime_env_report() -> dict:
    modules = {name: _module_status(name) for name in REQUIRED_RUNTIME_MODULES}
    ready = all(item["available"] for item in modules.values())
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "required_modules": modules,
        "runtime_ready_for_qwen_trace": ready,
        "notes": [
            "This report only checks Python-side dependencies.",
            "A False runtime_ready_for_qwen_trace means the project cannot yet run real Hugging Face trace capture from this workspace environment.",
        ],
    }

