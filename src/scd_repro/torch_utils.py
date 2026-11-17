from __future__ import annotations

import gc
import os
from pathlib import Path
from typing import Any

try:
    import resource
except Exception:  # pragma: no cover - Windows 不提供 resource
    resource = None

try:
    import torch
except Exception:  # pragma: no cover - 允许在无 torch 环境下延迟失败
    torch = None


def require_torch():
    if torch is None:
        raise RuntimeError("当前环境缺少 torch，无法执行 GPU/张量路径")
    return torch



def configure_torch_acceleration(accel_mode: str | None = "balanced") -> None:
    """配置全局 Torch 计算加速开关。"""
    t = require_torch()
    mode = str(accel_mode or "balanced").strip().lower()
    if mode not in {"off", "balanced", "aggressive"}:
        mode = "balanced"

    tf32_enabled = mode != "off"
    try:
        t.backends.cuda.matmul.allow_tf32 = tf32_enabled
    except Exception:
        pass
    try:
        t.backends.cudnn.allow_tf32 = tf32_enabled
    except Exception:
        pass
    try:
        t.set_float32_matmul_precision("high" if tf32_enabled else "highest")
    except Exception:
        pass
def resolve_torch_dtype(dtype: str | None):
    t = require_torch()
    key = str(dtype or "float32").lower()
    mapping = {
        "float16": t.float16,
        "fp16": t.float16,
        "bfloat16": t.bfloat16,
        "bf16": t.bfloat16,
        "float32": t.float32,
        "fp32": t.float32,
        "float64": t.float64,
        "fp64": t.float64,
    }
    if key not in mapping:
        raise ValueError(f"不支持的 dtype: {dtype}")
    return mapping[key]


def resolve_device(device: str | None, cpu_fallback: bool = True):
    t = require_torch()
    key = str(device or "cpu").strip().lower()
    if not key:
        key = "cpu"
    if key.startswith("gpu"):
        key = key.replace("gpu", "cuda", 1)
    if key.startswith("cuda") and not t.cuda.is_available():
        if cpu_fallback:
            return t.device("cpu")
        raise RuntimeError(f"请求使用 {device}，但当前环境不可用 CUDA")
    return t.device(key)


def tensor_to_cpu(value: Any, dtype: str | None = None):
    t = require_torch()
    target_dtype = resolve_torch_dtype(dtype) if dtype else None
    if t.is_tensor(value):
        out = value.detach().to(device="cpu")
        if target_dtype is not None and (out.is_floating_point() or out.is_complex()):
            out = out.to(dtype=target_dtype)
        return out.contiguous()
    out = t.as_tensor(value)
    if target_dtype is not None and (out.is_floating_point() or out.is_complex()):
        out = out.to(dtype=target_dtype)
    return out.to(device="cpu").contiguous()


def tensor_to_device(value: Any, device: str | None, dtype: str | None = None, cpu_fallback: bool = True):
    t = require_torch()
    target = resolve_device(device, cpu_fallback=cpu_fallback)
    target_dtype = resolve_torch_dtype(dtype) if dtype else None
    if t.is_tensor(value):
        return value.detach().to(device=target, dtype=target_dtype or value.dtype).contiguous()
    return t.as_tensor(value, device=target, dtype=target_dtype).contiguous()


def tensor_to_numpy(value: Any):
    t = require_torch()
    if t.is_tensor(value):
        out = value.detach().to(device="cpu")
        if out.dtype == t.bfloat16:
            out = out.to(dtype=t.float32)
        return out.numpy()
    return value


def current_rss_mb() -> float:
    if resource is not None:
        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if os.name == "posix":
            return float(usage) / 1024.0
        return float(usage) / (1024.0 * 1024.0)
    return 0.0


def gpu_memory_allocated_mb(device: str | None) -> float:
    t = require_torch()
    dev = resolve_device(device, cpu_fallback=True)
    if dev.type != "cuda":
        return 0.0
    return float(t.cuda.memory_allocated(dev)) / (1024.0 * 1024.0)


def clear_torch_cache() -> None:
    t = require_torch()
    gc.collect()
    if t.cuda.is_available():
        t.cuda.empty_cache()


def ensure_parent(path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p

