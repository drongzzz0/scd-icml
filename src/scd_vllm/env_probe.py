from __future__ import annotations

import argparse
import importlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .kv_spec import KVSpecError, assert_consumer_kv_compatible, kv_spec_from_hf_config


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe environment and model KV specs for SCD-vLLM MVP")
    parser.add_argument("--consumer-model", default="", help="Consumer model id/path; weights are not loaded")
    parser.add_argument("--producer-model", default="", help="Producer model id/path; weights are not loaded")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--out", default="runs/scd_vllm/env_probe.json")
    parser.add_argument("--report", default="reports/scd_vllm_status.md")
    parser.add_argument("--skip-model-config", action="store_true", help="Only probe package/runtime versions")
    return parser.parse_args(argv)


def probe_environment(args: argparse.Namespace) -> dict[str, Any]:
    data: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {
            "torch": _module_version("torch"),
            "transformers": _module_version("transformers"),
            "vllm": _module_version("vllm"),
            "numpy": _module_version("numpy"),
        },
        "cuda": _probe_cuda(),
        "nvidia_smi": _probe_nvidia_smi(),
        "models": {
            "producer": args.producer_model or None,
            "consumer": args.consumer_model or None,
            "dtype": args.dtype,
        },
        "kv_specs": {},
        "compatibility": {"checked": False, "compatible": None, "errors": []},
        "notes": [],
    }
    if data["packages"]["vllm"] is None:
        data["notes"].append("vLLM is not installed in this environment; runtime phases require a Linux CUDA environment with vLLM installed.")
    if args.skip_model_config:
        data["notes"].append("model config probing skipped by request")
        return data

    specs: dict[str, Any] = {}
    for role, model in (("producer", args.producer_model), ("consumer", args.consumer_model)):
        if not model:
            data["notes"].append(f"{role} model path/id not provided")
            continue
        try:
            specs[role] = kv_spec_from_hf_config(model, args.dtype).to_dict()
        except Exception as exc:
            data["compatibility"]["errors"].append(f"{role} config probe failed: {type(exc).__name__}: {exc}")
    data["kv_specs"] = specs
    if "producer" in specs and "consumer" in specs:
        data["compatibility"]["checked"] = True
        try:
            # vLLM can only consume consumer-shaped KV.  For the first MVP the
            # producer and consumer must therefore have the same KV spec.
            assert_consumer_kv_compatible(
                kv_spec_from_hf_config(args.producer_model, args.dtype),
                kv_spec_from_hf_config(args.consumer_model, args.dtype),
            )
            data["compatibility"]["compatible"] = True
        except KVSpecError as exc:
            data["compatibility"]["compatible"] = False
            data["compatibility"]["errors"].append(str(exc))
        except Exception as exc:
            data["compatibility"]["compatible"] = False
            data["compatibility"]["errors"].append(f"compatibility check failed: {type(exc).__name__}: {exc}")
    return data


def _module_version(name: str) -> str | None:
    try:
        module = importlib.import_module(name)
    except Exception:
        return None
    return str(getattr(module, "__version__", "unknown"))


def _probe_cuda() -> dict[str, Any]:
    try:
        import torch
    except Exception:
        return {"torch_available": False, "cuda_available": False}
    out: dict[str, Any] = {
        "torch_available": True,
        "cuda_available": bool(torch.cuda.is_available()),
        "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
    }
    if torch.cuda.is_available():
        out["devices"] = [
            {"index": idx, "name": torch.cuda.get_device_name(idx)}
            for idx in range(int(torch.cuda.device_count()))
        ]
    return out


def _probe_nvidia_smi() -> dict[str, Any]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        proc = subprocess.run(cmd, check=False, text=True, capture_output=True, timeout=10)
    except Exception as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    if proc.returncode != 0:
        return {"available": False, "error": proc.stderr.strip() or proc.stdout.strip()}
    rows = []
    for line in proc.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 5:
            rows.append(
                {
                    "index": int(parts[0]),
                    "name": parts[1],
                    "memory_total_mb": int(parts[2]),
                    "memory_used_mb": int(parts[3]),
                    "utilization_gpu_percent": int(parts[4]),
                }
            )
    return {"available": True, "gpus": rows}


def write_status(report_path: str | Path, probe: dict[str, Any]) -> None:
    path = Path(report_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    compatibility = probe.get("compatibility", {})
    lines = [
        "# SCD vLLM Integration Status",
        "",
        "## Phase 1 Env Probe",
        "",
        f"- created_at: `{probe.get('created_at')}`",
        f"- platform: `{probe.get('platform')}`",
        f"- torch: `{probe.get('packages', {}).get('torch')}`",
        f"- transformers: `{probe.get('packages', {}).get('transformers')}`",
        f"- vllm: `{probe.get('packages', {}).get('vllm')}`",
        f"- cuda_available: `{probe.get('cuda', {}).get('cuda_available')}`",
        f"- compatibility_checked: `{compatibility.get('checked')}`",
        f"- compatible: `{compatibility.get('compatible')}`",
        "",
    ]
    errors = compatibility.get("errors") or []
    if errors:
        lines.extend(["### Errors", ""])
        lines.extend(f"- {err}" for err in errors)
        lines.append("")
    notes = probe.get("notes") or []
    if notes:
        lines.extend(["### Notes", ""])
        lines.extend(f"- {note}" for note in notes)
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    probe = probe_environment(args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(probe, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_status(args.report, probe)
    print(f"wrote {out}")
    print(f"wrote {args.report}")
    if probe.get("compatibility", {}).get("errors"):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
