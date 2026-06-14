#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Preflight checks for the SCD-ICML Qwen-32B A800 run."
    )
    parser.add_argument("--models-config", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--datasets", default="cmrc2018")
    parser.add_argument("--required-gpus", type=int, default=2)
    parser.add_argument("--required-free-mb", type=int, default=70000)
    parser.add_argument("--monitor-seconds", type=int, default=60)
    parser.add_argument("--sample-interval", type=int, default=10)
    parser.add_argument("--out", default="")
    parser.add_argument("--write-env", default="")
    parser.add_argument(
        "--skip-autoconfig",
        action="store_true",
        help="Skip transformers AutoConfig checks and only inspect config.json files.",
    )
    return parser


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def model_summary(model_root: Path) -> dict[str, Any]:
    cfg_path = model_root / "config.json"
    summary: dict[str, Any] = {
        "path": str(model_root),
        "exists": model_root.exists(),
        "config_exists": cfg_path.is_file(),
    }
    if cfg_path.is_file():
        cfg = read_json(cfg_path)
        summary.update(
            {
                "model_type": cfg.get("model_type"),
                "architectures": cfg.get("architectures", []),
                "num_hidden_layers": cfg.get("num_hidden_layers")
                or cfg.get("n_layer")
                or cfg.get("num_layers"),
                "hidden_size": cfg.get("hidden_size") or cfg.get("n_embd"),
                "num_attention_heads": cfg.get("num_attention_heads") or cfg.get("n_head"),
                "num_key_value_heads": cfg.get("num_key_value_heads"),
                "rope_theta": cfg.get("rope_theta"),
            }
        )
    return summary


def is_vl_or_non_text(summary: dict[str, Any]) -> bool:
    text = " ".join(
        [
            str(summary.get("model_type") or ""),
            " ".join(str(x) for x in summary.get("architectures", [])),
        ]
    ).lower()
    return "vl" in text or "vision" in text or "visual" in text


def is_probably_causal_lm(summary: dict[str, Any]) -> bool:
    arch = " ".join(str(x) for x in summary.get("architectures", []))
    model_type = str(summary.get("model_type") or "").lower()
    if "ForCausalLM" in arch:
        return True
    return model_type in {"qwen2", "qwen3", "llama", "mistral", "gpt_neox"}


def check_autoconfig(path: Path) -> dict[str, Any]:
    try:
        from transformers import AutoConfig
    except Exception as exc:
        return {"available": False, "error": f"transformers import failed: {exc}"}
    try:
        cfg = AutoConfig.from_pretrained(str(path), trust_remote_code=True)
    except Exception as exc:
        return {"available": False, "error": f"AutoConfig failed: {exc}"}
    return {
        "available": True,
        "model_type": getattr(cfg, "model_type", None),
        "architectures": getattr(cfg, "architectures", None),
        "num_hidden_layers": getattr(cfg, "num_hidden_layers", None),
        "hidden_size": getattr(cfg, "hidden_size", None),
        "num_attention_heads": getattr(cfg, "num_attention_heads", None),
        "num_key_value_heads": getattr(cfg, "num_key_value_heads", None),
    }


def run_nvidia_smi() -> list[dict[str, int]]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,utilization.gpu,memory.used,memory.free,memory.total",
        "--format=csv,noheader,nounits",
    ]
    proc = subprocess.run(cmd, check=True, text=True, capture_output=True)
    rows: list[dict[str, int]] = []
    reader = csv.reader(proc.stdout.splitlines())
    for row in reader:
        if len(row) < 5:
            continue
        rows.append(
            {
                "index": int(row[0].strip()),
                "util": int(row[1].strip()),
                "memory_used_mb": int(row[2].strip()),
                "memory_free_mb": int(row[3].strip()),
                "memory_total_mb": int(row[4].strip()),
            }
        )
    return rows


def monitor_gpus(seconds: int, interval: int) -> list[list[dict[str, int]]]:
    seconds = max(0, int(seconds))
    interval = max(1, int(interval))
    if seconds <= 0:
        return [run_nvidia_smi()]
    samples: list[list[dict[str, int]]] = []
    deadline = time.time() + seconds
    while True:
        samples.append(run_nvidia_smi())
        if time.time() >= deadline:
            break
        time.sleep(min(interval, max(0.0, deadline - time.time())))
    return samples


def select_sustained_idle_gpus(
    samples: list[list[dict[str, int]]],
    *,
    required_gpus: int,
    required_free_mb: int,
) -> list[int]:
    by_index: dict[int, list[dict[str, int]]] = {}
    for sample in samples:
        for row in sample:
            by_index.setdefault(row["index"], []).append(row)
    candidates: list[tuple[int, int, int]] = []
    sample_count = len(samples)
    for idx, rows in by_index.items():
        if len(rows) != sample_count:
            continue
        sustained_zero_util = all(row["util"] == 0 for row in rows)
        min_free = min(row["memory_free_mb"] for row in rows)
        max_used = max(row["memory_used_mb"] for row in rows)
        if sustained_zero_util and min_free >= required_free_mb:
            candidates.append((idx, min_free, max_used))
    candidates.sort(key=lambda item: (-item[1], item[2], item[0]))
    return [idx for idx, _free, _used in candidates[:required_gpus]]


def check_data_root(data_root: Path, datasets: list[str]) -> dict[str, Any]:
    checks: dict[str, Any] = {"data_root": str(data_root), "exists": data_root.exists(), "datasets": {}}
    for dataset in datasets:
        key = dataset.strip().lower()
        if not key:
            continue
        if key == "cmrc2018":
            required = [
                data_root / "cmrc2018" / "cmrc2018_train.json",
                data_root / "cmrc2018" / "cmrc2018_dev.json",
            ]
        elif key in {"hotpotqa", "hotpotqa_compressed"}:
            required = [data_root / "HotpotQA" / "raw" / "hotpot_dev_distractor_v1.json"]
        elif key in {"wikitext2", "wikitext-2"}:
            required = [data_root / "wikitext-2" / "wiki.test.txt"]
        else:
            required = []
        checks["datasets"][key] = {
            "required": [str(path) for path in required],
            "present": [path.is_file() for path in required],
            "ok": all(path.is_file() for path in required) if required else False,
        }
    return checks


def main() -> None:
    args = build_parser().parse_args()
    errors: list[str] = []
    warnings: list[str] = []

    models_config = Path(args.models_config)
    cfg = read_json(models_config)
    model_a = model_summary(Path(cfg["model_a"]["path"]))
    model_b = model_summary(Path(cfg["model_b"]["path"]))
    for label, summary in [("model_a", model_a), ("model_b", model_b)]:
        if not summary["exists"]:
            errors.append(f"{label} path does not exist: {summary['path']}")
        if not summary["config_exists"]:
            errors.append(f"{label} config.json missing: {summary['path']}")
        if is_vl_or_non_text(summary):
            errors.append(f"{label} looks like a VL/non-text checkpoint, not a text CausalLM: {summary['path']}")
        if summary["config_exists"] and not is_probably_causal_lm(summary):
            warnings.append(f"{label} does not advertise a clear CausalLM architecture: {summary['path']}")

    comparable_fields = ["num_hidden_layers", "hidden_size", "num_attention_heads", "num_key_value_heads"]
    arch_mismatch = {
        field: [model_a.get(field), model_b.get(field)]
        for field in comparable_fields
        if model_a.get(field) and model_b.get(field) and model_a.get(field) != model_b.get(field)
    }
    if arch_mismatch:
        errors.append(f"producer/consumer architecture mismatch: {arch_mismatch}")

    autoconfig = {}
    if not args.skip_autoconfig:
        autoconfig = {
            "model_a": check_autoconfig(Path(cfg["model_a"]["path"])),
            "model_b": check_autoconfig(Path(cfg["model_b"]["path"])),
        }
        for label, row in autoconfig.items():
            if not row.get("available"):
                warnings.append(f"{label} AutoConfig unavailable: {row.get('error')}")

    datasets = [part.strip() for part in args.datasets.split(",") if part.strip()]
    data = check_data_root(Path(args.data_root), datasets)
    if not data["exists"]:
        errors.append(f"data root does not exist: {args.data_root}")
    for key, row in data["datasets"].items():
        if not row["ok"]:
            errors.append(f"dataset files missing for {key}: {row['required']}")

    try:
        gpu_samples = monitor_gpus(args.monitor_seconds, args.sample_interval)
        selected = select_sustained_idle_gpus(
            gpu_samples,
            required_gpus=args.required_gpus,
            required_free_mb=args.required_free_mb,
        )
        if len(selected) < args.required_gpus:
            errors.append(
                "not enough sustained-idle GPUs: "
                f"required_gpus={args.required_gpus}, required_free_mb={args.required_free_mb}, selected={selected}"
            )
    except Exception as exc:
        gpu_samples = []
        selected = []
        errors.append(f"nvidia-smi GPU check failed: {exc}")

    report = {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "models_config": str(models_config),
        "model_a": model_a,
        "model_b": model_b,
        "autoconfig": autoconfig,
        "data": data,
        "gpu_policy": {
            "required_gpus": args.required_gpus,
            "required_free_mb": args.required_free_mb,
            "monitor_seconds": args.monitor_seconds,
            "sample_interval": args.sample_interval,
            "selected_cuda_visible_devices": selected,
        },
        "gpu_samples": gpu_samples,
    }

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.write_env and report["ok"]:
        env_path = Path(args.write_env)
        env_path.parent.mkdir(parents=True, exist_ok=True)
        env_path.write_text(
            "export CUDA_VISIBLE_DEVICES=" + ",".join(str(x) for x in selected) + "\n",
            encoding="utf-8",
        )

    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["ok"] else 2)


if __name__ == "__main__":
    main()
