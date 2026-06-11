#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
from pathlib import Path

from scd_repro.cmrc import load_cmrc2018_samples


def load_json_cfg(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def save_json_cfg(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_cmd(cmd: list[str], env: dict[str, str], workdir: Path) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env, cwd=str(workdir))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="A800 CMRC2018 严格子集评测总控脚本")
    p.add_argument("--models-config", default="configs/models_a800.yaml")
    p.add_argument("--data-config", default="configs/data_a800.yaml")
    p.add_argument("--train-config", default="configs/train.yaml")
    p.add_argument("--eval-config", default="configs/eval.yaml")
    p.add_argument("--data-root", default="/home/student_t/maqianli/no4/data")
    p.add_argument("--output-root", default="/nas/mql/scd-paper-cmrc-7b2lite-200")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpu-devices", default="5,6")
    p.add_argument("--max-length", type=int, default=0)
    p.add_argument("--samples", type=int, default=200)
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--capture-batch-size", type=int, default=1)
    p.add_argument("--max-chunk-tokens", type=int, default=4096)
    p.add_argument("--bandwidth-gbps", type=float, default=200.0)
    p.add_argument("--quant-group-size", type=int, default=128)
    return p


def main() -> None:
    args = build_parser().parse_args()
    workdir = Path.cwd()
    models_cfg_path = (workdir / args.models_config).resolve()
    data_cfg_path = (workdir / args.data_config).resolve()
    train_cfg_path = (workdir / args.train_config).resolve()
    eval_cfg_path = (workdir / args.eval_config).resolve()

    models_cfg = load_json_cfg(models_cfg_path)
    data_cfg = load_json_cfg(data_cfg_path)
    eval_cfg = load_json_cfg(eval_cfg_path)

    model_a_path = str(models_cfg.get("model_a", {}).get("path", ""))
    model_b_path = str(models_cfg.get("model_b", {}).get("path", ""))
    if "Mistral-7B" not in model_a_path or "MistralLite" not in model_b_path:
        raise ValueError("模型方向校验失败：要求 model_a=Mistral-7B, model_b=MistralLite")

    data_root = Path(args.data_root)
    samples = load_cmrc2018_samples(data_root, limit=int(args.samples))
    prompts = [row.prompt for row in samples]

    output_root = Path(args.output_root)
    for sub in ["pairs", "ckpt_reuse", "ckpt_patch", "route", "reports"]:
        (output_root / sub).mkdir(parents=True, exist_ok=True)

    temp_cfg_dir = workdir / ".tmp_eval_cfg"
    temp_cfg_dir.mkdir(parents=True, exist_ok=True)

    per_data_cfg = copy.deepcopy(data_cfg)
    per_data_cfg.setdefault("artifact", {})["dtype"] = str(data_cfg.get("artifact", {}).get("dtype", "bfloat16"))
    per_data_cfg.setdefault("capture", {})["dataset"] = "cmrc2018"
    per_data_cfg["capture"]["split"] = "dev"
    per_data_cfg["capture"]["prompts"] = prompts
    per_data_cfg["capture"]["chunk_samples"] = max(1, int(args.capture_batch_size))
    per_data_cfg["capture"]["batch_size"] = max(1, int(args.capture_batch_size))
    per_data_cfg["capture"]["max_chunk_tokens"] = int(args.max_chunk_tokens)
    per_data_cfg["capture"]["max_length"] = int(args.max_length)
    per_data_cfg_path = temp_cfg_dir / "data_cmrc2018_strict.yaml"
    save_json_cfg(per_data_cfg_path, per_data_cfg)

    strict_eval_cfg = copy.deepcopy(eval_cfg)
    strict_eval_cfg.setdefault("network", {})["bandwidth_gbps"] = float(args.bandwidth_gbps)
    strict_eval_cfg_path = temp_cfg_dir / "eval_cmrc2018_strict.yaml"
    save_json_cfg(strict_eval_cfg_path, strict_eval_cfg)

    env = os.environ.copy()
    if args.gpu_devices:
        env["CUDA_VISIBLE_DEVICES"] = args.gpu_devices

    model_compute = models_cfg.get("compute", {})
    train_device = str(load_json_cfg(train_cfg_path).get("compute", {}).get("train_device", "cuda:0"))
    consumer_device = str(model_compute.get("model_b_device", "cuda:1"))

    pairs_dir = output_root / "pairs" / f"cmrc2018_seed{args.seed}"
    reuse_dir = output_root / "ckpt_reuse" / f"cmrc2018_seed{args.seed}"
    profile_path = output_root / "route" / f"profile_cmrc2018_seed{args.seed}.json"
    patch_dir = output_root / "ckpt_patch" / f"patch_cmrc2018_seed{args.seed}"
    route_path = output_root / "route" / f"route_cmrc2018_seed{args.seed}.json"
    strict_out = output_root / "reports" / "cmrc2018" / "strict_main"

    run_cmd(
        [
            sys.executable,
            "-m",
            "scd_repro.cli.capture_pair",
            "--backend",
            "transformers",
            "--models-config",
            str(models_cfg_path),
            "--data-config",
            str(per_data_cfg_path),
            "--out",
            str(pairs_dir),
            "--seed",
            str(args.seed),
            "--max-length",
            str(args.max_length),
            "--dtype",
            str(models_cfg.get("runtime", {}).get("dtype", "bfloat16")),
            "--model-a-device",
            str(model_compute.get("model_a_device", "cuda:0")),
            "--model-b-device",
            str(model_compute.get("model_b_device", "cuda:1")),
            "--artifact-dtype",
            str(data_cfg.get("artifact", {}).get("dtype", "bfloat16")),
            "--chunk-samples",
            str(max(1, int(args.capture_batch_size))),
            "--max-chunk-tokens",
            str(int(args.max_chunk_tokens)),
        ],
        env,
        workdir,
    )

    run_cmd(
        [
            sys.executable,
            "-m",
            "scd_repro.cli.train_reuse",
            "--pairs",
            str(pairs_dir),
            "--train-config",
            str(train_cfg_path),
            "--out",
            str(reuse_dir),
            "--seed",
            str(args.seed),
            "--train-device",
            train_device,
        ],
        env,
        workdir,
    )

    run_cmd(
        [
            sys.executable,
            "-m",
            "scd_repro.cli.profile_layers",
            "--pairs",
            str(pairs_dir),
            "--reuse-ckpt",
            str(reuse_dir),
            "--out",
            str(profile_path),
            "--train-device",
            train_device,
        ],
        env,
        workdir,
    )

    run_cmd(
        [
            sys.executable,
            "-m",
            "scd_repro.cli.train_patch",
            "--pairs",
            str(pairs_dir),
            "--profile",
            str(profile_path),
            "--train-config",
            str(train_cfg_path),
            "--out",
            str(patch_dir),
            "--seed",
            str(args.seed),
            "--train-device",
            train_device,
        ],
        env,
        workdir,
    )

    run_cmd(
        [
            sys.executable,
            "-m",
            "scd_repro.cli.build_route",
            "--profile",
            str(profile_path),
            "--eval-config",
            str(strict_eval_cfg_path),
            "--out",
            str(route_path),
        ],
        env,
        workdir,
    )

    run_cmd(
        [
            sys.executable,
            "-m",
            "scd_repro.cli.eval_cmrc_strict",
            "--models-config",
            str(models_cfg_path),
            "--data-root",
            str(data_root),
            "--route",
            str(route_path),
            "--reuse-ckpt",
            str(reuse_dir),
            "--patch-ckpt",
            str(patch_dir),
            "--out",
            str(strict_out),
            "--limit",
            str(int(args.samples)),
            "--max-length",
            str(int(args.max_length)),
            "--max-new-tokens",
            str(int(args.max_new_tokens)),
            "--bandwidth-gbps",
            str(float(args.bandwidth_gbps)),
            "--quant-group-size",
            str(int(args.quant_group_size)),
            "--producer-device",
            str(model_compute.get("model_a_device", "cuda:0")),
            "--consumer-device",
            consumer_device,
            "--dtype",
            str(models_cfg.get("runtime", {}).get("dtype", "bfloat16")),
            "--seed",
            str(args.seed),
        ],
        env,
        workdir,
    )

    print(f"\nCMRC2018 严格评测已完成，结果目录：{strict_out}", flush=True)


if __name__ == "__main__":
    main()
