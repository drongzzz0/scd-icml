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


def _load_route_patch_layers(path: Path) -> list[int]:
    route = load_json_cfg(path)
    selected = route.get("selected_patch_layers", route.get("patch_layers", []))
    if not isinstance(selected, list):
        return []
    out: list[int] = []
    seen: set[int] = set()
    for item in selected:
        try:
            layer = int(item)
        except Exception:
            continue
        if layer not in seen:
            seen.add(layer)
            out.append(layer)
    return sorted(out)


def run_cmd(cmd: list[str], env: dict[str, str], workdir: Path) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env, cwd=str(workdir))


def _resolve_eval_offset(train_offset: int, train_samples: int, eval_offset: int) -> int:
    return int(eval_offset) if int(eval_offset) >= 0 else int(train_offset) + int(train_samples)


def _slice_cmrc_window(samples: list, *, start: int, count: int, label: str) -> list:
    if int(start) < 0:
        raise ValueError(f"{label} offset must be >= 0, got {start}")
    if int(count) <= 0:
        raise ValueError(f"{label} samples must be > 0, got {count}")
    end = int(start) + int(count)
    if end > len(samples):
        raise ValueError(f"{label} window [{start}, {end}) exceeds loaded sample count {len(samples)}")
    return list(samples[int(start) : end])


def _prepare_cmrc_split(
    data_root: Path,
    *,
    train_offset: int,
    train_samples: int,
    eval_offset: int,
    eval_samples: int,
) -> tuple[list, list, int]:
    resolved_eval_offset = _resolve_eval_offset(int(train_offset), int(train_samples), int(eval_offset))
    load_limit = max(int(train_offset) + int(train_samples), resolved_eval_offset + int(eval_samples))
    all_samples = load_cmrc2018_samples(data_root, limit=load_limit)
    if len(all_samples) < load_limit:
        raise ValueError(f"CMRC sample count is insufficient: need {load_limit}, got {len(all_samples)}")
    train_rows = _slice_cmrc_window(all_samples, start=int(train_offset), count=int(train_samples), label="train")
    eval_rows = _slice_cmrc_window(all_samples, start=resolved_eval_offset, count=int(eval_samples), label="eval")
    overlap_qids = sorted({str(row.qid) for row in train_rows} & {str(row.qid) for row in eval_rows})
    if overlap_qids:
        preview = overlap_qids[:5]
        suffix = "..." if len(overlap_qids) > len(preview) else ""
        raise ValueError(f"train/eval subsets overlap on qids: {preview}{suffix}")
    return train_rows, eval_rows, load_limit


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Canonical Mistral strict default pipeline")
    p.add_argument("--models-config", default="configs/models_ser_mistrallite_to_mistral7_20260405.json")
    p.add_argument("--data-config", default="configs/data_ser.yaml")
    p.add_argument("--train-config", default="configs/train.yaml")
    p.add_argument("--eval-config", default="configs/eval.yaml")
    p.add_argument("--patch-eval-config", default="configs/eval_patch_only.yaml")
    p.add_argument("--route-template", default="configs/route_patch_only_all32.json")
    p.add_argument("--data-root", default="/disk_n/c101-3-1/projects/scd_main/legacy/no4/data")
    p.add_argument("--output-root", default="/disk_n/c101-3-1/projects/scd-default-mistral-strict")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpu-devices", default="")
    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument("--samples", type=int, default=200, help="Training sample count")
    p.add_argument("--eval-samples", type=int, default=200, help="Held-out eval sample count")
    p.add_argument("--train-offset", type=int, default=0)
    p.add_argument("--eval-offset", type=int, default=-1, help="Defaults to train_offset + samples")
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--capture-batch-size", type=int, default=2)
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
    patch_eval_cfg_path = (workdir / args.patch_eval_config).resolve()
    route_template_path = (workdir / args.route_template).resolve()

    models_cfg = load_json_cfg(models_cfg_path)
    data_cfg = load_json_cfg(data_cfg_path)
    train_cfg = load_json_cfg(train_cfg_path)
    eval_cfg = load_json_cfg(eval_cfg_path)
    patch_eval_cfg = load_json_cfg(patch_eval_cfg_path)
    model_a_name = str(models_cfg.get("model_a", {}).get("name", ""))
    model_b_name = str(models_cfg.get("model_b", {}).get("name", ""))
    if "MistralLite" not in model_a_name or "Mistral-7B" not in model_b_name:
        raise ValueError("strict default runner requires model_a=MistralLite and model_b=Mistral-7B")

    data_root = Path(args.data_root)
    train_rows, eval_rows, eval_limit = _prepare_cmrc_split(
        data_root,
        train_offset=int(args.train_offset),
        train_samples=int(args.samples),
        eval_offset=int(args.eval_offset),
        eval_samples=int(args.eval_samples),
    )
    resolved_eval_offset = _resolve_eval_offset(int(args.train_offset), int(args.samples), int(args.eval_offset))
    train_prompts = [row.prompt for row in train_rows]
    eval_qids = [str(row.qid) for row in eval_rows]

    output_root = Path(args.output_root)
    for sub in ["pairs", "ckpt_reuse", "profile", "ckpt_patch", "reports", "merged", "routes"]:
        (output_root / sub).mkdir(parents=True, exist_ok=True)

    split_manifest_path = output_root / "cmrc2018_split_manifest.json"
    save_json_cfg(
        split_manifest_path,
        {
            "dataset": "cmrc2018",
            "train_offset": int(args.train_offset),
            "train_samples": int(args.samples),
            "eval_offset": int(resolved_eval_offset),
            "eval_samples": int(args.eval_samples),
            "load_limit": int(eval_limit),
            "train_qids": [str(row.qid) for row in train_rows],
            "eval_qids": eval_qids,
        },
    )

    temp_cfg_dir = workdir / ".tmp_eval_cfg"
    temp_cfg_dir.mkdir(parents=True, exist_ok=True)

    per_data_cfg = copy.deepcopy(data_cfg)
    per_data_cfg.setdefault("artifact", {})["dtype"] = str(data_cfg.get("artifact", {}).get("dtype", "bfloat16"))
    per_data_cfg.setdefault("capture", {})["dataset"] = "cmrc2018"
    per_data_cfg["capture"]["split"] = "dev"
    per_data_cfg["capture"]["prompts"] = train_prompts
    per_data_cfg["capture"]["chunk_samples"] = max(1, int(args.capture_batch_size))
    per_data_cfg["capture"]["batch_size"] = max(1, int(args.capture_batch_size))
    per_data_cfg["capture"]["max_chunk_tokens"] = int(args.max_chunk_tokens)
    per_data_cfg["capture"]["max_length"] = int(args.max_length)
    per_data_cfg_path = temp_cfg_dir / "data_cmrc2018_mistral_strict.json"
    save_json_cfg(per_data_cfg_path, per_data_cfg)

    strict_eval_cfg = copy.deepcopy(eval_cfg)
    strict_eval_cfg.setdefault("network", {})["bandwidth_gbps"] = float(args.bandwidth_gbps)
    strict_eval_cfg.setdefault("runtime_eval", {})["qid_allowlist"] = eval_qids
    strict_eval_cfg_path = temp_cfg_dir / "eval_cmrc2018_strict_default.json"
    save_json_cfg(strict_eval_cfg_path, strict_eval_cfg)

    patch_only_eval_cfg = copy.deepcopy(patch_eval_cfg)
    patch_only_eval_cfg.setdefault("network", {})["bandwidth_gbps"] = float(args.bandwidth_gbps)
    patch_only_eval_cfg.setdefault("runtime_eval", {})["qid_allowlist"] = eval_qids
    patch_only_eval_cfg_path = temp_cfg_dir / "eval_cmrc2018_patch_only_default.json"
    save_json_cfg(patch_only_eval_cfg_path, patch_only_eval_cfg)

    env = os.environ.copy()
    if args.gpu_devices:
        env["CUDA_VISIBLE_DEVICES"] = args.gpu_devices

    model_compute = models_cfg.get("compute", {})
    train_device = str(train_cfg.get("compute", {}).get("train_device", "cuda:0"))
    producer_device = str(model_compute.get("model_a_device", "cuda:0"))
    consumer_device = str(model_compute.get("model_b_device", "cuda:1"))
    runtime_dtype = str(models_cfg.get("runtime", {}).get("dtype", "bfloat16"))
    use_sequential_runtime = producer_device == consumer_device

    pairs_dir = output_root / "pairs" / f"cmrc2018_seed{args.seed}"
    reuse_dir = output_root / "ckpt_reuse" / f"cmrc2018_seed{args.seed}"
    profile_path = output_root / "profile" / f"profile_cmrc2018_seed{args.seed}.json"
    patch_dir = output_root / "ckpt_patch" / f"patch_cmrc2018_seed{args.seed}"
    strict_main_out = output_root / "reports" / "cmrc2018" / "strict_baseline_reuse_only"
    patch_only_out = output_root / "reports" / "cmrc2018" / "strict_patch_only"
    merged_out = output_root / "merged" / "cmrc2018_strict_default"

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
            runtime_dtype,
            "--model-a-device",
            producer_device,
            "--model-b-device",
            consumer_device,
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
            "--route",
            str(route_template_path),
            "--train-config",
            str(train_cfg_path),
            "--models-config",
            str(models_cfg_path),
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
            "scd_repro.cli.eval_cmrc_strict",
            "--models-config",
            str(models_cfg_path),
            "--eval-config",
            str(strict_eval_cfg_path),
            "--data-root",
            str(data_root),
            "--route",
            str(route_template_path),
            "--reuse-ckpt",
            str(reuse_dir),
            "--patch-ckpt",
            str(patch_dir),
            "--out",
            str(strict_main_out),
            "--limit",
            str(int(eval_limit)),
            "--max-length",
            str(int(args.max_length)),
            "--max-new-tokens",
            str(int(args.max_new_tokens)),
            "--bandwidth-gbps",
            str(float(args.bandwidth_gbps)),
            "--quant-group-size",
            str(int(args.quant_group_size)),
            "--producer-device",
            producer_device,
            "--consumer-device",
            consumer_device,
            "--dtype",
            runtime_dtype,
            "--seed",
            str(args.seed),
            *(["--sequential-runtime"] if use_sequential_runtime else []),
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
            "--eval-config",
            str(patch_only_eval_cfg_path),
            "--data-root",
            str(data_root),
            "--route",
            str(route_template_path),
            "--reuse-ckpt",
            str(reuse_dir),
            "--patch-ckpt",
            str(patch_dir),
            "--out",
            str(patch_only_out),
            "--limit",
            str(int(eval_limit)),
            "--max-length",
            str(int(args.max_length)),
            "--max-new-tokens",
            str(int(args.max_new_tokens)),
            "--bandwidth-gbps",
            str(float(args.bandwidth_gbps)),
            "--quant-group-size",
            str(int(args.quant_group_size)),
            "--producer-device",
            producer_device,
            "--consumer-device",
            consumer_device,
            "--dtype",
            runtime_dtype,
            "--seed",
            str(args.seed),
            *(["--sequential-runtime"] if use_sequential_runtime else []),
        ],
        env,
        workdir,
    )

    merge_cmd = [
        sys.executable,
        "-m",
        "scd_repro.cli.merge_eval_cmrc_strict",
    ]
    for input_dir in [strict_main_out, patch_only_out]:
        merge_cmd.extend(["--input-dir", str(input_dir)])
    merge_cmd.extend(["--out", str(merged_out)])
    run_cmd(merge_cmd, env, workdir)

    print(f"\nMistral strict default evaluation finished. Merged result dir: {merged_out}", flush=True)


if __name__ == "__main__":
    main()
