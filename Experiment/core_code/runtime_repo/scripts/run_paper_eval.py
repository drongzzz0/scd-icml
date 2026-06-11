#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
from pathlib import Path


MAIN_METHODS = [
    "oracle",
    "raw_kv",
    "quantized_4bit",
    "selective_recompute",
    "reuse_only",
    "scd_full",
]

DEFAULT_SPLIT = {
    "cmrc2018": "dev",
    "hotpotqa": "validation",
    "wikitext2": "test",
}

DEFAULT_PROMPT = {
    "cmrc2018": "请阅读上下文并抽取最短答案。",
    "hotpotqa": "请进行多跳推理并给出最终答案。",
    "wikitext2": "请继续补全文本。",
}


def load_json_cfg(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def save_json_cfg(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_cmd(cmd: list[str], env: dict[str, str], workdir: Path) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env, cwd=str(workdir))


def load_structured_file(path: Path):
    text = path.read_text(encoding="utf-8-sig").strip()
    if not text:
        raise ValueError(f"empty file: {path}")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        rows = []
        for line in text.splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
        if rows:
            return rows
        raise


def sample_cmrc_prompts(data_root: Path, num_samples: int) -> list[str]:
    candidates = [
        data_root / "cmrc2018" / "cmrc2018_dev.json",
        data_root / "prompts" / "cmrc2018_dev.json",
        data_root / "cmrc2018" / "cmrc2018_train.json",
    ]
    prompts: list[str] = []
    for path in candidates:
        if not path.exists():
            continue
        try:
            obj = load_structured_file(path)
        except Exception:
            continue
        data = obj.get("data") if isinstance(obj, dict) else None
        if not isinstance(data, list):
            continue
        for article in data:
            for para in article.get("paragraphs", []):
                context = str(para.get("context", "")).strip()
                if not context:
                    continue
                for qa in para.get("qas", []):
                    question = str(qa.get("question", "")).strip()
                    if not question:
                        continue
                    prompts.append(f"阅读以下材料并回答问题。\n材料：{context}\n问题：{question}\n答案：")
                    if len(prompts) >= num_samples:
                        return prompts
    return prompts


def _hotpot_context_to_text(context) -> str:
    if isinstance(context, list):
        parts: list[str] = []
        for block in context:
            if isinstance(block, list) and len(block) >= 2:
                title = str(block[0])
                sents = block[1]
                if isinstance(sents, list):
                    parts.append(f"[{title}] " + " ".join(str(x) for x in sents))
                else:
                    parts.append(f"[{title}] {sents}")
            else:
                parts.append(str(block))
        return "\n".join(x for x in parts if x.strip())
    if isinstance(context, dict):
        return json.dumps(context, ensure_ascii=False)
    return str(context or "").strip()


def sample_hotpot_prompts(data_root: Path, num_samples: int) -> list[str]:
    candidates = [
        data_root / "HotpotQA" / "raw" / "hotpot_dev_distractor_v1.json",
        data_root / "droidspeak" / "hotpotqa.json",
    ]
    prompts: list[str] = []
    for path in candidates:
        if not path.exists():
            continue
        try:
            obj = load_structured_file(path)
        except Exception:
            continue
        if not isinstance(obj, list):
            continue
        for row in obj:
            question = str(row.get("question", "")).strip()
            if not question:
                continue
            evidence = _hotpot_context_to_text(row.get("context")) or "请基于常识进行多跳推理。"
            prompts.append(f"请阅读证据并回答问题。\n证据：{evidence}\n问题：{question}\n答案：")
            if len(prompts) >= num_samples:
                return prompts
    return prompts


def sample_wikitext_prompts(data_root: Path, num_samples: int) -> list[str]:
    candidates = [data_root / "wikitext", data_root / "WikiText-2", data_root / "wikitext-2"]
    prompts: list[str] = []
    for root in candidates:
        if not root.exists():
            continue
        for path in root.rglob("*.txt"):
            text = path.read_text(encoding="utf-8", errors="ignore")
            for line in text.splitlines():
                s = line.strip()
                if len(s) < 80:
                    continue
                prompts.append(f"请继续补全文本：\n{s}")
                if len(prompts) >= num_samples:
                    return prompts
    return prompts


def build_dataset_prompts(dataset: str, data_root: Path, num_samples: int) -> list[str]:
    if dataset == "cmrc2018":
        prompts = sample_cmrc_prompts(data_root, num_samples)
    elif dataset == "hotpotqa":
        prompts = sample_hotpot_prompts(data_root, num_samples)
    elif dataset == "wikitext2":
        prompts = sample_wikitext_prompts(data_root, num_samples)
    else:
        prompts = []
    return prompts or [DEFAULT_PROMPT.get(dataset, "请完成文本任务。")]


def run_eval_method(
    *,
    workdir: Path,
    env: dict[str, str],
    pairs: Path,
    reuse_ckpt: Path,
    patch_ckpt: Path,
    route: Path,
    eval_cfg: Path,
    method: str,
    out_dir: Path,
    seed: int,
    infer_device: str,
) -> None:
    run_cmd(
        [
            sys.executable,
            "-m",
            "scd_repro.cli.infer",
            "--pairs",
            str(pairs),
            "--reuse-ckpt",
            str(reuse_ckpt),
            "--patch-ckpt",
            str(patch_ckpt),
            "--route",
            str(route),
            "--eval-config",
            str(eval_cfg),
            "--method",
            method,
            "--out",
            str(out_dir),
            "--seed",
            str(seed),
            "--infer-device",
            infer_device,
        ],
        env,
        workdir,
    )
    run_cmd(
        [
            sys.executable,
            "-m",
            "scd_repro.cli.eval_cmd",
            "--pred",
            str(out_dir / "predictions.json"),
            "--oracle",
            str(out_dir / "oracle.json"),
            "--eval-config",
            str(eval_cfg),
            "--out",
            str(out_dir / "metrics.json"),
        ],
        env,
        workdir,
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="论文口径评测编排脚本（A=7B，B=Lite）")
    p.add_argument("--models-config", default="configs/models_a800.yaml")
    p.add_argument("--data-config", default="configs/data_a800.yaml")
    p.add_argument("--train-config", default="configs/train.yaml")
    p.add_argument("--eval-config", default="configs/eval.yaml")
    p.add_argument("--data-root", default="/home/student_t/maqianli/no4/data")
    p.add_argument("--output-root", default="/nas/mql/scd-repro")
    p.add_argument("--datasets", default="cmrc2018,hotpotqa,wikitext2")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpu-devices", default="")
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--samples-per-dataset", type=int, default=8)
    p.add_argument("--capture-batch-size", type=int, default=1)
    p.add_argument("--max-chunk-tokens", type=int, default=4096)
    p.add_argument("--bandwidths", default="100,150,200,400,800,1000")
    p.add_argument("--patch-budgets", default="0,1,3,5,8")
    p.add_argument("--skip-bandwidth-scan", action="store_true")
    p.add_argument("--skip-patch-budget-scan", action="store_true")
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
    train_cfg = load_json_cfg(train_cfg_path)
    eval_cfg = load_json_cfg(eval_cfg_path)

    path_a = str(models_cfg.get("model_a", {}).get("path", ""))
    path_b = str(models_cfg.get("model_b", {}).get("path", ""))
    if "Mistral-7B" not in path_a or "MistralLite" not in path_b:
        raise ValueError("model direction mismatch: need model_a=.../Mistral-7B*, model_b=.../MistralLite")

    output_root = Path(args.output_root)
    for sub in ["pairs", "ckpt_reuse", "ckpt_patch", "route", "reports"]:
        (output_root / sub).mkdir(parents=True, exist_ok=True)

    temp_cfg_dir = workdir / ".tmp_eval_cfg"
    temp_cfg_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    if args.gpu_devices:
        env["CUDA_VISIBLE_DEVICES"] = args.gpu_devices

    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]
    bandwidths = [float(x.strip()) for x in args.bandwidths.split(",") if x.strip()]
    patch_budgets = [int(x.strip()) for x in args.patch_budgets.split(",") if x.strip()]
    data_root = Path(args.data_root)

    model_compute = models_cfg.get("compute", {})
    train_device = str(train_cfg.get("compute", {}).get("train_device", "cuda:0"))
    infer_device = str(eval_cfg.get("compute", {}).get("infer_device", "cuda:0"))
    artifact_dtype = str(data_cfg.get("artifact", {}).get("dtype", "bfloat16"))

    for dataset in datasets:
        print(f"\n===== 开始数据集: {dataset} =====", flush=True)
        prompts = build_dataset_prompts(dataset, data_root, int(args.samples_per_dataset))

        per_data_cfg = copy.deepcopy(data_cfg)
        per_data_cfg.setdefault("artifact", {})["dtype"] = artifact_dtype
        per_data_cfg.setdefault("capture", {})["dataset"] = dataset
        per_data_cfg["capture"]["split"] = DEFAULT_SPLIT.get(dataset, "dev")
        per_data_cfg["capture"]["prompts"] = prompts
        per_data_cfg["capture"]["chunk_samples"] = max(1, int(args.capture_batch_size))
        per_data_cfg["capture"]["batch_size"] = max(1, int(args.capture_batch_size))
        per_data_cfg["capture"]["max_chunk_tokens"] = int(args.max_chunk_tokens)
        per_data_cfg["capture"]["max_length"] = int(args.max_length)
        per_data_cfg_path = temp_cfg_dir / f"data_{dataset}.yaml"
        save_json_cfg(per_data_cfg_path, per_data_cfg)

        pairs_dir = output_root / "pairs" / f"{dataset}_seed{args.seed}"
        reuse_dir = output_root / "ckpt_reuse" / f"{dataset}_seed{args.seed}"
        profile_path = output_root / "route" / f"profile_{dataset}_seed{args.seed}.json"
        patch_dir = output_root / "ckpt_patch" / f"patch_{dataset}_seed{args.seed}"
        route_path = output_root / "route" / f"route_{dataset}_seed{args.seed}.json"

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
                artifact_dtype,
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

        main_eval_cfg = copy.deepcopy(eval_cfg)
        main_eval_cfg.setdefault("network", {})["bandwidth_gbps"] = 200.0
        main_eval_cfg_path = temp_cfg_dir / f"eval_main_{dataset}.yaml"
        save_json_cfg(main_eval_cfg_path, main_eval_cfg)

        run_cmd(
            [sys.executable, "-m", "scd_repro.cli.build_route", "--profile", str(profile_path), "--eval-config", str(main_eval_cfg_path), "--out", str(route_path)],
            env,
            workdir,
        )

        main_runs: list[Path] = []
        for method in MAIN_METHODS:
            out_dir = output_root / "reports" / dataset / "main" / method
            out_dir.mkdir(parents=True, exist_ok=True)
            run_eval_method(
                workdir=workdir,
                env=env,
                pairs=pairs_dir,
                reuse_ckpt=reuse_dir,
                patch_ckpt=patch_dir,
                route=route_path,
                eval_cfg=main_eval_cfg_path,
                method=method,
                out_dir=out_dir,
                seed=args.seed,
                infer_device=infer_device,
            )
            main_runs.append(out_dir)

        run_cmd([sys.executable, "-m", "scd_repro.cli.report_cmd", "--runs", ",".join(str(x) for x in main_runs), "--out", str(output_root / "reports" / dataset / "main" / "final")], env, workdir)

        if not args.skip_bandwidth_scan:
            bw_runs: list[Path] = []
            for bw in bandwidths:
                bw_eval_cfg = copy.deepcopy(main_eval_cfg)
                bw_eval_cfg["network"]["bandwidth_gbps"] = float(bw)
                bw_eval_cfg_path = temp_cfg_dir / f"eval_bw_{dataset}_{int(bw)}.yaml"
                save_json_cfg(bw_eval_cfg_path, bw_eval_cfg)
                for method in ["scd_full", "selective_recompute", "raw_kv", "quantized_4bit"]:
                    out_dir = output_root / "reports" / dataset / "bandwidth_scan" / f"{int(bw)}gbps" / method
                    out_dir.mkdir(parents=True, exist_ok=True)
                    run_eval_method(
                        workdir=workdir,
                        env=env,
                        pairs=pairs_dir,
                        reuse_ckpt=reuse_dir,
                        patch_ckpt=patch_dir,
                        route=route_path,
                        eval_cfg=bw_eval_cfg_path,
                        method=method,
                        out_dir=out_dir,
                        seed=args.seed,
                        infer_device=infer_device,
                    )
                    bw_runs.append(out_dir)
            run_cmd([sys.executable, "-m", "scd_repro.cli.report_cmd", "--runs", ",".join(str(x) for x in bw_runs), "--out", str(output_root / "reports" / dataset / "bandwidth_scan" / "final")], env, workdir)

        if not args.skip_patch_budget_scan:
            budget_runs: list[Path] = []
            for k in patch_budgets:
                train_cfg_k = copy.deepcopy(train_cfg)
                train_cfg_k.setdefault("patch", {})["patch_budget"] = int(k)
                train_cfg_k_path = temp_cfg_dir / f"train_patch_{dataset}_k{k}.yaml"
                save_json_cfg(train_cfg_k_path, train_cfg_k)

                eval_cfg_k = copy.deepcopy(main_eval_cfg)
                eval_cfg_k.setdefault("route", {})["patch_budget"] = int(k)
                eval_cfg_k_path = temp_cfg_dir / f"eval_patch_{dataset}_k{k}.yaml"
                save_json_cfg(eval_cfg_k_path, eval_cfg_k)

                patch_dir_k = output_root / "ckpt_patch" / f"patch_{dataset}_seed{args.seed}_k{k}"
                route_path_k = output_root / "route" / f"route_{dataset}_seed{args.seed}_k{k}.json"

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
                        str(train_cfg_k_path),
                        "--out",
                        str(patch_dir_k),
                        "--seed",
                        str(args.seed),
                        "--train-device",
                        train_device,
                    ],
                    env,
                    workdir,
                )
                run_cmd([sys.executable, "-m", "scd_repro.cli.build_route", "--profile", str(profile_path), "--eval-config", str(eval_cfg_k_path), "--out", str(route_path_k)], env, workdir)

                out_dir = output_root / "reports" / dataset / "patch_budget" / f"k{k}" / "scd_full"
                out_dir.mkdir(parents=True, exist_ok=True)
                run_eval_method(
                    workdir=workdir,
                    env=env,
                    pairs=pairs_dir,
                    reuse_ckpt=reuse_dir,
                    patch_ckpt=patch_dir_k,
                    route=route_path_k,
                    eval_cfg=eval_cfg_k_path,
                    method="scd_full",
                    out_dir=out_dir,
                    seed=args.seed,
                    infer_device=infer_device,
                )
                budget_runs.append(out_dir)
            run_cmd([sys.executable, "-m", "scd_repro.cli.report_cmd", "--runs", ",".join(str(x) for x in budget_runs), "--out", str(output_root / "reports" / dataset / "patch_budget" / "final")], env, workdir)

    print("\n全部评测流程已完成。", flush=True)


if __name__ == "__main__":
    main()
