from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..io_utils import load_json, save_json
from .common import load_cfg, resolve_logger


@dataclass(frozen=True)
class LayerBlock:
    block_id: int
    layers: list[int]

    @property
    def label(self) -> str:
        if not self.layers:
            return f"block_{self.block_id:02d}_empty"
        return f"block_{self.block_id:02d}_{self.layers[0]:04d}_{self.layers[-1]:04d}"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="按 block 真实评测搜索 reuse/patch 混合路由")
    p.add_argument("--models-config", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--route-template", required=True)
    p.add_argument("--reuse-ckpt", required=True)
    p.add_argument("--patch-ckpt", required=True)
    p.add_argument("--patch-eval-config", required=True, help="仅运行 patch_only 的 eval config")
    p.add_argument("--hybrid-eval-config", default="", help="仅运行 reuse_patch 的 eval config；为空时自动生成")
    p.add_argument("--baseline-reuse-eval-config", default="", help="后续最终对比时可复用的 baseline+reuse-only config")
    p.add_argument("--out", required=True)
    p.add_argument("--block-size", type=int, default=4)
    p.add_argument("--limit", type=int, default=200)
    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--producer-device", default="")
    p.add_argument("--consumer-device", default="")
    p.add_argument("--worker-pairs", default="", help="候选评测用的设备对，格式如 cuda:0,cuda:1;cuda:2,cuda:3")
    p.add_argument("--dtype", default="")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-f1-drop", type=float, default=1.0)
    p.add_argument("--max-em-drop", type=float, default=0.10)
    p.add_argument("--min-ttft-gain-ms", type=float, default=1.0)
    p.add_argument("--max-accepted-blocks", type=int, default=0, help="0 表示不设上限")
    p.add_argument("--replay-top-k", type=int, default=0, help="对 first-pass 中排序最靠前的 top-k eligible 候选做二次 replay 复核；0 表示关闭")
    p.add_argument(
        "--replay-selection-mode",
        default="quality_threshold_ttft_first",
        choices=[
            "ttft_first",
            "quality_threshold_ttft_first",
            "quality_nonregression_ttft_first",
            "quality_threshold_f1_first_ttft_second",
        ],
        help="replay 后的最终选层规则",
    )
    p.add_argument("--replay-decision-max-f1-drop", type=float, default=-1.0, help="replay 最终决策时允许相对 patch-only 的最大绝对 F1 下降；<0 表示关闭")
    p.add_argument("--replay-decision-max-f1-drop-ratio", type=float, default=0.05, help="replay 最终决策时允许相对 patch-only 的最大相对 F1 下降比例；<0 表示关闭")
    p.add_argument("--replay-decision-max-em-drop", type=float, default=-1.0, help="replay 最终决策时允许相对 patch-only 的最大 EM 下降；<0 表示关闭")
    p.add_argument("--python-bin", default=sys.executable)
    p.add_argument("--log", default="")
    return p


def _runtime_eval_cfg_for_hybrid(template_eval_cfg: dict[str, Any]) -> dict[str, Any]:
    cfg = json.loads(json.dumps(template_eval_cfg))
    runtime = cfg.get("runtime_eval", {}) if isinstance(cfg.get("runtime_eval", {}), dict) else {}
    runtime.update(
        {
            "run_full_consumer_baseline": False,
            "run_reuse_only": False,
            "run_reuse_fallback": False,
            "run_patch_only": False,
            "run_reuse_patch": True,
            "run_reuse_patch_recompute": False,
            "run_oracle_xnorm_suffix": False,
            "run_oracle_state_suffix": False,
            "run_oracle_kv_patchonly": False,
            "max_eval_samples": int(runtime.get("max_eval_samples", 0) or 0),
            "strict_mode": True,
            "fail_on_missing_route_fields": False,
            "fail_on_missing_patch_fields": True,
            "align_full_consumer_first_decode": True,
        }
    )
    cfg["runtime_eval"] = runtime
    return cfg


def _runtime_eval_cfg_for_patch_anchor(template_eval_cfg: dict[str, Any]) -> dict[str, Any]:
    cfg = json.loads(json.dumps(template_eval_cfg))
    runtime = cfg.get("runtime_eval", {}) if isinstance(cfg.get("runtime_eval", {}), dict) else {}
    runtime.update(
        {
            "run_full_consumer_baseline": False,
            "run_reuse_only": False,
            "run_reuse_fallback": False,
            "run_patch_only": True,
            "run_reuse_patch": False,
            "run_reuse_patch_recompute": False,
            "run_oracle_xnorm_suffix": False,
            "run_oracle_state_suffix": False,
            "run_oracle_kv_patchonly": False,
            "strict_mode": True,
            "fail_on_missing_route_fields": False,
            "fail_on_missing_patch_fields": True,
            "align_full_consumer_first_decode": True,
        }
    )
    cfg["runtime_eval"] = runtime
    return cfg


def _force_eval_limit(eval_cfg: dict[str, Any], limit: int) -> dict[str, Any]:
    cfg = json.loads(json.dumps(eval_cfg))
    runtime = cfg.get("runtime_eval", {}) if isinstance(cfg.get("runtime_eval", {}), dict) else {}
    runtime["max_eval_samples"] = int(limit)
    cfg["runtime_eval"] = runtime
    return cfg


def _group_layers(layers: list[int], block_size: int) -> list[LayerBlock]:
    uniq = sorted({int(layer) for layer in layers})
    if block_size <= 0:
        raise ValueError("block_size 必须为正整数")
    return [LayerBlock(block_id=i, layers=uniq[i * block_size : (i + 1) * block_size]) for i in range((len(uniq) + block_size - 1) // block_size)]


def _route_with_patch_layers(route_template: dict[str, Any], patch_layers: list[int]) -> dict[str, Any]:
    patch_sorted = sorted({int(layer) for layer in patch_layers})
    route = json.loads(json.dumps(route_template))
    route["patch_layers"] = patch_sorted
    route["selected_patch_layers"] = patch_sorted
    route["fallback_layers"] = []
    route["selected_recompute_layers"] = []
    route["recompute_segment_length_by_layer"] = {}
    route["reuse_layers"] = sorted(
        set(int(layer) for layer in route.get("reuse_layers", [])) | set(range(int(route.get("num_layers", len(patch_sorted) or 0))))
    )
    if route.get("num_layers") is not None:
        num_layers = int(route["num_layers"])
        route["reuse_layers"] = [layer for layer in range(num_layers) if layer not in set(patch_sorted)]
    return route


def _metric_row(mode_row: dict[str, Any]) -> dict[str, float]:
    return {
        "f1": float(mode_row.get("f1", 0.0)),
        "em": float(mode_row.get("em", 0.0)),
        "ttft_ms": float(mode_row.get("ttft_ms", 0.0)),
        "avg_logp": float(mode_row.get("avg_logp", 0.0)),
        "avg_nll": float(mode_row.get("avg_nll", 0.0)),
        "encode_ms": float(mode_row.get("encode_ms", 0.0)),
        "transfer_ms": float(mode_row.get("transfer_ms", 0.0)),
        "rebuild_ms": float(mode_row.get("rebuild_ms", 0.0)),
        "patch_infer_ms": float(mode_row.get("patch_infer_ms", 0.0)),
        "decode_first_ms": float(mode_row.get("decode_first_ms", 0.0)),
        "payload_mb": float(mode_row.get("payload_mb", 0.0)),
    }


def _parse_worker_pairs(raw: str, producer_device: str, consumer_device: str) -> list[tuple[str, str]]:
    text = str(raw or "").strip()
    if not text:
        return [(str(producer_device or ""), str(consumer_device or ""))]
    pairs: list[tuple[str, str]] = []
    for part in text.split(";"):
        piece = part.strip()
        if not piece:
            continue
        items = [item.strip() for item in piece.split(",") if item.strip()]
        if len(items) != 2:
            raise ValueError(f"非法 worker pair: {piece}")
        pairs.append((items[0], items[1]))
    if not pairs:
        raise ValueError("worker_pairs 不能为空")
    return pairs


def _build_eval_cmd(
    *,
    python_bin: str,
    repo_root: Path,
    models_config: str,
    eval_config: str,
    data_root: str,
    route_path: str,
    reuse_ckpt: str,
    patch_ckpt: str,
    out_dir: Path,
    limit: int,
    max_length: int,
    max_new_tokens: int,
    producer_device: str,
    consumer_device: str,
    dtype: str,
    seed: int,
    log_path: Path,
) -> tuple[list[str], dict[str, str]]:
    cmd = [
        python_bin,
        "-m",
        "scd_repro.cli.eval_cmrc_strict",
        "--models-config",
        models_config,
        "--eval-config",
        eval_config,
        "--data-root",
        data_root,
        "--route",
        route_path,
        "--reuse-ckpt",
        reuse_ckpt,
        "--patch-ckpt",
        patch_ckpt,
        "--out",
        str(out_dir),
        "--limit",
        str(limit),
        "--max-length",
        str(max_length),
        "--max-new-tokens",
        str(max_new_tokens),
        "--seed",
        str(seed),
        "--log",
        str(log_path),
    ]
    if producer_device:
        cmd.extend(["--producer-device", producer_device])
    if consumer_device:
        cmd.extend(["--consumer-device", consumer_device])
    if dtype:
        cmd.extend(["--dtype", dtype])
    env = os.environ.copy()
    src_path = str(repo_root / "src")
    env["PYTHONPATH"] = src_path if not env.get("PYTHONPATH") else src_path + ":" + env["PYTHONPATH"]
    return cmd, env


def _call_eval(
    *,
    python_bin: str,
    repo_root: Path,
    models_config: str,
    eval_config: str,
    data_root: str,
    route_path: str,
    reuse_ckpt: str,
    patch_ckpt: str,
    out_dir: Path,
    limit: int,
    max_length: int,
    max_new_tokens: int,
    producer_device: str,
    consumer_device: str,
    dtype: str,
    seed: int,
    log_path: Path,
) -> dict[str, Any]:
    cmd, env = _build_eval_cmd(
        python_bin=python_bin,
        repo_root=repo_root,
        models_config=models_config,
        eval_config=eval_config,
        data_root=data_root,
        route_path=route_path,
        reuse_ckpt=reuse_ckpt,
        patch_ckpt=patch_ckpt,
        out_dir=out_dir,
        limit=limit,
        max_length=max_length,
        max_new_tokens=max_new_tokens,
        producer_device=producer_device,
        consumer_device=consumer_device,
        dtype=dtype,
        seed=seed,
        log_path=log_path,
    )
    subprocess.run(cmd, cwd=repo_root, check=True, env=env)
    return load_json(out_dir / "final_comparison.json")


def _pick_mode_payload(result: dict[str, Any], expected_method: str) -> dict[str, Any]:
    modes = result.get("modes", {})
    if expected_method not in modes:
        raise ValueError(f"结果中未找到 method={expected_method}")
    return dict(modes[expected_method])


def _candidate_record(
    *,
    block: LayerBlock,
    kept_patch_layers: list[int],
    current_metrics: dict[str, float],
    anchor_metrics: dict[str, float],
    candidate_metrics: dict[str, float],
) -> dict[str, Any]:
    return {
        "block_id": block.block_id,
        "block_label": block.label,
        "block_layers": list(block.layers),
        "kept_patch_layers": list(kept_patch_layers),
        "candidate_metrics": candidate_metrics,
        "delta_vs_current": {
            "ttft_gain_ms": float(current_metrics["ttft_ms"] - candidate_metrics["ttft_ms"]),
            "f1_drop": float(current_metrics["f1"] - candidate_metrics["f1"]),
            "em_drop": float(current_metrics["em"] - candidate_metrics["em"]),
        },
        "delta_vs_anchor_patch": {
            "ttft_gain_ms": float(anchor_metrics["ttft_ms"] - candidate_metrics["ttft_ms"]),
            "f1_drop": float(anchor_metrics["f1"] - candidate_metrics["f1"]),
            "em_drop": float(anchor_metrics["em"] - candidate_metrics["em"]),
        },
    }


def _candidate_sort_mode(replay_selection_mode: str) -> str:
    if replay_selection_mode == "quality_threshold_f1_first_ttft_second":
        return "f1_first_ttft_second"
    return "ttft_first"


def _sort_candidates(rows: list[dict[str, Any]], mode: str = "ttft_first") -> list[dict[str, Any]]:
    if mode == "f1_first_ttft_second":
        return sorted(
            rows,
            key=lambda row: (
                float(row["delta_vs_anchor_patch"]["f1_drop"]),
                float(row["delta_vs_anchor_patch"]["em_drop"]),
                -float(row["delta_vs_current"]["ttft_gain_ms"]),
                float(row["candidate_metrics"]["ttft_ms"]),
                -float(row["candidate_metrics"]["avg_logp"]),
            ),
        )
    return sorted(
        rows,
        key=lambda row: (
            float(row["delta_vs_current"]["ttft_gain_ms"]),
            -float(row["delta_vs_anchor_patch"]["f1_drop"]),
            -float(row["delta_vs_anchor_patch"]["em_drop"]),
            float(row["candidate_metrics"]["avg_logp"]),
        ),
        reverse=True,
    )


def main() -> None:
    args = build_parser().parse_args()
    logger = resolve_logger(args.log)
    repo_root = Path(__file__).resolve().parents[3]
    out_root = Path(args.out).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    route_template = load_json(args.route_template)
    worker_pairs = _parse_worker_pairs(args.worker_pairs, args.producer_device, args.consumer_device)
    patch_layers = sorted({int(layer) for layer in route_template.get("selected_patch_layers", route_template.get("patch_layers", []))})
    if not patch_layers:
        raise ValueError("route_template 缺少 patch_layers/selected_patch_layers")
    blocks = _group_layers(patch_layers, int(args.block_size))
    logger.info("blockwise hybrid search: patch_layers=%s blocks=%s", patch_layers, [block.layers for block in blocks])

    template_patch_eval_cfg = load_cfg(args.patch_eval_config)
    anchor_eval_cfg_path = out_root / "generated_eval_patch_only.json"
    anchor_eval_cfg = _force_eval_limit(_runtime_eval_cfg_for_patch_anchor(template_patch_eval_cfg), int(args.limit))
    save_json(anchor_eval_cfg_path, anchor_eval_cfg)

    hybrid_eval_cfg_path = Path(args.hybrid_eval_config).resolve() if args.hybrid_eval_config else out_root / "generated_eval_reuse_patch_only.json"
    if args.hybrid_eval_config:
        hybrid_eval_cfg = _force_eval_limit(load_cfg(args.hybrid_eval_config), int(args.limit))
        save_json(hybrid_eval_cfg_path, hybrid_eval_cfg)
    else:
        hybrid_eval_cfg = _force_eval_limit(_runtime_eval_cfg_for_hybrid(template_patch_eval_cfg), int(args.limit))
        save_json(hybrid_eval_cfg_path, hybrid_eval_cfg)

    anchor_route_path = out_root / "route_anchor_patch_only.json"
    save_json(anchor_route_path, _route_with_patch_layers(route_template, patch_layers))

    anchor_out = out_root / "eval_anchor_patch_only"
    logger.info("evaluating anchor patch-only on first %d samples", int(args.limit))
    anchor_result = _call_eval(
        python_bin=args.python_bin,
        repo_root=repo_root,
        models_config=args.models_config,
        eval_config=str(anchor_eval_cfg_path),
        data_root=args.data_root,
        route_path=str(anchor_route_path),
        reuse_ckpt=args.reuse_ckpt,
        patch_ckpt=args.patch_ckpt,
        out_dir=anchor_out,
        limit=int(args.limit),
        max_length=int(args.max_length),
        max_new_tokens=int(args.max_new_tokens),
        producer_device=args.producer_device,
        consumer_device=args.consumer_device,
        dtype=args.dtype,
        seed=int(args.seed),
        log_path=out_root / "anchor_patch_only.log",
    )
    anchor_metrics = _metric_row(_pick_mode_payload(anchor_result, "patch_only"))

    search_trace: list[dict[str, Any]] = []
    evaluation_records: list[dict[str, Any]] = []
    remaining_blocks = list(blocks)
    current_patch_layers = list(patch_layers)
    current_metrics = dict(anchor_metrics)
    accepted_blocks: list[int] = []
    candidate_sort_mode = _candidate_sort_mode(str(args.replay_selection_mode))

    step = 0
    while remaining_blocks:
        if int(args.max_accepted_blocks) > 0 and len(accepted_blocks) >= int(args.max_accepted_blocks):
            break
        step += 1
        candidate_records: list[dict[str, Any]] = []
        replay_records: list[dict[str, Any]] = []
        logger.info("search step %d: current_patch_layers=%s", step, current_patch_layers)
        job_specs: list[dict[str, Any]] = []
        for block in remaining_blocks:
            kept_patch_layers = [layer for layer in current_patch_layers if layer not in set(block.layers)]
            if not kept_patch_layers:
                continue
            route_path = out_root / f"route_step{step:02d}_{block.label}.json"
            save_json(route_path, _route_with_patch_layers(route_template, kept_patch_layers))
            eval_out = out_root / f"eval_step{step:02d}_{block.label}"
            job_specs.append(
                {
                    "block": block,
                    "kept_patch_layers": kept_patch_layers,
                    "route_path": route_path,
                    "eval_out": eval_out,
                    "log_path": out_root / f"step{step:02d}_{block.label}.log",
                }
            )

        for batch_start in range(0, len(job_specs), len(worker_pairs)):
            batch = job_specs[batch_start : batch_start + len(worker_pairs)]
            processes: list[tuple[dict[str, Any], tuple[str, str], subprocess.Popen[Any]]] = []
            for job, worker_pair in zip(batch, worker_pairs):
                cmd, env = _build_eval_cmd(
                    python_bin=args.python_bin,
                    repo_root=repo_root,
                    models_config=args.models_config,
                    eval_config=str(hybrid_eval_cfg_path),
                    data_root=args.data_root,
                    route_path=str(job["route_path"]),
                    reuse_ckpt=args.reuse_ckpt,
                    patch_ckpt=args.patch_ckpt,
                    out_dir=job["eval_out"],
                    limit=int(args.limit),
                    max_length=int(args.max_length),
                    max_new_tokens=int(args.max_new_tokens),
                    producer_device=worker_pair[0],
                    consumer_device=worker_pair[1],
                    dtype=args.dtype,
                    seed=int(args.seed),
                    log_path=job["log_path"],
                )
                logger.info("launch step %d candidate %s on %s -> %s", step, job["block"].label, worker_pair[0], worker_pair[1])
                processes.append((job, worker_pair, subprocess.Popen(cmd, cwd=repo_root, env=env)))

            for job, worker_pair, proc in processes:
                ret = proc.wait()
                if ret != 0:
                    raise subprocess.CalledProcessError(ret, proc.args)
                result = load_json(Path(job["eval_out"]) / "final_comparison.json")
                candidate_metrics = _metric_row(_pick_mode_payload(result, "reuse_patch"))
                record = _candidate_record(
                    block=job["block"],
                    kept_patch_layers=job["kept_patch_layers"],
                    current_metrics=current_metrics,
                    anchor_metrics=anchor_metrics,
                    candidate_metrics=candidate_metrics,
                )
                record["eval_out"] = str(job["eval_out"])
                record["worker_pair"] = {"producer_device": worker_pair[0], "consumer_device": worker_pair[1]}
                record["phase"] = "first_pass"
                candidate_records.append(record)
                evaluation_records.append(record)

        if not candidate_records:
            break

        eligible = [
            row
            for row in candidate_records
            if float(row["delta_vs_current"]["ttft_gain_ms"]) >= float(args.min_ttft_gain_ms)
            and float(row["delta_vs_anchor_patch"]["f1_drop"]) <= float(args.max_f1_drop)
            and float(row["delta_vs_anchor_patch"]["em_drop"]) <= float(args.max_em_drop)
        ]
        if not eligible:
            search_trace.append(
                {
                    "step": step,
                    "stop_reason": "no_eligible_block",
                    "current_patch_layers": list(current_patch_layers),
                    "current_metrics": dict(current_metrics),
                    "evaluated_blocks": [row["block_label"] for row in candidate_records],
                    "eligible_count": 0,
                }
            )
            break

        eligible = _sort_candidates(eligible, candidate_sort_mode)
        decision_candidates = list(eligible)
        decision_source = "first_pass"
        decision_quality_gate: dict[str, Any] | None = None
        replay_top_k = max(int(args.replay_top_k), 0)
        replay_input_records: list[dict[str, Any]] = []
        if replay_top_k > 0:
            replay_input_records = list(eligible[:replay_top_k])
            replay_job_specs: list[dict[str, Any]] = []
            for row in replay_input_records:
                replay_job_specs.append(
                    {
                        "seed_record": row,
                        "block": LayerBlock(block_id=int(row["block_id"]), layers=[int(layer) for layer in row["block_layers"]]),
                        "kept_patch_layers": [int(layer) for layer in row["kept_patch_layers"]],
                        "route_path": out_root / f"route_step{step:02d}_{row['block_label']}_replay.json",
                        "eval_out": out_root / f"eval_step{step:02d}_{row['block_label']}_replay",
                        "log_path": out_root / f"step{step:02d}_{row['block_label']}_replay.log",
                    }
                )
            for job in replay_job_specs:
                save_json(job["route_path"], _route_with_patch_layers(route_template, job["kept_patch_layers"]))

            for batch_start in range(0, len(replay_job_specs), len(worker_pairs)):
                batch = replay_job_specs[batch_start : batch_start + len(worker_pairs)]
                processes: list[tuple[dict[str, Any], tuple[str, str], subprocess.Popen[Any]]] = []
                for job, worker_pair in zip(batch, worker_pairs):
                    cmd, env = _build_eval_cmd(
                        python_bin=args.python_bin,
                        repo_root=repo_root,
                        models_config=args.models_config,
                        eval_config=str(hybrid_eval_cfg_path),
                        data_root=args.data_root,
                        route_path=str(job["route_path"]),
                        reuse_ckpt=args.reuse_ckpt,
                        patch_ckpt=args.patch_ckpt,
                        out_dir=job["eval_out"],
                        limit=int(args.limit),
                        max_length=int(args.max_length),
                        max_new_tokens=int(args.max_new_tokens),
                        producer_device=worker_pair[0],
                        consumer_device=worker_pair[1],
                        dtype=args.dtype,
                        seed=int(args.seed),
                        log_path=job["log_path"],
                    )
                    logger.info("launch replay step %d candidate %s on %s -> %s", step, job["block"].label, worker_pair[0], worker_pair[1])
                    processes.append((job, worker_pair, subprocess.Popen(cmd, cwd=repo_root, env=env)))

                for job, worker_pair, proc in processes:
                    ret = proc.wait()
                    if ret != 0:
                        raise subprocess.CalledProcessError(ret, proc.args)
                    result = load_json(Path(job["eval_out"]) / "final_comparison.json")
                    replay_metrics = _metric_row(_pick_mode_payload(result, "reuse_patch"))
                    replay_record = _candidate_record(
                        block=job["block"],
                        kept_patch_layers=job["kept_patch_layers"],
                        current_metrics=current_metrics,
                        anchor_metrics=anchor_metrics,
                        candidate_metrics=replay_metrics,
                    )
                    replay_record["eval_out"] = str(job["eval_out"])
                    replay_record["worker_pair"] = {"producer_device": worker_pair[0], "consumer_device": worker_pair[1]}
                    replay_record["phase"] = "replay"
                    replay_record["first_pass_eval_out"] = str(job["seed_record"].get("eval_out", ""))
                    replay_record["first_pass_candidate_metrics"] = dict(job["seed_record"]["candidate_metrics"])
                    replay_record["first_pass_delta_vs_current"] = dict(job["seed_record"]["delta_vs_current"])
                    replay_record["first_pass_delta_vs_anchor_patch"] = dict(job["seed_record"]["delta_vs_anchor_patch"])
                    replay_records.append(replay_record)
                    evaluation_records.append(replay_record)

            if replay_records:
                replay_sorted = _sort_candidates(replay_records, candidate_sort_mode)
                if str(args.replay_selection_mode) in (
                    "quality_threshold_ttft_first",
                    "quality_nonregression_ttft_first",
                    "quality_threshold_f1_first_ttft_second",
                ):
                    f1_gate_values: list[float] = []
                    if float(args.replay_decision_max_f1_drop) >= 0.0:
                        f1_gate_values.append(float(args.replay_decision_max_f1_drop))
                    if float(args.replay_decision_max_f1_drop_ratio) >= 0.0:
                        f1_gate_values.append(float(anchor_metrics["f1"]) * float(args.replay_decision_max_f1_drop_ratio))
                    allowed_f1_drop = min(f1_gate_values) if f1_gate_values else None
                    allowed_em_drop = float(args.replay_decision_max_em_drop) if float(args.replay_decision_max_em_drop) >= 0.0 else None
                    quality_pass = [
                        row
                        for row in replay_sorted
                        if (allowed_f1_drop is None or float(row["delta_vs_anchor_patch"]["f1_drop"]) <= allowed_f1_drop)
                        and (allowed_em_drop is None or float(row["delta_vs_anchor_patch"]["em_drop"]) <= allowed_em_drop)
                    ]
                    decision_quality_gate = {
                        "mode": str(args.replay_selection_mode),
                        "max_f1_drop": None if allowed_f1_drop is None else float(allowed_f1_drop),
                        "max_f1_drop_abs_arg": float(args.replay_decision_max_f1_drop),
                        "max_f1_drop_ratio_arg": float(args.replay_decision_max_f1_drop_ratio),
                        "max_em_drop": None if allowed_em_drop is None else float(allowed_em_drop),
                        "quality_pass_blocks": [str(row["block_label"]) for row in quality_pass],
                    }
                    if quality_pass:
                        decision_candidates = list(quality_pass)
                        if str(args.replay_selection_mode) == "quality_threshold_f1_first_ttft_second":
                            decision_source = "replay_quality_threshold_f1_first_ttft_second"
                        else:
                            decision_source = "replay_quality_threshold_ttft_first"
                    else:
                        search_trace.append(
                            {
                                "step": step,
                                "stop_reason": "no_replay_quality_pass",
                                "current_patch_layers": list(current_patch_layers),
                                "current_metrics": dict(current_metrics),
                                "replay_top_k": replay_top_k,
                                "replayed_blocks": [str(row["block_label"]) for row in replay_records],
                                "decision_quality_gate": decision_quality_gate,
                            }
                        )
                        logger.info(
                            "stop search at step %d: no replay candidate passed quality gate under mode=%s",
                            step,
                            str(args.replay_selection_mode),
                        )
                        remaining_blocks = []
                        break
                else:
                    decision_candidates = replay_sorted
                    decision_source = "replay_ttft_first"

        if not remaining_blocks:
            break

        best = decision_candidates[0]
        accepted_blocks.append(int(best["block_id"]))
        current_patch_layers = list(best["kept_patch_layers"])
        current_metrics = dict(best["candidate_metrics"])
        search_trace.append(
            {
                "step": step,
                "accepted_block_id": int(best["block_id"]),
                "accepted_block_label": str(best["block_label"]),
                "accepted_block_layers": list(best["block_layers"]),
                "current_patch_layers": list(current_patch_layers),
                "current_metrics": dict(current_metrics),
                "delta_vs_anchor_patch": dict(best["delta_vs_anchor_patch"]),
                "delta_vs_previous": dict(best["delta_vs_current"]),
                "first_pass_eligible_count": len(eligible),
                "replay_top_k": replay_top_k,
                "replayed_blocks": [str(row["block_label"]) for row in replay_records],
                "decision_source": decision_source,
                "decision_quality_gate": decision_quality_gate,
            }
        )
        remaining_blocks = [block for block in remaining_blocks if block.block_id != int(best["block_id"])]

    final_route = _route_with_patch_layers(route_template, current_patch_layers)
    final_route["search_policy"] = {
        "kind": "blockwise_greedy_real_eval_v2_replay" if int(args.replay_top_k) > 0 else "blockwise_greedy_real_eval_v1",
        "block_size": int(args.block_size),
        "anchor_method": "patch_only",
        "hybrid_method": "reuse_patch",
        "max_f1_drop": float(args.max_f1_drop),
        "max_em_drop": float(args.max_em_drop),
        "min_ttft_gain_ms": float(args.min_ttft_gain_ms),
        "accepted_blocks": list(accepted_blocks),
        "replay_top_k": int(args.replay_top_k),
        "replay_selection_mode": str(args.replay_selection_mode),
        "candidate_sort_mode": candidate_sort_mode,
        "replay_decision_max_f1_drop": float(args.replay_decision_max_f1_drop),
        "replay_decision_max_f1_drop_ratio": float(args.replay_decision_max_f1_drop_ratio),
        "replay_decision_max_em_drop": float(args.replay_decision_max_em_drop),
    }
    final_route_path = out_root / "selected_hybrid_route.json"
    save_json(final_route_path, final_route)

    report = {
        "search_config": {
            "models_config": args.models_config,
            "data_root": args.data_root,
            "route_template": args.route_template,
            "reuse_ckpt": args.reuse_ckpt,
            "patch_ckpt": args.patch_ckpt,
            "patch_eval_config": args.patch_eval_config,
            "generated_patch_eval_config": str(anchor_eval_cfg_path),
            "hybrid_eval_config": str(hybrid_eval_cfg_path),
            "limit": int(args.limit),
            "block_size": int(args.block_size),
            "producer_device": args.producer_device,
            "consumer_device": args.consumer_device,
            "worker_pairs": [{"producer_device": pair[0], "consumer_device": pair[1]} for pair in worker_pairs],
            "dtype": args.dtype,
            "seed": int(args.seed),
            "max_f1_drop": float(args.max_f1_drop),
            "max_em_drop": float(args.max_em_drop),
            "min_ttft_gain_ms": float(args.min_ttft_gain_ms),
            "max_accepted_blocks": int(args.max_accepted_blocks),
            "replay_top_k": int(args.replay_top_k),
            "replay_selection_mode": str(args.replay_selection_mode),
            "candidate_sort_mode": candidate_sort_mode,
            "replay_decision_max_f1_drop": float(args.replay_decision_max_f1_drop),
            "replay_decision_max_f1_drop_ratio": float(args.replay_decision_max_f1_drop_ratio),
            "replay_decision_max_em_drop": float(args.replay_decision_max_em_drop),
        },
        "anchor_patch_only": {
            "route_path": str(anchor_route_path),
            "eval_out": str(anchor_out),
            "metrics": anchor_metrics,
        },
        "blocks": [{"block_id": block.block_id, "label": block.label, "layers": block.layers} for block in blocks],
        "search_trace": search_trace,
        "evaluations": evaluation_records,
        "selected_hybrid_route_path": str(final_route_path),
        "selected_hybrid_patch_layers": list(current_patch_layers),
        "selected_hybrid_metrics": current_metrics,
        "accepted_block_ids": list(accepted_blocks),
    }
    save_json(out_root / "search_report.json", report)
    logger.info(
        "blockwise hybrid search done: accepted_blocks=%s final_patch_layers=%s final_metrics=%s",
        accepted_blocks,
        current_patch_layers,
        current_metrics,
    )


if __name__ == "__main__":
    main()
