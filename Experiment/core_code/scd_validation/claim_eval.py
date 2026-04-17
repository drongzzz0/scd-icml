from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from .common import bootstrap_mean_ci, write_csv, write_jsonl


CLAIM_ANCHORS = {
    "table1_qwen32b_primary": {
        "ttft_ratio_target": 0.4839,
        "quality_retention_target": 0.9669,
        "kl_ceiling_target": 0.31,
        "ppl_ratio_target": None,
        "payload_ratio_target_vs_raw_kv": None,
    },
    "table2_mistrallite_mistral7b_200gbps": {
        "ttft_ratio_target": 0.3772,
        "quality_retention_target": 0.9685,
        "kl_ceiling_target": 0.1105,
        "ppl_ratio_target": 1.1205,
        "payload_ratio_target_vs_raw_kv": 0.7591,
    },
}

RUN_MANIFEST_FIELDS = [
    "run_id",
    "timestamp_utc",
    "git_commit",
    "pair_id",
    "producer_model",
    "consumer_model",
    "method",
    "task",
    "split",
    "seed",
    "context_len",
    "decode_len",
    "bandwidth_gbps",
    "patch_budget_k",
    "patch_layers",
    "rank_k",
    "rank_v",
    "rank_h",
    "claim_anchor",
    "status",
    "artifact_paths",
    "notes",
]

PROMPT_LEVEL_FIELDS = [
    "run_id",
    "pair_id",
    "pair_bucket",
    "claim_anchor",
    "method",
    "task",
    "split",
    "seed",
    "prompt_id",
    "prefix_id",
    "context_len",
    "decode_len",
    "bandwidth_gbps",
    "patch_budget_k",
    "patch_layers",
    "rank_k",
    "rank_v",
    "rank_h",
    "payload_bytes",
    "producer_prefill_ms",
    "producer_encode_ms",
    "transfer_ms",
    "consumer_rebuild_ms",
    "consumer_patch_ms",
    "consumer_recompute_ms",
    "first_token_decode_ms",
    "ttft_ms",
    "end_to_end_ms",
    "task_metric_name",
    "task_metric_value",
    "token_kl",
    "token_tv",
    "ppl",
    "native_consumer_ttft_ms",
    "native_consumer_task_metric_value",
    "native_consumer_ppl",
    "raw_kv_payload_bytes",
]

RUN_LEVEL_FIELDS = [
    "run_id",
    "pair_id",
    "pair_bucket",
    "claim_anchor",
    "method",
    "task",
    "split",
    "seed",
    "n_prompts",
    "mean_ttft_ms",
    "ttft_ci95_low",
    "ttft_ci95_high",
    "mean_end_to_end_ms",
    "mean_task_metric_value",
    "task_metric_ci95_low",
    "task_metric_ci95_high",
    "mean_token_kl",
    "kl_ci95_low",
    "kl_ci95_high",
    "mean_token_tv",
    "mean_ppl",
    "ppl_ci95_low",
    "ppl_ci95_high",
    "mean_payload_bytes",
    "mean_payload_ratio_vs_raw_kv_local",
    "mean_ttft_ratio_local",
    "mean_quality_retention_local",
    "mean_ppl_ratio_local",
]

CLAIM_GAP_FIELDS = [
    "pair_id",
    "pair_bucket",
    "claim_anchor",
    "method",
    "task",
    "ttft_ratio_target",
    "quality_retention_target",
    "kl_ceiling_target",
    "ppl_ratio_target",
    "payload_ratio_target_vs_raw_kv",
    "ttft_ratio_local",
    "quality_retention_local",
    "kl_local",
    "ppl_ratio_local",
    "payload_ratio_vs_raw_kv_local",
    "gap_ttft",
    "gap_quality",
    "gap_kl",
    "gap_ppl",
    "gap_payload",
    "gap_fidelity",
    "claim_gap_total",
    "pass_label",
    "ci_overlap_flag",
]

LATENCY_FIELDS = [
    "run_id",
    "method",
    "producer_prefill_ms",
    "producer_encode_ms",
    "transfer_ms",
    "consumer_rebuild_ms",
    "consumer_patch_ms",
    "consumer_recompute_ms",
    "first_token_decode_ms",
    "ttft_ms",
    "end_to_end_ms",
]

ROUTE_RESCUE_FIELDS = [
    "pair_id",
    "claim_anchor",
    "task",
    "original_method_claim_gap_total",
    "route_method",
    "route_claim_gap_total",
    "gap_reduction_ratio",
    "ttft_delta_vs_original",
    "quality_delta_vs_original",
    "kl_delta_vs_original",
    "ppl_delta_vs_original",
    "rescue_label",
]


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _subset_key(selected: list[int]) -> str:
    return ",".join(str(layer) for layer in sorted(selected))


def select_patch_layers(selection_cfg: dict) -> dict:
    restore_one_gain = {
        int(layer): float(gain)
        for layer, gain in selection_cfg.get("restore_one_gain", {}).items()
    }
    greedy_marginal_gain = {
        subset: {int(layer): float(gain) for layer, gain in gains.items()}
        for subset, gains in selection_cfg.get("greedy_marginal_gain", {}).items()
    }
    if not restore_one_gain:
        return {
            "patch_layers": selection_cfg.get("patch_layers", []),
            "candidate_pool": [],
            "selection_trace": [],
            "budget_efficiency": [],
        }
    candidate_pool_top_m = int(selection_cfg.get("candidate_pool_top_m", len(restore_one_gain)))
    budget_candidates = [int(value) for value in selection_cfg.get("budget_candidates", [])]
    if not budget_candidates:
        budget_candidates = [int(selection_cfg.get("patch_budget_k", 0))]
    layer_cost = float(selection_cfg.get("layer_cost", 1.0))
    candidate_pool = [
        layer
        for layer, _ in sorted(
            restore_one_gain.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:candidate_pool_top_m]
    ]
    best_plan = None
    for budget in budget_candidates:
        selected: list[int] = []
        selection_trace: list[dict] = []
        total_gain = 0.0
        for step in range(min(budget, len(candidate_pool))):
            remaining = [layer for layer in candidate_pool if layer not in selected]
            subset_key = _subset_key(selected)
            step_gains = greedy_marginal_gain.get(subset_key, {})
            best_layer = max(
                remaining,
                key=lambda layer: step_gains.get(layer, restore_one_gain[layer]),
            )
            marginal_gain = step_gains.get(best_layer, restore_one_gain[best_layer])
            selected.append(best_layer)
            total_gain += marginal_gain
            selection_trace.append(
                {
                    "step": step + 1,
                    "selected_layer": best_layer,
                    "marginal_gain": marginal_gain,
                }
            )
        efficiency = total_gain / max(layer_cost * len(selected), 1e-9) if selected else 0.0
        plan = {
            "patch_layers": selected,
            "candidate_pool": candidate_pool,
            "selection_trace": selection_trace,
            "budget": len(selected),
            "total_gain": total_gain,
            "efficiency": efficiency,
        }
        if best_plan is None or plan["efficiency"] > best_plan["efficiency"]:
            best_plan = plan
    return {
        "patch_layers": best_plan["patch_layers"] if best_plan else [],
        "candidate_pool": candidate_pool,
        "selection_trace": best_plan["selection_trace"] if best_plan else [],
        "budget_efficiency": [
            {
                "budget": int(value),
                "efficiency": (
                    next(
                        (
                            item["marginal_gain"]
                            for item in (best_plan["selection_trace"] if best_plan else [])
                            if item["step"] == int(value)
                        ),
                        None,
                    )
                ),
            }
            for value in budget_candidates
        ],
    }


def _normalize_gap(gap: float, target: float | None) -> float:
    if target is None or target == 0:
        return 0.0
    return gap / abs(target)


def _ci_overlaps(limit: float, low: float | None, high: float | None) -> bool:
    if low is None or high is None:
        return False
    return low <= limit <= high


def _pass_label(run_summary: dict, claim_row: dict) -> tuple[str, bool]:
    ttft_limit = 1.10 * claim_row["ttft_ratio_target"]
    quality_limit = (
        claim_row["quality_retention_target"] - 0.02
        if claim_row["quality_retention_target"] is not None
        else None
    )
    kl_limit = (
        1.10 * claim_row["kl_ceiling_target"]
        if claim_row["kl_ceiling_target"] is not None
        else None
    )
    ppl_limit = (
        1.10 * claim_row["ppl_ratio_target"]
        if claim_row["ppl_ratio_target"] is not None
        else None
    )
    payload_limit = (
        1.10 * claim_row["payload_ratio_target_vs_raw_kv"]
        if claim_row["payload_ratio_target_vs_raw_kv"] is not None
        else None
    )

    failed_checks: list[str] = []
    ci_overlap = False

    if run_summary["mean_ttft_ratio_local"] > ttft_limit:
        failed_checks.append("ttft")
        ci_overlap |= _ci_overlaps(ttft_limit, run_summary["ttft_ci95_low_ratio"], run_summary["ttft_ci95_high_ratio"])
    if quality_limit is not None and run_summary["mean_quality_retention_local"] < quality_limit:
        failed_checks.append("quality")
        ci_overlap |= _ci_overlaps(quality_limit, run_summary["task_metric_ci95_low_ratio"], run_summary["task_metric_ci95_high_ratio"])
    if kl_limit is not None and run_summary["mean_token_kl"] > kl_limit:
        failed_checks.append("kl")
        ci_overlap |= _ci_overlaps(kl_limit, run_summary["kl_ci95_low"], run_summary["kl_ci95_high"])
    if ppl_limit is not None and run_summary["mean_ppl_ratio_local"] > ppl_limit:
        failed_checks.append("ppl")
        ci_overlap |= _ci_overlaps(ppl_limit, run_summary["ppl_ci95_low_ratio"], run_summary["ppl_ci95_high_ratio"])
    if payload_limit is not None and run_summary["mean_payload_ratio_vs_raw_kv_local"] > payload_limit:
        failed_checks.append("payload")

    if not failed_checks:
        return "strict_pass", False
    if len(failed_checks) == 1 and ci_overlap:
        return "soft_pass", True
    return "fail", ci_overlap


def aggregate_prompt_rows(prompt_rows: list[dict], *, claim_anchor: str) -> dict:
    ttft_values = [float(row["ttft_ms"]) for row in prompt_rows]
    end_to_end_values = [float(row["end_to_end_ms"]) for row in prompt_rows]
    task_values = [float(row["task_metric_value"]) for row in prompt_rows]
    kl_values = [float(row["token_kl"]) for row in prompt_rows]
    tv_values = [float(row["token_tv"]) for row in prompt_rows]
    ppl_values = [float(row["ppl"]) for row in prompt_rows]
    payload_values = [float(row["payload_bytes"]) for row in prompt_rows]
    native_ttft = [float(row["native_consumer_ttft_ms"]) for row in prompt_rows]
    native_quality = [float(row["native_consumer_task_metric_value"]) for row in prompt_rows]
    native_ppl = [float(row["native_consumer_ppl"]) for row in prompt_rows]
    raw_kv_bytes = [float(row["raw_kv_payload_bytes"]) for row in prompt_rows]

    ttft_mean = _mean(ttft_values)
    task_mean = _mean(task_values)
    kl_mean = _mean(kl_values)
    ppl_mean = _mean(ppl_values)
    payload_mean = _mean(payload_values)

    ttft_ci = bootstrap_mean_ci(ttft_values, seed=17)
    task_ci = bootstrap_mean_ci(task_values, seed=19)
    kl_ci = bootstrap_mean_ci(kl_values, seed=23)
    ppl_ci = bootstrap_mean_ci(ppl_values, seed=29)

    native_ttft_mean = _mean(native_ttft)
    native_quality_mean = _mean(native_quality)
    native_ppl_mean = _mean(native_ppl)
    raw_kv_mean = _mean(raw_kv_bytes)

    run_summary = {
        "run_id": prompt_rows[0]["run_id"],
        "pair_id": prompt_rows[0]["pair_id"],
        "pair_bucket": prompt_rows[0]["pair_bucket"],
        "claim_anchor": claim_anchor,
        "method": prompt_rows[0]["method"],
        "task": prompt_rows[0]["task"],
        "split": prompt_rows[0]["split"],
        "seed": prompt_rows[0]["seed"],
        "n_prompts": len(prompt_rows),
        "mean_ttft_ms": ttft_mean,
        "ttft_ci95_low": ttft_ci[0],
        "ttft_ci95_high": ttft_ci[1],
        "mean_end_to_end_ms": _mean(end_to_end_values),
        "mean_task_metric_value": task_mean,
        "task_metric_ci95_low": task_ci[0],
        "task_metric_ci95_high": task_ci[1],
        "mean_token_kl": kl_mean,
        "kl_ci95_low": kl_ci[0],
        "kl_ci95_high": kl_ci[1],
        "mean_token_tv": _mean(tv_values),
        "mean_ppl": ppl_mean,
        "ppl_ci95_low": ppl_ci[0],
        "ppl_ci95_high": ppl_ci[1],
        "mean_payload_bytes": payload_mean,
        "mean_payload_ratio_vs_raw_kv_local": payload_mean / raw_kv_mean if raw_kv_mean else None,
        "mean_ttft_ratio_local": ttft_mean / native_ttft_mean if native_ttft_mean else None,
        "mean_quality_retention_local": task_mean / native_quality_mean if native_quality_mean else None,
        "mean_ppl_ratio_local": ppl_mean / native_ppl_mean if native_ppl_mean else None,
        "ttft_ci95_low_ratio": ttft_ci[0] / native_ttft_mean if ttft_ci[0] is not None and native_ttft_mean else None,
        "ttft_ci95_high_ratio": ttft_ci[1] / native_ttft_mean if ttft_ci[1] is not None and native_ttft_mean else None,
        "task_metric_ci95_low_ratio": task_ci[0] / native_quality_mean if task_ci[0] is not None and native_quality_mean else None,
        "task_metric_ci95_high_ratio": task_ci[1] / native_quality_mean if task_ci[1] is not None and native_quality_mean else None,
        "ppl_ci95_low_ratio": ppl_ci[0] / native_ppl_mean if ppl_ci[0] is not None and native_ppl_mean else None,
        "ppl_ci95_high_ratio": ppl_ci[1] / native_ppl_mean if ppl_ci[1] is not None and native_ppl_mean else None,
    }
    return run_summary


def build_claim_gap_row(run_summary: dict, *, claim_anchor: str) -> dict:
    claim_row = CLAIM_ANCHORS[claim_anchor]
    ttft_limit = 1.10 * claim_row["ttft_ratio_target"]
    quality_limit = (
        claim_row["quality_retention_target"] - 0.02
        if claim_row["quality_retention_target"] is not None
        else None
    )
    kl_limit = (
        1.10 * claim_row["kl_ceiling_target"]
        if claim_row["kl_ceiling_target"] is not None
        else None
    )
    ppl_limit = (
        1.10 * claim_row["ppl_ratio_target"]
        if claim_row["ppl_ratio_target"] is not None
        else None
    )
    payload_limit = (
        1.10 * claim_row["payload_ratio_target_vs_raw_kv"]
        if claim_row["payload_ratio_target_vs_raw_kv"] is not None
        else None
    )

    gap_ttft = max(0.0, float(run_summary["mean_ttft_ratio_local"]) - ttft_limit)
    gap_quality = (
        max(0.0, quality_limit - float(run_summary["mean_quality_retention_local"]))
        if quality_limit is not None
        else 0.0
    )
    gap_kl = (
        max(0.0, float(run_summary["mean_token_kl"]) - kl_limit)
        if kl_limit is not None
        else 0.0
    )
    gap_ppl = (
        max(0.0, float(run_summary["mean_ppl_ratio_local"]) - ppl_limit)
        if ppl_limit is not None
        else 0.0
    )
    gap_payload = (
        max(0.0, float(run_summary["mean_payload_ratio_vs_raw_kv_local"]) - payload_limit)
        if payload_limit is not None
        else 0.0
    )
    gap_fidelity = max(gap_quality, gap_kl, gap_ppl)
    claim_gap_total = (
        0.45 * _normalize_gap(gap_ttft, claim_row["ttft_ratio_target"])
        + 0.45
        * _normalize_gap(
            gap_fidelity,
            max(
                value
                for value in [
                    claim_row["quality_retention_target"],
                    claim_row["kl_ceiling_target"],
                    claim_row["ppl_ratio_target"],
                    1.0,
                ]
                if value is not None
            ),
        )
        + 0.10 * _normalize_gap(gap_payload, claim_row["payload_ratio_target_vs_raw_kv"] or 1.0)
    )
    pass_label, ci_overlap = _pass_label(run_summary, claim_row)
    return {
        "pair_id": run_summary["pair_id"],
        "pair_bucket": run_summary["pair_bucket"],
        "claim_anchor": claim_anchor,
        "method": run_summary["method"],
        "task": run_summary["task"],
        "ttft_ratio_target": claim_row["ttft_ratio_target"],
        "quality_retention_target": claim_row["quality_retention_target"],
        "kl_ceiling_target": claim_row["kl_ceiling_target"],
        "ppl_ratio_target": claim_row["ppl_ratio_target"],
        "payload_ratio_target_vs_raw_kv": claim_row["payload_ratio_target_vs_raw_kv"],
        "ttft_ratio_local": run_summary["mean_ttft_ratio_local"],
        "quality_retention_local": run_summary["mean_quality_retention_local"],
        "kl_local": run_summary["mean_token_kl"],
        "ppl_ratio_local": run_summary["mean_ppl_ratio_local"],
        "payload_ratio_vs_raw_kv_local": run_summary["mean_payload_ratio_vs_raw_kv_local"],
        "gap_ttft": gap_ttft,
        "gap_quality": gap_quality,
        "gap_kl": gap_kl,
        "gap_ppl": gap_ppl,
        "gap_payload": gap_payload,
        "gap_fidelity": gap_fidelity,
        "claim_gap_total": claim_gap_total,
        "pass_label": pass_label,
        "ci_overlap_flag": ci_overlap,
    }


def build_latency_row(prompt_rows: list[dict]) -> dict:
    def mean_of(field: str) -> float | None:
        return _mean([float(row[field]) for row in prompt_rows])

    return {
        "run_id": prompt_rows[0]["run_id"],
        "method": prompt_rows[0]["method"],
        "producer_prefill_ms": mean_of("producer_prefill_ms"),
        "producer_encode_ms": mean_of("producer_encode_ms"),
        "transfer_ms": mean_of("transfer_ms"),
        "consumer_rebuild_ms": mean_of("consumer_rebuild_ms"),
        "consumer_patch_ms": mean_of("consumer_patch_ms"),
        "consumer_recompute_ms": mean_of("consumer_recompute_ms"),
        "first_token_decode_ms": mean_of("first_token_decode_ms"),
        "ttft_ms": mean_of("ttft_ms"),
        "end_to_end_ms": mean_of("end_to_end_ms"),
    }


def prepare_prompt_rows(config: dict, *, run_id: str, patch_layers: list[int]) -> list[dict]:
    prompt_rows: list[dict] = []
    for raw_row in config.get("prompt_rows", []):
        prompt_rows.append(
            {
                "run_id": run_id,
                "pair_id": config["pair_id"],
                "pair_bucket": config["pair_bucket"],
                "claim_anchor": config["claim_anchor"],
                "method": config.get("method", "scd_original"),
                "task": config["task"],
                "split": config["split"],
                "seed": int(config["seed"]),
                "prompt_id": raw_row["prompt_id"],
                "prefix_id": raw_row["prefix_id"],
                "context_len": int(config["context_len"]),
                "decode_len": int(config["decode_len"]),
                "bandwidth_gbps": float(config["bandwidth_gbps"]),
                "patch_budget_k": int(config["patch_budget_k"]),
                "patch_layers": patch_layers,
                "rank_k": int(config["rank_k"]),
                "rank_v": int(config["rank_v"]),
                "rank_h": int(config["rank_h"]),
                "payload_bytes": float(raw_row["payload_bytes"]),
                "producer_prefill_ms": float(raw_row["producer_prefill_ms"]),
                "producer_encode_ms": float(raw_row["producer_encode_ms"]),
                "transfer_ms": float(raw_row["transfer_ms"]),
                "consumer_rebuild_ms": float(raw_row["consumer_rebuild_ms"]),
                "consumer_patch_ms": float(raw_row["consumer_patch_ms"]),
                "consumer_recompute_ms": float(raw_row.get("consumer_recompute_ms", 0.0)),
                "first_token_decode_ms": float(raw_row["first_token_decode_ms"]),
                "ttft_ms": float(raw_row["ttft_ms"]),
                "end_to_end_ms": float(raw_row["end_to_end_ms"]),
                "task_metric_name": raw_row["task_metric_name"],
                "task_metric_value": float(raw_row["task_metric_value"]),
                "token_kl": float(raw_row["token_kl"]),
                "token_tv": float(raw_row["token_tv"]),
                "ppl": float(raw_row["ppl"]),
                "native_consumer_ttft_ms": float(raw_row["native_consumer_ttft_ms"]),
                "native_consumer_task_metric_value": float(raw_row["native_consumer_task_metric_value"]),
                "native_consumer_ppl": float(raw_row["native_consumer_ppl"]),
                "raw_kv_payload_bytes": float(raw_row["raw_kv_payload_bytes"]),
            }
        )
    return prompt_rows


def write_baseline_outputs(config: dict, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = config.get(
        "run_id",
        f"{config['pair_id']}__{config.get('method', 'scd_original')}__{config['task']}__seed{config['seed']}",
    )
    patch_selection = select_patch_layers(config.get("patch_selection", {}))
    patch_layers = patch_selection["patch_layers"]
    prompt_rows = prepare_prompt_rows(config, run_id=run_id, patch_layers=patch_layers)
    run_summary = aggregate_prompt_rows(prompt_rows, claim_anchor=config["claim_anchor"])
    claim_gap_row = build_claim_gap_row(run_summary, claim_anchor=config["claim_anchor"])
    latency_row = build_latency_row(prompt_rows)

    run_manifest = [
        {
            "run_id": run_id,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": config.get("git_commit", ""),
            "pair_id": config["pair_id"],
            "producer_model": config["producer_model"],
            "consumer_model": config["consumer_model"],
            "method": config.get("method", "scd_original"),
            "task": config["task"],
            "split": config["split"],
            "seed": int(config["seed"]),
            "context_len": int(config["context_len"]),
            "decode_len": int(config["decode_len"]),
            "bandwidth_gbps": float(config["bandwidth_gbps"]),
            "patch_budget_k": int(config["patch_budget_k"]),
            "patch_layers": patch_layers,
            "rank_k": int(config["rank_k"]),
            "rank_v": int(config["rank_v"]),
            "rank_h": int(config["rank_h"]),
            "claim_anchor": config["claim_anchor"],
            "status": "completed",
            "artifact_paths": {
                "prompt_level_metrics": str(output_dir / "prompt_level_metrics.csv"),
                "run_level_summary": str(output_dir / "run_level_summary.csv"),
                "claim_gap_summary": str(output_dir / "claim_gap_summary.csv"),
                "latency_breakdown_summary": str(output_dir / "latency_breakdown_summary.csv"),
            },
            "notes": config.get("notes", ""),
        }
    ]

    write_jsonl(output_dir / "run_manifest.jsonl", run_manifest)
    write_csv(output_dir / "prompt_level_metrics.csv", PROMPT_LEVEL_FIELDS, prompt_rows)
    write_csv(output_dir / "run_level_summary.csv", RUN_LEVEL_FIELDS, [run_summary])
    write_csv(output_dir / "claim_gap_summary.csv", CLAIM_GAP_FIELDS, [claim_gap_row])
    write_csv(output_dir / "latency_breakdown_summary.csv", LATENCY_FIELDS, [latency_row])
    write_csv(output_dir / "route_rescue_summary.csv", ROUTE_RESCUE_FIELDS, [])

    evaluation_readme = "\n".join(
        [
            "# Evaluation Readme",
            "",
            f"- mode: {config.get('mode', 'configured')}",
            f"- run_id: {run_id}",
            f"- pair_id: {config['pair_id']}",
            f"- pair_bucket: {config['pair_bucket']}",
            f"- method: {config.get('method', 'scd_original')}",
            f"- claim_anchor: {config['claim_anchor']}",
            f"- patch_layers: {patch_layers}",
            f"- selection_trace: {patch_selection['selection_trace']}",
            "- significance_note: single-seed outputs are descriptive rather than confirmatory.",
            "- route_rescue_summary.csv is intentionally empty until the original baseline fails and route A/B/C are run on the same matched pair.",
        ]
    )
    (output_dir / "evaluation_readme.md").write_text(evaluation_readme + "\n")
    return {
        "run_id": run_id,
        "patch_layers": patch_layers,
        "claim_gap_row": claim_gap_row,
        "run_summary": run_summary,
    }

