from __future__ import annotations

from .common import read_jsonl
from .inventory import build_candidate_pairs, inventory_index, validate_inventory


PAIR_METRIC_REQUIRED_FIELDS = [
    "pair_id",
    "median_r2_h",
    "median_r2_kv",
    "median_cos_h",
    "normalized_kl_token",
    "oracle_gap_task",
]

PROMPT_MEASUREMENT_REQUIRED_FIELDS = [
    "prompt_id",
    "prefix_id",
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


def _require_fields(row: dict, required: list[str], *, where: str) -> None:
    missing = [field for field in required if field not in row]
    if missing:
        raise ValueError(f"missing fields in {where}: {missing}")


def load_pair_metric_rows(path) -> list[dict]:
    rows = read_jsonl(path)
    for idx, row in enumerate(rows):
        _require_fields(row, PAIR_METRIC_REQUIRED_FIELDS, where=f"pair_metrics[{idx}]")
    return rows


def load_prompt_measurement_rows(path) -> list[dict]:
    rows = read_jsonl(path)
    for idx, row in enumerate(rows):
        _require_fields(row, PROMPT_MEASUREMENT_REQUIRED_FIELDS, where=f"prompt_measurements[{idx}]")
    return rows


def build_pair_screening_config(
    inventory: dict,
    pair_metric_rows: list[dict],
    *,
    bucket_policy: dict | None = None,
    include_missing_pairs: bool = False,
) -> dict:
    validate_inventory(inventory)
    pair_metrics = {row["pair_id"]: row for row in pair_metric_rows}
    candidates = []
    for pair in build_candidate_pairs(inventory, inventory.get("pair_policy", {})):
        metric_row = pair_metrics.get(pair["pair_id"])
        if metric_row is None and not include_missing_pairs:
            continue
        metric_row = metric_row or {
            "median_r2_h": 0.0,
            "median_r2_kv": 0.0,
            "median_cos_h": 0.0,
            "normalized_kl_token": 1.0,
            "oracle_gap_task": 0.0,
            "notes": "missing measured pair metrics",
        }
        candidates.append(
            {
                "pair_id": pair["pair_id"],
                "producer_model": pair["producer_model"],
                "consumer_model": pair["consumer_model"],
                "tokenizer_shared": pair["tokenizer_shared"],
                "architecture_compatible": pair["architecture_compatible"],
                "metrics": {
                    "median_r2_h": metric_row["median_r2_h"],
                    "median_r2_kv": metric_row["median_r2_kv"],
                    "median_cos_h": metric_row["median_cos_h"],
                    "normalized_kl_token": metric_row["normalized_kl_token"],
                    "oracle_gap_task": metric_row["oracle_gap_task"],
                },
                "notes": " | ".join(
                    piece for piece in [pair.get("notes", ""), metric_row.get("notes", "")] if piece
                ),
            }
        )
    return {
        "project": inventory.get("project", "scd-similar-pair-validation"),
        "bucket_policy": bucket_policy or inventory.get("bucket_policy", {"close_top_k": 1, "far_bottom_k": 1}),
        "candidates": candidates,
    }


def build_original_scd_config(request: dict, prompt_rows: list[dict], inventory: dict | None = None) -> dict:
    inventory_idx = inventory_index(inventory) if inventory else {}
    producer_model = request["producer_model"]
    consumer_model = request["consumer_model"]
    if inventory_idx:
        if producer_model not in inventory_idx or consumer_model not in inventory_idx:
            raise KeyError("producer_model or consumer_model is not present in inventory")

    return {
        "mode": request.get("mode", "configured"),
        "pair_id": request["pair_id"],
        "pair_bucket": request.get("pair_bucket", "close"),
        "producer_model": producer_model,
        "consumer_model": consumer_model,
        "method": request.get("method", "scd_original"),
        "task": request["task"],
        "split": request["split"],
        "claim_anchor": request["claim_anchor"],
        "seed": request["seed"],
        "context_len": request["context_len"],
        "decode_len": request["decode_len"],
        "bandwidth_gbps": request["bandwidth_gbps"],
        "patch_budget_k": request["patch_budget_k"],
        "rank_k": request["rank_k"],
        "rank_v": request["rank_v"],
        "rank_h": request["rank_h"],
        "patch_selection": request.get("patch_selection", {}),
        "prompt_rows": prompt_rows,
        "notes": request.get("notes", ""),
    }

