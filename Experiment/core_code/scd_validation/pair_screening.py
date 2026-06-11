from __future__ import annotations

from pathlib import Path

from .common import write_csv


PAIR_SCREEN_FIELDS = [
    "pair_id",
    "producer_model",
    "consumer_model",
    "tokenizer_shared",
    "architecture_compatible",
    "median_r2_h",
    "median_r2_kv",
    "median_cos_h",
    "normalized_kl_token",
    "oracle_gap_task",
    "similarity_score",
    "pair_bucket",
    "selected_for_baseline",
    "notes",
]


def compute_similarity_score(metrics: dict) -> float:
    return (
        0.30 * float(metrics.get("median_r2_h", 0.0))
        + 0.30 * float(metrics.get("median_r2_kv", 0.0))
        + 0.20 * float(metrics.get("median_cos_h", 0.0))
        + 0.20 * (1.0 - float(metrics.get("normalized_kl_token", 1.0)))
    )


def assign_buckets(candidates: list[dict], close_top_k: int = 1, far_bottom_k: int = 1) -> list[dict]:
    ranked = sorted(candidates, key=lambda item: item["similarity_score"], reverse=True)
    n = len(ranked)
    if n == 1:
        ranked[0]["pair_bucket"] = "close"
        ranked[0]["selected_for_baseline"] = True
        return ranked
    if n == 2:
        ranked[0]["pair_bucket"] = "close"
        ranked[0]["selected_for_baseline"] = True
        ranked[1]["pair_bucket"] = "far"
        ranked[1]["selected_for_baseline"] = False
        return ranked
    for idx, row in enumerate(ranked):
        if idx < close_top_k:
            row["pair_bucket"] = "close"
            row["selected_for_baseline"] = True
        elif idx >= n - far_bottom_k:
            row["pair_bucket"] = "far"
            row["selected_for_baseline"] = False
        else:
            row["pair_bucket"] = "medium"
            row["selected_for_baseline"] = False
    return ranked


def build_pair_rows(config: dict) -> list[dict]:
    candidates: list[dict] = []
    for item in config.get("candidates", []):
        metrics = item.get("metrics", {})
        candidates.append(
            {
                "pair_id": item["pair_id"],
                "producer_model": item["producer_model"],
                "consumer_model": item["consumer_model"],
                "tokenizer_shared": bool(item.get("tokenizer_shared", True)),
                "architecture_compatible": bool(item.get("architecture_compatible", True)),
                "median_r2_h": float(metrics.get("median_r2_h", 0.0)),
                "median_r2_kv": float(metrics.get("median_r2_kv", 0.0)),
                "median_cos_h": float(metrics.get("median_cos_h", 0.0)),
                "normalized_kl_token": float(metrics.get("normalized_kl_token", 1.0)),
                "oracle_gap_task": float(metrics.get("oracle_gap_task", 0.0)),
                "similarity_score": compute_similarity_score(metrics),
                "notes": item.get("notes", ""),
            }
        )
    bucket_policy = config.get("bucket_policy", {})
    return assign_buckets(
        candidates,
        close_top_k=int(bucket_policy.get("close_top_k", 1)),
        far_bottom_k=int(bucket_policy.get("far_bottom_k", 1)),
    )


def write_pair_screen_summary(config: dict, output_path: Path) -> list[dict]:
    rows = build_pair_rows(config)
    write_csv(output_path, PAIR_SCREEN_FIELDS, rows)
    return rows

