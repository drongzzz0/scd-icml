"""Validate both full CMRC cohorts, score with the official evaluator, and summarize latency."""

import argparse
from collections import Counter
import json
import math
from pathlib import Path

from benchmark_a800_cmrc_raw_apc import ARMS, STAGE_KEYS, validate_paired_rows
from benchmark_a800_cmrc_quality import CMRC_DEV_SHA256, DEV26_PARSER_SHA256, read_json, write_json
import score_cmrc2018_official as official
from summarize_a800_cmrc_apc import STOP_REASONS, _latency_stats


TIMINGS = (*STAGE_KEYS, "transfer_gpu_ms", "kv_load_gpu_ms")
ADDITIVE_STAGES = ("prefill_ms", "conversion_ms", "transfer_ms", "target_setup_ms", "kv_load_ms", "decode_ms")
TOPOLOGY = "cross_process_same_gpu_cuda_ipc_explicit_d2d_copy"


def _check_attempt(row, geometry=(32, 8, 128)):
    layers, heads, head_dim = geometry
    if (row["stop_reason"] not in STOP_REASONS or not 1 <= len(row["token_ids"]) <= 32
            or row["loaded_layers"] != layers or row["loaded_tokens"] != row["prompt_tokens"] - 1
            or row["transfer_bytes"] != row["loaded_tokens"] * layers * heads * head_dim * 4
            or not 0 <= row["source_cached_tokens"] < row["prompt_tokens"]
            or row["source_cached_tokens"] % 16):
        raise ValueError(f"invalid generation, import or APC receipt for {row['qid']}")
    if any(not math.isfinite(row[k]) or row[k] < 0 for k in TIMINGS):
        raise ValueError(f"non-finite or negative timing for {row['qid']}")
    completion = math.fsum(row[stage] for stage in ADDITIVE_STAGES)
    if (not math.isclose(row["completion_ms"], completion, abs_tol=1e-6)
            or not math.isclose(row["request_wall_ms"], completion + row["cleanup_ms"], abs_tol=1e-6)
            or not completion - row["decode_ms"] <= row["ttft_ms"] <= row["completion_ms"]
            or (row["arm"] == "raw" and row["conversion_ms"] != 0)):
        raise ValueError(f"request boundary mismatch for {row['qid']}")


def collect(shards, qids, *, arms=ARMS):
    rows = []
    headers = []
    for path in shards:
        header = read_json(path / "summary.json")
        if (header["status"] != "completed" or header["smoke_qids"]
                or header["source_apc"] is not True or header["target_apc"] is not False
                or tuple(header.get("arms", ARMS)) != arms
                or header["dataset_sha256"] != CMRC_DEV_SHA256 or header["parser_sha256"] != DEV26_PARSER_SHA256
                or header["transport"] != TOPOLOGY
                or (path / "exit_code.txt").read_text(encoding="utf-8").strip() != "0"):
            raise ValueError(f"expected a successful full paired APC shard at {path}")
        with (path / "attempts.jsonl").open(encoding="utf-8") as stream:
            attempts = [json.loads(line, object_pairs_hook=official._unique_json_object) for line in stream]
        geometry = header.get("geometry", (32, 8, 128))
        if len(geometry) != 3 or any(type(value) is not int or value <= 0 for value in geometry):
            raise ValueError(f"invalid model KV geometry in {path}")
        for arm in arms:
            predicted = official._read_predictions(path / f"{arm}_predictions.json", qids)
            if (predicted != {r["qid"]: r["prediction"] for r in attempts if r["arm"] == arm}
                    or len(predicted) != header["pairs"]):
                raise ValueError(f"{arm} predictions or count differ from attempts in {path}")
        for row in attempts:
            _check_attempt(row, geometry)
        rows.extend(attempts)
        headers.append(header)
    if (sorted(h["shard_index"] for h in headers) != list(range(len(shards)))
            or any(h["num_shards"] != len(shards) for h in headers)
            or len({(h["profile_sha256"], h["vllm"], h.get("source_model"), h.get("target_model"),
                     tuple(h.get("geometry", (32, 8, 128))), h.get("retokenize", False))
                    for h in headers}) != 1):
        raise ValueError("incompatible, missing or duplicated shards")
    return validate_paired_rows(rows, qids, arms=arms), headers


def summarize_arm(rows):
    if not rows:
        return {"N": 0}
    return {
        "N": len(rows),
        "mean_output_tokens": sum(len(r["token_ids"]) for r in rows) / len(rows),
        "mean_payload_bytes": sum(r["transfer_bytes"] for r in rows) / len(rows),
        "stop_reasons": dict(Counter(r["stop_reason"] for r in rows)),
        "blank_predictions": sum(not r["prediction"] for r in rows),
        "total_request_seconds": math.fsum(r["request_wall_ms"] for r in rows) / 1000,
        **{k: _latency_stats([r[k] for r in rows]) for k in TIMINGS},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--scorer", type=Path, required=True)
    parser.add_argument("--shard", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--raw-only", action="store_const", dest="only_arm", const="raw")
    selection.add_argument("--scd-only", action="store_const", dest="only_arm", const="scd")
    args = parser.parse_args()
    dataset = json.loads(official._read_pinned(args.dataset, "dataset", official.DEV_SHA256))
    ids = [qa["id"] for article in dataset["data"] for para in article["paragraphs"] for qa in para["qas"]]
    qids = set(ids)
    if len(ids) != len(qids) or len(qids) != official.DEV_N:
        raise ValueError("expected all 3,219 unique official CMRC dev QIDs")
    arms = (args.only_arm,) if args.only_arm else ARMS
    pairs, headers = collect(args.shard, qids, arms=arms)
    args.out.mkdir(parents=True, exist_ok=True)
    predictions = {arm: {qid: pairs[qid][arm]["prediction"] for qid in ids} for arm in arms}
    for arm, values in predictions.items():
        write_json(args.out / f"{arm}_predictions.json", values)
    scores_path = args.out / "official_scores.json"
    evaluate, scorer = official._load_evaluator(args.scorer)
    scores = {
        "dataset": {"source": official.DEV_URL, "sha256": official.DEV_SHA256},
        "scorer": scorer,
        **{arm: official._score(evaluate, dataset, values) for arm, values in predictions.items()},
    }
    if any(scores[arm]["N"] != len(qids) or scores[arm]["SKIP"] for arm in arms):
        raise ValueError("official scorer did not cover the complete cohort")
    if arms == ARMS:
        scores["delta"] = {metric: scores["scd"][metric] - scores["raw"][metric] for metric in ("F1", "EM")}
    write_json(scores_path, scores)
    source = [pair[arms[0]] for pair in pairs.values()]
    hits = sum(r["source_cached_tokens"] > 0 for r in source)
    metrics = {
        "status": "completed",
        "dataset": "CMRC2018 official dev",
        "source_model": headers[0].get("source_model"),
        "target_model": headers[0].get("target_model"),
        "quality": scores,
        "source_apc": {
            "hits": hits, "misses": len(source) - hits,
            "token_hit_fraction": sum(r["source_cached_tokens"] for r in source) / sum(r["prompt_tokens"] for r in source),
            "hit_prefill_ms": _latency_stats([r["prefill_ms"] for r in source if r["source_cached_tokens"]]),
            "miss_prefill_ms": _latency_stats([r["prefill_ms"] for r in source if not r["source_cached_tokens"]]),
        },
        "timing": {arm: summarize_arm([pair[arm] for pair in pairs.values()]) for arm in arms},
        "shards": headers,
    }
    write_json(args.out / "metrics.json", metrics)


if __name__ == "__main__":
    main()
