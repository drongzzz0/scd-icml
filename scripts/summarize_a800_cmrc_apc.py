#!/usr/bin/env python3
"""Build paired official scores and latency summaries from complete CMRC APC shards."""

from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import redirect_stdout
import io
import json
import math
from pathlib import Path
import re
from tempfile import TemporaryDirectory

if __package__:
    from . import score_cmrc2018_official as official
else:
    import score_cmrc2018_official as official


ARMS = ("native", "scd")
REPLAY_DATASET_SHA256 = "5cfe4414c28a8ecbb51670f78c0dc7d1049f286c2d5769b52f1f94bcc0752cf1"
PARSER_SHA256 = "450a9a05618a0288de91397976c070f158d9475a89d84996e066a155d541e49c"
LATENCY_SCOPE = "outer_request_to_target_submit_first_token_stop_decision_and_return"
PHASES = ("pre_target_submit_ms", "target_to_first_token_ms", "decode_after_first_ms", "stop_to_return_ms")
TOTALS = ("ttft_ms", "completion_ms", "request_wall_ms")
STOP_REASONS = ("stop_token", "text_boundary", "context_span_boundary", "max_new_tokens")


def _validate_summary(summary):
    if summary.get("status") != "completed" or summary.get("native_apc") is not True or summary.get("smoke_qids"):
        raise ValueError("expected a completed full-cohort APC shard")
    arms = summary["arms"]
    if not isinstance(arms, list) or not arms or any(arm not in ARMS for arm in arms) or len(set(arms)) != len(arms):
        raise ValueError("invalid arms in summary.json")
    index, count, pairs = summary["shard_index"], summary["num_shards"], summary["pairs"]
    if any(type(x) is not int for x in (index, count, pairs)) or not 0 <= index < count <= official.DEV_N or not 0 < pairs <= official.DEV_N:
        raise ValueError("invalid shard index, shard count or question count")
    if summary["dataset_sha256"] != REPLAY_DATASET_SHA256 or summary["parser_sha256"] != PARSER_SHA256:
        raise ValueError("expected the verified CMRC replay dataset and answer parser")
    if not re.fullmatch(r"[a-f0-9]{64}", summary["profile_sha256"]):
        raise ValueError("missing or invalid profile SHA-256")
    if summary["latency_scope"] != LATENCY_SCOPE:
        raise ValueError("expected the paired request latency scope")
    for key in ("runtime", "vllm"):
        if not isinstance(summary[key], str) or not summary[key].strip():
            raise ValueError(f"missing {key} in summary.json")


def _validate_number(number, label):
    if type(number) not in (int, float) or not math.isfinite(number) or number < 0:
        raise ValueError(f"{label} must be finite and nonnegative")


def _validate_attempt(row):
    qid, arm = row["qid"], row["arm"]
    if row["stop_reason"] not in STOP_REASONS or not isinstance(row["prediction"], str):
        raise ValueError(f"missing completed prediction for {arm}/{qid}")
    tokens = row["token_ids"]
    if not isinstance(tokens, list) or not 1 <= len(tokens) <= 32 or any(type(token) is not int or token < 0 for token in tokens):
        raise ValueError(f"invalid generated tokens for {arm}/{qid}")
    prompts = row["prompt_tokens"]
    if type(prompts) is not int or prompts < 1:
        raise ValueError(f"invalid prompt token count for {arm}/{qid}")
    cached = row["native_cached_tokens"]
    if arm == "native" and (type(cached) is not int or not 0 <= cached <= prompts):
        raise ValueError(f"invalid native APC cached token count for {qid}")
    if arm == "scd" and cached is not None:
        raise ValueError(f"unexpected native cached tokens on SCD row {qid}")

    for label in PHASES + TOTALS:
        _validate_number(row[label], f"{arm}/{qid}/{label}")
    parts = [row[label] for label in PHASES]
    for length, label in enumerate(TOTALS, start=2):
        if not math.isclose(row[label], math.fsum(parts[:length]), rel_tol=1e-9, abs_tol=1e-6):
            raise ValueError(f"non-additive timing for {arm}/{qid}: {label}")

    stages = row.get("scd_stage_ms")
    if stages is not None:
        if arm != "scd" or not isinstance(stages, dict):
            raise ValueError(f"invalid SCD diagnostic stages for {arm}/{qid}")
        for label, value in stages.items():
            if type(value) in (int, float):
                _validate_number(value, f"{arm}/{qid}/scd_stage_ms/{label}")


def _read_shard(directory, qids):
    if (directory / "exit_code.txt").read_text(encoding="utf-8-sig").strip() != "0":
        raise ValueError("exit code differs from 0")
    summary = json.loads((directory / "summary.json").read_bytes(), object_pairs_hook=official._unique_json_object)
    _validate_summary(summary)
    rows = {arm: {} for arm in summary["arms"]}
    with (directory / "attempts.jsonl").open(encoding="utf-8-sig") as attempts:
        for line_no, line in enumerate(attempts, start=1):
            try:
                row = json.loads(line, object_pairs_hook=official._unique_json_object)
                qid, arm = row["qid"], row["arm"]
                if qid not in qids or arm not in rows:
                    raise ValueError(f"unknown question or undeclared arm: {qid!r}, {arm!r}")
                if qid in rows[arm]:
                    raise ValueError(f"duplicate attempt for {arm}/{qid}")
                _validate_attempt(row)
                rows[arm][qid] = row
            except (KeyError, TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"attempts.jsonl line {line_no}: {exc}") from exc
    for arm in ARMS:
        path = directory / f"{arm}_predictions.json"
        if arm in rows:
            predictions = official._read_predictions(path, qids)
            if predictions != {qid: row["prediction"] for qid, row in rows[arm].items()}:
                raise ValueError(f"prediction mismatch between {path.name} and attempts.jsonl")
            if len(predictions) != summary["pairs"]:
                raise ValueError(f"{arm} question count differs from summary.json")
        elif path.exists():
            raise ValueError(f"{path.name} belongs to an undeclared arm")
    if len(rows) == 2 and rows["native"].keys() != rows["scd"].keys():
        raise ValueError("paired shard arms contain different questions")
    return summary, rows


def _collect_shards(shards, qids):
    cohort = {arm: {} for arm in ARMS}
    layouts = {arm: {} for arm in ARMS}
    sources = []
    common = None
    for directory in shards:
        try:
            summary, rows = _read_shard(directory, qids)
            identity = {key: summary[key] for key in (
                "dataset_sha256", "parser_sha256", "profile_sha256", "runtime", "vllm", "latency_scope",
            )}
            if common is None:
                common = identity
            elif identity != common:
                different = [key for key in identity if identity[key] != common[key]]
                raise ValueError(f"shard provenance differs in {', '.join(different)}")

            index = summary["shard_index"]
            for arm, attempts in rows.items():
                if index in layouts[arm]:
                    raise ValueError(f"duplicate shard index for {arm}: {index}")
                duplicates = cohort[arm].keys() & attempts.keys()
                if duplicates:
                    raise ValueError(f"duplicate question for {arm}: {min(duplicates)}")
                layouts[arm][index] = summary["num_shards"]
                cohort[arm].update(attempts)
            sources.append({"path": str(directory.resolve()), **{
                key: summary[key] for key in ("arms", "shard_index", "num_shards", "pairs")
            }})
        except (OSError, AttributeError, KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{directory}: {exc}") from exc

    for arm, layout in layouts.items():
        totals = set(layout.values())
        if len(totals) != 1 or set(layout) != set(range(next(iter(totals)))):
            raise ValueError(f"missing shard or inconsistent shard count for {arm}")
        missing = qids - cohort[arm].keys()
        if missing:
            raise ValueError(f"{arm} is missing {len(missing)} official questions (first: {min(missing)})")
    for qid in qids:
        if cohort["native"][qid]["prompt_tokens"] != cohort["scd"][qid]["prompt_tokens"]:
            raise ValueError(f"prompt token count differs between arms for {qid}")

    provenance = {**common, "official_dev_sha256": official.DEV_SHA256, "shards": sources}
    return cohort, provenance


def _latency_stats(values):
    ordered = sorted(values)

    def percentile(p):
        index = (len(ordered) - 1) * p
        before = math.floor(index)
        after = math.ceil(index)
        return ordered[before] + (index - before) * (ordered[after] - ordered[before])

    return {"mean": math.fsum(ordered) / len(ordered), "p50": percentile(0.5), "p95": percentile(0.95)}


def _summarize_arm(rows):
    n = len(rows)
    if not n:
        return {"N": 0}
    summary = {
        "N": n,
        "mean_output_tokens": sum(len(row["token_ids"]) for row in rows) / n,
        **{label: _latency_stats([row[label] for row in rows]) for label in PHASES + TOTALS},
    }
    stages = defaultdict(list)
    for row in rows:
        for label, value in (row.get("scd_stage_ms") or {}).items():
            if type(value) in (int, float):
                stages[label].append(value)
    if stages:
        summary["scd_stage_ms"] = {label: {"N": len(values), **_latency_stats(values)} for label, values in stages.items()}
    return summary


def _summarize_latency(cohort, provenance, qids):
    native = cohort["native"]
    hits = [qid for qid in qids if native[qid]["native_cached_tokens"] > 0]
    misses = [qid for qid in qids if native[qid]["native_cached_tokens"] == 0]
    cached_tokens = sum(row["native_cached_tokens"] for row in native.values())
    prompt_tokens = sum(row["prompt_tokens"] for row in native.values())

    def paired(ids):
        return {"N": len(ids), **{arm: _summarize_arm([cohort[arm][qid] for qid in ids]) for arm in ARMS}}

    return {
        **paired(qids),
        "native_apc": {
            "hits": len(hits),
            "misses": len(misses),
            "hit_rate": len(hits) / len(qids),
            "mean_cached_tokens": cached_tokens / len(qids),
            "cached_token_fraction": cached_tokens / prompt_tokens,
            "hit": paired(hits),
            "miss": paired(misses),
        },
        "provenance": {"percentiles": "linear interpolation at (N-1)*p", **provenance},
    }


def _write_reports(args, latency, cohort):
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".cmrc-report-", dir=args.out_dir.resolve()) as workspace:
        temporary = Path(workspace)
        predictions = {arm: temporary / f"{arm}_predictions.json" for arm in ARMS}
        for arm, path in predictions.items():
            values = {qid: row["prediction"] for qid, row in cohort[arm].items()}
            path.write_text(json.dumps(values, ensure_ascii=False), encoding="utf-8")

        scores = temporary / "official_scores.json"
        with redirect_stdout(io.StringIO()):
            official.main([
                "--dataset", str(args.dataset), "--scorer", str(args.scorer), "--out", str(scores),
                "--native-predictions", str(predictions["native"]), "--scd-predictions", str(predictions["scd"]),
            ])
        latency_path = temporary / "latency_summary.json"
        latency_path.write_text(json.dumps(latency, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        for output in (scores, latency_path):
            output.replace(args.out_dir / output.name)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--scorer", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("shard_dir", type=Path, nargs="+")
    args = parser.parse_args(argv)
    try:
        dataset = json.loads(official._read_pinned(args.dataset, "dataset", official.DEV_SHA256))
        qids = [qa["id"] for article in dataset["data"] for para in article["paragraphs"] for qa in para["qas"]]
        if len(qids) != official.DEV_N or len(set(qids)) != official.DEV_N:
            raise ValueError(f"official dev must contain {official.DEV_N} unique questions")
        cohort, provenance = _collect_shards(args.shard_dir, set(qids))
        latency = _summarize_latency(cohort, provenance, qids)
        _write_reports(args, latency, cohort)
    except (OSError, KeyError, TypeError, ValueError, OverflowError) as exc:
        parser.error(str(exc))
    print(f"wrote official_scores.json and latency_summary.json for {len(qids)} paired questions in {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
