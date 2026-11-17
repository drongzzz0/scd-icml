from __future__ import annotations

import hashlib
import json

import pytest


@pytest.fixture
def apc_shards(tmp_path, monkeypatch):
    from scripts import score_cmrc2018_official, summarize_a800_cmrc_apc

    def write_json(path, contents):
        path.write_text(json.dumps(contents), encoding="utf-8")
        return path

    dataset = write_json(
        tmp_path / "dev.json",
        {"data": [{"paragraphs": [{"qas": [{"id": qid} for qid in ("a", "b", "c")]}]}]},
    )
    scorer = tmp_path / "scorer.py"
    scorer.write_text(
        """\
from types import SimpleNamespace

nltk = SimpleNamespace(__version__="fixture")

def evaluate(dataset, predictions):
    n = len(dataset["data"][0]["paragraphs"][0]["qas"])
    percent = 100 * sum(value == "answer" for value in predictions.values()) / n
    return percent, percent, n, n - len(predictions)
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(score_cmrc2018_official, "DEV_N", 3)
    monkeypatch.setattr(score_cmrc2018_official, "DEV_SHA256", hashlib.sha256(dataset.read_bytes()).hexdigest())
    monkeypatch.setattr(score_cmrc2018_official, "SCORER_SHA256", hashlib.sha256(scorer.read_bytes()).hexdigest())

    def attempt(qid, arm, first, cached):
        row = {
            "qid": qid,
            "arm": arm,
            "prediction": "answer" if arm == "scd" or qid != "c" else "different",
            "stop_reason": "stop_token",
            "token_ids": [7, 9],
            "prompt_tokens": 128,
            "native_cached_tokens": cached,
            "pre_target_submit_ms": 0.5,
            "target_to_first_token_ms": first,
            "decode_after_first_ms": 2.0,
            "stop_to_return_ms": 1.0,
            "ttft_ms": first + 0.5,
            "completion_ms": first + 2.5,
            "request_wall_ms": first + 3.5,
        }
        if arm == "scd":
            row["scd_stage_ms"] = {"source_prefill_ms": 0.2, "tag": "diagnostic"}
        return row

    def shard(name, index, rows):
        directory = tmp_path / name
        directory.mkdir()
        arms = {row["arm"] for row in rows}
        (directory / "exit_code.txt").write_text("0\n", encoding="utf-8")
        (directory / "attempts.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        write_json(directory / "summary.json", {
            "status": "completed", "pairs": len({row["qid"] for row in rows}),
            "shard_index": index, "num_shards": 2, "smoke_qids": None,
            "dataset_sha256": "5cfe4414c28a8ecbb51670f78c0dc7d1049f286c2d5769b52f1f94bcc0752cf1",
            "profile_sha256": "a" * 64,
            "parser_sha256": "450a9a05618a0288de91397976c070f158d9475a89d84996e066a155d541e49c",
            "runtime": "fixture-runtime", "vllm": "fixture-vllm", "arms": sorted(arms), "native_apc": True,
            "latency_scope": "outer_request_to_target_submit_first_token_stop_decision_and_return",
        })
        for arm in arms:
            write_json(directory / f"{arm}_predictions.json", {
                row["qid"]: row["prediction"] for row in rows if row["arm"] == arm
            })
        return directory

    both = shard("both", 0, [
        attempt("a", "native", 1.0, 0), attempt("b", "native", 3.0, 32),
        attempt("a", "scd", 2.0, None), attempt("b", "scd", 6.0, None),
    ])
    native = shard("native", 1, [attempt("c", "native", 5.0, 64)])
    scd = shard("scd", 1, [attempt("c", "scd", 10.0, None)])
    out = tmp_path / "out"
    argv = ["--dataset", str(dataset), "--scorer", str(scorer), "--out-dir", str(out), str(both), str(native), str(scd)]
    return summarize_a800_cmrc_apc, argv, out, both, native, scd


def test_summary_uses_complete_cohort_and_pairs_apc_groups(apc_shards):
    script, argv, out, _, _, _ = apc_shards
    assert script.main(argv) == 0

    assert {path.name for path in out.iterdir()} == {"official_scores.json", "latency_summary.json"}
    latency = json.loads((out / "latency_summary.json").read_bytes())
    quality = json.loads((out / "official_scores.json").read_bytes())
    assert quality["scd"] == {"F1": 100.0, "EM": 100.0, "N": 3, "SKIP": 0}
    assert quality["native"]["F1"] == 200 / 3
    assert latency["N"] == latency["native"]["N"] == latency["scd"]["N"] == 3
    assert latency["native"]["mean_output_tokens"] == 2
    assert latency["native"]["target_to_first_token_ms"] == {"mean": 3.0, "p50": 3.0, "p95": 4.8}
    assert latency["native"]["request_wall_ms"]["mean"] == 6.5
    assert latency["scd"]["scd_stage_ms"]["source_prefill_ms"]["N"] == 3
    apc = latency["native_apc"]
    assert apc["hits"] == 2
    assert apc["misses"] == 1
    assert apc["cached_token_fraction"] == 0.25
    assert apc["hit"]["scd"]["target_to_first_token_ms"]["mean"] == 8.0
    assert apc["miss"]["scd"]["target_to_first_token_ms"]["mean"] == 2.0


@pytest.mark.parametrize("invalid", ["exit code", "prediction mismatch", "non-additive timing", "missing shard"])
def test_summary_refuses_incorrect_or_incomplete_runs(apc_shards, capsys, invalid):
    script, argv, out, _, native, scd = apc_shards
    if invalid == "exit code":
        (scd / "exit_code.txt").write_text("1\n", encoding="utf-8")
    elif invalid == "prediction mismatch":
        (native / "native_predictions.json").write_text('{"c": "other text"}', encoding="utf-8")
    elif invalid == "non-additive timing":
        path = scd / "attempts.jsonl"
        row = json.loads(path.read_bytes())
        row["decode_after_first_ms"] = 10
        path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    else:
        argv.pop()

    with pytest.raises(SystemExit, match="2"):
        script.main(argv)

    assert invalid in capsys.readouterr().err
    assert not out.exists()
