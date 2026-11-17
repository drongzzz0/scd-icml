from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from benchmark_a800_cmrc_raw_apc import paired_request_timings, retokenize_cases, validate_paired_rows


def test_timing_counts_prefill_once_in_each_arm_and_keeps_transfer_and_load_distinct():
    ms = 1_000_000
    times = paired_request_timings(
        12 * ms, 200 * ms, 200 * ms, 202 * ms, 205 * ms, 206 * ms,
        209 * ms, 234 * ms, 237 * ms,
    )
    assert times == {
        "prefill_ms": 12, "conversion_ms": 0, "transfer_ms": 2,
        "target_setup_ms": 3, "kv_load_ms": 1, "decode_ms": 28,
        "ttft_ms": 21, "completion_ms": 46, "cleanup_ms": 3,
        "request_wall_ms": 49,
    }
    with pytest.raises(ValueError, match="timestamps"):
        paired_request_timings(
            12 * ms, 200 * ms, 200 * ms, 202 * ms, 206 * ms, 205 * ms,
            209 * ms, 234 * ms, 237 * ms,
        )


def test_pairs_require_same_actual_source_and_no_missing_questions():
    row = {"qid": "qid-1", "prompt_tokens": 10, "loaded_tokens": 9,
           "source_cached_tokens": 0, "prefill_ms": 1.0}
    raw = {**row, "arm": "raw"}
    scd = {**row, "arm": "scd"}
    assert list(validate_paired_rows([scd, raw], {"qid-1"})) == ["qid-1"]
    with pytest.raises(ValueError, match="coverage"):
        validate_paired_rows([scd, raw], {"qid-1", "qid-2"})
    with pytest.raises(ValueError, match="source"):
        validate_paired_rows([scd, {**raw, "prefill_ms": 1.5}], {"qid-1"})
    with pytest.raises(ValueError, match="duplicate"):
        validate_paired_rows([scd, raw, raw], {"qid-1"})


@pytest.mark.parametrize("arm", ("raw", "scd"))
def test_single_arm_run_cannot_claim_paired_coverage(arm):
    row = {"qid": "qid-1", "arm": arm}
    assert validate_paired_rows([row], {"qid-1"}, arms=(arm,)) == {"qid-1": {arm: row}}
    with pytest.raises(ValueError, match="coverage"):
        validate_paired_rows([row], {"qid-1"})


def test_retokenize_cases_checks_alignment_and_rebinds_target_stops():
    case = {"qid": "qid-1", "prompt_text": "prompt", "generation": {"stop_token_ids": [2]}}
    source = SimpleNamespace(encode=lambda prompt, *, add_special_tokens: [4, 5, 6])
    target = SimpleNamespace(encode=lambda prompt, *, add_special_tokens: [4, 7, 6], all_special_ids=[9])
    with pytest.raises(ValueError, match="qid-1"):
        retokenize_cases([case], source, target)
    target.encode = source.encode
    retokenize_cases([case], source, target)
    assert case["prompt_text"] == "prompt"
    assert case["source_input_ids"] == case["prompt_ids"] == [4, 5, 6]
    assert case["prefix_ids"] == [4, 5]
    assert case["generation"]["stop_token_ids"] == [9]
