from __future__ import annotations

from scd_vllm.report import _injection_note, _injection_passed, _select_injection, _select_truekv_oracle, _truekv_oracle_note


def test_truekv_report_prefers_passing_oracle_over_inspection() -> None:
    inspection = {
        "path": "runs/scd_vllm/inspection",
        "mode": "truekv_oracle_inspection",
        "oracle_pass": False,
        "vllm_version": "0.18.0",
        "blocker": "inspection only",
    }
    oracle = {
        "path": "runs/scd_vllm/oracle",
        "mode": "truekv_oracle",
        "oracle_pass": True,
        "vllm_version": "0.18.0",
        "save_safetensor_files": 32,
        "load_safetensor_files": 32,
        "greedy_1tok_match": True,
        "greedy_16tok_match": True,
        "blocker": None,
    }

    selected = _select_truekv_oracle([oracle, inspection])
    note = _truekv_oracle_note([oracle, inspection])

    assert selected is oracle
    assert "runs/scd_vllm/oracle" in note
    assert "oracle_pass=True" in note
    assert "save_files=32" in note
    assert "greedy_16tok=True" in note


def test_injection_report_keeps_full_consumer_smoke_pending() -> None:
    smoke = {
        "path": "runs/scd_vllm/smoke",
        "bundle_type": "full_consumer",
        "adapter_smoke_pass": True,
        "scd_injection_pass": False,
        "aligned_tokens": 32,
        "num_layers": 2,
        "payload_bytes": 1024,
        "greedy_1tok_match": True,
        "greedy_ntok_match": True,
    }

    assert _injection_passed([smoke]) is False
    note = _injection_note([smoke])

    assert "bundle_type=full_consumer" in note
    assert "adapter_smoke_pass=True" in note
    assert "scd_injection_pass=False" in note


def test_injection_report_prefers_newest_passing_reconstructed_run() -> None:
    old = {
        "path": "runs/scd_vllm/old",
        "bundle_type": "reuse_patch",
        "adapter_smoke_pass": True,
        "scd_injection_pass": True,
        "reconstructed_scd": True,
        "mtime": 1.0,
    }
    new = {
        "path": "runs/scd_vllm/new",
        "bundle_type": "reuse_patch",
        "adapter_smoke_pass": True,
        "scd_injection_pass": True,
        "reconstructed_scd": True,
        "mtime": 2.0,
    }

    assert _select_injection([new, old]) is new
    assert _select_injection([old, new]) is new
