from __future__ import annotations

from pathlib import Path

from scd_vllm.truekv_oracle import build_truekv_oracle_metrics, describe_class, extract_source_evidence


class _FakeConnector:
    def start_load_kv(self, forward_context, **kwargs):
        return None

    def wait_for_layer_load(self, layer_name: str) -> None:
        return None


def test_describe_class_records_available_method_signatures() -> None:
    desc = describe_class(_FakeConnector, ["start_load_kv", "wait_for_layer_load", "missing_method"])

    assert desc["available"] is True
    assert desc["methods"]["start_load_kv"]["available"] is True
    assert "forward_context" in desc["methods"]["start_load_kv"]["signature"]
    assert desc["methods"]["missing_method"]["available"] is False


def test_extract_source_evidence_finds_slot_mapping_lines() -> None:
    lines = [
        "def f():\n",
        "    slot_mapping = request.slot_mapping\n",
        "    dst_kv_cache_layer[slot_mapping] = src\n",
    ]

    evidence = extract_source_evidence(lines, ["slot_mapping", "dst_kv_cache_layer"], start_line=10)

    assert evidence == [
        {"line": 11, "text": "slot_mapping = request.slot_mapping"},
        {"line": 12, "text": "dst_kv_cache_layer[slot_mapping] = src"},
    ]


def test_build_truekv_oracle_metrics_passes_on_matching_records(tmp_path: Path) -> None:
    record = {
        "prompt_token_ids": [1, 2, 3],
        "prompt_token_count": 3,
        "prompt_token_hash": "abc",
        "storage": {"safetensor_files": 32},
        "one_token": {"output_token_ids": [4], "text": "x"},
        "n_token": {"output_token_ids": [4, 5], "text": "xy"},
    }

    metrics = build_truekv_oracle_metrics(
        model="m",
        tokenizer="t",
        prompt="hello",
        storage_dir=tmp_path,
        records={"baseline": record, "save": record, "load": record},
        max_new_tokens=2,
    )

    assert metrics["oracle_pass"] is True
    assert metrics["same_token_ids"] is True
    assert metrics["greedy_16tok_match"] is True


def test_build_truekv_oracle_metrics_fails_without_saved_files(tmp_path: Path) -> None:
    record = {
        "prompt_token_ids": [1, 2, 3],
        "prompt_token_count": 3,
        "prompt_token_hash": "abc",
        "storage": {"safetensor_files": 0},
        "one_token": {"output_token_ids": [4], "text": "x"},
        "n_token": {"output_token_ids": [4, 5], "text": "xy"},
    }

    metrics = build_truekv_oracle_metrics(
        model="m",
        tokenizer="t",
        prompt="hello",
        storage_dir=tmp_path,
        records={"baseline": record, "save": record, "load": record},
        max_new_tokens=2,
    )

    assert metrics["oracle_pass"] is False
