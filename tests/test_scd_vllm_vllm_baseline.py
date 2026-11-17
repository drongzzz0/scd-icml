from __future__ import annotations

from scd_vllm.vllm_baseline import extract_request_output, object_to_plain, summarize_records


class _Metrics:
    def __init__(self) -> None:
        self.arrival_time = 10.0
        self.first_token_time = 10.25
        self.finished_time = 10.5


class _Completion:
    def __init__(self) -> None:
        self.token_ids = [7, 8, 9]
        self.finish_reason = "length"
        self.text = "abc"


class _RequestOutput:
    def __init__(self) -> None:
        self.outputs = [_Completion()]
        self.prompt_token_ids = [1, 2, 3]
        self.metrics = _Metrics()


def test_extract_request_output_reads_metrics_and_tokens() -> None:
    record = extract_request_output(_RequestOutput(), generation_seconds=0.75, index=2, kind="measure")

    assert record["index"] == 2
    assert record["output_token_count"] == 3
    assert record["ttft_seconds"] == 0.25
    assert record["prompt_token_ids"] == [1, 2, 3]
    assert record["text"] == "abc"


def test_summarize_records_computes_latency_stats() -> None:
    summary = summarize_records(
        [
            {"generation_seconds": 1.0, "output_token_count": 10, "ttft_seconds": 0.2},
            {"generation_seconds": 2.0, "output_token_count": 10, "ttft_seconds": 0.3},
        ],
        ttft_proxy_records=[
            {"generation_seconds": 0.4, "output_token_count": 1, "ttft_seconds": None},
            {"generation_seconds": 0.2, "output_token_count": 1, "ttft_seconds": None},
        ],
    )

    assert summary["generation_seconds_mean"] == 1.5
    assert summary["tokens_per_second_mean"] == 10.0 / 1.5
    assert summary["ttft_seconds_mean"] == 0.25
    assert summary["ttft_source"] == "vllm_request_metrics"
    assert summary["ttft_proxy_seconds_mean"] == 0.30000000000000004
    assert summary["ttft_proxy_source"] == "one_token_generate_wall_time"


def test_object_to_plain_uses_public_dict_fields() -> None:
    plain = object_to_plain(_Metrics())

    assert plain["arrival_time"] == 10.0
    assert plain["first_token_time"] == 10.25
