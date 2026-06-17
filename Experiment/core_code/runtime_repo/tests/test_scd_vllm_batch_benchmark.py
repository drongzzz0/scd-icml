from __future__ import annotations

import json

import numpy as np

from scd_vllm.batch_benchmark import build_batch_benchmark_metrics, parse_args
from scd_vllm.bundle import make_manifest, write_manifest


def test_batch_benchmark_metrics_pass_for_loaded_repeated_prefix(tmp_path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    token_ids = np.asarray([[1, 2, 3, 4]], dtype=np.int64)
    manifest = make_manifest(
        consumer_model="consumer",
        producer_model="producer",
        tokenizer_id="tok",
        token_ids=token_ids,
        bundle_type="reuse_patch",
        dtype="float32",
        kv_spec={"num_layers": 2, "num_kv_heads": 2, "head_dim": 4},
        route_id="r",
        route={"name": "r"},
    )
    write_manifest(bundle / "manifest.json", manifest)
    storage = tmp_path / "storage"
    load = {
        "request_count": 8,
        "output_tokens": 64,
        "tokens_per_second": 320.0,
        "requests_per_second": 40.0,
        "prompt_token_hash": manifest.token_hash,
        "storage": {"exists": True, "safetensor_files": 2},
    }
    baseline = {
        "request_count": 8,
        "output_tokens": 64,
        "tokens_per_second": 160.0,
        "requests_per_second": 20.0,
        "prompt_token_hash": manifest.token_hash,
        "storage": {"exists": True, "safetensor_files": 2},
    }
    prefix_cache = {
        "request_count": 8,
        "output_tokens": 64,
        "tokens_per_second": 240.0,
        "requests_per_second": 30.0,
        "prompt_token_hash": manifest.token_hash,
        "storage": {"exists": True, "safetensor_files": 2},
    }

    metrics = build_batch_benchmark_metrics(
        model="m",
        tokenizer="t",
        bundle_dir=bundle,
        storage_dir=storage,
        records={"baseline": baseline, "prefix-cache": prefix_cache, "load": load},
        batch_size=4,
        num_batches=2,
        warmup_batches=1,
        max_new_tokens=8,
    )

    assert metrics["benchmark_pass"] is True
    assert metrics["bundle_type"] == "reuse_patch"
    assert metrics["bundle_token_match"] is True
    assert metrics["storage_ready"] is True
    assert metrics["prefix_cache_generation_ok"] is True
    assert metrics["speedup_tokens_per_second"] == 2.0
    assert metrics["prefix_cache_vs_baseline_tokens_per_second"] == 1.5
    assert metrics["load_vs_prefix_cache_tokens_per_second"] == 320.0 / 240.0
    assert "repeated-prefix trusted microbenchmark" in metrics["claim_boundary"]


def test_batch_benchmark_metrics_fail_on_token_hash_mismatch(tmp_path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    token_ids = np.asarray([[1, 2, 3, 4]], dtype=np.int64)
    manifest = make_manifest(
        consumer_model="consumer",
        producer_model="producer",
        tokenizer_id="tok",
        token_ids=token_ids,
        bundle_type="reuse_patch",
        dtype="float32",
        kv_spec={"num_layers": 2, "num_kv_heads": 2, "head_dim": 4},
    )
    write_manifest(bundle / "manifest.json", manifest)
    load = {
        "request_count": 8,
        "output_tokens": 64,
        "tokens_per_second": 320.0,
        "requests_per_second": 40.0,
        "prompt_token_hash": "wrong",
        "storage": {"exists": True, "safetensor_files": 2},
    }

    metrics = build_batch_benchmark_metrics(
        model="m",
        tokenizer="t",
        bundle_dir=bundle,
        storage_dir=tmp_path / "storage",
        records={"load": load},
        batch_size=4,
        num_batches=2,
        warmup_batches=0,
        max_new_tokens=8,
    )

    assert metrics["benchmark_pass"] is False
    assert metrics["bundle_token_match"] is False


def test_batch_benchmark_parse_args_keeps_load_stage_hidden() -> None:
    args = parse_args(
        [
            "--stage",
            "load",
            "--model",
            "m",
            "--bundle-dir",
            "bundle",
            "--storage-dir",
            "storage",
            "--out",
            "out.json",
            "--batch-size",
            "8",
        ]
    )

    assert args.stage == "load"
    assert args.batch_size == 8


def test_batch_benchmark_parse_args_supports_prefix_cache_baseline() -> None:
    args = parse_args(
        [
            "--stage",
            "prefix-cache",
            "--model",
            "m",
            "--bundle-dir",
            "bundle",
            "--storage-dir",
            "storage",
            "--out",
            "out.json",
            "--include-prefix-cache-baseline",
        ]
    )

    assert args.stage == "prefix-cache"
    assert args.include_prefix_cache_baseline is True
