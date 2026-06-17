from __future__ import annotations

import json

import numpy as np

from scd_vllm.batch_benchmark import (
    _collect_preloaded_connector_stats,
    _effective_disable_hybrid_kv_cache_manager,
    build_batch_benchmark_metrics,
    build_kv_transfer_config,
    parse_args,
)
from scd_vllm.bundle import make_manifest, write_manifest


def test_collect_preloaded_connector_stats_aggregates_workers(tmp_path) -> None:
    stats_dir = tmp_path / "stats"
    stats_dir.mkdir()
    (stats_dir / "scd_preloaded_connector_11.json").write_text(
        json.dumps(
            {
                "cache_hits": 10,
                "cache_misses": 2,
                "cached_files": 2,
                "cached_bytes": 100,
                "final": True,
            }
        ),
        encoding="utf-8",
    )
    (stats_dir / "scd_preloaded_connector_12.json").write_text(
        json.dumps(
            {
                "cache_hits": 5,
                "cache_misses": 3,
                "cached_files": 3,
                "cached_bytes": 200,
                "final": False,
            }
        ),
        encoding="utf-8",
    )

    stats = _collect_preloaded_connector_stats(stats_dir)

    assert stats["exists"] is True
    assert stats["worker_count"] == 2
    assert stats["cache_hits"] == 15
    assert stats["cache_misses"] == 5
    assert stats["cache_events"] == 20
    assert stats["cache_hit_rate"] == 0.75
    assert stats["cached_files"] == 5
    assert stats["cached_bytes"] == 300
    assert stats["final_workers"] == 1
    assert stats["files"] == ["scd_preloaded_connector_11.json", "scd_preloaded_connector_12.json"]


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
        "preloaded_connector_stats_dir": str(tmp_path / "stats"),
        "preloaded_connector_stats": {
            "exists": True,
            "worker_count": 1,
            "cache_hits": 12,
            "cache_misses": 4,
            "cache_events": 16,
            "cache_hit_rate": 0.75,
            "cached_files": 4,
            "cached_bytes": 4096,
            "final_workers": 1,
            "files": ["scd_preloaded_connector_1.json"],
        },
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
    assert metrics["preloaded_connector_worker_count"] == 1
    assert metrics["preloaded_connector_cache_hits"] == 12
    assert metrics["preloaded_connector_cache_misses"] == 4
    assert metrics["preloaded_connector_cache_events"] == 16
    assert metrics["preloaded_connector_cache_hit_rate"] == 0.75
    assert metrics["preloaded_connector_cached_files"] == 4
    assert metrics["preloaded_connector_cached_bytes"] == 4096
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


def test_batch_benchmark_builds_external_connector_config() -> None:
    args = parse_args(
        [
            "--model",
            "m",
            "--bundle-dir",
            "bundle",
            "--storage-dir",
            "storage",
            "--out",
            "out",
            "--kv-connector",
            "SCDPreloadedConnector",
            "--kv-connector-module-path",
            "scd_vllm.preloaded_connector",
            "--connector-cache-device",
            "cuda",
        ]
    )

    config = build_kv_transfer_config(args)

    assert config["kv_connector"] == "SCDPreloadedConnector"
    assert config["kv_connector_module_path"] == "scd_vllm.preloaded_connector"
    assert config["kv_connector_extra_config"] == {
        "shared_storage_path": "storage",
        "cache_device": "cuda",
    }


def test_preloaded_connector_load_stage_disables_hybrid_kv_cache_manager() -> None:
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
            "--kv-connector",
            "SCDPreloadedConnector",
            "--kv-connector-module-path",
            "scd_vllm.preloaded_connector",
        ]
    )

    assert _effective_disable_hybrid_kv_cache_manager(args) is True
