from __future__ import annotations

import numpy as np

from scd_vllm.bundle import make_manifest, write_manifest
from scd_vllm.injection import (
    align_to_vllm_block,
    build_injection_metrics,
    convert_bundle_to_example_storage,
    example_connector_folder_hash,
)


def test_align_to_vllm_block_matches_example_connector_rule() -> None:
    assert align_to_vllm_block(14, 16) == 0
    assert align_to_vllm_block(33, 16) == 32
    assert align_to_vllm_block(229, 16) == 224


def test_convert_bundle_to_example_storage_writes_vllm_layer_files(tmp_path) -> None:
    torch, save_file, safe_open = _tensor_runtime()
    bundle = tmp_path / "bundle"
    (bundle / "kv").mkdir(parents=True)
    token_ids = np.arange(33, dtype=np.int64).reshape(1, 33)
    np.save(bundle / "token_ids.npy", token_ids)
    (bundle / "prefix.txt").write_text("hello\n", encoding="utf-8")
    manifest = make_manifest(
        consumer_model="consumer",
        producer_model="producer",
        tokenizer_id="tok",
        token_ids=token_ids,
        bundle_type="full_consumer",
        dtype="float32",
        kv_spec={"num_layers": 2, "num_kv_heads": 2, "head_dim": 4},
    )
    write_manifest(bundle / "manifest.json", manifest)
    for layer in range(2):
        k = torch.arange(33 * 2 * 4, dtype=torch.float32).reshape(33, 2, 4) + layer
        v = k + 1000
        save_file({"k": k, "v": v}, str(bundle / "kv" / f"layer_{layer:03d}.safetensors"))

    storage = tmp_path / "storage"
    metrics = convert_bundle_to_example_storage(bundle_dir=bundle, storage_dir=storage, block_size=16)

    folder_hash = example_connector_folder_hash(token_ids, 32)
    assert metrics["aligned_tokens"] == 32
    assert metrics["folder_hash"] == folder_hash
    target = storage / folder_hash / "model.layers.0.self_attn.attn.safetensors"
    with safe_open(str(target), framework="pt", device="cpu") as f:
        kv_cache = f.get_tensor("kv_cache")
    assert tuple(kv_cache.shape) == (2, 32, 8)
    assert float(kv_cache[0, 0, 0]) == 0.0
    assert float(kv_cache[1, 0, 0]) == 1000.0


def test_build_injection_metrics_full_consumer_is_adapter_smoke_not_scd_pass() -> None:
    conversion = {
        "bundle_type": "full_consumer",
        "token_hash": "tok",
        "aligned_tokens": 32,
        "block_size": 16,
        "num_layers": 2,
        "payload_bytes": 1024,
    }
    record = _record("tok", [1, 2, 3], [4], [4, 5])

    metrics = build_injection_metrics(
        model="m",
        tokenizer="t",
        prompt="hello",
        conversion=conversion,
        records={"baseline": record, "load": record},
        max_new_tokens=2,
    )

    assert metrics["adapter_smoke_pass"] is True
    assert metrics["scd_injection_pass"] is False
    assert metrics["reconstructed_scd"] is False


def test_build_injection_metrics_reconstructed_bundle_can_pass() -> None:
    conversion = {
        "bundle_type": "reuse_patch",
        "token_hash": "tok",
        "aligned_tokens": 32,
        "block_size": 16,
        "num_layers": 2,
        "payload_bytes": 1024,
    }
    record = _record("tok", [1, 2, 3], [4], [4, 5])

    metrics = build_injection_metrics(
        model="m",
        tokenizer="t",
        prompt="hello",
        conversion=conversion,
        records={"baseline": record, "load": record},
        max_new_tokens=2,
    )

    assert metrics["adapter_smoke_pass"] is True
    assert metrics["scd_injection_pass"] is True
    assert metrics["reconstructed_scd"] is True


def test_build_injection_metrics_reconstructed_bundle_does_not_require_exact_greedy_match() -> None:
    conversion = {
        "bundle_type": "reuse_patch",
        "token_hash": "tok",
        "aligned_tokens": 32,
        "block_size": 16,
        "num_layers": 2,
        "payload_bytes": 1024,
    }
    baseline = _record("tok", [1, 2, 3], [4], [4, 5])
    loaded = _record("tok", [1, 2, 3], [7], [7, 8])

    metrics = build_injection_metrics(
        model="m",
        tokenizer="t",
        prompt="hello",
        conversion=conversion,
        records={"baseline": baseline, "load": loaded},
        max_new_tokens=2,
    )

    assert metrics["exact_match_required"] is False
    assert metrics["loaded_generation_ok"] is True
    assert metrics["greedy_1tok_match"] is False
    assert metrics["adapter_smoke_pass"] is True
    assert metrics["scd_injection_pass"] is True


def _record(token_hash: str, prompt_ids: list[int], one_ids: list[int], n_ids: list[int]) -> dict:
    return {
        "prompt_token_hash": token_hash,
        "prompt_token_ids": prompt_ids,
        "prompt_token_count": len(prompt_ids),
        "one_token": {"output_token_ids": one_ids},
        "n_token": {"output_token_ids": n_ids},
    }


def _tensor_runtime():
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    return torch, save_file, safe_open
