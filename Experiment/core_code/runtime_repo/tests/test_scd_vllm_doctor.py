from __future__ import annotations

import json

from scd_vllm.bundle import SCHEMA_VERSION, stable_json_hash
from scd_vllm_plugin.cli import main as plugin_main
from scd_vllm_plugin.doctor import run_doctor


def _write_route(path, *, num_layers=4, patch=None, reuse=None, fallback=None, recompute=None) -> dict:
    payload = {
        "name": "test_route",
        "num_layers": num_layers,
        "patch_layers": [0, 2] if patch is None else patch,
        "reuse_layers": [1, 3] if reuse is None else reuse,
        "fallback_layers": [] if fallback is None else fallback,
        "selected_recompute_layers": [],
        "recompute_segment_length_by_layer": {} if recompute is None else recompute,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def _write_bundle(path, *, route_payload: dict, num_layers: int = 4) -> None:
    kv_dir = path / "kv"
    kv_dir.mkdir(parents=True)
    (path / "token_ids.npy").write_bytes(b"placeholder")
    (path / "prefix.txt").write_text("hello", encoding="utf-8")
    for layer in range(num_layers):
        (kv_dir / f"layer_{layer:03d}.safetensors").write_bytes(b"placeholder")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": "2026-06-18T00:00:00+00:00",
        "producer_model": "producer",
        "consumer_model": "consumer",
        "tokenizer_id": "consumer",
        "token_hash": "dummy",
        "bundle_type": "reuse_patch",
        "dtype": "bfloat16",
        "device_origin": "cuda:0",
        "num_tokens": 16,
        "kv_spec": {"num_layers": num_layers, "num_kv_heads": 8, "head_dim": 128},
        "route_id": "test_route",
        "route_hash": stable_json_hash(route_payload),
        "quality_gate": {"gate_name": "doctor_test", "thresholds": {}},
    }
    (path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_doctor_passes_for_strict_mixed_route(tmp_path) -> None:
    reuse = tmp_path / "reuse"
    patch = tmp_path / "patch"
    route = tmp_path / "route.json"
    reuse.mkdir()
    patch.mkdir()
    _write_route(route)

    metrics = run_doctor(
        reuse_ckpt=str(reuse),
        patch_ckpt=str(patch),
        route=str(route),
        skip_vllm_import=True,
    )

    assert metrics["pass"] is True
    assert metrics["failure_count"] == 0
    route_check = next(item for item in metrics["checks"] if item["name"] == "route_strictness")
    assert route_check["details"]["patch_count"] == 2
    assert route_check["details"]["reuse_count"] == 2


def test_doctor_rejects_fallback_and_missing_coverage(tmp_path) -> None:
    reuse = tmp_path / "reuse"
    patch = tmp_path / "patch"
    route = tmp_path / "route.json"
    reuse.mkdir()
    patch.mkdir()
    _write_route(route, patch=[0], reuse=[1], fallback=[2])

    metrics = run_doctor(
        reuse_ckpt=str(reuse),
        patch_ckpt=str(patch),
        route=str(route),
        skip_vllm_import=True,
    )

    assert metrics["pass"] is False
    route_check = next(item for item in metrics["checks"] if item["name"] == "route_strictness")
    assert "fallback_layers must be empty" in route_check["message"]
    assert "do not cover layers" in route_check["message"]


def test_doctor_cli_writes_metrics(tmp_path) -> None:
    reuse = tmp_path / "reuse"
    patch = tmp_path / "patch"
    route = tmp_path / "route.json"
    out = tmp_path / "doctor.json"
    reuse.mkdir()
    patch.mkdir()
    _write_route(route)

    rc = plugin_main(
        [
            "doctor",
            "--reuse-ckpt",
            str(reuse),
            "--patch-ckpt",
            str(patch),
            "--route",
            str(route),
            "--skip-vllm-import",
            "--out",
            str(out),
        ]
    )

    assert rc == 0
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert saved["mode"] == "scd_vllm_doctor"
    assert saved["pass"] is True


def test_doctor_accepts_existing_bundle_without_raw_banks(tmp_path) -> None:
    route = tmp_path / "route.json"
    bundle = tmp_path / "bundle"
    route_payload = _write_route(route)
    _write_bundle(bundle, route_payload=route_payload)

    metrics = run_doctor(
        bundle_dir=str(bundle),
        route=str(route),
        skip_vllm_import=True,
    )

    assert metrics["pass"] is True
    bundle_check = next(item for item in metrics["checks"] if item["name"] == "bundle_manifest")
    assert bundle_check["status"] == "pass"
    assert bundle_check["details"]["route_hash_match"] is True
    assert bundle_check["details"]["num_layers"] == 4


def test_doctor_rejects_bundle_route_hash_mismatch(tmp_path) -> None:
    route = tmp_path / "route.json"
    bundle = tmp_path / "bundle"
    _write_bundle(bundle, route_payload={"name": "different", "num_layers": 4})
    _write_route(route)

    metrics = run_doctor(
        bundle_dir=str(bundle),
        route=str(route),
        skip_vllm_import=True,
    )

    assert metrics["pass"] is False
    bundle_check = next(item for item in metrics["checks"] if item["name"] == "bundle_manifest")
    assert "route_hash does not match" in bundle_check["message"]
