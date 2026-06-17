from __future__ import annotations

import json

import numpy as np

from scd_vllm.bundle import make_manifest, write_manifest
from scd_vllm_plugin.cli import main
from scd_vllm_plugin.engine import SCDVLLMConfig, SCDVLLMEngine, build_preloaded_launch_spec


def test_plugin_engine_run_smoke_pipeline_delegates_steps(monkeypatch, tmp_path) -> None:
    calls: list[tuple[str, str]] = []

    def fake_export(**kwargs):
        calls.append(("export", str(kwargs["out_dir"])))
        return {"mode": "reuse_patch_reconstructed_bundle", "bundle_type": "reuse_patch"}

    def fake_smoke(args):
        calls.append(("smoke", str(args.bundle_dir)))
        return {"scd_injection_pass": True, "bundle_type": "reuse_patch"}

    monkeypatch.setattr("scd_vllm_plugin.engine.export_reconstructed_bundle", fake_export)
    monkeypatch.setattr("scd_vllm_plugin.engine.run_injection_smoke", fake_smoke)

    engine = SCDVLLMEngine(
        SCDVLLMConfig(
            producer_model="producer",
            consumer_model="consumer",
            reuse_ckpt="reuse",
            patch_ckpt="patch",
            route="route.json",
        )
    )
    metrics = engine.run_smoke_pipeline(out_dir=tmp_path / "run", prompt="hello")

    assert metrics["pass"] is True
    assert calls == [
        ("export", str(tmp_path / "run" / "reuse_patch_bundle")),
        ("smoke", str(tmp_path / "run" / "reuse_patch_bundle")),
    ]


def test_plugin_cli_run_writes_pipeline_metrics(monkeypatch, tmp_path, capsys) -> None:
    def fake_export(**kwargs):
        return {"mode": "reuse_patch_reconstructed_bundle", "out_dir": str(kwargs["out_dir"])}

    def fake_smoke(args):
        return {"scd_injection_pass": True, "out": str(args.out)}

    monkeypatch.setattr("scd_vllm_plugin.engine.export_reconstructed_bundle", fake_export)
    monkeypatch.setattr("scd_vllm_plugin.engine.run_injection_smoke", fake_smoke)
    out_dir = tmp_path / "plugin_run"

    rc = main(
        [
            "run",
            "--producer-model",
            "producer",
            "--consumer-model",
            "consumer",
            "--reuse-ckpt",
            "reuse",
            "--patch-ckpt",
            "patch",
            "--route",
            "route.json",
            "--prompt",
            "hello",
            "--out-dir",
            str(out_dir),
        ]
    )

    assert rc == 0
    saved = json.loads((out_dir / "plugin_metrics.json").read_text(encoding="utf-8"))
    assert saved["pass"] is True
    assert saved["bundle_dir"].endswith("reuse_patch_bundle")
    assert "scd_vllm_plugin_smoke_pipeline" in capsys.readouterr().out


def test_build_preloaded_launch_spec_for_official_vllm_kwargs(tmp_path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    manifest = make_manifest(
        consumer_model="consumer",
        producer_model="producer",
        tokenizer_id="tok",
        token_ids=np.asarray([[1, 2, 3, 4]], dtype=np.int64),
        bundle_type="reuse_patch",
        dtype="bfloat16",
        kv_spec={"num_layers": 2, "num_kv_heads": 2, "head_dim": 4},
        route_id="route-a",
        route={"name": "route-a"},
    )
    write_manifest(bundle / "manifest.json", manifest)
    storage = tmp_path / "storage" / "hash"
    storage.mkdir(parents=True)
    (storage / "model.layers.0.self_attn.attn.safetensors").write_bytes(b"x")

    payload = build_preloaded_launch_spec(
        model="consumer",
        tokenizer="tok",
        bundle_dir=bundle,
        storage_dir=tmp_path / "storage",
        max_model_len=1024,
        gpu_memory_utilization=0.5,
        stats_dir=tmp_path / "stats",
    )

    assert payload["launch_spec_pass"] is True
    assert payload["storage_ready"] is True
    assert payload["bundle"]["route_id"] == "route-a"
    assert payload["safety"] == {
        "multi_tenant_safe": False,
        "dynamic_request_routing": False,
        "trusted_bundle_required": True,
    }
    assert payload["environment"] == {"SCD_PRELOADED_STATS_DIR": str(tmp_path / "stats")}
    llm_kwargs = payload["llm_kwargs"]
    assert llm_kwargs["model"] == "consumer"
    assert llm_kwargs["tokenizer"] == "tok"
    assert llm_kwargs["max_model_len"] == 1024
    assert llm_kwargs["gpu_memory_utilization"] == 0.5
    assert llm_kwargs["disable_hybrid_kv_cache_manager"] is True
    assert llm_kwargs["kv_transfer_config"]["kv_connector"] == "SCDPreloadedConnector"
    assert llm_kwargs["kv_transfer_config"]["kv_connector_module_path"] == "scd_vllm.preloaded_connector"
    assert llm_kwargs["kv_transfer_config"]["kv_connector_extra_config"] == {
        "shared_storage_path": str(tmp_path / "storage"),
        "cache_device": "cuda",
    }


def test_plugin_cli_launch_spec_writes_json(tmp_path, capsys) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    manifest = make_manifest(
        consumer_model="consumer",
        producer_model="producer",
        tokenizer_id="tok",
        token_ids=np.asarray([[1, 2, 3, 4]], dtype=np.int64),
        bundle_type="reuse_patch",
        dtype="bfloat16",
        kv_spec={"num_layers": 2, "num_kv_heads": 2, "head_dim": 4},
    )
    write_manifest(bundle / "manifest.json", manifest)
    storage = tmp_path / "storage" / "hash"
    storage.mkdir(parents=True)
    (storage / "model.layers.0.self_attn.attn.safetensors").write_bytes(b"x")
    out = tmp_path / "launch_spec.json"

    rc = main(
        [
            "launch-spec",
            "--model",
            "consumer",
            "--bundle-dir",
            str(bundle),
            "--storage-dir",
            str(tmp_path / "storage"),
            "--out",
            str(out),
        ]
    )

    assert rc == 0
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert saved["mode"] == "scd_vllm_launch_spec"
    assert saved["llm_kwargs"]["disable_hybrid_kv_cache_manager"] is True
    assert "scd_vllm_launch_spec" in capsys.readouterr().out


def test_plugin_cli_delegates_existing_commands(monkeypatch) -> None:
    calls: list[tuple[str, list[str]]] = []

    monkeypatch.setattr("scd_vllm_plugin.cli.reconstructed_export.main", lambda argv: calls.append(("export", list(argv))) or 0)
    monkeypatch.setattr("scd_vllm_plugin.cli.injection.main", lambda argv: calls.append(("inject", list(argv))) or 0)
    monkeypatch.setattr("scd_vllm_plugin.cli.batch_benchmark.main", lambda argv: calls.append(("batch", list(argv))) or 0)

    assert main(["export-bundle", "--out", "bundle"]) == 0
    assert main(["convert-bundle", "--bundle-dir", "bundle"]) == 0
    assert main(["smoke-test", "--bundle-dir", "bundle"]) == 0
    assert main(["batch-benchmark", "--bundle-dir", "bundle"]) == 0

    assert calls == [
        ("export", ["--out", "bundle"]),
        ("inject", ["convert", "--bundle-dir", "bundle"]),
        ("inject", ["run", "--bundle-dir", "bundle"]),
        ("batch", ["--bundle-dir", "bundle"]),
    ]
