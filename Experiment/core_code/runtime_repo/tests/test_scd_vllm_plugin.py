from __future__ import annotations

import json

from scd_vllm_plugin.cli import main
from scd_vllm_plugin.engine import SCDVLLMConfig, SCDVLLMEngine


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
