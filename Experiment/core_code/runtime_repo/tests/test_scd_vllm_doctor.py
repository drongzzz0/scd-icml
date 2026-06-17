from __future__ import annotations

import json

from scd_vllm_plugin.cli import main as plugin_main
from scd_vllm_plugin.doctor import run_doctor


def _write_route(path, *, num_layers=4, patch=None, reuse=None, fallback=None, recompute=None) -> None:
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

