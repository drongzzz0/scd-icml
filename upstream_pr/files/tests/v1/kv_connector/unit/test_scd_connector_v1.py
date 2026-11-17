# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the thin SCDConnectorV1 wrapper (runs without scd-icml installed)."""

from types import ModuleType, SimpleNamespace
import sys

import pytest
import torch

from vllm.config import KVTransferConfig, VllmConfig
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.distributed.kv_transfer.kv_connector.v1.scd_connector import (
    SCDConnectorV1,
)


def _make_vllm_config(extra: dict | None = None) -> VllmConfig:
    ktc = KVTransferConfig(
        kv_connector="SCDConnectorV1",
        kv_role="kv_consumer",
        kv_load_failure_policy="fail",
        kv_connector_extra_config=extra or {"mode": "file"},
    )
    return SimpleNamespace(kv_transfer_config=ktc)  # type: ignore[return-value]


def test_scd_connector_v1_is_registered_in_factory() -> None:
    cls = KVConnectorFactory.get_connector_class_by_name("SCDConnectorV1")
    assert cls is SCDConnectorV1


def test_scd_connector_v1_raises_helpful_import_error_when_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "scd_vllm", None)
    monkeypatch.setitem(sys.modules, "scd_vllm.vllm_v1_adapter", None)
    cfg = _make_vllm_config()
    with pytest.raises(ImportError, match="scd-icml"):
        SCDConnectorV1(cfg, KVConnectorRole.WORKER, kv_cache_config=None)


def test_scd_connector_v1_delegates_scheduler_and_worker_hooks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple] = []

    class FakeImpl:
        def __init__(self, vllm_config, role, kv_cache_config=None, parent=None):
            calls.append(("init", role, kv_cache_config, parent is not None))

        def register_kv_caches(self, kv_caches):
            calls.append(("register_kv_caches", list(kv_caches)))

        def bind_connector_metadata(self, meta):
            calls.append(("bind_connector_metadata", type(meta).__name__))

        def clear_connector_metadata(self):
            calls.append(("clear_connector_metadata",))

        def start_load_kv(self, forward_context, **kwargs):
            calls.append(("start_load_kv", forward_context))

        def wait_for_layer_load(self, layer_name: str):
            calls.append(("wait_for_layer_load", layer_name))

        def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs):
            calls.append(("save_kv_layer", layer_name))

        def wait_for_save(self):
            calls.append(("wait_for_save",))

        def get_finished(self, finished_req_ids):
            calls.append(("get_finished", set(finished_req_ids)))
            return None, {"req-1"}

        def get_num_new_matched_tokens(self, request, num_computed_tokens):
            calls.append(("get_num_new_matched_tokens", num_computed_tokens))
            return 32, False

        def update_state_after_alloc(self, request, blocks, num_external_tokens):
            calls.append(("update_state_after_alloc", num_external_tokens))

        def build_connector_meta(self, scheduler_output):
            calls.append(("build_connector_meta",))
            return KVConnectorMetadata()

        def request_finished(self, request, block_ids):
            calls.append(("request_finished", list(block_ids)))
            return False, None

        def shutdown(self):
            calls.append(("shutdown",))

    pkg = ModuleType("scd_vllm")
    pkg.__path__ = []  # type: ignore[attr-defined]
    adapter_mod = ModuleType("scd_vllm.vllm_v1_adapter")
    adapter_mod.SCDConnectorV1Impl = FakeImpl  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "scd_vllm", pkg)
    monkeypatch.setitem(sys.modules, "scd_vllm.vllm_v1_adapter", adapter_mod)

    cfg = _make_vllm_config({"mode": "resident"})
    conn = SCDConnectorV1(cfg, KVConnectorRole.WORKER, kv_cache_config=None)

    assert conn.get_num_new_matched_tokens(SimpleNamespace(), 0) == (32, False)
    conn.update_state_after_alloc(SimpleNamespace(), SimpleNamespace(), 32)
    meta = conn.build_connector_meta(SimpleNamespace())
    assert isinstance(meta, KVConnectorMetadata)

    conn.register_kv_caches({"layer.0": torch.empty(1)})
    conn.bind_connector_metadata(meta)
    conn.start_load_kv("ctx")
    conn.wait_for_layer_load("layer.0")
    conn.save_kv_layer("layer.0", torch.empty(1), SimpleNamespace())
    conn.wait_for_save()
    assert conn.get_finished({"req-1"}) == (None, {"req-1"})
    assert conn.request_finished(SimpleNamespace(), [1, 2]) == (False, None)
    conn.clear_connector_metadata()
    conn.shutdown()

    recorded_names = [entry[0] for entry in calls]
    assert recorded_names == [
        "init",
        "get_num_new_matched_tokens",
        "update_state_after_alloc",
        "build_connector_meta",
        "register_kv_caches",
        "bind_connector_metadata",
        "start_load_kv",
        "wait_for_layer_load",
        "save_kv_layer",
        "wait_for_save",
        "get_finished",
        "request_finished",
        "clear_connector_metadata",
        "shutdown",
    ]
