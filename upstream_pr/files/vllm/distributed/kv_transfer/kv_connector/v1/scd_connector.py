# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

import torch

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.logger import init_logger
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.outputs import KVConnectorOutput

if TYPE_CHECKING:
    from vllm.distributed.kv_events import KVCacheEvent
    from vllm.forward_context import ForwardContext
    from vllm.v1.attention.backend import AttentionMetadata
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_logger(__name__)


class SCDConnectorV1(KVConnectorBase_V1):
    """V1 KV Connector for Semantic Cache Distillation (SCD) cross-model KV transfer.

    SCD converts prefix activations/KV from a lightweight producer model into
    target-model paged KV cache blocks via hybrid layerwise Reuse + learned
    residual Patch projections, supporting file-store, preloaded, and
    same-worker GPU-resident transport modes without modifying target model
    weights or disabling CUDA graphs.

    Installation:
        Install the external ``scd-icml`` runtime package::

            pip install scd-icml

    Configuration:
        Pass ``kv_connector="SCDConnectorV1"`` via ``--kv-transfer-config``::

            --kv-transfer-config \\
            '{"kv_connector":"SCDConnectorV1","kv_role":"kv_consumer","kv_load_failure_policy":"fail","kv_connector_extra_config":{"mode":"file","store_root":"/path/to/store","scd_profile":{...}}}'
    """

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig | None" = None,
    ) -> None:
        super().__init__(
            vllm_config=vllm_config,
            role=role,
            kv_cache_config=kv_cache_config,
        )
        try:
            from scd_vllm.vllm_v1_adapter import SCDConnectorV1Impl
        except ImportError as e:
            raise ImportError(
                "scd-icml (`scd_vllm`) is not installed. Please install "
                "`scd-icml` to use SCDConnectorV1."
            ) from e

        self._scd_connector = SCDConnectorV1Impl(
            vllm_config,
            role,
            kv_cache_config,
            parent=self,
        )

    def shutdown(self) -> None:
        shutdown_fn = getattr(self._scd_connector, "shutdown", None)
        if callable(shutdown_fn):
            shutdown_fn()

    # ==============================
    # Worker-side methods
    # ==============================
    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        self._scd_connector.register_kv_caches(kv_caches)

    def bind_connector_metadata(self, connector_metadata: KVConnectorMetadata) -> None:
        super().bind_connector_metadata(connector_metadata)
        self._scd_connector.bind_connector_metadata(connector_metadata)

    def clear_connector_metadata(self) -> None:
        self._scd_connector.clear_connector_metadata()
        super().clear_connector_metadata()

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:
        self._scd_connector.start_load_kv(forward_context, **kwargs)

    def wait_for_layer_load(self, layer_name: str) -> None:
        self._scd_connector.wait_for_layer_load(layer_name)

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: "AttentionMetadata",
        **kwargs: Any,
    ) -> None:
        self._scd_connector.save_kv_layer(
            layer_name, kv_layer, attn_metadata, **kwargs
        )

    def wait_for_save(self) -> None:
        self._scd_connector.wait_for_save()

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[set[str] | None, set[str] | None]:
        return self._scd_connector.get_finished(finished_req_ids)

    def get_block_ids_with_load_errors(self) -> set[int]:
        method = getattr(self._scd_connector, "get_block_ids_with_load_errors", None)
        if callable(method):
            return method()
        return set()

    # ==============================
    # Scheduler-side methods
    # ==============================
    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        return self._scd_connector.get_num_new_matched_tokens(
            request, num_computed_tokens
        )

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ) -> None:
        self._scd_connector.update_state_after_alloc(
            request, blocks, num_external_tokens
        )

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        return self._scd_connector.build_connector_meta(scheduler_output)

    def update_connector_output(self, connector_output: KVConnectorOutput) -> None:
        method = getattr(self._scd_connector, "update_connector_output", None)
        if callable(method):
            method(connector_output)

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        return self._scd_connector.request_finished(request, block_ids)

    def take_events(self) -> Iterable["KVCacheEvent"]:
        method = getattr(self._scd_connector, "take_events", None)
        if callable(method):
            yield from method()
