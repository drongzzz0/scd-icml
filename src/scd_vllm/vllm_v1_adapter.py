"""Upstream-compatible V1 adapter delegating ``SCDConnectorV1`` across SCD transport modes.

Follows the official vLLM third-party connector delegation convention used by
``LMCacheConnectorV1`` and ``FlexKVConnectorV1`` so that upstream vLLM only needs
a thin wrapper importing ``SCDConnectorV1Impl`` from ``scd_vllm.vllm_v1_adapter``,
and unmodified vLLM installs auto-register ``SCDConnectorV1`` via ``vllm.general_plugins``.
"""
from __future__ import annotations

from typing import Any

try:
    from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorBase_V1
except ImportError:
    KVConnectorBase_V1 = object  # type: ignore[assignment,misc]


def _resolve_mode(kv_transfer_config: Any) -> str:
    if kv_transfer_config is None:
        return 'file'
    getter = getattr(kv_transfer_config, 'get_from_extra_config', None)
    extra = getattr(kv_transfer_config, 'kv_connector_extra_config', None) or {}

    def get_val(key: str, default: Any = None) -> Any:
        if callable(getter):
            return getter(key, default)
        return extra.get(key, default)

    explicit = get_val('mode', None) or get_val('transport_mode', None)
    if explicit is not None:
        mode = str(explicit).strip().lower()
    elif get_val('resident_session', None):
        mode = 'resident'
    elif get_val('shared_storage_path', None):
        mode = 'preloaded'
    else:
        mode = 'file'

    aliases = {
        'file': 'file',
        'store': 'file',
        'resident': 'resident',
        'gpu': 'resident',
        'preloaded': 'preloaded',
    }
    if mode not in aliases:
        raise ValueError(f'Unsupported SCDConnectorV1 mode: {explicit or mode!r}')
    return aliases[mode]


class SCDConnectorV1Impl(KVConnectorBase_V1):  # type: ignore[misc]
    """Unified V1 connector delegator for file-store, GPU-resident, and preloaded SCD modes."""

    def __init__(
        self,
        vllm_config: Any,
        role: Any,
        kv_cache_config: Any = None,
        parent: Any = None,
    ) -> None:
        extra = getattr(vllm_config, 'kv_transfer_config', None)
        self.mode = _resolve_mode(extra)
        self._parent = parent

        if self.mode == 'resident':
            from .resident_connector import SCDGPUConnectorV1

            self._impl: Any = SCDGPUConnectorV1(vllm_config, role, kv_cache_config)
        elif self.mode == 'preloaded':
            from .preloaded_connector import SCDPreloadedConnector

            self._impl = SCDPreloadedConnector(vllm_config, role, kv_cache_config)
        else:
            from .connector import SCDConnectorV1

            self._impl = SCDConnectorV1(vllm_config, role, kv_cache_config)

    def register_kv_caches(self, kv_caches: Any) -> None:
        return self._impl.register_kv_caches(kv_caches)

    def bind_connector_metadata(self, connector_metadata: Any) -> None:
        if hasattr(self._impl, 'bind_connector_metadata'):
            self._impl.bind_connector_metadata(connector_metadata)

    def clear_connector_metadata(self) -> None:
        if hasattr(self._impl, 'clear_connector_metadata'):
            self._impl.clear_connector_metadata()

    def start_load_kv(self, forward_context: Any, **kwargs: Any) -> None:
        return self._impl.start_load_kv(forward_context, **kwargs)

    def wait_for_layer_load(self, layer_name: str) -> None:
        return self._impl.wait_for_layer_load(layer_name)

    def save_kv_layer(self, layer_name: str, kv_layer: Any, attn_metadata: Any, **kwargs: Any) -> None:
        return self._impl.save_kv_layer(layer_name, kv_layer, attn_metadata, **kwargs)

    def wait_for_save(self) -> None:
        return self._impl.wait_for_save()

    def get_finished(self, finished_req_ids: set[str]) -> tuple[set[str] | None, set[str] | None]:
        if hasattr(self._impl, 'get_finished'):
            return self._impl.get_finished(finished_req_ids)
        return None, None

    def get_num_new_matched_tokens(self, request: Any, num_computed_tokens: int) -> tuple[int | None, bool]:
        return self._impl.get_num_new_matched_tokens(request, num_computed_tokens)

    def update_state_after_alloc(self, request: Any, blocks: Any, num_external_tokens: int) -> None:
        return self._impl.update_state_after_alloc(request, blocks, num_external_tokens)

    def build_connector_meta(self, scheduler_output: Any) -> Any:
        return self._impl.build_connector_meta(scheduler_output)

    def request_finished(self, request: Any, block_ids: list[int]) -> tuple[bool, dict[str, Any] | None]:
        if hasattr(self._impl, 'request_finished'):
            return self._impl.request_finished(request, block_ids)
        return False, None

    def __getattr__(self, name: str) -> Any:
        return getattr(self._impl, name)
