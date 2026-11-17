"""vLLM general_plugins entry point for automatic SCD-ICML KVConnector registration."""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_REGISTERED = False

REGISTERED_CONNECTORS: tuple[tuple[str, str, str], ...] = (
    ('SCDConnectorV1', 'scd_vllm.vllm_v1_adapter', 'SCDConnectorV1Impl'),
    ('SCDGPUConnectorV1', 'scd_vllm.resident_connector', 'SCDGPUConnectorV1'),
    ('SCDPreloadedConnector', 'scd_vllm.preloaded_connector', 'SCDPreloadedConnector'),
)


def register() -> list[str]:
    """Idempotently register SCD-ICML V1 KV connectors in vLLM's KVConnectorFactory.

    Loaded automatically by vLLM across frontend, EngineCore, and Worker
    processes via ``[project.entry-points."vllm.general_plugins"]``.
    Safe to call multiple times or when vLLM is not installed.
    """
    global _REGISTERED
    if _REGISTERED:
        return []
    try:
        from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
    except ImportError:
        logger.debug('vLLM not installed; skipping SCD KVConnectorFactory registration.')
        return []

    added: list[str] = []
    registry = getattr(KVConnectorFactory, '_registry', {})
    for name, module_path, class_name in REGISTERED_CONNECTORS:
        if name not in registry:
            KVConnectorFactory.register_connector(name, module_path, class_name)
            added.append(name)
    _REGISTERED = True
    return added
