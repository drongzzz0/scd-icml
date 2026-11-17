"""SCD-ICML vLLM V1 integration and KVConnector plugin package.

Provides:
- ``scd_vllm.plugin:register``: automatic ``vllm.general_plugins`` registration
  for ``SCDConnectorV1``, ``SCDGPUConnectorV1``, and ``SCDPreloadedConnector``.
- ``scd_vllm.vllm_v1_adapter:SCDConnectorV1Impl``: unified thin-delegator target
  for upstream ``vllm.distributed.kv_transfer.kv_connector.v1.scd_connector``.
- ``scd_vllm.connector`` / ``scd_vllm.resident_connector``: lease-bound file and
  same-worker GPU KV connectors for cross-model KV transfer.
"""

__version__ = "1.0.0"

__all__ = [
    "bundle",
    "connector",
    "env_probe",
    "kv_spec",
    "layout",
    "metrics",
    "plugin",
    "preloaded_connector",
    "repo_audit",
    "report",
    "resident_connector",
    "resident_worker",
    "vllm_v1_adapter",
]
