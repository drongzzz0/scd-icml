"""vLLM-facing SCD integration helpers.

This package is intentionally separate from ``scd_repro``.  The first
implementation phase provides audit, environment probing, KV spec validation,
bundle schemas, and metrics utilities.  Runtime vLLM KV injection should only
be added after a TrueKV oracle path is proven for the pinned vLLM version.
"""

__all__ = [
    "bundle",
    "env_probe",
    "kv_spec",
    "metrics",
    "repo_audit",
    "report",
]
