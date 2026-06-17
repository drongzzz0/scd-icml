#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"

python -m scd_vllm.report \
  --run-dir "${SCD_VLLM_RUN_DIR:-runs/scd_vllm}" \
  --repo-audit "${SCD_VLLM_REPO_AUDIT_OUT:-reports/scd_vllm_repo_audit.md}" \
  --env-probe "${SCD_VLLM_ENV_PROBE_OUT:-runs/scd_vllm/env_probe.json}" \
  --out "${SCD_VLLM_STATUS_OUT:-reports/scd_vllm_status.md}"
