#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"

python -m scd_vllm.repo_audit \
  --repo-root "$ROOT" \
  --out "${SCD_VLLM_REPO_AUDIT_OUT:-reports/scd_vllm_repo_audit.md}" \
  --json-out "${SCD_VLLM_REPO_AUDIT_JSON:-runs/scd_vllm/repo_audit.json}"
