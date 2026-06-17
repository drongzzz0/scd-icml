#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"

args=(
  --dtype "${SCD_VLLM_DTYPE:-bfloat16}"
  --out "${SCD_VLLM_ENV_PROBE_OUT:-runs/scd_vllm/env_probe.json}"
  --report "${SCD_VLLM_STATUS_OUT:-reports/scd_vllm_status.md}"
)

if [[ -n "${SCD_VLLM_PRODUCER_MODEL:-}" ]]; then
  args+=(--producer-model "$SCD_VLLM_PRODUCER_MODEL")
fi
if [[ -n "${SCD_VLLM_CONSUMER_MODEL:-}" ]]; then
  args+=(--consumer-model "$SCD_VLLM_CONSUMER_MODEL")
fi
if [[ "${SCD_VLLM_SKIP_MODEL_CONFIG:-0}" == "1" ]]; then
  args+=(--skip-model-config)
fi

python -m scd_vllm.env_probe "${args[@]}"
