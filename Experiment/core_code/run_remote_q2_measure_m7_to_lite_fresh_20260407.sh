#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_RUNTIME_ROOT="${SCRIPT_DIR}/runtime_repo"
REMOTE_ROOT="${REMOTE_ROOT:-/disk_n/c101-3-1/projects/scd-repro-codex-mistral-instruct-20260323}"
STAMP="$(date +%Y%m%d-%H%M%S)"
RUN_ROOT="${RUN_ROOT:-${REMOTE_ROOT}/runs/scd-q2-m7_to_lite-fresh-${STAMP}}"

CODE_VERSION_ID="${CODE_VERSION_ID:-$(
  cd "${LOCAL_RUNTIME_ROOT}"
  find src configs scripts -type f | sort | xargs shasum -a 256 | shasum -a 256 | awk '{print $1}'
)}"

echo "Syncing runtime repo to ser:${REMOTE_ROOT}"
rsync -az "${LOCAL_RUNTIME_ROOT}/src/" "ser:${REMOTE_ROOT}/src/"
rsync -az "${LOCAL_RUNTIME_ROOT}/configs/" "ser:${REMOTE_ROOT}/configs/"
rsync -az "${LOCAL_RUNTIME_ROOT}/scripts/" "ser:${REMOTE_ROOT}/scripts/"
rsync -az "${LOCAL_RUNTIME_ROOT}/pyproject.toml" "ser:${REMOTE_ROOT}/pyproject.toml"
rsync -az "${LOCAL_RUNTIME_ROOT}/README.md" "ser:${REMOTE_ROOT}/README.md"

echo "Launching fresh Q2 rerun on ser"
ssh ser "cd '${REMOTE_ROOT}' && env \
  RUN_ROOT='${RUN_ROOT}' \
  CODE_VERSION_ID='${CODE_VERSION_ID}' \
  CALIB_SAMPLES='${CALIB_SAMPLES:-200}' \
  EVAL_SAMPLES='${EVAL_SAMPLES:-200}' \
  MAX_LENGTH='${MAX_LENGTH:-256}' \
  MAX_NEW_TOKENS='${MAX_NEW_TOKENS:-32}' \
  BANDWIDTH_GBPS='${BANDWIDTH_GBPS:-200}' \
  QUANT_GROUP_SIZE='${QUANT_GROUP_SIZE:-128}' \
  PRODUCER_GPU='${PRODUCER_GPU:-}' \
  CONSUMER_GPU='${CONSUMER_GPU:-}' \
  TRAIN_GPU='${TRAIN_GPU:-}' \
  bash scripts/run_ser_q2_mistral7_to_mistrallite_fresh.sh"
