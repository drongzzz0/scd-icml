#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_RUNTIME_ROOT="${SCRIPT_DIR}/runtime_repo"
REMOTE_ROOT="${REMOTE_ROOT:-/disk_n/c101-3-1/projects/scd-repro-codex-mistral-instruct-20260323}"
STAMP="$(date +%Y%m%d-%H%M%S)"
RUN_ROOT="${RUN_ROOT:-${REMOTE_ROOT}/runs/mistral7_to_mistrallite_tunedfresh_${STAMP}}"

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

echo "Launching tuned fresh reverse rerun on ser"
ssh ser "cd '${REMOTE_ROOT}' && env \
  RUN_ROOT='${RUN_ROOT}' \
  CODE_VERSION_ID='${CODE_VERSION_ID}' \
  RESUME='${RESUME:-0}' \
  TRAIN_LIMIT='${TRAIN_LIMIT:-200}' \
  SEARCH_LIMIT='${SEARCH_LIMIT:-200}' \
  MAX_LENGTH='${MAX_LENGTH:-1024}' \
  MAX_NEW_TOKENS='${MAX_NEW_TOKENS:-32}' \
  BLOCK_SIZE='${BLOCK_SIZE:-1}' \
  LAYERS_PER_SHARD='${LAYERS_PER_SHARD:-4}' \
  REPLAY_TOP_K='${REPLAY_TOP_K:-5}' \
  REPLAY_SELECTION_MODE='${REPLAY_SELECTION_MODE:-quality_threshold_f1_first_ttft_second}' \
  REPLAY_DECISION_MAX_F1_DROP_RATIO='${REPLAY_DECISION_MAX_F1_DROP_RATIO:-0.05}' \
  REPLAY_DECISION_MAX_EM_DROP='${REPLAY_DECISION_MAX_EM_DROP:--1.0}' \
  MIN_TTFT_GAIN_MS='${MIN_TTFT_GAIN_MS:-0.0}' \
  MAX_ACCEPTED_BLOCKS='${MAX_ACCEPTED_BLOCKS:-0}' \
  PRODUCER_GPU='${PRODUCER_GPU:-}' \
  CONSUMER_GPU='${CONSUMER_GPU:-}' \
  TRAIN_GPU='${TRAIN_GPU:-}' \
  AUX_GPU='${AUX_GPU:-}' \
  WORKER_PAIRS='${WORKER_PAIRS:-}' \
  bash scripts/run_ser_mistral7_to_mistrallite_tuned_fresh.sh"
