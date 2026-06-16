#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export TRAIN_SAMPLES="${TRAIN_SAMPLES:-0}"
export EVAL_LIMIT="${EVAL_LIMIT:-0}"
export CHUNK_SAMPLES="${CHUNK_SAMPLES:-1}"
export MAX_CHUNK_TOKENS="${MAX_CHUNK_TOKENS:-4096}"

exec bash "${SCRIPT_DIR}/run_ser_qwen17_bf16_smoke_current.sh"
