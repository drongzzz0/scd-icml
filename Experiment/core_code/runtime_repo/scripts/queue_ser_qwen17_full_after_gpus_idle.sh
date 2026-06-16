#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/disk_n/c101-3-mql/projects/SCD-ICML/repro_current_from_codex_20260607-154030/runtime_repo}"
GPU_A="${GPU_A:-2}"
GPU_B="${GPU_B:-3}"
MEM_THRESHOLD_MB="${MEM_THRESHOLD_MB:-1000}"
CHECK_INTERVAL_SEC="${CHECK_INTERVAL_SEC:-60}"

TRAIN_SAMPLES="${TRAIN_SAMPLES:-0}"
EVAL_LIMIT="${EVAL_LIMIT:-0}"
CHUNK_SAMPLES="${CHUNK_SAMPLES:-2}"
MAX_CHUNK_TOKENS="${MAX_CHUNK_TOKENS:-4096}"
PRODUCER_DEVICE="${PRODUCER_DEVICE:-cuda:${GPU_A}}"
CONSUMER_DEVICE="${CONSUMER_DEVICE:-cuda:${GPU_B}}"
TRAIN_DEVICE="${TRAIN_DEVICE:-cuda:${GPU_B}}"

gpu_mem_mb() {
  local gpu_id="$1"
  nvidia-smi -i "${gpu_id}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d " "
}

cd "${ROOT}"

while true; do
  mem_a="$(gpu_mem_mb "${GPU_A}")"
  mem_b="$(gpu_mem_mb "${GPU_B}")"
  echo "$(date +%F_%T) gpu${GPU_A}_mem=${mem_a} gpu${GPU_B}_mem=${mem_b}"
  if [ "${mem_a}" -lt "${MEM_THRESHOLD_MB}" ] && [ "${mem_b}" -lt "${MEM_THRESHOLD_MB}" ]; then
    TRAIN_SAMPLES="${TRAIN_SAMPLES}" \
      EVAL_LIMIT="${EVAL_LIMIT}" \
      CHUNK_SAMPLES="${CHUNK_SAMPLES}" \
      MAX_CHUNK_TOKENS="${MAX_CHUNK_TOKENS}" \
      PRODUCER_DEVICE="${PRODUCER_DEVICE}" \
      CONSUMER_DEVICE="${CONSUMER_DEVICE}" \
      TRAIN_DEVICE="${TRAIN_DEVICE}" \
      WAIT=0 \
      bash scripts/run_ser_qwen17_bf16_full_current.sh
    exit 0
  fi
  sleep "${CHECK_INTERVAL_SEC}"
done
