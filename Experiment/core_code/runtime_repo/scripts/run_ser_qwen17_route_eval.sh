#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATA_SOURCE_DIR="/disk_n/c101-3-mql/projects/scd_main/legacy/no4/data/cmrc2018"
VENV="/disk_n/c101-3-mql/venvs/A800"

SEED="${SEED:-42}"
MAX_LENGTH="${MAX_LENGTH:-1024}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"
SAMPLES="${SAMPLES:-200}"
BANDWIDTH_GBPS="${BANDWIDTH_GBPS:-200}"
PRODUCER_DEVICE="${PRODUCER_DEVICE:-cuda:0}"
CONSUMER_DEVICE="${CONSUMER_DEVICE:-cuda:1}"
ROUTE_PATH="${ROUTE_PATH:-}"
OUT_NAME="${OUT_NAME:-}"
EVAL_CONFIG="${EVAL_CONFIG:-configs/eval_reuse_fallback_main200.json}"

RUN_ROOT="${ROOT}/runs/qwen17_to_bf16_cmrc"
REUSE_DIR="${RUN_ROOT}/ckpt_reuse_seed${SEED}"
PATCH_DIR="${RUN_ROOT}/ckpt_patch_seed${SEED}"
TMP_DIR="${ROOT}/.tmp_eval_cfg_qwen17_bf16"
DATA_ROOT="${ROOT}/.tmp_data_root_qwen17_bf16"
LOCAL_CMRC_DIR="${DATA_ROOT}/cmrc2018"

if [ -z "${ROUTE_PATH}" ]; then
  echo "ROUTE_PATH is required" >&2
  exit 1
fi

if [ -z "${OUT_NAME}" ]; then
  echo "OUT_NAME is required" >&2
  exit 1
fi

OUT_DIR="${RUN_ROOT}/${OUT_NAME}"

cd "${ROOT}"
source "${VENV}/bin/activate"
export PYTHONPATH="${ROOT}/src"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

if [ -L "${LOCAL_CMRC_DIR}" ]; then
  rm "${LOCAL_CMRC_DIR}"
fi

mkdir -p "${TMP_DIR}" "${DATA_ROOT}" "${LOCAL_CMRC_DIR}" "${OUT_DIR}"
cp "${DATA_SOURCE_DIR}/cmrc2018_dev.json" "${LOCAL_CMRC_DIR}/cmrc2018_dev.json"
cp "${DATA_SOURCE_DIR}/cmrc2018_train.json" "${LOCAL_CMRC_DIR}/cmrc2018_train.json"
if [ -f "${DATA_SOURCE_DIR}/README.md" ]; then
  cp "${DATA_SOURCE_DIR}/README.md" "${LOCAL_CMRC_DIR}/README.md"
fi

python -m scd_repro.cli.eval_cmrc_strict \
  --models-config configs/models_ser_qwen17_bf16.yaml \
  --eval-config "${EVAL_CONFIG}" \
  --data-root "${DATA_ROOT}" \
  --route "${ROUTE_PATH}" \
  --reuse-ckpt "${REUSE_DIR}" \
  --patch-ckpt "${PATCH_DIR}" \
  --out "${OUT_DIR}" \
  --limit "${SAMPLES}" \
  --max-length "${MAX_LENGTH}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --bandwidth-gbps "${BANDWIDTH_GBPS}" \
  --quant-group-size 128 \
  --producer-device "${PRODUCER_DEVICE}" \
  --consumer-device "${CONSUMER_DEVICE}" \
  --dtype bfloat16 \
  --seed "${SEED}"
