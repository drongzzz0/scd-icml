#!/usr/bin/env bash
set -euo pipefail

ROOT="/disk_n/c101-3-mql/projects/scd-repro-codex-mistral-instruct-20260323"
VENV="/disk_n/c101-3-mql/venvs/A800"

SEED="${SEED:-42}"
TRAIN_DEVICE="${TRAIN_DEVICE:-cuda:1}"
MODELS_CONFIG="${MODELS_CONFIG:-configs/models_ser_qwen17_bf16.yaml}"
TRAIN_CFG="${TRAIN_CFG:-configs/train_reuse_qwen17_self_layerwise.yaml}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/runs/qwen17_to_bf16_cmrc_clean_orig_s42}"
PAIR_DIR="${PAIR_DIR:-${RUN_ROOT}/pairs_cmrc_seed${SEED}}"
REUSE_DIR="${REUSE_DIR:-${RUN_ROOT}/ckpt_reuse_layerwise_seed${SEED}}"
FORCE_RETRAIN="${FORCE_RETRAIN:-0}"

cd "${ROOT}"
source "${VENV}/bin/activate"
export PYTHONPATH="${ROOT}/src"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

if [ ! -d "${PAIR_DIR}" ]; then
  echo "MISSING_PAIR_DIR ${PAIR_DIR}" >&2
  exit 1
fi

if [ -d "${REUSE_DIR}" ] && [ "${FORCE_RETRAIN}" != "1" ]; then
  echo "EXISTS ${REUSE_DIR}"
  echo "Set FORCE_RETRAIN=1 to overwrite."
  exit 0
fi

rm -rf "${REUSE_DIR}"

python -m scd_repro.cli.train_reuse \
  --pairs "${PAIR_DIR}" \
  --models-config "${MODELS_CONFIG}" \
  --train-config "${TRAIN_CFG}" \
  --out "${REUSE_DIR}" \
  --seed "${SEED}" \
  --train-device "${TRAIN_DEVICE}"

echo "RUN_DONE ${REUSE_DIR}"
