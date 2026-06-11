#!/usr/bin/env bash
set -euo pipefail

ROOT="/disk_n/c101-3-1/projects/scd-repro-codex-mistral-instruct-20260323"
DATA_SOURCE_DIR="/disk_n/c101-3-1/projects/scd_main/legacy/no4/data/cmrc2018"
VENV="/disk_n/c101-3-1/venvs/A800"

SEED="${SEED:-42}"
MAX_LENGTH="${MAX_LENGTH:-1024}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"
SAMPLES_MAIN="${SAMPLES_MAIN:-200}"
BANDWIDTH_GBPS="${BANDWIDTH_GBPS:-200}"
PRODUCER_DEVICE="${PRODUCER_DEVICE:-cuda:3}"
CONSUMER_DEVICE="${CONSUMER_DEVICE:-cuda:1}"
TRAIN_DEVICE="${TRAIN_DEVICE:-cuda:1}"

RUN_ROOT="${ROOT}/runs/qwen17_to_bf16_cmrc_clean_orig_s42"
PAIR_DIR="${RUN_ROOT}/pairs_cmrc_seed${SEED}"
REUSE_DIR="${RUN_ROOT}/ckpt_reuse_seed${SEED}"
PROFILE_PATH="${RUN_ROOT}/profile_cmrc_seed${SEED}.json"
ROUTE_PATH="${RUN_ROOT}/route_cmrc_seed${SEED}.json"
PATCH_DIR="${RUN_ROOT}/ckpt_patch_routea_state_native_seed${SEED}"
STRICT_OUT="${RUN_ROOT}/strict_routea_state_native_main${SAMPLES_MAIN}"

TMP_DIR="${ROOT}/.tmp_eval_cfg_qwen17_bf16_clean_orig"
DATA_ROOT="${ROOT}/.tmp_data_root_qwen17_bf16_clean_orig"
LOCAL_CMRC_DIR="${DATA_ROOT}/cmrc2018"
TRAIN_CFG="${ROOT}/configs/train_qwen17_route_a_state_native.json"
EVAL_CFG="${ROOT}/configs/eval_qwen17_route_a_state_native.json"

cd "${ROOT}"
source "${VENV}/bin/activate"
export PYTHONPATH="${ROOT}/src"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

mkdir -p "${RUN_ROOT}" "${TMP_DIR}" "${DATA_ROOT}" "${LOCAL_CMRC_DIR}" "${PATCH_DIR}" "${STRICT_OUT}"
cp "${DATA_SOURCE_DIR}/cmrc2018_dev.json" "${LOCAL_CMRC_DIR}/cmrc2018_dev.json"
cp "${DATA_SOURCE_DIR}/cmrc2018_train.json" "${LOCAL_CMRC_DIR}/cmrc2018_train.json"
if [ -f "${DATA_SOURCE_DIR}/README.md" ]; then
  cp "${DATA_SOURCE_DIR}/README.md" "${LOCAL_CMRC_DIR}/README.md"
fi

python -m scd_repro.cli.train_patch \
  --pairs "${PAIR_DIR}" \
  --profile "${PROFILE_PATH}" \
  --route "${ROUTE_PATH}" \
  --models-config configs/models_ser_qwen17_bf16.yaml \
  --train-config "${TRAIN_CFG}" \
  --out "${PATCH_DIR}" \
  --seed "${SEED}" \
  --train-device "${TRAIN_DEVICE}"

python -m scd_repro.cli.eval_cmrc_strict \
  --models-config configs/models_ser_qwen17_bf16.yaml \
  --eval-config "${EVAL_CFG}" \
  --data-root "${DATA_ROOT}" \
  --route "${ROUTE_PATH}" \
  --reuse-ckpt "${REUSE_DIR}" \
  --patch-ckpt "${PATCH_DIR}" \
  --out "${STRICT_OUT}" \
  --limit "${SAMPLES_MAIN}" \
  --max-length "${MAX_LENGTH}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --bandwidth-gbps "${BANDWIDTH_GBPS}" \
  --quant-group-size 128 \
  --producer-device "${PRODUCER_DEVICE}" \
  --consumer-device "${CONSUMER_DEVICE}" \
  --dtype bfloat16 \
  --seed "${SEED}"

echo "RUN_DONE ${STRICT_OUT}"
