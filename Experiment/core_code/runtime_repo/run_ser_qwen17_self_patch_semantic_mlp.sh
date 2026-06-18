#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
DATA_SOURCE_DIR="/disk_n/c101-3-mql/projects/scd_main/legacy/no4/data/cmrc2018"
VENV="/disk_n/c101-3-mql/venvs/A800"

SEED="${SEED:-42}"
MAX_LENGTH="${MAX_LENGTH:-1024}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"
SAMPLES_MAIN="${SAMPLES_MAIN:-50}"
BANDWIDTH_GBPS="${BANDWIDTH_GBPS:-200}"
PRODUCER_DEVICE="${PRODUCER_DEVICE:-cuda:0}"
CONSUMER_DEVICE="${CONSUMER_DEVICE:-cuda:1}"
TRAIN_DEVICE="${TRAIN_DEVICE:-cuda:1}"
RUN_LAYER0_ONLY="${RUN_LAYER0_ONLY:-0}"
RUN_TAG="${RUN_TAG:-20260331}"

RUN_ROOT="${ROOT}/runs/qwen17_self_cmrc"
PAIR_DIR="${RUN_ROOT}/pairs_cmrc_layerwise_len1024_seed${SEED}"
REUSE_DIR="${RUN_ROOT}/ckpt_reuse_layerwise_seed${SEED}"
PROFILE_PATH="${RUN_ROOT}/profile_patchonly_all28_seed42.json"
ROUTE_ALL28="${RUN_ROOT}/route_patchonly_all28_seed42.json"
ROUTE_EXCL0="${RUN_ROOT}/route_patchonly_excl0_seed42.json"
ROUTE_LAYER0="${RUN_ROOT}/route_patchonly_layer0_seed42.json"
PATCH_DIR="${RUN_ROOT}/ckpt_patch_semantic_mlp_all28_len1024_seed${SEED}"
STRICT_OUT_ALL28="${RUN_ROOT}/strict_diag_self_patchonly_semantic_mlp_all28_len1024_${SAMPLES_MAIN}_${RUN_TAG}"
STRICT_OUT_EXCL0="${RUN_ROOT}/strict_diag_self_patchonly_semantic_mlp_excl0_len1024_${SAMPLES_MAIN}_${RUN_TAG}"
STRICT_OUT_LAYER0="${RUN_ROOT}/strict_diag_self_patchonly_semantic_mlp_layer0_len1024_${SAMPLES_MAIN}_${RUN_TAG}"

DATA_ROOT="${ROOT}/.tmp_data_root_qwen17_self_cmrc"
LOCAL_CMRC_DIR="${DATA_ROOT}/cmrc2018"
TRAIN_CFG="${ROOT}/configs/train_qwen17_self_patch_semantic_mlp.json"
EVAL_CFG_ALL28="${RUN_ROOT}/eval_patchonly_diag50.json"
EVAL_CFG_EXCL0="${RUN_ROOT}/eval_patchonly_excl0_diag50.json"
EVAL_CFG_LAYER0="${RUN_ROOT}/eval_patchonly_layer0_oracle_diag50.json"

cd "${ROOT}"
source "${VENV}/bin/activate"
export PYTHONPATH="${ROOT}/src"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

mkdir -p "${RUN_ROOT}" "${DATA_ROOT}" "${LOCAL_CMRC_DIR}" "${PATCH_DIR}" \
  "${STRICT_OUT_ALL28}" "${STRICT_OUT_EXCL0}"

if [ ! -d "${PAIR_DIR}" ]; then
  echo "MISSING_PAIR_DIR ${PAIR_DIR}" >&2
  exit 1
fi

if [ ! -d "${REUSE_DIR}" ]; then
  echo "MISSING_REUSE_DIR ${REUSE_DIR}" >&2
  exit 1
fi

cp "${DATA_SOURCE_DIR}/cmrc2018_dev.json" "${LOCAL_CMRC_DIR}/cmrc2018_dev.json"
cp "${DATA_SOURCE_DIR}/cmrc2018_train.json" "${LOCAL_CMRC_DIR}/cmrc2018_train.json"
if [ -f "${DATA_SOURCE_DIR}/README.md" ]; then
  cp "${DATA_SOURCE_DIR}/README.md" "${LOCAL_CMRC_DIR}/README.md"
fi

python -m scd_repro.cli.train_patch \
  --pairs "${PAIR_DIR}" \
  --profile "${PROFILE_PATH}" \
  --route "${ROUTE_ALL28}" \
  --models-config configs/models_ser_qwen17_self.yaml \
  --train-config "${TRAIN_CFG}" \
  --out "${PATCH_DIR}" \
  --seed "${SEED}" \
  --train-device "${TRAIN_DEVICE}"

python -m scd_repro.cli.eval_cmrc_strict \
  --models-config configs/models_ser_qwen17_self.yaml \
  --eval-config "${EVAL_CFG_ALL28}" \
  --data-root "${DATA_ROOT}" \
  --route "${ROUTE_ALL28}" \
  --reuse-ckpt "${REUSE_DIR}" \
  --patch-ckpt "${PATCH_DIR}" \
  --out "${STRICT_OUT_ALL28}" \
  --limit "${SAMPLES_MAIN}" \
  --max-length "${MAX_LENGTH}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --bandwidth-gbps "${BANDWIDTH_GBPS}" \
  --quant-group-size 128 \
  --producer-device "${PRODUCER_DEVICE}" \
  --consumer-device "${CONSUMER_DEVICE}" \
  --dtype bfloat16 \
  --seed "${SEED}"

python -m scd_repro.cli.eval_cmrc_strict \
  --models-config configs/models_ser_qwen17_self.yaml \
  --eval-config "${EVAL_CFG_EXCL0}" \
  --data-root "${DATA_ROOT}" \
  --route "${ROUTE_EXCL0}" \
  --reuse-ckpt "${REUSE_DIR}" \
  --patch-ckpt "${PATCH_DIR}" \
  --out "${STRICT_OUT_EXCL0}" \
  --limit "${SAMPLES_MAIN}" \
  --max-length "${MAX_LENGTH}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --bandwidth-gbps "${BANDWIDTH_GBPS}" \
  --quant-group-size 128 \
  --producer-device "${PRODUCER_DEVICE}" \
  --consumer-device "${CONSUMER_DEVICE}" \
  --dtype bfloat16 \
  --seed "${SEED}"

if [ "${RUN_LAYER0_ONLY}" = "1" ]; then
  mkdir -p "${STRICT_OUT_LAYER0}"
  python -m scd_repro.cli.eval_cmrc_strict \
    --models-config configs/models_ser_qwen17_self.yaml \
    --eval-config "${EVAL_CFG_LAYER0}" \
    --data-root "${DATA_ROOT}" \
    --route "${ROUTE_LAYER0}" \
    --reuse-ckpt "${REUSE_DIR}" \
    --patch-ckpt "${PATCH_DIR}" \
    --out "${STRICT_OUT_LAYER0}" \
    --limit "${SAMPLES_MAIN}" \
    --max-length "${MAX_LENGTH}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --bandwidth-gbps "${BANDWIDTH_GBPS}" \
    --quant-group-size 128 \
    --producer-device "${PRODUCER_DEVICE}" \
    --consumer-device "${CONSUMER_DEVICE}" \
    --dtype bfloat16 \
    --seed "${SEED}"
fi

echo "RUN_DONE ${PATCH_DIR}"
echo "RUN_DONE ${STRICT_OUT_ALL28}"
echo "RUN_DONE ${STRICT_OUT_EXCL0}"
if [ "${RUN_LAYER0_ONLY}" = "1" ]; then
  echo "RUN_DONE ${STRICT_OUT_LAYER0}"
fi
