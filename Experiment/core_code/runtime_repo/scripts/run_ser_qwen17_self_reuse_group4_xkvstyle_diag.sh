#!/usr/bin/env bash
set -euo pipefail

ROOT="/disk_n/c101-3-1/projects/scd-repro-codex-mistral-instruct-20260323"
DATA_SOURCE_DIR="/disk_n/c101-3-1/projects/scd_main/legacy/no4/data/cmrc2018"
VENV="/disk_n/c101-3-1/venvs/A800"

SEED="${SEED:-42}"
MAX_LENGTH="${MAX_LENGTH:-1024}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"
SAMPLES="${SAMPLES:-20}"
BANDWIDTH_GBPS="${BANDWIDTH_GBPS:-200}"
PRODUCER_DEVICE="${PRODUCER_DEVICE:-cuda:0}"
CONSUMER_DEVICE="${CONSUMER_DEVICE:-cuda:1}"
TRAIN_DEVICE="${TRAIN_DEVICE:-cuda:1}"
MODELS_CONFIG="${MODELS_CONFIG:-configs/models_ser_qwen17_self.yaml}"
TRAIN_CONFIG="${TRAIN_CONFIG:-configs/train_reuse_qwen17_self_group4_xkvstyle.yaml}"
ROUTE_PATH="${ROUTE_PATH:-configs/routes/ser_qwen17_self_all_reuse.json}"
EVAL_CONFIG="${EVAL_CONFIG:-configs/eval_reuse_fallback_diag50.json}"
OUT_NAME="${OUT_NAME:-strict_diag_self_all_reuse_group4_xkvstyle_20}"

RUN_ROOT="${ROOT}/runs/qwen17_self_cmrc"
PAIR_DIR="${RUN_ROOT}/pairs_cmrc_group4_xkvstyle_seed${SEED}"
REUSE_DIR="${RUN_ROOT}/ckpt_reuse_group4_xkvstyle_seed${SEED}"
DUMMY_PATCH_DIR="${ROOT}/runs/qwen17_to_bf16_cmrc/ckpt_patch_seed${SEED}"
TMP_DIR="${ROOT}/.tmp_eval_cfg_qwen17_self_group4_xkvstyle"
DATA_ROOT="${ROOT}/.tmp_data_root_qwen17_self_group4_xkvstyle"
LOCAL_CMRC_DIR="${DATA_ROOT}/cmrc2018"
PROMPTS_JSON="${TMP_DIR}/cmrc_prompts_${SAMPLES}.json"
STRICT_OUT="${RUN_ROOT}/${OUT_NAME}"

cd "${ROOT}"
source "${VENV}/bin/activate"
export PYTHONPATH="${ROOT}/src"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

if [ -L "${LOCAL_CMRC_DIR}" ]; then
  rm "${LOCAL_CMRC_DIR}"
fi

mkdir -p "${RUN_ROOT}" "${TMP_DIR}" "${DATA_ROOT}" "${LOCAL_CMRC_DIR}" "${STRICT_OUT}"
cp "${DATA_SOURCE_DIR}/cmrc2018_dev.json" "${LOCAL_CMRC_DIR}/cmrc2018_dev.json"
cp "${DATA_SOURCE_DIR}/cmrc2018_train.json" "${LOCAL_CMRC_DIR}/cmrc2018_train.json"
if [ -f "${DATA_SOURCE_DIR}/README.md" ]; then
  cp "${DATA_SOURCE_DIR}/README.md" "${LOCAL_CMRC_DIR}/README.md"
fi

python - <<PY
import json
from pathlib import Path
from scd_repro.cmrc import load_cmrc2018_samples

rows = load_cmrc2018_samples("${DATA_ROOT}", limit=int("${SAMPLES}"))
Path("${PROMPTS_JSON}").write_text(
    json.dumps([x.prompt for x in rows], ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
PY

python -m scd_repro.cli.capture_pair \
  --backend transformers \
  --models-config "${MODELS_CONFIG}" \
  --data-config configs/data_ser.yaml \
  --out "${PAIR_DIR}" \
  --seed "${SEED}" \
  --prompt-file "${PROMPTS_JSON}" \
  --max-length "${MAX_LENGTH}" \
  --dtype bfloat16 \
  --model-a-device "${PRODUCER_DEVICE}" \
  --model-b-device "${CONSUMER_DEVICE}" \
  --artifact-dtype bfloat16 \
  --chunk-samples 1 \
  --max-chunk-tokens 4096

python -m scd_repro.cli.train_reuse \
  --pairs "${PAIR_DIR}" \
  --train-config "${TRAIN_CONFIG}" \
  --out "${REUSE_DIR}" \
  --seed "${SEED}" \
  --train-device "${TRAIN_DEVICE}"

python -m scd_repro.cli.eval_cmrc_strict \
  --models-config "${MODELS_CONFIG}" \
  --eval-config "${EVAL_CONFIG}" \
  --data-root "${DATA_ROOT}" \
  --route "${ROUTE_PATH}" \
  --reuse-ckpt "${REUSE_DIR}" \
  --patch-ckpt "${DUMMY_PATCH_DIR}" \
  --out "${STRICT_OUT}" \
  --limit "${SAMPLES}" \
  --max-length "${MAX_LENGTH}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --bandwidth-gbps "${BANDWIDTH_GBPS}" \
  --quant-group-size 128 \
  --producer-device "${PRODUCER_DEVICE}" \
  --consumer-device "${CONSUMER_DEVICE}" \
  --dtype bfloat16 \
  --seed "${SEED}"
