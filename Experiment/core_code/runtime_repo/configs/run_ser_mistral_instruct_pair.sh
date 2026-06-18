#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATA_ROOT="/disk_n/c101-3-mql/projects/scd_main/legacy/no4/data"
VENV="/disk_n/c101-3-mql/venvs/A800"

SEED="${SEED:-42}"
MAX_LENGTH="${MAX_LENGTH:-256}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"
SAMPLES_MAIN="${SAMPLES_MAIN:-200}"
SAMPLES_RS="${SAMPLES_RS:-50}"
BANDWIDTH_GBPS="${BANDWIDTH_GBPS:-200}"
PRODUCER_DEVICE="${PRODUCER_DEVICE:-cuda:1}"
CONSUMER_DEVICE="${CONSUMER_DEVICE:-cuda:2}"
TRAIN_DEVICE="${TRAIN_DEVICE:-cuda:2}"

RUN_ROOT="${ROOT}/runs/mistral_to_instruct_cmrc"
PAIR_DIR="${RUN_ROOT}/pairs_cmrc_seed${SEED}"
REUSE_DIR="${RUN_ROOT}/ckpt_reuse_seed${SEED}"
PROFILE_PATH="${RUN_ROOT}/profile_cmrc_seed${SEED}.json"
PATCH_DIR="${RUN_ROOT}/ckpt_patch_seed${SEED}"
ROUTE_PATH="${RUN_ROOT}/route_cmrc_seed${SEED}.json"
STRICT_OUT="${RUN_ROOT}/strict_main"
ORACLE_OUT="${RUN_ROOT}/oracle_diag_rough50"
TMP_DIR="${ROOT}/.tmp_eval_cfg_mistral_instruct"
PROMPTS_JSON="${TMP_DIR}/cmrc_prompts_${SAMPLES_MAIN}.json"

cd "${ROOT}"
source "${VENV}/bin/activate"
export PYTHONPATH="${ROOT}/src"

mkdir -p "${RUN_ROOT}" "${TMP_DIR}"

python -c "import json; from pathlib import Path; from scd_repro.cmrc import load_cmrc2018_samples; rows=load_cmrc2018_samples('${DATA_ROOT}', limit=${SAMPLES_MAIN}); Path('${PROMPTS_JSON}').write_text(json.dumps([x.prompt for x in rows], ensure_ascii=False, indent=2)+'\n', encoding='utf-8')"

python -m scd_repro.cli.capture_pair \
  --backend transformers \
  --models-config configs/models_ser_mistral_instruct.yaml \
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
  --train-config configs/train.yaml \
  --out "${REUSE_DIR}" \
  --seed "${SEED}" \
  --train-device "${TRAIN_DEVICE}"

python -m scd_repro.cli.profile_layers \
  --pairs "${PAIR_DIR}" \
  --reuse-ckpt "${REUSE_DIR}" \
  --out "${PROFILE_PATH}" \
  --train-device "${TRAIN_DEVICE}"

python -m scd_repro.cli.train_patch \
  --pairs "${PAIR_DIR}" \
  --profile "${PROFILE_PATH}" \
  --train-config configs/train.yaml \
  --out "${PATCH_DIR}" \
  --seed "${SEED}" \
  --train-device "${TRAIN_DEVICE}"

python -m scd_repro.cli.build_route \
  --profile "${PROFILE_PATH}" \
  --eval-config configs/eval.yaml \
  --out "${ROUTE_PATH}"

python -m scd_repro.cli.eval_cmrc_strict \
  --models-config configs/models_ser_mistral_instruct.yaml \
  --eval-config configs/eval.yaml \
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

python -m scd_repro.cli.run_oracle_diag \
  --mode single_anchor \
  --models-config configs/models_ser_mistral_instruct.yaml \
  --eval-config configs/eval_oracle_diag.yaml \
  --data-root "${DATA_ROOT}" \
  --reuse-ckpt "${REUSE_DIR}" \
  --patch-ckpt "${PATCH_DIR}" \
  --out-root "${ORACLE_OUT}" \
  --layers 12,14,16,18 \
  --recompute-lengths 16 \
  --limit "${SAMPLES_RS}" \
  --max-length "${MAX_LENGTH}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --bandwidth-gbps "${BANDWIDTH_GBPS}" \
  --producer-device "${PRODUCER_DEVICE}" \
  --consumer-device "${CONSUMER_DEVICE}" \
  --dtype bfloat16 \
  --seed "${SEED}"
