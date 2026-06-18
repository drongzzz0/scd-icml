#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
VENV="/disk_n/c101-3-mql/venvs/A800"

SEED="${SEED:-42}"
MAX_LENGTH="${MAX_LENGTH:-256}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"
SAMPLES_MAIN="${SAMPLES_MAIN:-200}"
SAMPLES_RS="${SAMPLES_RS:-50}"
BANDWIDTH_GBPS="${BANDWIDTH_GBPS:-200}"
PRODUCER_DEVICE="${PRODUCER_DEVICE:-cuda:1}"
CONSUMER_DEVICE="${CONSUMER_DEVICE:-cuda:2}"

RUN_ROOT="${ROOT}/runs/mistral_to_instruct_cmrc"
REUSE_DIR="${RUN_ROOT}/ckpt_reuse_seed${SEED}"
PATCH_DIR="${RUN_ROOT}/ckpt_patch_seed${SEED}"
ROUTE_PATH="${RUN_ROOT}/route_cmrc_seed${SEED}.json"
STRICT_OUT="${RUN_ROOT}/strict_main"
ORACLE_OUT="${RUN_ROOT}/oracle_diag_rough50"
DATA_ROOT="${ROOT}/.tmp_data_root"

cd "${ROOT}"
source "${VENV}/bin/activate"
export PYTHONPATH="${ROOT}/src"

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
