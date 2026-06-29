#!/usr/bin/env bash
set -euo pipefail

ROOT="/disk_n/c101-3-1/projects/scd-repro-codex-mistral-instruct-20260323"
VENV="/disk_n/c101-3-1/venvs/A800"
DATA_ROOT="/disk_n/c101-3-1/projects/scd_main/legacy/no4/data"

SEED="${SEED:-42}"
LIMIT="${LIMIT:-50}"
MAX_LENGTH="${MAX_LENGTH:-1024}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"
LAYERS_PER_SHARD="${LAYERS_PER_SHARD:-4}"
PRODUCER_DEVICE="${PRODUCER_DEVICE:-cuda:0}"
CONSUMER_DEVICE="${CONSUMER_DEVICE:-cuda:1}"
TRAIN_DEVICE="${TRAIN_DEVICE:-cuda:2}"

if [ $# -lt 2 ]; then
  echo "usage: $0 <models-config> <run-root>" >&2
  exit 1
fi

MODELS_CONFIG="$1"
RUN_ROOT="$2"

REUSE_TRAIN_CONFIG="${REUSE_TRAIN_CONFIG:-configs/train_reuse_mistral_self_layerwise_20260404.json}"
PATCH_TRAIN_CONFIG="${PATCH_TRAIN_CONFIG:-configs/mistral_self_patch_xnorm_nativekv_shared_residual_mlp_alllayers_e1_h2048_r512_b2_20260404.json}"
EVAL_CONFIG="${EVAL_CONFIG:-configs/eval_baseline_reuse_patchonly_50_20260404.json}"
ROUTE_CONFIG="${ROUTE_CONFIG:-configs/route_reuse_patchonly_all32_20260404.json}"

PROMPTS_JSON="${RUN_ROOT}/cmrc_prompts_${LIMIT}.json"
PAIR_DIR="${RUN_ROOT}/pairs_cmrc_limit${LIMIT}_seed${SEED}"
REUSE_DIR="${RUN_ROOT}/ckpt_reuse_layerwise_seed${SEED}"
PROFILE_PATH="${RUN_ROOT}/profile_cmrc_limit${LIMIT}_seed${SEED}.json"
PATCH_DIR="${RUN_ROOT}/ckpt_patch_xnorm_nativekv_shared_residual_mlp_all32_e1_h2048_r512_b2_seed${SEED}"
PATCH_DIR_SHARDED="${PATCH_DIR}_sharded_w${LAYERS_PER_SHARD}"
OUT_DIR="${RUN_ROOT}/strict_baseline_reuse_patchonly_all32_limit${LIMIT}_20260404"

cd "${ROOT}"
source "${VENV}/bin/activate"
export PYTHONPATH="${ROOT}/src"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-32}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-32}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-32}"

mkdir -p "${RUN_ROOT}"

python - <<PY
import json
from pathlib import Path
from scd_repro.cmrc import load_cmrc2018_samples
rows = load_cmrc2018_samples("${DATA_ROOT}", limit=${LIMIT})
Path("${PROMPTS_JSON}").write_text(json.dumps([x.prompt for x in rows], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY

if [ ! -f "${PAIR_DIR}/manifest.json" ]; then
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
fi

if [ ! -d "${REUSE_DIR}" ]; then
  python -m scd_repro.cli.train_reuse \
    --pairs "${PAIR_DIR}" \
    --train-config "${REUSE_TRAIN_CONFIG}" \
    --out "${REUSE_DIR}" \
    --seed "${SEED}" \
    --train-device "${TRAIN_DEVICE}" \
    --log "${RUN_ROOT}/train_reuse_layerwise_seed${SEED}.log"
fi

if [ ! -f "${PROFILE_PATH}" ]; then
  python -m scd_repro.cli.profile_layers \
    --pairs "${PAIR_DIR}" \
    --reuse-ckpt "${REUSE_DIR}" \
    --out "${PROFILE_PATH}" \
    --train-device "${TRAIN_DEVICE}"
fi

if [ ! -d "${PATCH_DIR}" ]; then
  python -m scd_repro.cli.train_patch \
    --pairs "${PAIR_DIR}" \
    --profile "${PROFILE_PATH}" \
    --route "${ROUTE_CONFIG}" \
    --train-config "${PATCH_TRAIN_CONFIG}" \
    --models-config "${MODELS_CONFIG}" \
    --out "${PATCH_DIR}" \
    --seed "${SEED}" \
    --train-device "${TRAIN_DEVICE}" \
    --log "${RUN_ROOT}/train_patch_shared_seed${SEED}.log"
fi

if [ ! -d "${PATCH_DIR_SHARDED}" ]; then
  python - <<PY
from scd_repro.pipeline import reshard_patch_ckpt
reshard_patch_ckpt(
    "${PATCH_DIR}",
    "${PATCH_DIR_SHARDED}",
    layers_per_shard=${LAYERS_PER_SHARD},
    drop_identity_residual=False,
)
PY
fi

python -m scd_repro.cli.eval_cmrc_strict \
  --models-config "${MODELS_CONFIG}" \
  --eval-config "${EVAL_CONFIG}" \
  --data-root "${DATA_ROOT}" \
  --route "${ROUTE_CONFIG}" \
  --reuse-ckpt "${REUSE_DIR}" \
  --patch-ckpt "${PATCH_DIR_SHARDED}" \
  --out "${OUT_DIR}" \
  --limit "${LIMIT}" \
  --max-length "${MAX_LENGTH}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --producer-device "${PRODUCER_DEVICE}" \
  --consumer-device "${CONSUMER_DEVICE}" \
  --dtype bfloat16 \
  --seed "${SEED}" \
  --log "${RUN_ROOT}/strict_baseline_reuse_patchonly_all32_limit${LIMIT}_20260404.log"

echo "RUN_DONE ${OUT_DIR}"
