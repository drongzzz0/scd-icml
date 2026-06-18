#!/usr/bin/env bash
set -euo pipefail

ROOT="/disk_n/c101-3-mql/projects/scd-repro-codex-mistral-instruct-20260323"
VENV="/disk_n/c101-3-mql/venvs/A800"

RUN_ROOT="${ROOT}/runs/qwen17_to_gray_cmrc"
SEED="${SEED:-42}"
LAYERS_PER_SHARD="${LAYERS_PER_SHARD:-4}"
LIMIT="${LIMIT:-50}"

MODELS_CONFIG="${MODELS_CONFIG:-configs/models_ser_qwen17_gray.yaml}"
TRAIN_CONFIG="${TRAIN_CONFIG:-configs/qwen17_to_gray_patch_xnorm_nativekv_shared_residual_mlp_alllayers_subset12_e1_h2048_r512_b2_20260403.json}"
PAIRS_DIR="${PAIRS_DIR:-${RUN_ROOT}/pairs_cmrc_seed42_subset12}"
PROFILE_PATH="${PROFILE_PATH:-${RUN_ROOT}/profile_cmrc_seed42.json}"
ROUTE_PATH="${ROUTE_PATH:-${RUN_ROOT}/route_reuse_patchonly_all28_20260403.json}"
REUSE_CKPT="${REUSE_CKPT:-${RUN_ROOT}/ckpt_reuse_layerwise_seed42}"
PATCH_CKPT="${PATCH_CKPT:-${RUN_ROOT}/ckpt_patch_xnorm_nativekv_shared_residual_mlp_alllayers_subset12_e1_h2048_r512_b2_20260403}"
PATCH_CKPT_SHARDED="${PATCH_CKPT_SHARDED:-${RUN_ROOT}/ckpt_patch_xnorm_nativekv_shared_residual_mlp_alllayers_subset12_e1_h2048_r512_b2_20260403_sharded_w4}"
EVAL_CONFIG="${EVAL_CONFIG:-${RUN_ROOT}/eval_baseline_reuse_patchonly_all28_50_20260403.json}"
OUT_DIR="${OUT_DIR:-${RUN_ROOT}/strict_baseline_reuse_patchonly_all28_shared_subset12e1_50_20260403}"
DATA_ROOT="${DATA_ROOT:-${ROOT}/.tmp_data_root_qwen17_self_cmrc}"

PATCH_LOG="${PATCH_LOG:-${RUN_ROOT}/train_patch_xnorm_nativekv_shared_residual_mlp_alllayers_subset12_e1_h2048_r512_b2_20260403.log}"
EVAL_LOG="${EVAL_LOG:-${RUN_ROOT}/strict_baseline_reuse_patchonly_all28_shared_subset12e1_50_20260403.log}"

cd "${ROOT}"
source "${VENV}/bin/activate"
export PYTHONPATH="${ROOT}/src"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

if [ ! -d "${REUSE_CKPT}" ]; then
  echo "MISSING_REUSE_CKPT ${REUSE_CKPT}" >&2
  exit 1
fi

if [ ! -f "${PROFILE_PATH}" ]; then
  echo "MISSING_PROFILE ${PROFILE_PATH}" >&2
  exit 1
fi

if [ ! -f "${ROUTE_PATH}" ]; then
  echo "MISSING_ROUTE ${ROUTE_PATH}" >&2
  exit 1
fi

if [ ! -f "${PAIRS_DIR}/manifest.json" ]; then
  echo "MISSING_PAIRS_MANIFEST ${PAIRS_DIR}/manifest.json" >&2
  exit 1
fi

if [ ! -d "${PATCH_CKPT}" ]; then
  python -m scd_repro.cli.train_patch \
    --pairs "${PAIRS_DIR}" \
    --profile "${PROFILE_PATH}" \
    --route "${ROUTE_PATH}" \
    --train-config "${TRAIN_CONFIG}" \
    --models-config "${MODELS_CONFIG}" \
    --out "${PATCH_CKPT}" \
    --seed "${SEED}" \
    --log "${PATCH_LOG}"
fi

if [ ! -d "${PATCH_CKPT_SHARDED}" ]; then
  python3 - <<PY
from scd_repro.pipeline import reshard_patch_ckpt
reshard_patch_ckpt(
    "${PATCH_CKPT}",
    "${PATCH_CKPT_SHARDED}",
    layers_per_shard=${LAYERS_PER_SHARD},
    drop_identity_residual=False,
)
PY
fi

python -m scd_repro.cli.eval_cmrc_strict \
  --models-config "${MODELS_CONFIG}" \
  --eval-config "${EVAL_CONFIG}" \
  --data-root "${DATA_ROOT}" \
  --route "${ROUTE_PATH}" \
  --reuse-ckpt "${REUSE_CKPT}" \
  --patch-ckpt "${PATCH_CKPT_SHARDED}" \
  --out "${OUT_DIR}" \
  --limit "${LIMIT}" \
  --producer-device cuda:0 \
  --consumer-device cuda:1 \
  --dtype bfloat16 \
  --seed "${SEED}" \
  --log "${EVAL_LOG}"

echo "RUN_DONE ${OUT_DIR}"
