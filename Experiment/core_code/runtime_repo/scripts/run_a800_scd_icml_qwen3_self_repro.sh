#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

export MODELS_CONFIG="${MODELS_CONFIG:-configs/models_a800_qwen3_17_self.json}"
export REUSE_TRAIN_CONFIG="${REUSE_TRAIN_CONFIG:-configs/train_reuse_qwen32_layerwise_20260614.json}"
export PATCH_TRAIN_CONFIG="${PATCH_TRAIN_CONFIG:-configs/qwen3_17_self_patch_xnorm_nativekv_shared_residual_mlp_alllayers_e1_h2048_r512_b1.json}"
export EVAL_CONFIG_TEMPLATE="${EVAL_CONFIG_TEMPLATE:-configs/eval_scd_icml_qwen32_no_fallback_no_recompute.json}"
export DATA_ROOT="${DATA_ROOT:-/home/c101-800-mql/maqianli/no4/data}"
export DATASETS="${DATASETS:-cmrc2018,hotpotqa}"
export TRAIN_LIMIT="${TRAIN_LIMIT:-200}"
export EVAL_LIMIT="${EVAL_LIMIT:-200}"
export PATCH_BUDGET="${PATCH_BUDGET:-6}"
export REQUIRED_GPUS="${REQUIRED_GPUS:-2}"
export REQUIRED_FREE_MB="${REQUIRED_FREE_MB:-70000}"
export GPU_MONITOR_SECONDS="${GPU_MONITOR_SECONDS:-60}"
export GPU_SAMPLE_INTERVAL="${GPU_SAMPLE_INTERVAL:-10}"
export MAX_LENGTH="${MAX_LENGTH:-0}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"
export BANDWIDTH_GBPS="${BANDWIDTH_GBPS:-200}"
export QUANT_GROUP_SIZE="${QUANT_GROUP_SIZE:-128}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"

exec bash "${ROOT}/scripts/run_a800_scd_icml_qwen32_repro.sh"
