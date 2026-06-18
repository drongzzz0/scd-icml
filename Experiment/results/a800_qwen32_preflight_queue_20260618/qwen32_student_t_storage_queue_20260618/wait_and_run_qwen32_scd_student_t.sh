#!/usr/bin/env bash
set -euo pipefail

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate cache

REPO="${REPO:-/home/student_t/maqianli/SCD-ICML/runtime_repo_codex_storage_20260618}"
QUEUE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_PARENT="${RUN_PARENT:-/home/c101-800-mql/nas/SCD-ICML/runs}"
SCD_STORAGE_ROOT="${SCD_STORAGE_ROOT:-/home/c101-800-mql/nas/SCD-ICML}"
MODEL_CACHE_ROOT="${MODEL_CACHE_ROOT:-${SCD_STORAGE_ROOT}/model_cache}"
MODELS_CONFIG="${MODELS_CONFIG:-${REPO}/configs/models_a800_qwen25_32b_base_to_instruct_template.json}"
DATA_ROOT="${DATA_ROOT:-/home/c101-800-mql/maqianli/no4/data}"
REQUIRED_GPUS="${REQUIRED_GPUS:-2}"
REQUIRED_FREE_MB="${REQUIRED_FREE_MB:-70000}"
GPU_MONITOR_SECONDS="${GPU_MONITOR_SECONDS:-60}"
GPU_SAMPLE_INTERVAL="${GPU_SAMPLE_INTERVAL:-10}"
MAX_WAIT_SECONDS="${MAX_WAIT_SECONDS:-43200}"
TRAIN_LIMIT="${TRAIN_LIMIT:-200}"
EVAL_LIMIT="${EVAL_LIMIT:-200}"
TRAIN_CMRC_SPLIT="${TRAIN_CMRC_SPLIT:-train}"
EVAL_CMRC_SPLIT="${EVAL_CMRC_SPLIT:-dev}"
DATASETS="${DATASETS:-cmrc2018}"
MAX_LENGTH="${MAX_LENGTH:-0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"

mkdir -p "${QUEUE_ROOT}" "${RUN_PARENT}" "${MODEL_CACHE_ROOT}"

for model_dir in \
  /home/c101-800-mql/nas/models/Qwen2.5-32B \
  /home/c101-800-mql/nas/models/Qwen2.5-32B-Instruct
do
  until [ -f "${model_dir}/config.json" ] && [ "$(find "${model_dir}" -maxdepth 1 -name '*.safetensors' 2>/dev/null | wc -l)" -ge 17 ]; do
    echo "$(date -Is) waiting for complete NAS model at ${model_dir}"
    sleep 60
  done
done

cd "${REPO}"
start="$(date +%s)"
attempt=0
while true; do
  now="$(date +%s)"
  if [ $((now - start)) -gt "${MAX_WAIT_SECONDS}" ]; then
    echo "timeout waiting for ${REQUIRED_GPUS} sustained-idle GPUs with free>=${REQUIRED_FREE_MB}MB" >&2
    exit 3
  fi
  attempt=$((attempt + 1))
  out="${QUEUE_ROOT}/preflight_attempt_${attempt}.json"
  if python scripts/preflight_qwen32_a800.py \
    --models-config "${MODELS_CONFIG}" \
    --data-root "${DATA_ROOT}" \
    --datasets "${DATASETS}" \
    --required-gpus "${REQUIRED_GPUS}" \
    --required-free-mb "${REQUIRED_FREE_MB}" \
    --monitor-seconds "${GPU_MONITOR_SECONDS}" \
    --sample-interval "${GPU_SAMPLE_INTERVAL}" \
    --out "${out}" \
    --write-env "${QUEUE_ROOT}/selected_gpus.env"; then
    echo "$(date -Is) preflight passed on attempt ${attempt}"
    break
  fi
  echo "$(date -Is) waiting; preflight failed on attempt ${attempt}"
  sleep 30
done

export RUN_PARENT
export SCD_STORAGE_ROOT
export MODEL_CACHE_ROOT
export MODELS_CONFIG
export DATA_ROOT
export REQUIRED_GPUS
export REQUIRED_FREE_MB
export GPU_MONITOR_SECONDS
export GPU_SAMPLE_INTERVAL
export TRAIN_LIMIT
export EVAL_LIMIT
export TRAIN_CMRC_SPLIT
export EVAL_CMRC_SPLIT
export DATASETS
export MAX_LENGTH
export MAX_NEW_TOKENS
export CONDA_ENV=cache
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export HF_HOME="${HF_HOME:-${MODEL_CACHE_ROOT}/hf_home}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${MODEL_CACHE_ROOT}/hf_hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${MODEL_CACHE_ROOT}/transformers}"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-${MODEL_CACHE_ROOT}/modelscope}"
export TORCH_HOME="${TORCH_HOME:-${MODEL_CACHE_ROOT}/torch}"

bash scripts/run_a800_scd_icml_qwen32_repro.sh
