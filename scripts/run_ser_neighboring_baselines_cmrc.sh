#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
VENV="${VENV:-/disk_n/c101-3-mql/venvs/A800}"
DATA_ROOT="${DATA_ROOT:-/disk_n/c101-3-mql/projects/SCD-ICML/kv/snapshots/scd-github-f7d7854}"
MODELS_CONFIG="${MODELS_CONFIG:-configs/models_ser_mistrallite_to_mistral7_mql.json}"
PAIR_DIR="${PAIR_DIR:-/disk_n/c101-3-mql/projects/SCD-ICML/kv/snapshots/scd-github-f7d7854/runs/mistral_strict_default_gpu1_200_b2_20260421/pairs/cmrc2018_seed42}"
SCD_SUMMARY="${SCD_SUMMARY:-}"
RUN_ROOT="${RUN_ROOT:-/disk_n/c101-3-mql/projects/SCD-ICML/runs/neighboring_baselines_cmrc${LIMIT:-200}_$(date +%Y%m%d_%H%M%S)}"

LIMIT="${LIMIT:-200}"
MAX_LENGTH="${MAX_LENGTH:-256}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"
BANDWIDTH_GBPS="${BANDWIDTH_GBPS:-200}"
SVDLLM_RANK="${SVDLLM_RANK:-16}"
SLICE_KEEP_DIM="${SLICE_KEEP_DIM:-96}"
SLICE_CALIB_LIMIT="${SLICE_CALIB_LIMIT:-64}"
TENSOR_TOKEN_KEEP_RATIO="${TENSOR_TOKEN_KEEP_RATIO:-0.75}"
TENSOR_HEAD_DIM_RANK="${TENSOR_HEAD_DIM_RANK:-96}"
TENSOR_STACK_RANK="${TENSOR_STACK_RANK:-1}"

SVDLLM_GPUS="${SVDLLM_GPUS:-0,1}"
TENSORLLM_GPUS="${TENSORLLM_GPUS:-2,3}"
SLICEGPT_GPUS="${SLICEGPT_GPUS:-0,1}"
PRODUCER_DEVICE="${PRODUCER_DEVICE:-cuda:0}"
CONSUMER_DEVICE="${CONSUMER_DEVICE:-cuda:1}"

if [[ -f "${VENV}/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${VENV}/bin/activate"
fi

cd "${ROOT}"
mkdir -p "${RUN_ROOT}/logs"

echo "run_root=${RUN_ROOT}"
echo "data_root=${DATA_ROOT}"
echo "pair_dir=${PAIR_DIR}"
echo "scd_summary=${SCD_SUMMARY:-<unset>}"

(
  set -euo pipefail
  export CUDA_VISIBLE_DEVICES="${SVDLLM_GPUS}"
  python scripts/run_svdllm_style_kv_cmrc.py \
    --project-root "${ROOT}" \
    --models-config "${MODELS_CONFIG}" \
    --data-root "${DATA_ROOT}" \
    --out "${RUN_ROOT}/svdllm" \
    --limit "${LIMIT}" \
    --max-length "${MAX_LENGTH}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --bandwidth-gbps "${BANDWIDTH_GBPS}" \
    --producer-device "${PRODUCER_DEVICE}" \
    --consumer-device "${CONSUMER_DEVICE}" \
    --rank "${SVDLLM_RANK}"
) >"${RUN_ROOT}/logs/svdllm.log" 2>&1 &
svdllm_pid=$!

(
  set -euo pipefail
  export CUDA_VISIBLE_DEVICES="${TENSORLLM_GPUS}"
  python scripts/run_tensorllm_style_kv_cmrc.py \
    --project-root "${ROOT}" \
    --models-config "${MODELS_CONFIG}" \
    --data-root "${DATA_ROOT}" \
    --out "${RUN_ROOT}/tensorllm" \
    --limit "${LIMIT}" \
    --max-length "${MAX_LENGTH}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --bandwidth-gbps "${BANDWIDTH_GBPS}" \
    --producer-device "${PRODUCER_DEVICE}" \
    --consumer-device "${CONSUMER_DEVICE}" \
    --token-keep-ratio "${TENSOR_TOKEN_KEEP_RATIO}" \
    --head-dim-rank "${TENSOR_HEAD_DIM_RANK}" \
    --stack-rank "${TENSOR_STACK_RANK}"
) >"${RUN_ROOT}/logs/tensorllm.log" 2>&1 &
tensorllm_pid=$!

status=0
wait "${svdllm_pid}" || status=1
wait "${tensorllm_pid}" || status=1
if [[ "${status}" -ne 0 ]]; then
  echo "SVD-LLM-style or TensorLLM-style run failed; see ${RUN_ROOT}/logs" >&2
  exit "${status}"
fi

(
  set -euo pipefail
  export CUDA_VISIBLE_DEVICES="${SLICEGPT_GPUS}"
  python scripts/run_slicegpt_style_kv_cmrc.py \
    --project-root "${ROOT}" \
    --models-config "${MODELS_CONFIG}" \
    --data-root "${DATA_ROOT}" \
    --pair-dir "${PAIR_DIR}" \
    --out "${RUN_ROOT}/slicegpt" \
    --limit "${LIMIT}" \
    --max-length "${MAX_LENGTH}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --bandwidth-gbps "${BANDWIDTH_GBPS}" \
    --producer-device "${PRODUCER_DEVICE}" \
    --consumer-device "${CONSUMER_DEVICE}" \
    --keep-dim "${SLICE_KEEP_DIM}" \
    --calib-limit "${SLICE_CALIB_LIMIT}"
) >"${RUN_ROOT}/logs/slicegpt.log" 2>&1

if [[ -z "${SCD_SUMMARY}" ]]; then
  cat >"${RUN_ROOT}/logs/build_table.log" <<EOF
SCD_SUMMARY is not set.

The adapted neighboring baselines completed, but the consolidated table was
not generated because the SCD row must use the same protocol, especially
LIMIT=${LIMIT}, MAX_LENGTH=${MAX_LENGTH}, MAX_NEW_TOKENS=${MAX_NEW_TOKENS}, and
BANDWIDTH_GBPS=${BANDWIDTH_GBPS}.

After producing a same-protocol SCD summary, rerun:

python scripts/build_neighboring_baseline_table.py \\
  --svdllm-summary "${RUN_ROOT}/svdllm/summary.json" \\
  --slicegpt-summary "${RUN_ROOT}/slicegpt/summary.json" \\
  --tensorllm-summary "${RUN_ROOT}/tensorllm/summary.json" \\
  --scd-summary "\${SCD_SUMMARY}" \\
  --out-dir "${RUN_ROOT}/table"
EOF
  echo "neighboring_baselines_table=pending_scd_summary"
  echo "see ${RUN_ROOT}/logs/build_table.log"
else
  python scripts/build_neighboring_baseline_table.py \
    --svdllm-summary "${RUN_ROOT}/svdllm/summary.json" \
    --slicegpt-summary "${RUN_ROOT}/slicegpt/summary.json" \
    --tensorllm-summary "${RUN_ROOT}/tensorllm/summary.json" \
    --scd-summary "${SCD_SUMMARY}" \
    --require-compatible-protocol \
    --out-dir "${RUN_ROOT}/table" \
    >"${RUN_ROOT}/logs/build_table.log" 2>&1

  echo "neighboring_baselines_table=${RUN_ROOT}/table/neighboring_baselines.md"
fi
