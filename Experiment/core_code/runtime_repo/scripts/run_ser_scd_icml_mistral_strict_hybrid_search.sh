#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
VENV="${VENV:-/disk_n/c101-3-mql/venvs/A800}"
DATA_ROOT="${DATA_ROOT:-/disk_n/c101-3-mql/projects/SCD-ICML/kv/snapshots/scd-github-f7d7854}"
CMRC_DATA_ROOT="${CMRC_DATA_ROOT:-${DATA_ROOT}}"

BASE_RUN_ROOT="${BASE_RUN_ROOT:-}"
DIRECTION="${DIRECTION:-lite_to_7b}"
SEED="${SEED:-42}"
SEARCH_LIMIT="${SEARCH_LIMIT:-200}"
BLOCK_SIZE="${BLOCK_SIZE:-1}"
REPLAY_TOP_K="${REPLAY_TOP_K:-3}"
REPLAY_SELECTION_MODE="${REPLAY_SELECTION_MODE:-quality_threshold_ttft_first}"
REPLAY_DECISION_MAX_F1_DROP="${REPLAY_DECISION_MAX_F1_DROP:--1.0}"
REPLAY_DECISION_MAX_F1_DROP_RATIO="${REPLAY_DECISION_MAX_F1_DROP_RATIO:-0.05}"
REPLAY_DECISION_MAX_EM_DROP="${REPLAY_DECISION_MAX_EM_DROP:--1.0}"
STRICT_SEARCH_REQUIRE_MIXED="${STRICT_SEARCH_REQUIRE_MIXED:-0}"
MAX_F1_DROP="${MAX_F1_DROP:-1.0}"
MAX_EM_DROP="${MAX_EM_DROP:-0.10}"
MIN_TTFT_GAIN_MS="${MIN_TTFT_GAIN_MS:-1.0}"
MAX_ACCEPTED_BLOCKS="${MAX_ACCEPTED_BLOCKS:-0}"
MAX_LENGTH="${MAX_LENGTH:-1024}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"
PRODUCER_DEVICE="${PRODUCER_DEVICE:-cuda:0}"
CONSUMER_DEVICE="${CONSUMER_DEVICE:-cuda:1}"
WORKER_PAIRS="${WORKER_PAIRS:-${PRODUCER_DEVICE},${CONSUMER_DEVICE}}"
WAIT_SEARCH_FREE_MB="${WAIT_SEARCH_FREE_MB:-18000}"
WAIT_GPU_POLL_SEC="${WAIT_GPU_POLL_SEC:-120}"
INSTALL_EDITABLE="${INSTALL_EDITABLE:-1}"

MODELS_CONFIG="${MODELS_CONFIG:-configs/models_ser_mistrallite_to_mistral7_mql.json}"
PATCH_EVAL_CONFIG="${PATCH_EVAL_CONFIG:-configs/eval_scd_icml_no_fallback_no_recompute.json}"

resolve_repo_path() {
  case "$1" in
    /*) echo "$1" ;;
    *) echo "${ROOT}/$1" ;;
  esac
}

if [ -z "${BASE_RUN_ROOT}" ]; then
  echo "BASE_RUN_ROOT is required. Point it at a completed run from run_ser_scd_icml_mistral_repro.sh." >&2
  exit 2
fi

DIR_ROOT="${BASE_RUN_ROOT}/${DIRECTION}"
REUSE_DIR="${REUSE_DIR:-${DIR_ROOT}/ckpt_reuse/cmrc2018_seed${SEED}}"
PATCH_DIR="${PATCH_DIR:-${DIR_ROOT}/ckpt_patch/patch_cmrc2018_seed${SEED}}"
ROUTE_TEMPLATE="${ROUTE_TEMPLATE:-${DIR_ROOT}/route/route_cmrc2018_seed${SEED}.json}"
SEARCH_OUT="${SEARCH_OUT:-${DIR_ROOT}/strict_hybrid_search/strictstop_b${BLOCK_SIZE}_replayk${REPLAY_TOP_K}_f1pct5}"
MODELS_CONFIG="$(resolve_repo_path "${MODELS_CONFIG}")"
PATCH_EVAL_CONFIG="$(resolve_repo_path "${PATCH_EVAL_CONFIG}")"
ROUTE_TEMPLATE="$(resolve_repo_path "${ROUTE_TEMPLATE}")"

for path in "${REUSE_DIR}" "${PATCH_DIR}" "${ROUTE_TEMPLATE}" "${PATCH_EVAL_CONFIG}"; do
  if [ ! -e "${path}" ]; then
    echo "required input missing: ${path}" >&2
    exit 2
  fi
done

logical_to_physical_gpu() {
  local logical_idx="$1"
  if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    echo "${logical_idx}"
    return 0
  fi
  IFS=',' read -r -a visible_devices <<< "${CUDA_VISIBLE_DEVICES}"
  echo "${visible_devices[${logical_idx}]}"
}

worker_pair_cuda_indices() {
  echo "${WORKER_PAIRS}" | tr ';,' '\n' | sed -n 's/^cuda:\([0-9][0-9]*\)$/\1/p' | awk '!seen[$0]++'
}

wait_visible_gpus_free() {
  local required_free_mb="$1"
  shift
  if [ "${required_free_mb}" -le 0 ] || [ "$#" -eq 0 ]; then
    return 0
  fi
  while true; do
    local ok=1
    local report=()
    local logical_idx physical_idx free_mb
    for logical_idx in "$@"; do
      physical_idx="$(logical_to_physical_gpu "${logical_idx}")"
      free_mb="$(nvidia-smi --id="${physical_idx}" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -n 1 | tr -d ' ')"
      if [ -z "${free_mb}" ]; then
        free_mb=0
      fi
      report+=("cuda:${logical_idx}->gpu:${physical_idx} free=${free_mb}MB")
      if [ "${free_mb}" -lt "${required_free_mb}" ]; then
        ok=0
      fi
    done
    if [ "${ok}" -eq 1 ]; then
      echo "GPU memory gate passed (${required_free_mb}MB): ${report[*]}"
      return 0
    fi
    echo "waiting for GPU free memory >= ${required_free_mb}MB: ${report[*]}"
    sleep "${WAIT_GPU_POLL_SEC}"
  done
}

mkdir -p "${SEARCH_OUT}"
cd "${ROOT}"
source "${VENV}/bin/activate"
if [ "${INSTALL_EDITABLE}" = "1" ]; then
  python -m pip install -e .
fi
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mapfile -t _wait_indices < <(worker_pair_cuda_indices)
wait_visible_gpus_free "${WAIT_SEARCH_FREE_MB}" "${_wait_indices[@]}"

cat > "${SEARCH_OUT}/run_config.json" <<EOF
{
  "root": "${ROOT}",
  "venv": "${VENV}",
  "base_run_root": "${BASE_RUN_ROOT}",
  "direction": "${DIRECTION}",
  "data_root": "${DATA_ROOT}",
  "cmrc_data_root": "${CMRC_DATA_ROOT}",
  "models_config": "${MODELS_CONFIG}",
  "reuse_dir": "${REUSE_DIR}",
  "patch_dir": "${PATCH_DIR}",
  "route_template": "${ROUTE_TEMPLATE}",
  "patch_eval_config": "${PATCH_EVAL_CONFIG}",
  "search_out": "${SEARCH_OUT}",
  "seed": ${SEED},
  "search_limit": ${SEARCH_LIMIT},
  "block_size": ${BLOCK_SIZE},
  "replay_top_k": ${REPLAY_TOP_K},
  "replay_decision_max_f1_drop_ratio": ${REPLAY_DECISION_MAX_F1_DROP_RATIO},
  "worker_pairs": "${WORKER_PAIRS}",
  "wait_search_free_mb": ${WAIT_SEARCH_FREE_MB}
}
EOF

python -m scd_repro.cli.search_blockwise_hybrid \
  --models-config "${MODELS_CONFIG}" \
  --data-root "${CMRC_DATA_ROOT}" \
  --route-template "${ROUTE_TEMPLATE}" \
  --reuse-ckpt "${REUSE_DIR}" \
  --patch-ckpt "${PATCH_DIR}" \
  --patch-eval-config "${PATCH_EVAL_CONFIG}" \
  --out "${SEARCH_OUT}" \
  --limit "${SEARCH_LIMIT}" \
  --block-size "${BLOCK_SIZE}" \
  --max-length "${MAX_LENGTH}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --worker-pairs "${WORKER_PAIRS}" \
  --max-f1-drop "${MAX_F1_DROP}" \
  --max-em-drop "${MAX_EM_DROP}" \
  --min-ttft-gain-ms "${MIN_TTFT_GAIN_MS}" \
  --max-accepted-blocks "${MAX_ACCEPTED_BLOCKS}" \
  --replay-top-k "${REPLAY_TOP_K}" \
  --replay-selection-mode "${REPLAY_SELECTION_MODE}" \
  --replay-decision-max-f1-drop "${REPLAY_DECISION_MAX_F1_DROP}" \
  --replay-decision-max-f1-drop-ratio "${REPLAY_DECISION_MAX_F1_DROP_RATIO}" \
  --replay-decision-max-em-drop "${REPLAY_DECISION_MAX_EM_DROP}" \
  --seed "${SEED}" \
  --log "${SEARCH_OUT}/search.log"

python scripts/validate_scd_route.py --route "${SEARCH_OUT}/selected_hybrid_route.json"
_strict_search_args=()
if [ "${STRICT_SEARCH_REQUIRE_MIXED}" = "1" ]; then
  _strict_search_args+=(--require-mixed)
fi
python scripts/validate_strict_hybrid_search.py \
  --search-root "${SEARCH_OUT}" \
  --max-f1-drop-ratio "${REPLAY_DECISION_MAX_F1_DROP_RATIO}" \
  "${_strict_search_args[@]}"

echo "strict_hybrid_route=${SEARCH_OUT}/selected_hybrid_route.json"
echo "strict_hybrid_report=${SEARCH_OUT}/search_report.json"
