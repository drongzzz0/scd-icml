#!/usr/bin/env bash
set -euo pipefail

RUNTIME_ROOT="${RUNTIME_ROOT:-$(pwd)}"
SOURCE_RUN_ROOT="${SOURCE_RUN_ROOT:?Set SOURCE_RUN_ROOT}"
RUN_ROOT="${RUN_ROOT:?Set RUN_ROOT}"
VENV="${VENV:-/disk_n/c101-3-mql/venvs/A800}"
DIRECTION="${DIRECTION:-lite_to_7b}"
SEED="${SEED:-42}"
SHARD_NAMES="${SHARD_NAMES:-shard_10_12;shard_13_15;shard_16_18;shard_19_20}"
BUDGETS="${BUDGETS:-5,6,8,10,11}"
EVAL_LIMIT="${EVAL_LIMIT:-50}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"
WAIT_EVAL_FREE_MB="${WAIT_EVAL_FREE_MB:-18000}"
REPORT_PREFIX="${REPORT_PREFIX:-selection_cmrc50_f1}"

FINAL_PATCH_DIR="${FINAL_PATCH_DIR:-${RUN_ROOT}/${DIRECTION}/ckpt_patch/patch_cmrc2018_seed${SEED}}"
PROFILE="${PROFILE:-${SOURCE_RUN_ROOT}/${DIRECTION}/route/profile_cmrc2018_seed${SEED}.json}"
CANDIDATE_DIR="${CANDIDATE_DIR:-${RUN_ROOT}/${DIRECTION}/route/candidates_residual_top11}"
EVAL_ROOT="${EVAL_ROOT:-${RUN_ROOT}/${DIRECTION}/evals_cmrc50}"
REPORT_DIR="${REPORT_DIR:-${RUN_ROOT}/${DIRECTION}/quality_gate}"

cd "${RUNTIME_ROOT}"
source "${VENV}/bin/activate"
export PYTHONPATH="${RUNTIME_ROOT}/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
export OPENBLAS_MAIN_FREE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "${EVAL_ROOT}" "${REPORT_DIR}"

IFS=";" read -r -a shard_names <<< "${SHARD_NAMES}"
declare -a shard_dirs=()
echo "candidate postprocess waiting at $(date -Is)"
for name in "${shard_names[@]}"; do
  [[ -z "${name}" ]] && continue
  shard_dir="${RUN_ROOT}/${DIRECTION}/ckpt_patch_shards/${name}"
  shard_dirs+=("${shard_dir}")
  while [[ ! -f "${shard_dir}/meta.json" ]]; do
    sleep 120
  done
  echo "found ${shard_dir}/meta.json at $(date -Is)"
done

python scripts/merge_patch_ckpts.py --out "${FINAL_PATCH_DIR}" "${shard_dirs[@]}"
python scripts/make_paper_scd_candidate_routes.py \
  --profile "${PROFILE}" \
  --patch-dir "${FINAL_PATCH_DIR}" \
  --models-config configs/models_ser_mistrallite_to_mistral7_mql.json \
  --eval-config configs/eval_scd_icml_no_fallback_no_recompute.json \
  --out-dir "${CANDIDATE_DIR}" \
  --budgets "${BUDGETS}" \
  --include-trained-max

for route in "${CANDIDATE_DIR}"/route_k*.json; do
  python scripts/validate_scd_route.py --route "${route}" --require-mixed
done

run_eval() {
  local k="$1"
  local gpus="$2"
  local route="${CANDIDATE_DIR}/route_k${k}.json"
  local out="${EVAL_ROOT}/eval_k${k}"
  [[ -f "${route}" ]] || return 0
  echo "starting candidate CMRC${EVAL_LIMIT} eval k=${k} gpus=${gpus} at $(date -Is)"
  CUDA_VISIBLE_DEVICES="${gpus}" \
    ROOT="${RUNTIME_ROOT}" \
    VENV="${VENV}" \
    SOURCE_RUN_ROOT="${SOURCE_RUN_ROOT}" \
    PATCH_DIR="${FINAL_PATCH_DIR}" \
    ROUTE_PATH="${route}" \
    RUN_ROOT="${out}" \
    DATASETS=cmrc2018 \
    EVAL_LIMIT="${EVAL_LIMIT}" \
    MAX_NEW_TOKENS="${MAX_NEW_TOKENS}" \
    WAIT_EVAL_FREE_MB="${WAIT_EVAL_FREE_MB}" \
    AUTO_GPU_COUNT=2 \
    INSTALL_EDITABLE=0 \
    OMP_NUM_THREADS="${OMP_NUM_THREADS}" \
    MKL_NUM_THREADS="${MKL_NUM_THREADS}" \
    OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS}" \
    nice -n 10 bash scripts/run_ser_scd_icml_eval_existing_route.sh
  echo "finished candidate CMRC${EVAL_LIMIT} eval k=${k} at $(date -Is)"
}

run_eval 11 0,1 & p1=$!
run_eval 08 2,3 & p2=$!
wait "${p1}" "${p2}"
run_eval 05 0,1 & p3=$!
run_eval 06 2,3 & p4=$!
wait "${p3}" "${p4}"
run_eval 10 0,1

python scripts/select_quality_gated_scd_route.py \
  --eval-glob "${EVAL_ROOT}/eval_k*" \
  --direction lite_to_7b \
  --dataset cmrc2018 \
  --metric f1 \
  --retention-threshold 0.95 \
  --out-report "${REPORT_DIR}/${REPORT_PREFIX}_ret95.json" \
  --out-route "${REPORT_DIR}/selected_route_ret95.json" || true
python scripts/select_quality_gated_scd_route.py \
  --eval-glob "${EVAL_ROOT}/eval_k*" \
  --direction lite_to_7b \
  --dataset cmrc2018 \
  --metric f1 \
  --retention-threshold 0.90 \
  --out-report "${REPORT_DIR}/${REPORT_PREFIX}_ret90.json" \
  --out-route "${REPORT_DIR}/selected_route_ret90.json" || true

echo "candidate postprocess finished at $(date -Is)"
