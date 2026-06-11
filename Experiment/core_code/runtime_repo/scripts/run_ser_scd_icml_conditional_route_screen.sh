#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
VENV="${VENV:-/disk_n/c101-3-mql/venvs/A800}"
SOURCE_RUN_ROOT="${SOURCE_RUN_ROOT:-/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_paper_mixed_repro_20260608_013851}"
PATCH_DIR="${PATCH_DIR:?Set PATCH_DIR to a merged Patch checkpoint}"
REUSE_DIR="${REUSE_DIR:-}"
BASE_ROUTE="${BASE_ROUTE:?Set BASE_ROUTE to the current all-Patch or mixed route}"
RUN_ROOT="${RUN_ROOT:-/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_conditional_route_screen_$(date +%Y%m%d_%H%M%S)}"
DIRECTION="${DIRECTION:-lite_to_7b}"
SEED="${SEED:-42}"
DATASETS="${DATASETS:-cmrc2018}"
REPORT_DATASET="${REPORT_DATASET:-cmrc2018}"
CMRC_SPLIT="${CMRC_SPLIT:-dev}"
EVAL_LIMIT="${EVAL_LIMIT:-10}"
MAX_LENGTH="${MAX_LENGTH:-1024}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"
BANDWIDTH_GBPS="${BANDWIDTH_GBPS:-200}"
GPU_PAIRS="${GPU_PAIRS:-0,1;2,3}"
PATCH_CKPT_DEVICE="${PATCH_CKPT_DEVICE:-cpu}"
RUNTIME_PATCH_WINDOW_SIZE="${RUNTIME_PATCH_WINDOW_SIZE:-8}"
RUNTIME_PATCH_PERSIST_SHARDS="${RUNTIME_PATCH_PERSIST_SHARDS:-false}"
RUNTIME_PATCH_PERSIST_MODULES="${RUNTIME_PATCH_PERSIST_MODULES:-false}"
CANDIDATE_LAYERS="${CANDIDATE_LAYERS:-}"
PREFIX_BUDGETS="${PREFIX_BUDGETS:-}"
REUSE_SETS="${REUSE_SETS:-}"
EVAL_CONFIG_TEMPLATE="${EVAL_CONFIG_TEMPLATE:-configs/eval_scd_icml_no_fallback_no_recompute.json}"
WAIT_EVAL_FREE_MB="${WAIT_EVAL_FREE_MB:-18000}"

cd "${ROOT}"
source "${VENV}/bin/activate"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
export OPENBLAS_MAIN_FREE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

CANDIDATE_DIR="${RUN_ROOT}/${DIRECTION}/route/conditional_candidates"
EVAL_ROOT="${RUN_ROOT}/${DIRECTION}/conditional_evals"
REPORT_DIR="${RUN_ROOT}/${DIRECTION}/conditional_report"
mkdir -p "${CANDIDATE_DIR}" "${EVAL_ROOT}" "${REPORT_DIR}" "${RUN_ROOT}/logs"

python scripts/generate_conditional_reuse_routes.py \
  --base-route "${BASE_ROUTE}" \
  --out-dir "${CANDIDATE_DIR}" \
  --candidate-layers "${CANDIDATE_LAYERS}" \
  --prefix-budgets "${PREFIX_BUDGETS}" \
  --reuse-sets "${REUSE_SETS}"

mapfile -t ROUTES < <(python - "${CANDIDATE_DIR}/conditional_reuse_manifest.json" <<'PY'
import json, sys
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8-sig"))
for row in payload.get("candidates", []):
    print(row["route"])
PY
)
if [[ "${#ROUTES[@]}" -eq 0 ]]; then
  echo "no candidate routes generated" >&2
  exit 2
fi

IFS=";" read -r -a PAIRS <<< "${GPU_PAIRS}"
if [[ "${#PAIRS[@]}" -eq 0 ]]; then
  echo "GPU_PAIRS is empty" >&2
  exit 2
fi

run_one() {
  local route="$1"
  local gpus="$2"
  local name
  name="$(basename "${route}" .json)"
  local out="${EVAL_ROOT}/${name}"
  echo "starting ${name} on CUDA_VISIBLE_DEVICES=${gpus} at $(date -Is)"
  CUDA_VISIBLE_DEVICES="${gpus}" \
    ROOT="${ROOT}" \
    VENV="${VENV}" \
    SOURCE_RUN_ROOT="${SOURCE_RUN_ROOT}" \
    PATCH_DIR="${PATCH_DIR}" \
    REUSE_DIR="${REUSE_DIR}" \
    ROUTE_PATH="${route}" \
    RUN_ROOT="${out}" \
    DIRECTION="${DIRECTION}" \
    SEED="${SEED}" \
    DATASETS="${DATASETS}" \
    CMRC_SPLIT="${CMRC_SPLIT}" \
    EVAL_LIMIT="${EVAL_LIMIT}" \
    MAX_LENGTH="${MAX_LENGTH}" \
    MAX_NEW_TOKENS="${MAX_NEW_TOKENS}" \
    BANDWIDTH_GBPS="${BANDWIDTH_GBPS}" \
    PATCH_CKPT_DEVICE="${PATCH_CKPT_DEVICE}" \
    RUNTIME_PATCH_WINDOW_SIZE="${RUNTIME_PATCH_WINDOW_SIZE}" \
    RUNTIME_PATCH_PERSIST_SHARDS="${RUNTIME_PATCH_PERSIST_SHARDS}" \
    RUNTIME_PATCH_PERSIST_MODULES="${RUNTIME_PATCH_PERSIST_MODULES}" \
    EVAL_CONFIG_TEMPLATE="${EVAL_CONFIG_TEMPLATE}" \
    WAIT_EVAL_FREE_MB="${WAIT_EVAL_FREE_MB}" \
    AUTO_GPU_COUNT=2 \
    INSTALL_EDITABLE=0 \
    OMP_NUM_THREADS="${OMP_NUM_THREADS}" \
    MKL_NUM_THREADS="${MKL_NUM_THREADS}" \
    OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS}" \
    nice -n 10 bash scripts/run_ser_scd_icml_eval_existing_route.sh \
      > "${RUN_ROOT}/logs/${name}.stdout.log" \
      2> "${RUN_ROOT}/logs/${name}.stderr.log"
  echo "finished ${name} at $(date -Is)"
}

QUEUE_FILE="${RUN_ROOT}/logs/route_queue.txt"
QUEUE_LOCK="${RUN_ROOT}/logs/route_queue.lock"
FAILED_FILE="${RUN_ROOT}/logs/route_queue.failed"
printf "%s\n" "${ROUTES[@]}" > "${QUEUE_FILE}"
rm -f "${FAILED_FILE}"

dequeue_route() {
  local route=""
  {
    flock -x 200
    route="$(python - "${QUEUE_FILE}" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1])
items = [line.rstrip("\n") for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
if items:
    print(items[0])
    path.write_text(("\n".join(items[1:]) + "\n") if len(items) > 1 else "", encoding="utf-8")
PY
)"
  } 200>"${QUEUE_LOCK}"
  [[ -n "${route}" ]] || return 1
  printf "%s\n" "${route}"
}

worker_loop() {
  local pair="$1"
  local worker_id="$2"
  local route
  echo "worker ${worker_id} ready on CUDA_VISIBLE_DEVICES=${pair} at $(date -Is)"
  while route="$(dequeue_route)"; do
    if ! run_one "${route}" "${pair}"; then
      {
        flock -x 201
        echo "worker=${worker_id} gpus=${pair} route=${route} failed_at=$(date -Is)" >> "${FAILED_FILE}"
      } 201>"${FAILED_FILE}.lock"
      return 1
    fi
  done
  echo "worker ${worker_id} idle on CUDA_VISIBLE_DEVICES=${pair} at $(date -Is)"
}

pids=()
for pair_idx in "${!PAIRS[@]}"; do
  worker_loop "${PAIRS[$pair_idx]}" "${pair_idx}" &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "${pid}" || status=1
done
if [[ "${status}" -ne 0 ]]; then
  echo "one or more route workers failed" >&2
  [[ -f "${FAILED_FILE}" ]] && cat "${FAILED_FILE}" >&2
  exit "${status}"
fi

python scripts/collect_route_eval_metrics.py \
  --eval-glob "${EVAL_ROOT}/*" \
  --direction "${DIRECTION}" \
  --dataset "${REPORT_DATASET}" \
  --method scd_full \
  --out-json "${REPORT_DIR}/route_metrics.json" \
  --out-jsonl "${REPORT_DIR}/route_metrics.jsonl"

echo "conditional route screen finished."
echo "RUN_ROOT=${RUN_ROOT}"
echo "metrics=${REPORT_DIR}/route_metrics.json"
