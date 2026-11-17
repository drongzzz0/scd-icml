#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
VENV="${VENV:-/disk_n/c101-3-mql/venvs/A800}"
DATA_ROOT="${DATA_ROOT:-/disk_n/c101-3-mql/projects/SCD-ICML/kv/snapshots/scd-github-f7d7854}"

SOURCE_RUN_ROOT="${SOURCE_RUN_ROOT:-}"
PATCH_DIR="${PATCH_DIR:-}"
ROUTE_PATH="${ROUTE_PATH:-}"
RUN_ROOT="${RUN_ROOT:-/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_eval_existing_route_$(date +%Y%m%d_%H%M%S)}"
DIRECTION="${DIRECTION:-lite_to_7b}"
SEED="${SEED:-42}"
EVAL_LIMIT="${EVAL_LIMIT:-200}"
MAX_LENGTH="${MAX_LENGTH:-0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"
BANDWIDTH_GBPS="${BANDWIDTH_GBPS:-200}"
QUANT_GROUP_SIZE="${QUANT_GROUP_SIZE:-128}"
DATASETS="${DATASETS:-cmrc2018,hotpotqa,wikitext2}"
CMRC_SPLIT="${CMRC_SPLIT:-dev}"
EVAL_CONFIG_TEMPLATE="${EVAL_CONFIG_TEMPLATE:-configs/eval_scd_icml_no_fallback_no_recompute.json}"
MODELS_CONFIG="${MODELS_CONFIG:-configs/models_ser_mistrallite_to_mistral7_mql.json}"
REUSE_DIR="${REUSE_DIR:-}"
PRODUCER_DEVICE="${PRODUCER_DEVICE:-cuda:0}"
CONSUMER_DEVICE="${CONSUMER_DEVICE:-cuda:1}"
WAIT_GPU_POLL_SEC="${WAIT_GPU_POLL_SEC:-120}"
WAIT_EVAL_FREE_MB="${WAIT_EVAL_FREE_MB:-18000}"
AUTO_GPU_COUNT="${AUTO_GPU_COUNT:-2}"
INSTALL_EDITABLE="${INSTALL_EDITABLE:-0}"
PATCH_CKPT_DEVICE="${PATCH_CKPT_DEVICE:-cpu}"
RUNTIME_PATCH_WINDOW_SIZE="${RUNTIME_PATCH_WINDOW_SIZE:-8}"
RUNTIME_PATCH_PERSIST_SHARDS="${RUNTIME_PATCH_PERSIST_SHARDS:-false}"
RUNTIME_PATCH_PERSIST_MODULES="${RUNTIME_PATCH_PERSIST_MODULES:-false}"

if [[ -z "${SOURCE_RUN_ROOT}" && -z "${REUSE_DIR}" ]]; then
  echo "Set SOURCE_RUN_ROOT or REUSE_DIR." >&2
  exit 2
fi
if [[ -z "${PATCH_DIR}" || ! -d "${PATCH_DIR}" ]]; then
  echo "Set PATCH_DIR to an existing merged Patch checkpoint." >&2
  exit 2
fi
if [[ -z "${ROUTE_PATH}" || ! -f "${ROUTE_PATH}" ]]; then
  echo "Set ROUTE_PATH to an existing route JSON." >&2
  exit 2
fi
if [[ -z "${REUSE_DIR}" ]]; then
  REUSE_DIR="${SOURCE_RUN_ROOT}/${DIRECTION}/ckpt_reuse/cmrc2018_seed${SEED}"
fi
if [[ ! -d "${REUSE_DIR}" ]]; then
  echo "missing Reuse checkpoint: ${REUSE_DIR}" >&2
  exit 2
fi

pick_free_gpus() {
  local required_free_mb="$1"
  local count="$2"
  nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits | \
    awk -F',' -v required="${required_free_mb}" '{gsub(/ /, "", $1); gsub(/ /, "", $2); gsub(/ /, "", $3); if ($2 >= required) print $1 "," $2 "," $3}' | \
    sort -t',' -k2,2nr -k3,3n | \
    awk -F',' -v count="${count}" 'NR <= count {print $1}'
}

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  while true; do
    mapfile -t _picked_gpus < <(pick_free_gpus "${WAIT_EVAL_FREE_MB}" "${AUTO_GPU_COUNT}")
    if [[ "${#_picked_gpus[@]}" -ge "${AUTO_GPU_COUNT}" ]]; then
      export CUDA_VISIBLE_DEVICES="$(IFS=','; echo "${_picked_gpus[*]}")"
      echo "selected CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
      break
    fi
    echo "waiting to select ${AUTO_GPU_COUNT} GPU(s) with free memory >= ${WAIT_EVAL_FREE_MB}MB"
    nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits
    sleep "${WAIT_GPU_POLL_SEC}"
  done
fi

mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/tmp_cfg" "${RUN_ROOT}/${DIRECTION}/route" "${RUN_ROOT}/${DIRECTION}/strict_eval"
cd "${ROOT}"
source "${VENV}/bin/activate"
if [[ "${INSTALL_EDITABLE}" == "1" ]]; then
  python -m pip install -e .
fi
export PYTHONPATH="${ROOT}/src"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

TARGET_ROUTE="${RUN_ROOT}/${DIRECTION}/route/route_cmrc2018_seed${SEED}.json"
cp "${ROUTE_PATH}" "${TARGET_ROUTE}"
python scripts/validate_scd_route.py --route "${TARGET_ROUTE}" --require-mixed

EVAL_CONFIG="${RUN_ROOT}/tmp_cfg/eval_${DIRECTION}.json"
python - \
  "${EVAL_CONFIG_TEMPLATE}" \
  "${EVAL_CONFIG}" \
  "${EVAL_LIMIT}" \
  "${BANDWIDTH_GBPS}" \
  "${PATCH_CKPT_DEVICE}" \
  "${RUNTIME_PATCH_WINDOW_SIZE}" \
  "${RUNTIME_PATCH_PERSIST_SHARDS}" \
  "${RUNTIME_PATCH_PERSIST_MODULES}" \
  "${CMRC_SPLIT}" <<'PY'
import json
import sys
from pathlib import Path

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
limit = int(sys.argv[3])
bandwidth = float(sys.argv[4])
patch_ckpt_device = sys.argv[5].strip()
runtime_patch_window_size = sys.argv[6].strip()
runtime_patch_persist_shards = sys.argv[7].strip()
runtime_patch_persist_modules = sys.argv[8].strip()
cmrc_split = sys.argv[9].strip() or "dev"

def parse_bool(value):
    text = str(value).strip().lower()
    if text in {"", "none", "null"}:
        return None
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"invalid boolean override: {value!r}")

cfg = json.loads(src.read_text(encoding="utf-8-sig"))
runtime_eval = cfg.setdefault("runtime_eval", {})
runtime_eval["max_eval_samples"] = limit
runtime_eval["cmrc_split"] = cmrc_split
if patch_ckpt_device:
    runtime_eval["patch_ckpt_device"] = patch_ckpt_device
if runtime_patch_window_size:
    runtime_eval["runtime_patch_window_size"] = int(runtime_patch_window_size)
persist_shards = parse_bool(runtime_patch_persist_shards)
persist_modules = parse_bool(runtime_patch_persist_modules)
if persist_shards is not None:
    runtime_eval["runtime_patch_persist_shards"] = persist_shards
if persist_modules is not None:
    runtime_eval["runtime_patch_persist_modules"] = persist_modules
cfg.setdefault("network", {})["bandwidth_gbps"] = bandwidth
dst.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY

cat > "${RUN_ROOT}/run_config.json" <<EOF
{
  "root": "${ROOT}",
  "source_run_root": "${SOURCE_RUN_ROOT}",
  "reuse_dir": "${REUSE_DIR}",
  "patch_dir": "${PATCH_DIR}",
  "route_source": "${ROUTE_PATH}",
  "route": "${TARGET_ROUTE}",
  "run_root": "${RUN_ROOT}",
  "direction": "${DIRECTION}",
  "datasets": "${DATASETS}",
  "cmrc_split": "${CMRC_SPLIT}",
  "eval_limit": ${EVAL_LIMIT},
  "max_length": ${MAX_LENGTH},
  "max_new_tokens": ${MAX_NEW_TOKENS},
  "cuda_visible_devices": "${CUDA_VISIBLE_DEVICES:-}",
  "producer_device": "${PRODUCER_DEVICE}",
  "consumer_device": "${CONSUMER_DEVICE}",
  "patch_ckpt_device": "${PATCH_CKPT_DEVICE}",
  "runtime_patch_window_size": "${RUNTIME_PATCH_WINDOW_SIZE}",
  "runtime_patch_persist_shards": "${RUNTIME_PATCH_PERSIST_SHARDS}",
  "runtime_patch_persist_modules": "${RUNTIME_PATCH_PERSIST_MODULES}"
}
EOF

run_logged() {
  local name="$1"
  shift
  local stage_dir="${RUN_ROOT}/logs/${name}"
  mkdir -p "${stage_dir}"
  echo "+ $*" | tee "${stage_dir}/command.txt"
  /usr/bin/time -v -o "${stage_dir}/time.txt" "$@" >"${stage_dir}/stdout.log" 2>"${stage_dir}/stderr.log"
}

IFS=',' read -r -a DATASET_LIST <<< "${DATASETS}"
for dataset in "${DATASET_LIST[@]}"; do
  dataset="$(echo "${dataset}" | xargs)"
  [[ -z "${dataset}" ]] && continue
  OUT_DIR="${RUN_ROOT}/${DIRECTION}/strict_eval/${dataset}"
  run_logged "${DIRECTION}_eval_${dataset}" \
    python -m scd_repro.cli.eval_text_strict \
      --dataset "${dataset}" \
      --models-config "${MODELS_CONFIG}" \
      --eval-config "${EVAL_CONFIG}" \
      --data-root "${DATA_ROOT}" \
      --route "${TARGET_ROUTE}" \
      --reuse-ckpt "${REUSE_DIR}" \
      --patch-ckpt "${PATCH_DIR}" \
      --out "${OUT_DIR}" \
      --limit "${EVAL_LIMIT}" \
      --max-length "${MAX_LENGTH}" \
      --max-new-tokens "${MAX_NEW_TOKENS}" \
      --bandwidth-gbps "${BANDWIDTH_GBPS}" \
      --quant-group-size "${QUANT_GROUP_SIZE}" \
      --producer-device "${PRODUCER_DEVICE}" \
      --consumer-device "${CONSUMER_DEVICE}" \
      --dtype bfloat16 \
      --cmrc-split "${CMRC_SPLIT}" \
      --seed "${SEED}"
done

python -m scd_repro.cli.summarize_mistral_repro \
  --run-root "${RUN_ROOT}" \
  --eval-config "${EVAL_CONFIG_TEMPLATE}" \
  --expected-directions "${DIRECTION}" \
  --expected-datasets "${DATASETS}"

echo "existing-route evaluation finished."
echo "RUN_ROOT=${RUN_ROOT}"
