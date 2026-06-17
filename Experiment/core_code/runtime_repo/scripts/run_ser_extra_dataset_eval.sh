#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
VENV="${VENV:-/disk_n/c101-3-mql/venvs/A800}"
DATA_ROOT="${DATA_ROOT:-/disk_n/c101-3-mql/projects/SCD-ICML/kv/snapshots/scd-github-f7d7854/data}"
BANK_RUN_ROOT="${BANK_RUN_ROOT:-/disk_n/c101-3-mql/projects/SCD-ICML/runs/open_build_native_fixedroute_cmrc200_20260616_133222}"

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_BASE="${RUN_BASE:-/disk_n/c101-3-mql/projects/SCD-ICML/runs/extra_datasets_cmrcbank_${STAMP}}"
HOTPOT_LIMIT="${HOTPOT_LIMIT:-100}"
WIKITEXT_LIMIT="${WIKITEXT_LIMIT:-200}"
MAX_LENGTH="${MAX_LENGTH:-0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"
BANDWIDTH_GBPS="${BANDWIDTH_GBPS:-200}"
QUANT_GROUP_SIZE="${QUANT_GROUP_SIZE:-128}"
DTYPE="${DTYPE:-bfloat16}"
SEED="${SEED:-42}"
RUN_HOTPOT="${RUN_HOTPOT:-1}"
RUN_WIKITEXT="${RUN_WIKITEXT:-1}"
WAIT="${WAIT:-0}"

HOTPOT_PRODUCER_GPU="${HOTPOT_PRODUCER_GPU:-0}"
HOTPOT_CONSUMER_GPU="${HOTPOT_CONSUMER_GPU:-1}"
WIKITEXT_PRODUCER_GPU="${WIKITEXT_PRODUCER_GPU:-2}"
WIKITEXT_CONSUMER_GPU="${WIKITEXT_CONSUMER_GPU:-3}"

MODELS_CONFIG="${MODELS_CONFIG:-configs/models_ser_mistrallite_to_mistral7_mql.json}"
EVAL_CONFIG="${EVAL_CONFIG:-${BANK_RUN_ROOT}/configs/eval_fixed_patchbudget13.json}"
ROUTE_PATH="${ROUTE_PATH:-${BANK_RUN_ROOT}/route/route_cmrc2018_fixed_patchbudget13.json}"
REUSE_CKPT="${REUSE_CKPT:-${BANK_RUN_ROOT}/ckpt_reuse/layerwise_seed42}"
PATCH_CKPT="${PATCH_CKPT:-${BANK_RUN_ROOT}/ckpt_patch/native_alllayers_seed42}"

PYTHON_BIN="${VENV}/bin/python"
mkdir -p "${RUN_BASE}/logs" "${RUN_BASE}/configs"

require_file() {
  local path="$1"
  if [ ! -f "${path}" ]; then
    echo "missing required file: ${path}" >&2
    exit 1
  fi
}

require_dir() {
  local path="$1"
  if [ ! -d "${path}" ]; then
    echo "missing required directory: ${path}" >&2
    exit 1
  fi
}

require_file "${ROOT}/${MODELS_CONFIG}"
require_file "${EVAL_CONFIG}"
require_file "${ROUTE_PATH}"
require_dir "${REUSE_CKPT}"
require_dir "${PATCH_CKPT}"
require_dir "${DATA_ROOT}"
require_file "${PYTHON_BIN}"

"${PYTHON_BIN}" - <<PY
import json
from pathlib import Path

route_path = Path("${ROUTE_PATH}")
route = json.loads(route_path.read_text(encoding="utf-8-sig"))
num_layers = int(route.get("num_layers") or 32)
reuse = {int(v) for v in route.get("reuse_layers", [])}
patch = {int(v) for v in route.get("selected_patch_layers", route.get("patch_layers", []))}
fallback = {int(v) for v in route.get("fallback_layers", [])}
selected_recompute = {int(v) for v in route.get("selected_recompute_layers", [])}
recompute_map = route.get("recompute_segment_length_by_layer", {}) or {}
errors = []
if fallback:
    errors.append(f"fallback_layers must be empty, got {sorted(fallback)}")
if selected_recompute or recompute_map:
    errors.append(
        "recompute must be disabled, "
        f"got selected_recompute_layers={sorted(selected_recompute)} "
        f"recompute_segment_length_by_layer={recompute_map}"
    )
overlap = reuse & patch
if overlap:
    errors.append(f"reuse_layers and patch_layers overlap: {sorted(overlap)}")
missing = set(range(num_layers)) - (reuse | patch)
if missing:
    errors.append(f"reuse_layers + patch_layers must cover all layers, missing {sorted(missing)}")
if errors:
    raise SystemExit("invalid strict no-fallback/no-recompute route "
                     + str(route_path) + ": " + "; ".join(errors))
print(
    "validated strict route:",
    route_path,
    f"reuse={len(reuse)}",
    f"patch={len(patch)}",
    "fallback=0",
    "recompute=0",
)
PY

cat > "${RUN_BASE}/run_config.json" <<EOF
{
  "run_base": "${RUN_BASE}",
  "root": "${ROOT}",
  "data_root": "${DATA_ROOT}",
  "bank_run_root": "${BANK_RUN_ROOT}",
  "models_config": "${MODELS_CONFIG}",
  "eval_config": "${EVAL_CONFIG}",
  "route": "${ROUTE_PATH}",
  "reuse_ckpt": "${REUSE_CKPT}",
  "patch_ckpt": "${PATCH_CKPT}",
  "hotpot_limit": ${HOTPOT_LIMIT},
  "wikitext_limit": ${WIKITEXT_LIMIT},
  "max_length": ${MAX_LENGTH},
  "max_new_tokens": ${MAX_NEW_TOKENS},
  "bandwidth_gbps": ${BANDWIDTH_GBPS},
  "quant_group_size": ${QUANT_GROUP_SIZE},
  "dtype": "${DTYPE}",
  "seed": ${SEED}
}
EOF

start_eval() {
  local dataset="$1"
  local limit="$2"
  local out_name="$3"
  local producer_gpu="$4"
  local consumer_gpu="$5"
  local out_dir="${RUN_BASE}/${out_name}"
  local log_prefix="${RUN_BASE}/logs/${out_name}"
  local eval_cfg="${RUN_BASE}/configs/eval_${out_name}.json"
  "${PYTHON_BIN}" - <<PY
import json
from pathlib import Path

src = Path("${EVAL_CONFIG}")
dst = Path("${eval_cfg}")
cfg = json.loads(src.read_text(encoding="utf-8-sig"))
runtime_eval = cfg.setdefault("runtime_eval", {})
runtime_eval["max_eval_samples"] = int("${limit}")
dst.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\\n", encoding="utf-8")
PY
  mkdir -p "${out_dir}"
  echo "start ${dataset}: out=${out_dir} gpus=${producer_gpu},${consumer_gpu} limit=${limit} eval_config=${eval_cfg}"
  nohup env PYTHONPATH="${ROOT}/src" "${PYTHON_BIN}" -m scd_repro.cli.eval_text_strict \
    --dataset "${dataset}" \
    --models-config "${MODELS_CONFIG}" \
    --eval-config "${eval_cfg}" \
    --data-root "${DATA_ROOT}" \
    --route "${ROUTE_PATH}" \
    --reuse-ckpt "${REUSE_CKPT}" \
    --patch-ckpt "${PATCH_CKPT}" \
    --out "${out_dir}" \
    --limit "${limit}" \
    --max-length "${MAX_LENGTH}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --bandwidth-gbps "${BANDWIDTH_GBPS}" \
    --quant-group-size "${QUANT_GROUP_SIZE}" \
    --producer-device "cuda:${producer_gpu}" \
    --consumer-device "cuda:${consumer_gpu}" \
    --dtype "${DTYPE}" \
    --seed "${SEED}" \
    --log "${log_prefix}.cli.log" \
    > "${log_prefix}.stdout" 2> "${log_prefix}.stderr" &
  echo "$!" > "${log_prefix}.pid"
}

pids=()
if [ "${RUN_HOTPOT}" = "1" ]; then
  start_eval "hotpotqa" "${HOTPOT_LIMIT}" "hotpotqa_full_n${HOTPOT_LIMIT}" "${HOTPOT_PRODUCER_GPU}" "${HOTPOT_CONSUMER_GPU}"
  pids+=("$(cat "${RUN_BASE}/logs/hotpotqa_full_n${HOTPOT_LIMIT}.pid")")
fi
if [ "${RUN_WIKITEXT}" = "1" ]; then
  start_eval "wikitext2" "${WIKITEXT_LIMIT}" "wikitext2_n${WIKITEXT_LIMIT}" "${WIKITEXT_PRODUCER_GPU}" "${WIKITEXT_CONSUMER_GPU}"
  pids+=("$(cat "${RUN_BASE}/logs/wikitext2_n${WIKITEXT_LIMIT}.pid")")
fi

printf "%s\n" "${pids[@]}" > "${RUN_BASE}/pids.txt"
echo "RUN_BASE=${RUN_BASE}"
echo "PIDS=$(paste -sd, "${RUN_BASE}/pids.txt")"

if [ "${WAIT}" = "1" ]; then
  for pid in "${pids[@]}"; do
    wait "${pid}"
  done
fi
