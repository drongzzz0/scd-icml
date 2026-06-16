#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
VENV="${VENV:-/disk_n/c101-3-mql/venvs/A800}"
DATA_SOURCE_DIR="${DATA_SOURCE_DIR:-/disk_n/c101-3-mql/projects/SCD-ICML/kv/snapshots/scd-github-f7d7854/data/cmrc2018}"

STAMP="$(date +%Y%m%d-%H%M%S)"
RUN_ROOT="${RUN_ROOT:-${ROOT}/runs/build_reuse_patch_route_cmrc2018_${STAMP}}"
SEED="${SEED:-42}"
CALIB_SAMPLES="${CALIB_SAMPLES:-200}"
EVAL_SAMPLES="${EVAL_SAMPLES:-200}"
MAX_LENGTH="${MAX_LENGTH:-0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"
BANDWIDTH_GBPS="${BANDWIDTH_GBPS:-200}"
QUANT_GROUP_SIZE="${QUANT_GROUP_SIZE:-128}"
ROUTE_PATCH_BUDGET="${ROUTE_PATCH_BUDGET:-13}"
ROUTE_TOP_M_CANDIDATES="${ROUTE_TOP_M_CANDIDATES:-0}"
ROUTE_MAX_EVALS="${ROUTE_MAX_EVALS:-0}"
CAPTURE_SKIP_VALIDATE="${CAPTURE_SKIP_VALIDATE:-1}"
RESUME="${RESUME:-0}"

MODELS_CONFIG="${MODELS_CONFIG:-configs/models_ser_mistrallite_to_mistral7_mql.json}"
CAPTURE_DATA_CONFIG="${CAPTURE_DATA_CONFIG:-configs/data_ser.yaml}"
REUSE_TRAIN_CONFIG="${REUSE_TRAIN_CONFIG:-configs/train_reuse_mistral_self_layerwise_20260404.json}"
PATCH_TRAIN_CONFIG="${PATCH_TRAIN_CONFIG:-configs/train_scd_icml_patch_semantic_alllayers_20260616.json}"
ROUTE_EVAL_TEMPLATE="${ROUTE_EVAL_TEMPLATE:-configs/eval_scd_icml_no_fallback_no_recompute.json}"

MEASURE_DIR="${RUN_ROOT}/measure"
CALIB_ROOT="${RUN_ROOT}/data_calib_train"
EVAL_ROOT="${RUN_ROOT}/data_eval_dev"
CALIB_CMRC_DIR="${CALIB_ROOT}/cmrc2018"
EVAL_CMRC_DIR="${EVAL_ROOT}/cmrc2018"
PROMPTS_JSON="${RUN_ROOT}/prompts/cmrc2018_train_prompts_seed${SEED}_n${CALIB_SAMPLES}.json"
PAIRS_DIR="${RUN_ROOT}/pairs/cmrc2018_train_seed${SEED}_n${CALIB_SAMPLES}"
REUSE_DIR="${RUN_ROOT}/ckpt_reuse/layerwise_seed${SEED}"
PROFILE_PATH="${RUN_ROOT}/profile/profile_cmrc2018_train_seed${SEED}.json"
PATCH_DIR="${RUN_ROOT}/ckpt_patch/semantic_alllayers_seed${SEED}"
GENERATED_EVAL_CONFIG="${RUN_ROOT}/configs/eval_task_route_patchbudget${ROUTE_PATCH_BUDGET}.json"
ROUTE_PATH="${RUN_ROOT}/route/route_cmrc2018_tasklevel_patchbudget${ROUTE_PATCH_BUDGET}.json"
STRICT_OUT="${RUN_ROOT}/strict_eval/cmrc2018_dev_full_vs_scd"

mkdir -p \
  "${MEASURE_DIR}" \
  "${CALIB_CMRC_DIR}" \
  "${EVAL_CMRC_DIR}" \
  "$(dirname "${PROMPTS_JSON}")" \
  "$(dirname "${PROFILE_PATH}")" \
  "$(dirname "${GENERATED_EVAL_CONFIG}")" \
  "$(dirname "${ROUTE_PATH}")" \
  "${STRICT_OUT}"

require_file() {
  local path="$1"
  if [ ! -f "${path}" ]; then
    echo "missing required file: ${path}" >&2
    exit 1
  fi
}

require_file "${DATA_SOURCE_DIR}/cmrc2018_train.json"
require_file "${DATA_SOURCE_DIR}/cmrc2018_dev.json"
require_file "${ROOT}/${MODELS_CONFIG}"
require_file "${ROOT}/${CAPTURE_DATA_CONFIG}"
require_file "${ROOT}/${REUSE_TRAIN_CONFIG}"
require_file "${ROOT}/${PATCH_TRAIN_CONFIG}"
require_file "${ROOT}/${ROUTE_EVAL_TEMPLATE}"

cd "${ROOT}"
source "${VENV}/bin/activate"
export PYTHONPATH="${ROOT}/src"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"

nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu --format=csv > "${MEASURE_DIR}/gpu_initial.csv"

pick_gpus() {
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits | \
    awk -F',' '{idx=$1; mem=$2; util=$3; gsub(/ /, "", idx); gsub(/ /, "", mem); gsub(/ /, "", util); print util, mem, idx}' | \
    sort -n -k1,1 -k2,2 | \
    awk '{print $3}'
}

if [ -z "${PRODUCER_GPU:-}" ] || [ -z "${CONSUMER_GPU:-}" ] || [ -z "${TRAIN_GPU:-}" ]; then
  mapfile -t _picked_gpus < <(pick_gpus)
  if [ "${#_picked_gpus[@]}" -lt 2 ]; then
    echo "need at least two visible GPUs for producer/consumer capture and strict eval" >&2
    exit 1
  fi
  PRODUCER_GPU="${PRODUCER_GPU:-${_picked_gpus[0]}}"
  CONSUMER_GPU="${CONSUMER_GPU:-${_picked_gpus[1]}}"
  TRAIN_GPU="${TRAIN_GPU:-${_picked_gpus[2]:-${_picked_gpus[0]}}}"
fi

PRODUCER_DEVICE="cuda:${PRODUCER_GPU}"
CONSUMER_DEVICE="cuda:${CONSUMER_GPU}"
TRAIN_DEVICE="cuda:${TRAIN_GPU}"

cat > "${MEASURE_DIR}/gpu_assignment.json" <<EOF
{
  "producer_gpu": ${PRODUCER_GPU},
  "consumer_gpu": ${CONSUMER_GPU},
  "train_gpu": ${TRAIN_GPU},
  "producer_device": "${PRODUCER_DEVICE}",
  "consumer_device": "${CONSUMER_DEVICE}",
  "train_device": "${TRAIN_DEVICE}"
}
EOF

cp "${DATA_SOURCE_DIR}/cmrc2018_train.json" "${CALIB_CMRC_DIR}/cmrc2018_train.json"
cp "${DATA_SOURCE_DIR}/cmrc2018_train.json" "${CALIB_CMRC_DIR}/cmrc2018_dev.json"
cp "${DATA_SOURCE_DIR}/cmrc2018_train.json" "${EVAL_CMRC_DIR}/cmrc2018_train.json"
cp "${DATA_SOURCE_DIR}/cmrc2018_dev.json" "${EVAL_CMRC_DIR}/cmrc2018_dev.json"

python - <<PY
import json
from pathlib import Path
from scd_repro.cmrc import load_cmrc2018_samples

rows = load_cmrc2018_samples("${CALIB_ROOT}", limit=int("${CALIB_SAMPLES}"))
out = Path("${PROMPTS_JSON}")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps([row.prompt for row in rows], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"prepared_train_prompts={len(rows)}")
PY

python - <<PY
import json
from pathlib import Path

template_path = Path("${ROOT}") / "${ROUTE_EVAL_TEMPLATE}"
cfg = json.loads(template_path.read_text(encoding="utf-8-sig"))
route = cfg.setdefault("route", {})
route["enable_task_level_profile"] = True
route["patch_budget"] = int("${ROUTE_PATCH_BUDGET}")
route["top_m_candidates"] = int("${ROUTE_TOP_M_CANDIDATES}")
route["max_evals"] = int("${ROUTE_MAX_EVALS}")
route["allow_patch_only"] = True
route["allow_patch_plus_recompute"] = False
route["enable_short_recompute"] = False
route["recompute_segment_lengths"] = []
route["recompute_eval_mode"] = "disabled"
network = cfg.setdefault("network", {})
network["bandwidth_gbps"] = float("${BANDWIDTH_GBPS}")
runtime_eval = cfg.setdefault("runtime_eval", {})
for key in list(runtime_eval):
    if key.startswith("run_"):
        runtime_eval[key] = False
runtime_eval["run_full_consumer_baseline"] = True
runtime_eval["run_reuse_only"] = True
runtime_eval["run_scd_full"] = True
runtime_eval["strict_mode"] = True
runtime_eval["fail_on_missing_route_fields"] = True
runtime_eval["fail_on_missing_patch_fields"] = True
runtime_eval["measure_teacher_forced_metrics"] = True
runtime_eval["measure_generation_metrics"] = True
runtime_eval["measure_ttft"] = True
runtime_eval["measure_transfer_bytes"] = True
runtime_eval["max_eval_samples"] = int("${EVAL_SAMPLES}")
out = Path("${GENERATED_EVAL_CONFIG}")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"generated_eval_config={out}")
PY

start_gpu_monitor() {
  local trace_path="$1"
  local gpu_ids_csv="$2"
  (
    echo "timestamp,gpu_index,memory_used_mb,utilization_gpu_pct"
    while true; do
      local ts
      ts="$(date +%s.%N)"
      nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits | \
        while IFS=',' read -r idx mem util; do
          idx="$(echo "${idx}" | xargs)"
          mem="$(echo "${mem}" | xargs)"
          util="$(echo "${util}" | xargs)"
          case ",${gpu_ids_csv}," in
            *,"${idx}",*) echo "${ts},${idx},${mem},${util}" ;;
          esac
        done
      sleep 1
    done
  ) > "${trace_path}" &
  echo $!
}

run_stage() {
  local stage_name="$1"
  local gpu_ids_csv="$2"
  shift 2
  local stage_dir="${MEASURE_DIR}/${stage_name}"
  local stdout_file="${stage_dir}/stdout.log"
  local stderr_file="${stage_dir}/stderr.log"
  local time_file="${stage_dir}/time.txt"
  local trace_file="${stage_dir}/gpu_trace.csv"
  local command_file="${stage_dir}/command.txt"
  local summary_file="${stage_dir}/stage_summary.json"
  local done_file="${stage_dir}/.done"
  local mon_pid=""
  local exit_code=0

  if [ "${RESUME}" = "1" ] && [ -f "${done_file}" ]; then
    echo "skip ${stage_name}: ${done_file} exists"
    return 0
  fi

  rm -rf "${stage_dir}"
  mkdir -p "${stage_dir}"
  printf "%q " "$@" > "${command_file}"
  printf "\n" >> "${command_file}"

  local start_epoch
  local end_epoch
  local start_iso
  local end_iso
  start_epoch="$(python -c 'import time; print(f"{time.time():.6f}")')"
  start_iso="$(date -Is)"

  if [ -n "${gpu_ids_csv}" ]; then
    mon_pid="$(start_gpu_monitor "${trace_file}" "${gpu_ids_csv}")"
  fi

  if /usr/bin/time -v -o "${time_file}" "$@" > "${stdout_file}" 2> "${stderr_file}"; then
    exit_code=0
  else
    exit_code=$?
  fi

  if [ -n "${mon_pid}" ]; then
    kill "${mon_pid}" >/dev/null 2>&1 || true
    wait "${mon_pid}" 2>/dev/null || true
  fi

  end_epoch="$(python -c 'import time; print(f"{time.time():.6f}")')"
  end_iso="$(date -Is)"

  python - <<PY
import csv
import json
import re
from pathlib import Path

trace_file = Path("${trace_file}")
time_file = Path("${time_file}")
peak_by_gpu = {}
if trace_file.exists():
    with trace_file.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx = int(row["gpu_index"])
            mem = float(row["memory_used_mb"])
            peak_by_gpu[str(idx)] = max(mem, peak_by_gpu.get(str(idx), 0.0))
max_rss_kb = None
if time_file.exists():
    text = time_file.read_text(encoding="utf-8", errors="ignore")
    match = re.search(r"Maximum resident set size \\(kbytes\\):\\s*(\\d+)", text)
    if match:
        max_rss_kb = int(match.group(1))
payload = {
    "stage": "${stage_name}",
    "success": ${exit_code} == 0,
    "exit_code": int(${exit_code}),
    "start_time_iso": "${start_iso}",
    "end_time_iso": "${end_iso}",
    "elapsed_sec": float("${end_epoch}") - float("${start_epoch}"),
    "gpu_ids": [int(x) for x in "${gpu_ids_csv}".split(",") if x.strip()],
    "peak_memory_mb_by_gpu": peak_by_gpu,
    "max_rss_kb": max_rss_kb,
    "stdout": str(Path("${stdout_file}")),
    "stderr": str(Path("${stderr_file}")),
    "command": str(Path("${command_file}")),
}
Path("${summary_file}").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY

  if [ "${exit_code}" -ne 0 ]; then
    echo "stage failed: ${stage_name}" >&2
    tail -n 80 "${stderr_file}" >&2 || true
    exit "${exit_code}"
  fi
  touch "${done_file}"
}

CAPTURE_ARGS=(
  python -m scd_repro.cli.capture_pair
  --backend transformers
  --models-config "${MODELS_CONFIG}"
  --data-config "${CAPTURE_DATA_CONFIG}"
  --out "${PAIRS_DIR}"
  --seed "${SEED}"
  --log "${MEASURE_DIR}/capture_pair/capture_pair.cli.log"
  --prompt-file "${PROMPTS_JSON}"
  --max-length "${MAX_LENGTH}"
  --dtype bfloat16
  --model-a-device "${PRODUCER_DEVICE}"
  --model-b-device "${CONSUMER_DEVICE}"
  --artifact-dtype bfloat16
  --chunk-samples 1
  --max-chunk-tokens 4096
)
if [ "${CAPTURE_SKIP_VALIDATE}" = "1" ]; then
  CAPTURE_ARGS+=(--skip-validate)
fi

run_stage "capture_pair" "${PRODUCER_GPU},${CONSUMER_GPU}" "${CAPTURE_ARGS[@]}"

run_stage "train_reuse" "${TRAIN_GPU}" \
  python -m scd_repro.cli.train_reuse \
    --pairs "${PAIRS_DIR}" \
    --train-config "${REUSE_TRAIN_CONFIG}" \
    --out "${REUSE_DIR}" \
    --seed "${SEED}" \
    --train-device "${TRAIN_DEVICE}" \
    --log "${MEASURE_DIR}/train_reuse/train_reuse.cli.log"

run_stage "profile" "${TRAIN_GPU}" \
  python -m scd_repro.cli.profile_layers \
    --pairs "${PAIRS_DIR}" \
    --reuse-ckpt "${REUSE_DIR}" \
    --models-config "${MODELS_CONFIG}" \
    --out "${PROFILE_PATH}" \
    --train-device "${TRAIN_DEVICE}" \
    --log "${MEASURE_DIR}/profile/profile.cli.log"

run_stage "train_patch" "${TRAIN_GPU}" \
  python -m scd_repro.cli.train_patch \
    --pairs "${PAIRS_DIR}" \
    --profile "${PROFILE_PATH}" \
    --models-config "${MODELS_CONFIG}" \
    --train-config "${PATCH_TRAIN_CONFIG}" \
    --out "${PATCH_DIR}" \
    --seed "${SEED}" \
    --train-device "${TRAIN_DEVICE}" \
    --log "${MEASURE_DIR}/train_patch/train_patch.cli.log"

run_stage "build_route_tasklevel" "${TRAIN_GPU}" \
  python -m scd_repro.cli.build_route \
    --profile "${PROFILE_PATH}" \
    --pairs "${PAIRS_DIR}" \
    --patch-ckpt "${PATCH_DIR}" \
    --eval-config "${GENERATED_EVAL_CONFIG}" \
    --out "${ROUTE_PATH}" \
    --search-device "${TRAIN_DEVICE}" \
    --log "${MEASURE_DIR}/build_route_tasklevel/build_route.cli.log"

run_stage "validate_route" "" \
  python scripts/validate_scd_route.py \
    --route "${ROUTE_PATH}" \
    --require-mixed

run_stage "strict_eval" "${PRODUCER_GPU},${CONSUMER_GPU}" \
  python -m scd_repro.cli.eval_cmrc_strict \
    --models-config "${MODELS_CONFIG}" \
    --eval-config "${GENERATED_EVAL_CONFIG}" \
    --data-root "${EVAL_ROOT}" \
    --cmrc-split dev \
    --route "${ROUTE_PATH}" \
    --reuse-ckpt "${REUSE_DIR}" \
    --patch-ckpt "${PATCH_DIR}" \
    --out "${STRICT_OUT}" \
    --limit "${EVAL_SAMPLES}" \
    --max-length "${MAX_LENGTH}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --bandwidth-gbps "${BANDWIDTH_GBPS}" \
    --quant-group-size "${QUANT_GROUP_SIZE}" \
    --producer-device "${PRODUCER_DEVICE}" \
    --consumer-device "${CONSUMER_DEVICE}" \
    --dtype bfloat16 \
    --seed "${SEED}" \
    --log "${MEASURE_DIR}/strict_eval/strict_eval.cli.log"

python - <<PY
import json
from pathlib import Path

route = json.loads(Path("${ROUTE_PATH}").read_text(encoding="utf-8-sig"))
per_mode_path = Path("${STRICT_OUT}") / "per_mode_summary.json"
summary_path = Path("${MEASURE_DIR}") / "reuse_patch_build_summary.json"
payload = {
    "run_root": "${RUN_ROOT}",
    "models_config": "${MODELS_CONFIG}",
    "reuse_train_config": "${REUSE_TRAIN_CONFIG}",
    "patch_train_config": "${PATCH_TRAIN_CONFIG}",
    "generated_eval_config": "${GENERATED_EVAL_CONFIG}",
    "pairs": "${PAIRS_DIR}",
    "reuse_ckpt": "${REUSE_DIR}",
    "patch_ckpt": "${PATCH_DIR}",
    "profile": "${PROFILE_PATH}",
    "route": "${ROUTE_PATH}",
    "strict_eval": "${STRICT_OUT}",
    "route_counts": {
        "num_layers": int(route.get("num_layers", 0)),
        "reuse": len(route.get("reuse_layers", [])),
        "patch": len(route.get("patch_layers", [])),
        "fallback": len(route.get("fallback_layers", [])),
    },
    "per_mode_summary": json.loads(per_mode_path.read_text(encoding="utf-8-sig")) if per_mode_path.exists() else None,
}
summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"summary={summary_path}")
PY

echo "SCD reuse+patch construction completed."
echo "RUN_ROOT=${RUN_ROOT}"
echo "SUMMARY_JSON=${MEASURE_DIR}/reuse_patch_build_summary.json"
echo "ROUTE_JSON=${ROUTE_PATH}"
echo "STRICT_PER_MODE_JSON=${STRICT_OUT}/per_mode_summary.json"
