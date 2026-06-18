#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
VENV="${VENV:-/disk_n/c101-3-mql/venvs/A800}"
DATA_SOURCE_DIR="${DATA_SOURCE_DIR:-/disk_n/c101-3-mql/projects/scd_main/legacy/no4/data/cmrc2018}"

STAMP="$(date +%Y%m%d-%H%M%S)"
RUN_ROOT="${RUN_ROOT:-${ROOT}/runs/scd-q2-m7_to_lite-fresh-${STAMP}}"
SEED="${SEED:-42}"
CALIB_SAMPLES="${CALIB_SAMPLES:-200}"
EVAL_SAMPLES="${EVAL_SAMPLES:-200}"
MAX_LENGTH="${MAX_LENGTH:-256}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"
BANDWIDTH_GBPS="${BANDWIDTH_GBPS:-200}"
QUANT_GROUP_SIZE="${QUANT_GROUP_SIZE:-128}"
CODE_VERSION_ID="${CODE_VERSION_ID:-unknown}"

MODELS_CONFIG="${MODELS_CONFIG:-configs/models_ser_mistral7_to_mistrallite_20260405.json}"
TRAIN_CONFIG="${TRAIN_CONFIG:-configs/train_q2_mistral7_to_mistrallite_fix7_20260407.json}"
EVAL_CONFIG="${EVAL_CONFIG:-configs/eval_q2_mistral7_to_mistrallite_fix7_20260407.json}"
CAPTURE_DATA_CONFIG="${CAPTURE_DATA_CONFIG:-configs/data_ser.yaml}"

pick_gpus() {
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | \
    sort -t',' -k2,2n | \
    awk -F',' '{gsub(/ /, "", $1); print $1}'
}

if [ -z "${PRODUCER_GPU:-}" ] || [ -z "${CONSUMER_GPU:-}" ] || [ -z "${TRAIN_GPU:-}" ]; then
  mapfile -t _picked_gpus < <(pick_gpus)
  PRODUCER_GPU="${PRODUCER_GPU:-${_picked_gpus[0]}}"
  CONSUMER_GPU="${CONSUMER_GPU:-${_picked_gpus[1]}}"
  TRAIN_GPU="${TRAIN_GPU:-${_picked_gpus[2]}}"
fi

PRODUCER_DEVICE="cuda:${PRODUCER_GPU}"
CONSUMER_DEVICE="cuda:${CONSUMER_GPU}"
TRAIN_DEVICE="cuda:${TRAIN_GPU}"

PAIRS_DIR="${RUN_ROOT}/pairs/cmrc2018_seed${SEED}"
MEASURE_DIR="${RUN_ROOT}/measure"
REUSE_DIR="${RUN_ROOT}/ckpt_reuse/xkvgroup_g4_r128_gpu${TRAIN_GPU}_seed${SEED}"
PATCH_DIR="${RUN_ROOT}/ckpt_patch/semantic_fix7_gpu${TRAIN_GPU}_seed${SEED}"
PROFILE_DIR="${RUN_ROOT}/profile"
PROFILE_PATH="${PROFILE_DIR}/profile_cmrc2018_seed${SEED}.json"
ROUTE_DIR="${RUN_ROOT}/route"
ROUTE_PATH="${ROUTE_DIR}/route_cmrc2018_seed${SEED}_fix7_xkvgroup_g4_r128_gpu${TRAIN_GPU}.json"
STRICT_OUT="${RUN_ROOT}/strict_eval/eval_stepc_fix7_semantic_rllm_gpu${PRODUCER_GPU}${CONSUMER_GPU}"
CALIB_ROOT="${RUN_ROOT}/data_calib_train"
EVAL_ROOT="${RUN_ROOT}/data_eval_dev"
CALIB_CMRC_DIR="${CALIB_ROOT}/cmrc2018"
EVAL_CMRC_DIR="${EVAL_ROOT}/cmrc2018"
PROMPTS_JSON="${RUN_ROOT}/prompts/cmrc2018_train_prompts_seed${SEED}_n${CALIB_SAMPLES}.json"

mkdir -p "${RUN_ROOT}" "${MEASURE_DIR}" "${PROFILE_DIR}" "${ROUTE_DIR}" "$(dirname "${PROMPTS_JSON}")" "${CALIB_CMRC_DIR}" "${EVAL_CMRC_DIR}"

cd "${ROOT}"
source "${VENV}/bin/activate"
export PYTHONPATH="${ROOT}/src"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-16}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-16}"

cp "${DATA_SOURCE_DIR}/cmrc2018_train.json" "${CALIB_CMRC_DIR}/cmrc2018_train.json"
cp "${DATA_SOURCE_DIR}/cmrc2018_dev.json" "${EVAL_CMRC_DIR}/cmrc2018_dev.json"
cp "${DATA_SOURCE_DIR}/cmrc2018_train.json" "${EVAL_CMRC_DIR}/cmrc2018_train.json"
if [ -f "${DATA_SOURCE_DIR}/README.md" ]; then
  cp "${DATA_SOURCE_DIR}/README.md" "${CALIB_CMRC_DIR}/README.md"
  cp "${DATA_SOURCE_DIR}/README.md" "${EVAL_CMRC_DIR}/README.md"
fi

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

python - <<PY
import json
from pathlib import Path
from scd_repro.cmrc import load_cmrc2018_samples

rows = load_cmrc2018_samples("${CALIB_ROOT}", limit=int("${CALIB_SAMPLES}"), split="train")
out = Path("${PROMPTS_JSON}")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps([row.prompt for row in rows], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"prepared_calibration_prompts={len(rows)}")
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
  local script_body="$3"
  local stage_dir="${MEASURE_DIR}/${stage_name}"
  local start_epoch
  local end_epoch
  local start_iso
  local end_iso
  local time_file="${stage_dir}/time.txt"
  local stdout_file="${stage_dir}/stdout.log"
  local stderr_file="${stage_dir}/stderr.log"
  local trace_file="${stage_dir}/gpu_trace.csv"
  local script_file="${stage_dir}/command.sh"
  local mon_pid=""
  local exit_code=0

  mkdir -p "${stage_dir}"
  cat > "${script_file}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "${ROOT}"
source "${VENV}/bin/activate"
export PYTHONPATH="${ROOT}/src"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-16}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-16}"
${script_body}
EOF
  chmod +x "${script_file}"

  start_epoch="$(python - <<'PY'
import time
print(f"{time.time():.6f}")
PY
)"
  start_iso="$(date -Is)"

  if [ -n "${gpu_ids_csv}" ]; then
    mon_pid="$(start_gpu_monitor "${trace_file}" "${gpu_ids_csv}")"
  fi

  if /usr/bin/time -v -o "${time_file}" bash "${script_file}" > "${stdout_file}" 2> "${stderr_file}"; then
    exit_code=0
  else
    exit_code=$?
  fi

  if [ -n "${mon_pid}" ]; then
    kill "${mon_pid}" >/dev/null 2>&1 || true
    wait "${mon_pid}" 2>/dev/null || true
  fi

  end_epoch="$(python - <<'PY'
import time
print(f"{time.time():.6f}")
PY
)"
  end_iso="$(date -Is)"

  python - <<PY
import csv
import json
import re
from pathlib import Path

stage_name = "${stage_name}"
gpu_ids = [int(x) for x in "${gpu_ids_csv}".split(",") if x.strip()]
start_epoch = float("${start_epoch}")
end_epoch = float("${end_epoch}")
time_file = Path("${time_file}")
trace_file = Path("${trace_file}")
stdout_file = Path("${stdout_file}")
stderr_file = Path("${stderr_file}")
script_file = Path("${script_file}")
summary_path = Path("${stage_dir}") / "stage_summary.json"

max_rss_kb = None
elapsed_wall = end_epoch - start_epoch
if time_file.exists():
    text = time_file.read_text(encoding="utf-8", errors="ignore")
    match = re.search(r"Maximum resident set size \\(kbytes\\):\\s*(\\d+)", text)
    if match:
        max_rss_kb = int(match.group(1))

peak_by_gpu = {}
if trace_file.exists():
    with trace_file.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx = int(row["gpu_index"])
            mem = float(row["memory_used_mb"])
            peak_by_gpu[idx] = max(mem, peak_by_gpu.get(idx, 0.0))

payload = {
    "stage": stage_name,
    "success": ${exit_code} == 0,
    "exit_code": int(${exit_code}),
    "start_time_iso": "${start_iso}",
    "end_time_iso": "${end_iso}",
    "start_epoch_sec": start_epoch,
    "end_epoch_sec": end_epoch,
    "wall_clock_sec": float(elapsed_wall),
    "gpu_ids": gpu_ids,
    "peak_gpu_memory_by_gpu_mb": {str(k): float(v) for k, v in sorted(peak_by_gpu.items())},
    "peak_gpu_memory_mb": None if not peak_by_gpu else float(max(peak_by_gpu.values())),
    "max_cpu_rss_kb": max_rss_kb,
    "stdout_log": str(stdout_file),
    "stderr_log": str(stderr_file),
    "time_log": str(time_file),
    "command_file": str(script_file),
}
summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(payload, ensure_ascii=False, indent=2))
PY

  if [ "${exit_code}" -ne 0 ]; then
    echo "stage ${stage_name} failed; see ${stage_dir}" >&2
    exit "${exit_code}"
  fi
}

run_stage "capture_pair" "${PRODUCER_GPU},${CONSUMER_GPU}" "$(cat <<EOF
python -m scd_repro.cli.capture_pair \
  --backend transformers \
  --models-config "${MODELS_CONFIG}" \
  --data-config "${CAPTURE_DATA_CONFIG}" \
  --out "${PAIRS_DIR}" \
  --seed "${SEED}" \
  --log "${MEASURE_DIR}/capture_pair/capture_pair.cli.log" \
  --prompt-file "${PROMPTS_JSON}" \
  --max-length "${MAX_LENGTH}" \
  --dtype bfloat16 \
  --model-a-device "${PRODUCER_DEVICE}" \
  --model-b-device "${CONSUMER_DEVICE}" \
  --artifact-dtype bfloat16 \
  --chunk-samples 1 \
  --max-chunk-tokens 4096 \
  --skip-validate
EOF
)"

run_stage "train_reuse" "${TRAIN_GPU}" "$(cat <<EOF
python -m scd_repro.cli.train_reuse \
  --pairs "${PAIRS_DIR}" \
  --train-config "${TRAIN_CONFIG}" \
  --out "${REUSE_DIR}" \
  --seed "${SEED}" \
  --train-device "${TRAIN_DEVICE}" \
  --log "${MEASURE_DIR}/train_reuse/train_reuse.cli.log"
EOF
)"

run_stage "profile" "${TRAIN_GPU}" "$(cat <<EOF
python -m scd_repro.cli.profile_layers \
  --pairs "${PAIRS_DIR}" \
  --reuse-ckpt "${REUSE_DIR}" \
  --out "${PROFILE_PATH}" \
  --train-device "${TRAIN_DEVICE}" \
  --log "${MEASURE_DIR}/profile/profile.cli.log"
EOF
)"

run_stage "train_patch" "${TRAIN_GPU}" "$(cat <<EOF
python -m scd_repro.cli.train_patch \
  --pairs "${PAIRS_DIR}" \
  --profile "${PROFILE_PATH}" \
  --models-config "${MODELS_CONFIG}" \
  --train-config "${TRAIN_CONFIG}" \
  --out "${PATCH_DIR}" \
  --seed "${SEED}" \
  --train-device "${TRAIN_DEVICE}" \
  --log "${MEASURE_DIR}/train_patch/train_patch.cli.log"
EOF
)"

run_stage "build_route" "${TRAIN_GPU}" "$(cat <<EOF
python -m scd_repro.cli.build_route \
  --profile "${PROFILE_PATH}" \
  --pairs "${PAIRS_DIR}" \
  --patch-ckpt "${PATCH_DIR}" \
  --eval-config "${EVAL_CONFIG}" \
  --out "${ROUTE_PATH}" \
  --search-device "${TRAIN_DEVICE}" \
  --log "${MEASURE_DIR}/build_route/build_route.cli.log"
EOF
)"

run_stage "strict_eval" "${PRODUCER_GPU},${CONSUMER_GPU}" "$(cat <<EOF
python -m scd_repro.cli.eval_cmrc_strict \
  --models-config "${MODELS_CONFIG}" \
  --eval-config "${EVAL_CONFIG}" \
  --data-root "${EVAL_ROOT}" \
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
EOF
)"

python scripts/summarize_q2_m7_to_mistrallite_fresh.py \
  --run-root "${RUN_ROOT}" \
  --models-config "${MODELS_CONFIG}" \
  --train-config "${TRAIN_CONFIG}" \
  --eval-config "${EVAL_CONFIG}" \
  --pairs "${PAIRS_DIR}" \
  --reuse-ckpt "${REUSE_DIR}" \
  --patch-ckpt "${PATCH_DIR}" \
  --profile "${PROFILE_PATH}" \
  --route "${ROUTE_PATH}" \
  --strict-out "${STRICT_OUT}" \
  --measure-root "${MEASURE_DIR}" \
  --code-version "${CODE_VERSION_ID}"

echo "Q2 fresh rerun completed."
echo "RUN_ROOT=${RUN_ROOT}"
echo "SUMMARY_JSON=${MEASURE_DIR}/rebuttal_summary.json"
echo "STRICT_SUMMARY_JSON=${STRICT_OUT}/summary.json"
echo "STRICT_PER_MODE_JSON=${STRICT_OUT}/per_mode_summary.json"
