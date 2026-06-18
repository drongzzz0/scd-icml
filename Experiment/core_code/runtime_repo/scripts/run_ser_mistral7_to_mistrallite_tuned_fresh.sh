#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
VENV="${VENV:-/disk_n/c101-3-mql/venvs/A800}"
DATA_SOURCE_DIR="${DATA_SOURCE_DIR:-/disk_n/c101-3-mql/projects/scd_main/legacy/no4/data/cmrc2018}"

STAMP="$(date +%Y%m%d-%H%M%S)"
RUN_ROOT="${RUN_ROOT:-${ROOT}/runs/mistral7_to_mistrallite_tunedfresh_${STAMP}}"
SEED="${SEED:-42}"
TRAIN_LIMIT="${TRAIN_LIMIT:-200}"
SEARCH_LIMIT="${SEARCH_LIMIT:-200}"
MAX_LENGTH="${MAX_LENGTH:-1024}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"
LAYERS_PER_SHARD="${LAYERS_PER_SHARD:-4}"
BLOCK_SIZE="${BLOCK_SIZE:-1}"
MAX_F1_DROP="${MAX_F1_DROP:-1.0}"
MAX_EM_DROP="${MAX_EM_DROP:-1.0}"
MIN_TTFT_GAIN_MS="${MIN_TTFT_GAIN_MS:-0.0}"
MAX_ACCEPTED_BLOCKS="${MAX_ACCEPTED_BLOCKS:-0}"
REPLAY_TOP_K="${REPLAY_TOP_K:-5}"
REPLAY_SELECTION_MODE="${REPLAY_SELECTION_MODE:-quality_threshold_f1_first_ttft_second}"
REPLAY_DECISION_MAX_F1_DROP="${REPLAY_DECISION_MAX_F1_DROP:--1.0}"
REPLAY_DECISION_MAX_F1_DROP_RATIO="${REPLAY_DECISION_MAX_F1_DROP_RATIO:-0.05}"
REPLAY_DECISION_MAX_EM_DROP="${REPLAY_DECISION_MAX_EM_DROP:--1.0}"
CAPTURE_SKIP_VALIDATE="${CAPTURE_SKIP_VALIDATE:-1}"
CODE_VERSION_ID="${CODE_VERSION_ID:-unknown}"
RESUME="${RESUME:-0}"

MODELS_CONFIG="${MODELS_CONFIG:-configs/models_ser_mistral7_to_mistrallite_20260405.json}"
REUSE_TRAIN_CONFIG="${REUSE_TRAIN_CONFIG:-configs/train_reuse_mistral_self_layerwise_20260404.json}"
PATCH_TRAIN_CONFIG="${PATCH_TRAIN_CONFIG:-configs/mistral_self_patch_xnorm_nativekv_shared_residual_mlp_alllayers_e1_h2048_r512_b2_20260404.json}"
ROUTE_TEMPLATE="${ROUTE_TEMPLATE:-configs/route_reuse_patchonly_all32_20260404.json}"
PATCH_EVAL_CONFIG="${PATCH_EVAL_CONFIG:-configs/eval_patch_only_train50_20260405.json}"

pick_gpus() {
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | \
    sort -t',' -k2,2n | \
    awk -F',' '{gsub(/ /, "", $1); print $1}'
}

if [ -z "${PRODUCER_GPU:-}" ] || [ -z "${CONSUMER_GPU:-}" ] || [ -z "${TRAIN_GPU:-}" ] || [ -z "${AUX_GPU:-}" ]; then
  mapfile -t _picked_gpus < <(pick_gpus)
  PRODUCER_GPU="${PRODUCER_GPU:-${_picked_gpus[0]}}"
  CONSUMER_GPU="${CONSUMER_GPU:-${_picked_gpus[1]}}"
  TRAIN_GPU="${TRAIN_GPU:-${_picked_gpus[2]}}"
  AUX_GPU="${AUX_GPU:-${_picked_gpus[3]}}"
fi

PRODUCER_DEVICE="cuda:${PRODUCER_GPU}"
CONSUMER_DEVICE="cuda:${CONSUMER_GPU}"
TRAIN_DEVICE="cuda:${TRAIN_GPU}"
AUX_DEVICE="cuda:${AUX_GPU}"
WORKER_PAIRS="${WORKER_PAIRS:-${PRODUCER_DEVICE},${CONSUMER_DEVICE};${TRAIN_DEVICE},${AUX_DEVICE}}"

PAIR_DIR="${RUN_ROOT}/pairs/cmrc2018_seed${SEED}"
MEASURE_DIR="${RUN_ROOT}/measure"
REUSE_DIR="${RUN_ROOT}/ckpt_reuse/layerwise_seed${SEED}"
PROFILE_PATH="${RUN_ROOT}/profile/profile_cmrc2018_seed${SEED}.json"
PATCH_DIR="${RUN_ROOT}/ckpt_patch/xnorm_nativekv_shared_residual_mlp_all32_e1_h2048_r512_b2_seed${SEED}"
PATCH_DIR_SHARDED="${PATCH_DIR}_sharded_w${LAYERS_PER_SHARD}"
TRAIN_ONLY_ROOT="${RUN_ROOT}/data_trainonly"
LOCAL_CMRC_DIR="${TRAIN_ONLY_ROOT}/cmrc2018"
EVAL_ROOT="${RUN_ROOT}/data_eval_dev"
EVAL_CMRC_DIR="${EVAL_ROOT}/cmrc2018"
PROMPTS_JSON="${RUN_ROOT}/prompts/cmrc2018_train_prompts_seed${SEED}_n${TRAIN_LIMIT}.json"
SEARCH_OUT="${RUN_ROOT}/blockwise_search/strictstop_b1_replayk${REPLAY_TOP_K}_f1pct5"
BASE_REUSE_CFG="${SEARCH_OUT}/generated_eval_baseline_reuse_only.json"
BASE_REUSE_OUT="${RUN_ROOT}/strict_eval/strict_baseline_reuse_only"
HYBRID_OUT="${RUN_ROOT}/strict_eval/strict_hybrid"

mkdir -p "${RUN_ROOT}" "${MEASURE_DIR}" "$(dirname "${PROFILE_PATH}")" "${LOCAL_CMRC_DIR}" "${EVAL_CMRC_DIR}" "$(dirname "${PROMPTS_JSON}")" "${SEARCH_OUT}"

cd "${ROOT}"
source "${VENV}/bin/activate"
export PYTHONPATH="${ROOT}/src"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-16}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-16}"

cp "${DATA_SOURCE_DIR}/cmrc2018_train.json" "${LOCAL_CMRC_DIR}/cmrc2018_train.json"
cp "${DATA_SOURCE_DIR}/cmrc2018_dev.json" "${EVAL_CMRC_DIR}/cmrc2018_dev.json"
cp "${DATA_SOURCE_DIR}/cmrc2018_train.json" "${EVAL_CMRC_DIR}/cmrc2018_train.json"
if [ -f "${DATA_SOURCE_DIR}/README.md" ]; then
  cp "${DATA_SOURCE_DIR}/README.md" "${LOCAL_CMRC_DIR}/README.md"
  cp "${DATA_SOURCE_DIR}/README.md" "${EVAL_CMRC_DIR}/README.md"
fi

cat > "${MEASURE_DIR}/gpu_assignment.json" <<EOF
{
  "producer_gpu": ${PRODUCER_GPU},
  "consumer_gpu": ${CONSUMER_GPU},
  "train_gpu": ${TRAIN_GPU},
  "aux_gpu": ${AUX_GPU},
  "producer_device": "${PRODUCER_DEVICE}",
  "consumer_device": "${CONSUMER_DEVICE}",
  "train_device": "${TRAIN_DEVICE}",
  "aux_device": "${AUX_DEVICE}",
  "worker_pairs": "${WORKER_PAIRS}"
}
EOF

python - <<PY
import json
from pathlib import Path
from scd_repro.cmrc import load_cmrc2018_samples

rows = load_cmrc2018_samples("${TRAIN_ONLY_ROOT}", limit=int("${TRAIN_LIMIT}"), split="train")
out = Path("${PROMPTS_JSON}")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps([row.prompt for row in rows], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"prepared_train_prompts={len(rows)}")
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
  local archive_suffix=""

  if [ -d "${stage_dir}" ]; then
    archive_suffix="$(date +%Y%m%d-%H%M%S)"
    mv "${stage_dir}" "${stage_dir}.prev_${archive_suffix}"
  fi
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

stage_succeeded() {
  local stage_name="$1"
  local summary_path="${MEASURE_DIR}/${stage_name}/stage_summary.json"
  if [ ! -f "${summary_path}" ]; then
    return 1
  fi
  python - <<PY
import json
from pathlib import Path
path = Path("${summary_path}")
try:
    payload = json.loads(path.read_text(encoding="utf-8"))
except Exception:
    raise SystemExit(1)
raise SystemExit(0 if bool(payload.get("success")) else 1)
PY
}

run_stage_if_needed() {
  local stage_name="$1"
  local gpu_ids_csv="$2"
  local script_body="$3"
  if [ "${RESUME}" = "1" ] && stage_succeeded "${stage_name}"; then
    echo "Skipping successful stage ${stage_name} because RESUME=1"
    return 0
  fi
  run_stage "${stage_name}" "${gpu_ids_csv}" "${script_body}"
}

run_stage_if_needed "capture_pair" "${PRODUCER_GPU},${CONSUMER_GPU}" "$(cat <<EOF
python -m scd_repro.cli.capture_pair \
  --backend transformers \
  --models-config "${MODELS_CONFIG}" \
  --data-config configs/data_ser.yaml \
  --out "${PAIR_DIR}" \
  --seed "${SEED}" \
  --prompt-file "${PROMPTS_JSON}" \
  --max-length "${MAX_LENGTH}" \
  --dtype bfloat16 \
  --model-a-device "${PRODUCER_DEVICE}" \
  --model-b-device "${CONSUMER_DEVICE}" \
  --artifact-dtype bfloat16 \
  --chunk-samples 1 \
  --max-chunk-tokens 4096 \
  $( [ "${CAPTURE_SKIP_VALIDATE}" = "1" ] && printf '%s' '--skip-validate' )
EOF
)"

run_stage_if_needed "train_reuse" "${TRAIN_GPU}" "$(cat <<EOF
python -m scd_repro.cli.train_reuse \
  --pairs "${PAIR_DIR}" \
  --train-config "${REUSE_TRAIN_CONFIG}" \
  --out "${REUSE_DIR}" \
  --seed "${SEED}" \
  --train-device "${TRAIN_DEVICE}" \
  --log "${MEASURE_DIR}/train_reuse/train_reuse.cli.log"
EOF
)"

run_stage_if_needed "profile" "${TRAIN_GPU}" "$(cat <<EOF
python -m scd_repro.cli.profile_layers \
  --pairs "${PAIR_DIR}" \
  --reuse-ckpt "${REUSE_DIR}" \
  --out "${PROFILE_PATH}" \
  --train-device "${TRAIN_DEVICE}" \
  --log "${MEASURE_DIR}/profile/profile.cli.log"
EOF
)"

run_stage_if_needed "train_patch" "${TRAIN_GPU}" "$(cat <<EOF
python -m scd_repro.cli.train_patch \
  --pairs "${PAIR_DIR}" \
  --profile "${PROFILE_PATH}" \
  --route "${ROUTE_TEMPLATE}" \
  --train-config "${PATCH_TRAIN_CONFIG}" \
  --models-config "${MODELS_CONFIG}" \
  --out "${PATCH_DIR}" \
  --seed "${SEED}" \
  --train-device "${TRAIN_DEVICE}" \
  --log "${MEASURE_DIR}/train_patch/train_patch.cli.log"
EOF
)"

run_stage_if_needed "reshard_patch" "${TRAIN_GPU}" "$(cat <<EOF
python - <<PY
from scd_repro.pipeline import reshard_patch_ckpt

reshard_patch_ckpt(
    "${PATCH_DIR}",
    "${PATCH_DIR_SHARDED}",
    layers_per_shard=int("${LAYERS_PER_SHARD}"),
    drop_identity_residual=False,
)
PY
EOF
)"

run_stage_if_needed "blockwise_search" "${PRODUCER_GPU},${CONSUMER_GPU},${TRAIN_GPU},${AUX_GPU}" "$(cat <<EOF
python -m scd_repro.cli.search_blockwise_hybrid \
  --models-config "${MODELS_CONFIG}" \
  --data-root "${TRAIN_ONLY_ROOT}" \
  --route-template "${ROUTE_TEMPLATE}" \
  --reuse-ckpt "${REUSE_DIR}" \
  --patch-ckpt "${PATCH_DIR_SHARDED}" \
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
  --producer-device "${PRODUCER_DEVICE}" \
  --consumer-device "${CONSUMER_DEVICE}" \
  --dtype bfloat16 \
  --seed "${SEED}" \
  --log "${MEASURE_DIR}/blockwise_search/search.cli.log"
EOF
)"

run_stage_if_needed "strict_baseline_reuse_only" "${PRODUCER_GPU},${CONSUMER_GPU}" "$(cat <<EOF
python - <<PY
import json
from pathlib import Path

src = Path("${PATCH_EVAL_CONFIG}")
cfg = json.loads(src.read_text(encoding="utf-8"))
runtime = cfg.get("runtime_eval", {})
runtime.update(
    {
        "run_full_consumer_baseline": True,
        "run_reuse_only": True,
        "run_reuse_fallback": False,
        "run_patch_only": False,
        "run_reuse_patch": False,
        "run_reuse_patch_recompute": False,
        "run_oracle_xnorm_suffix": False,
        "run_oracle_state_suffix": False,
        "run_oracle_kv_patchonly": False,
        "max_eval_samples": int("${SEARCH_LIMIT}"),
        "strict_mode": True,
        "fail_on_missing_route_fields": False,
        "fail_on_missing_patch_fields": True,
        "align_full_consumer_first_decode": True,
    }
)
cfg["runtime_eval"] = runtime
Path("${BASE_REUSE_CFG}").write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\\n", encoding="utf-8")
PY

python -m scd_repro.cli.eval_cmrc_strict \
  --models-config "${MODELS_CONFIG}" \
  --eval-config "${BASE_REUSE_CFG}" \
  --data-root "${EVAL_ROOT}" \
  --route "${ROUTE_TEMPLATE}" \
  --reuse-ckpt "${REUSE_DIR}" \
  --patch-ckpt "${PATCH_DIR_SHARDED}" \
  --out "${BASE_REUSE_OUT}" \
  --limit "${SEARCH_LIMIT}" \
  --max-length "${MAX_LENGTH}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --producer-device "${PRODUCER_DEVICE}" \
  --consumer-device "${CONSUMER_DEVICE}" \
  --dtype bfloat16 \
  --seed "${SEED}" \
  --log "${MEASURE_DIR}/strict_baseline_reuse_only/strict_baseline_reuse_only.cli.log"
EOF
)"

run_stage_if_needed "strict_hybrid" "${PRODUCER_GPU},${CONSUMER_GPU}" "$(cat <<EOF
python -m scd_repro.cli.eval_cmrc_strict \
  --models-config "${MODELS_CONFIG}" \
  --eval-config "${SEARCH_OUT}/generated_eval_reuse_patch_only.json" \
  --data-root "${EVAL_ROOT}" \
  --route "${SEARCH_OUT}/selected_hybrid_route.json" \
  --reuse-ckpt "${REUSE_DIR}" \
  --patch-ckpt "${PATCH_DIR_SHARDED}" \
  --out "${HYBRID_OUT}" \
  --limit "${SEARCH_LIMIT}" \
  --max-length "${MAX_LENGTH}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --producer-device "${PRODUCER_DEVICE}" \
  --consumer-device "${CONSUMER_DEVICE}" \
  --dtype bfloat16 \
  --seed "${SEED}" \
  --log "${MEASURE_DIR}/strict_hybrid/strict_hybrid.cli.log"
EOF
)"

python scripts/summarize_mistrallite_to_mistral7_tuned_fresh.py \
  --run-root "${RUN_ROOT}" \
  --models-config "${MODELS_CONFIG}" \
  --reuse-train-config "${REUSE_TRAIN_CONFIG}" \
  --patch-train-config "${PATCH_TRAIN_CONFIG}" \
  --route-template "${ROUTE_TEMPLATE}" \
  --pairs "${PAIR_DIR}" \
  --reuse-ckpt "${REUSE_DIR}" \
  --patch-ckpt "${PATCH_DIR_SHARDED}" \
  --profile "${PROFILE_PATH}" \
  --search-out "${SEARCH_OUT}" \
  --baseline-reuse-out "${BASE_REUSE_OUT}" \
  --hybrid-out "${HYBRID_OUT}" \
  --measure-root "${MEASURE_DIR}" \
  --code-version "${CODE_VERSION_ID}"

echo "Tuned fresh reverse rerun completed."
echo "RUN_ROOT=${RUN_ROOT}"
echo "SUMMARY_JSON=${MEASURE_DIR}/tuned_rebuttal_summary.json"
