#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
VENV="${VENV:-/disk_n/c101-3-mql/venvs/A800}"
DATA_ROOT="${DATA_ROOT:-/disk_n/c101-3-mql/projects/SCD-ICML/kv/snapshots/scd-github-f7d7854}"

STAMP="$(date +%Y%m%d-%H%M%S)"
RUN_ROOT="${RUN_ROOT:-/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_mistral_repro_${STAMP}}"
SEED="${SEED:-42}"
TRAIN_LIMIT="${TRAIN_LIMIT:-200}"
EVAL_LIMIT="${EVAL_LIMIT:-200}"
TRAIN_CMRC_SPLIT="${TRAIN_CMRC_SPLIT:-train}"
EVAL_CMRC_SPLIT="${EVAL_CMRC_SPLIT:-dev}"
MAX_LENGTH="${MAX_LENGTH:-0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"
BANDWIDTH_GBPS="${BANDWIDTH_GBPS:-200}"
QUANT_GROUP_SIZE="${QUANT_GROUP_SIZE:-128}"
# Paper-facing Mistral reproduction path. Set
# DIRECTIONS=lite_to_7b,7b_to_lite explicitly to run the optional reverse check.
DIRECTIONS="${DIRECTIONS:-lite_to_7b}"
DATASETS="${DATASETS:-cmrc2018}"
RESUME="${RESUME:-0}"
CAPTURE_A_DEVICE="${CAPTURE_A_DEVICE:-cuda:0}"
CAPTURE_B_DEVICE="${CAPTURE_B_DEVICE:-cuda:1}"
TRAIN_DEVICE="${TRAIN_DEVICE:-cuda:0}"
PRODUCER_DEVICE="${PRODUCER_DEVICE:-cuda:0}"
CONSUMER_DEVICE="${CONSUMER_DEVICE:-cuda:1}"
WAIT_GPU_POLL_SEC="${WAIT_GPU_POLL_SEC:-120}"
WAIT_CAPTURE_FREE_MB="${WAIT_CAPTURE_FREE_MB:-18000}"
WAIT_TRAIN_FREE_MB="${WAIT_TRAIN_FREE_MB:-6000}"
WAIT_EVAL_FREE_MB="${WAIT_EVAL_FREE_MB:-18000}"
AUTO_SELECTED_CUDA_VISIBLE_DEVICES="${AUTO_SELECTED_CUDA_VISIBLE_DEVICES:-0}"

REUSE_TRAIN_CONFIG="${REUSE_TRAIN_CONFIG:-configs/train_reuse_mistral_self_layerwise_20260404.json}"
PATCH_TRAIN_CONFIG="${PATCH_TRAIN_CONFIG:-configs/mistral_self_patch_xnorm_nativekv_shared_residual_mlp_alllayers_e1_h2048_r512_b2_20260404.json}"
EVAL_CONFIG_TEMPLATE="${EVAL_CONFIG_TEMPLATE:-configs/eval_scd_icml_full_scd_no_fallback_no_recompute.json}"
ROUTE_MODE="${ROUTE_MODE:-template}"
ROUTE_TEMPLATE="${ROUTE_TEMPLATE:-configs/routes/scd_icml_mistrallite_to_mistral7_reuse19_patch13_cmrc200.json}"
ROUTE_PATCH_BUDGET="${ROUTE_PATCH_BUDGET:-6}"
ROUTE_PATCH_LAYERS="${ROUTE_PATCH_LAYERS:-}"

pick_gpus() {
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits | \
    awk -F',' '{gsub(/ /, "", $1); gsub(/ /, "", $2); gsub(/ /, "", $3); print $1 "," $2 "," $3}' | \
    sort -t',' -k2,2n -k3,3n | \
    awk -F',' '{print $1}'
}

pick_free_gpus() {
  local required_free_mb="$1"
  local count="$2"
  nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits | \
    awk -F',' -v required="${required_free_mb}" '{gsub(/ /, "", $1); gsub(/ /, "", $2); gsub(/ /, "", $3); if ($2 >= required) print $1 "," $2 "," $3}' | \
    sort -t',' -k2,2nr -k3,3n | \
    awk -F',' -v count="${count}" 'NR <= count {print $1}'
}

cuda_logical_indices_from_devices() {
  local seen=","
  local dev idx
  for dev in "$@"; do
    dev="$(echo "${dev}" | xargs)"
    case "${dev}" in
      cuda) idx="0" ;;
      cuda:*) idx="${dev#cuda:}" ;;
      *) continue ;;
    esac
    case "${idx}" in
      ''|*[!0-9]*) continue ;;
    esac
    if [[ "${seen}" != *",${idx},"* ]]; then
      echo "${idx}"
      seen="${seen}${idx},"
    fi
  done
}

mapfile -t _auto_gpu_indices < <(
  cuda_logical_indices_from_devices \
    "${CAPTURE_A_DEVICE}" "${CAPTURE_B_DEVICE}" \
    "${PRODUCER_DEVICE}" "${CONSUMER_DEVICE}" "${TRAIN_DEVICE}"
)
AUTO_GPU_COUNT="${AUTO_GPU_COUNT:-${#_auto_gpu_indices[@]}}"
if [ "${AUTO_GPU_COUNT}" -lt 1 ]; then
  AUTO_GPU_COUNT=1
fi

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  while true; do
    if [ "${WAIT_CAPTURE_FREE_MB}" -gt 0 ]; then
      mapfile -t _picked_gpus < <(pick_free_gpus "${WAIT_CAPTURE_FREE_MB}" "${AUTO_GPU_COUNT}")
      if [ "${#_picked_gpus[@]}" -ge "${AUTO_GPU_COUNT}" ]; then
        break
      fi
      echo "waiting to select ${AUTO_GPU_COUNT} GPU(s) with free memory >= ${WAIT_CAPTURE_FREE_MB}MB"
      nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits
      sleep "${WAIT_GPU_POLL_SEC}"
    else
      mapfile -t _picked_gpus < <(pick_gpus)
      if [ "${#_picked_gpus[@]}" -ge "${AUTO_GPU_COUNT}" ]; then
        break
      fi
      echo "Need at least ${AUTO_GPU_COUNT} visible GPU(s) for Mistral pair evaluation." >&2
      exit 1
    fi
  done
  export CUDA_VISIBLE_DEVICES="$(IFS=','; echo "${_picked_gpus[*]}")"
  AUTO_SELECTED_CUDA_VISIBLE_DEVICES=1
  echo "selected CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
fi

logical_to_physical_gpu() {
  local logical_idx="$1"
  local visible="${CUDA_VISIBLE_DEVICES:-}"
  if [ -z "${visible}" ]; then
    echo "${logical_idx}"
    return 0
  fi
  IFS=',' read -r -a _visible_gpu_list <<< "${visible}"
  local physical="${_visible_gpu_list[${logical_idx}]:-${logical_idx}}"
  echo "${physical}" | xargs
}

wait_visible_gpus_free() {
  local required_free_mb="$1"
  shift
  if [ "${required_free_mb}" -le 0 ]; then
    return 0
  fi
  while true; do
    if [ "${AUTO_SELECTED_CUDA_VISIBLE_DEVICES:-0}" = "1" ] && [ "$#" -ge 2 ]; then
      mapfile -t _reselected_gpus < <(pick_free_gpus "${required_free_mb}" "$#")
      if [ "${#_reselected_gpus[@]}" -ge "$#" ]; then
        local new_visible
        new_visible="$(IFS=','; echo "${_reselected_gpus[*]}")"
        if [ "${new_visible}" != "${CUDA_VISIBLE_DEVICES:-}" ]; then
          export CUDA_VISIBLE_DEVICES="${new_visible}"
          echo "reselected CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
        fi
      fi
    fi
    local ok=1
    local report=()
    for logical_idx in "$@"; do
      local physical_idx free_mb
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

mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/tmp_cfg" "${RUN_ROOT}/prompts"

cd "${ROOT}"
source "${VENV}/bin/activate"
python -m pip install -e .
export PYTHONPATH="${ROOT}/src"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

IFS=',' read -r -a DIRECTION_LIST <<< "${DIRECTIONS}"
IFS=',' read -r -a DATASET_LIST <<< "${DATASETS}"

cat > "${RUN_ROOT}/run_config.json" <<EOF
{
  "root": "${ROOT}",
  "venv": "${VENV}",
  "data_root": "${DATA_ROOT}",
  "run_root": "${RUN_ROOT}",
  "seed": ${SEED},
  "train_limit": ${TRAIN_LIMIT},
  "eval_limit": ${EVAL_LIMIT},
  "train_cmrc_split": "${TRAIN_CMRC_SPLIT}",
  "eval_cmrc_split": "${EVAL_CMRC_SPLIT}",
  "max_length": ${MAX_LENGTH},
  "max_new_tokens": ${MAX_NEW_TOKENS},
  "bandwidth_gbps": ${BANDWIDTH_GBPS},
  "directions": "${DIRECTIONS}",
  "datasets": "${DATASETS}",
  "route_mode": "${ROUTE_MODE}",
  "route_template": "${ROUTE_TEMPLATE}",
  "route_patch_budget": ${ROUTE_PATCH_BUDGET},
  "route_patch_layers": "${ROUTE_PATCH_LAYERS}",
  "cuda_visible_devices": "${CUDA_VISIBLE_DEVICES}",
  "capture_a_device": "${CAPTURE_A_DEVICE}",
  "capture_b_device": "${CAPTURE_B_DEVICE}",
  "train_device": "${TRAIN_DEVICE}",
  "producer_device": "${PRODUCER_DEVICE}",
  "consumer_device": "${CONSUMER_DEVICE}",
  "auto_gpu_count": ${AUTO_GPU_COUNT},
  "wait_capture_free_mb": ${WAIT_CAPTURE_FREE_MB},
  "wait_train_free_mb": ${WAIT_TRAIN_FREE_MB},
  "wait_eval_free_mb": ${WAIT_EVAL_FREE_MB}
}
EOF

run_logged() {
  local name="$1"
  shift
  local stage_dir="${RUN_ROOT}/logs/${name}"
  local status_path="${stage_dir}/status.json"
  if [ "${RESUME}" = "1" ] && [ -f "${status_path}" ] && python - "${status_path}" <<'PY'
import json, sys
payload = json.loads(open(sys.argv[1], encoding="utf-8").read())
raise SystemExit(0 if payload.get("success") else 1)
PY
  then
    echo "skip ${name} (RESUME=1)"
    return 0
  fi
  rm -rf "${stage_dir}"
  mkdir -p "${stage_dir}"
  local start_ts end_ts exit_code
  start_ts="$(date +%s)"
  echo "+ $*" | tee "${stage_dir}/command.txt"
  if /usr/bin/time -v -o "${stage_dir}/time.txt" "$@" >"${stage_dir}/stdout.log" 2>"${stage_dir}/stderr.log"; then
    exit_code=0
  else
    exit_code=$?
  fi
  end_ts="$(date +%s)"
  python - "${status_path}" "${name}" "${exit_code}" "${start_ts}" "${end_ts}" <<'PY'
import json, sys, time
path, name, exit_code, start_ts, end_ts = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
payload = {
    "stage": name,
    "success": exit_code == 0,
    "exit_code": exit_code,
    "start_epoch_sec": start_ts,
    "end_epoch_sec": end_ts,
    "wall_clock_sec": max(0, end_ts - start_ts),
    "written_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
}
open(path, "w", encoding="utf-8").write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(payload, ensure_ascii=False, indent=2))
PY
  if [ "${exit_code}" -ne 0 ]; then
    echo "stage failed: ${name}; see ${stage_dir}" >&2
    exit "${exit_code}"
  fi
}

stage_succeeded() {
  local name="$1"
  local status_path="${RUN_ROOT}/logs/${name}/status.json"
  [ -f "${status_path}" ] && python - "${status_path}" <<'PY'
import json, sys
payload = json.loads(open(sys.argv[1], encoding="utf-8").read())
raise SystemExit(0 if payload.get("success") else 1)
PY
}

make_eval_config() {
  local out_path="$1"
  python - "${EVAL_CONFIG_TEMPLATE}" "${out_path}" "${EVAL_LIMIT}" "${BANDWIDTH_GBPS}" "${ROUTE_MODE}" "${ROUTE_PATCH_LAYERS}" "${EVAL_CMRC_SPLIT}" <<'PY'
import json, sys
template, out_path, eval_limit, bandwidth, route_mode, route_patch_layers, eval_cmrc_split = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4]), sys.argv[5], sys.argv[6], sys.argv[7]
cfg = json.load(open(template, encoding="utf-8-sig"))
cfg.setdefault("network", {})["bandwidth_gbps"] = bandwidth
cfg.setdefault("runtime_eval", {})["max_eval_samples"] = eval_limit
cfg.setdefault("runtime_eval", {})["cmrc_split"] = eval_cmrc_split or "dev"
if route_mode.strip() == "paper_profile" or route_patch_layers.strip():
    # Runtime patch_only is defined as all-layer Patch. The paper-facing SCD
    # route is sparse Patch + Reuse, so evaluate scd_full against reuse_only
    # instead of running patch_only on a mixed route.
    cfg.setdefault("runtime_eval", {})["run_patch_only"] = False
open(out_path, "w", encoding="utf-8").write(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n")
PY
}

make_train_prompts() {
  local out_path="$1"
  python - "${DATA_ROOT}" "${TRAIN_LIMIT}" "${out_path}" "${TRAIN_CMRC_SPLIT}" <<'PY'
import json, sys
from pathlib import Path
from scd_repro.text_benchmarks import load_text_benchmark_samples
data_root, limit, out_path, train_cmrc_split = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4] or "train"
samples = load_text_benchmark_samples("cmrc2018", data_root, limit=limit, cmrc_split=train_cmrc_split)
out = Path(out_path)
out.write_text(json.dumps([row.prompt for row in samples], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
meta_path = out.with_suffix(out.suffix + ".meta.json")
meta_path.write_text(
    json.dumps(
        {
            "dataset": "cmrc2018",
            "split": train_cmrc_split,
            "limit": limit,
            "sample_count": len(samples),
            "qids": [row.qid for row in samples],
        },
        ensure_ascii=False,
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)
print(f"wrote_prompts={len(samples)} split={train_cmrc_split} path={out_path} metadata={meta_path}")
PY
}

models_config_for_direction() {
  case "$1" in
    lite_to_7b) echo "${MODELS_CONFIG_LITE_TO_7B:-configs/models_ser_mistrallite_to_mistral7_mql.json}" ;;
    7b_to_lite) echo "${MODELS_CONFIG_7B_TO_LITE:-configs/models_ser_mistral7_to_mistrallite_mql.json}" ;;
    *) echo "Unknown direction: $1" >&2; exit 2 ;;
  esac
}

for direction in "${DIRECTION_LIST[@]}"; do
  direction="$(echo "${direction}" | xargs)"
  [ -n "${direction}" ] || continue
  MODELS_CONFIG="$(models_config_for_direction "${direction}")"
  DIR_ROOT="${RUN_ROOT}/${direction}"
  PAIR_DIR="${DIR_ROOT}/pairs/cmrc2018_train_seed${SEED}_n${TRAIN_LIMIT}"
  REUSE_DIR="${DIR_ROOT}/ckpt_reuse/cmrc2018_seed${SEED}"
  PROFILE_PATH="${DIR_ROOT}/route/profile_cmrc2018_seed${SEED}.json"
  PATCH_DIR="${DIR_ROOT}/ckpt_patch/patch_cmrc2018_seed${SEED}"
  ROUTE_PATH="${DIR_ROOT}/route/route_cmrc2018_seed${SEED}.json"
  PROMPTS_JSON="${RUN_ROOT}/prompts/cmrc2018_train_prompts_seed${SEED}_n${TRAIN_LIMIT}.json"
  EVAL_CONFIG="${RUN_ROOT}/tmp_cfg/eval_main_${direction}.json"

  mkdir -p "${DIR_ROOT}/route"
  make_eval_config "${EVAL_CONFIG}"
  make_train_prompts "${PROMPTS_JSON}"

  if ! stage_succeeded "${direction}_capture_pair"; then
    mapfile -t _capture_wait_indices < <(cuda_logical_indices_from_devices "${CAPTURE_A_DEVICE}" "${CAPTURE_B_DEVICE}")
    wait_visible_gpus_free "${WAIT_CAPTURE_FREE_MB}" "${_capture_wait_indices[@]}"
  fi
  run_logged "${direction}_capture_pair" \
    python -m scd_repro.cli.capture_pair \
      --backend transformers \
      --models-config "${MODELS_CONFIG}" \
      --data-config configs/data_ser.yaml \
      --out "${PAIR_DIR}" \
      --seed "${SEED}" \
      --prompt-file "${PROMPTS_JSON}" \
      --max-length "${MAX_LENGTH}" \
      --dtype bfloat16 \
      --model-a-device "${CAPTURE_A_DEVICE}" \
      --model-b-device "${CAPTURE_B_DEVICE}" \
      --artifact-dtype bfloat16 \
      --chunk-samples 1 \
      --max-chunk-tokens 4096 \
      --skip-validate

  if ! stage_succeeded "${direction}_train_reuse"; then
    wait_visible_gpus_free "${WAIT_TRAIN_FREE_MB}" 0
  fi
  run_logged "${direction}_train_reuse" \
    python -m scd_repro.cli.train_reuse \
      --pairs "${PAIR_DIR}" \
      --train-config "${REUSE_TRAIN_CONFIG}" \
      --out "${REUSE_DIR}" \
      --seed "${SEED}" \
      --train-device "${TRAIN_DEVICE}"

  if ! stage_succeeded "${direction}_profile_layers"; then
    wait_visible_gpus_free "${WAIT_TRAIN_FREE_MB}" 0
  fi
  run_logged "${direction}_profile_layers" \
    python -m scd_repro.cli.profile_layers \
      --pairs "${PAIR_DIR}" \
      --reuse-ckpt "${REUSE_DIR}" \
      --out "${PROFILE_PATH}" \
      --train-device "${TRAIN_DEVICE}"

  if ! stage_succeeded "${direction}_train_patch"; then
    wait_visible_gpus_free "${WAIT_TRAIN_FREE_MB}" 0
  fi
  run_logged "${direction}_train_patch" \
    python -m scd_repro.cli.train_patch \
      --pairs "${PAIR_DIR}" \
      --profile "${PROFILE_PATH}" \
      --train-config "${PATCH_TRAIN_CONFIG}" \
      --models-config "${MODELS_CONFIG}" \
      --out "${PATCH_DIR}" \
      --seed "${SEED}" \
      --train-device "${TRAIN_DEVICE}"

  if [ "${ROUTE_MODE}" = "build" ]; then
    run_logged "${direction}_build_route" \
      python -m scd_repro.cli.build_route \
        --profile "${PROFILE_PATH}" \
        --eval-config "${EVAL_CONFIG}" \
        --out "${ROUTE_PATH}"
  elif [ "${ROUTE_MODE}" = "paper_profile" ]; then
    _patch_layer_args=()
    if [ -n "${ROUTE_PATCH_LAYERS}" ]; then
      _patch_layer_args+=(--patch-layers "${ROUTE_PATCH_LAYERS}")
    fi
    run_logged "${direction}_route_paper_scd" \
      python scripts/make_paper_scd_route.py \
        --profile "${PROFILE_PATH}" \
        --patch-dir "${PATCH_DIR}" \
        --models-config "${MODELS_CONFIG}" \
        --eval-config "${EVAL_CONFIG}" \
        --patch-budget "${ROUTE_PATCH_BUDGET}" \
        "${_patch_layer_args[@]}" \
        --out "${ROUTE_PATH}"
  elif [ "${ROUTE_MODE}" = "patch_meta" ]; then
    _patch_layer_args=()
    if [ -n "${ROUTE_PATCH_LAYERS}" ]; then
      _patch_layer_args+=(--patch-layers "${ROUTE_PATCH_LAYERS}")
    fi
    run_logged "${direction}_route_patch_meta" \
      python scripts/make_patch_meta_route.py \
        --patch-dir "${PATCH_DIR}" \
        --models-config "${MODELS_CONFIG}" \
        "${_patch_layer_args[@]}" \
        --out "${ROUTE_PATH}"
  else
    run_logged "${direction}_route_template" \
      cp "${ROUTE_TEMPLATE}" "${ROUTE_PATH}"
  fi

  for dataset in "${DATASET_LIST[@]}"; do
    dataset="$(echo "${dataset}" | xargs)"
    [ -n "${dataset}" ] || continue
    OUT_DIR="${DIR_ROOT}/strict_eval/${dataset}"
    if ! stage_succeeded "${direction}_eval_${dataset}"; then
      mapfile -t _eval_wait_indices < <(cuda_logical_indices_from_devices "${PRODUCER_DEVICE}" "${CONSUMER_DEVICE}")
      wait_visible_gpus_free "${WAIT_EVAL_FREE_MB}" "${_eval_wait_indices[@]}"
    fi
    run_logged "${direction}_eval_${dataset}" \
      python -m scd_repro.cli.eval_text_strict \
        --dataset "${dataset}" \
        --models-config "${MODELS_CONFIG}" \
        --eval-config "${EVAL_CONFIG}" \
        --data-root "${DATA_ROOT}" \
        --route "${ROUTE_PATH}" \
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
        --cmrc-split "${EVAL_CMRC_SPLIT}" \
        --seed "${SEED}"
  done
done

python -m scd_repro.cli.summarize_mistral_repro \
  --run-root "${RUN_ROOT}" \
  --eval-config "${EVAL_CONFIG_TEMPLATE}" \
  --expected-directions "${DIRECTIONS}" \
  --expected-datasets "${DATASETS}"

echo "SCD-ICML Mistral reproduction finished."
echo "RUN_ROOT=${RUN_ROOT}"
