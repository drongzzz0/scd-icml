#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CONDA_ENV="${CONDA_ENV:-cache}"
VENV="${VENV:-}"
DATA_ROOT="${DATA_ROOT:-/home/c101-800-mql/maqianli/no4/data}"
MODELS_CONFIG="${MODELS_CONFIG:-configs/models_a800_qwen25_32b_base_to_instruct_template.json}"
SCD_STORAGE_ROOT="${SCD_STORAGE_ROOT:-/home/c101-800-mql/nas/SCD-ICML}"
MODEL_CACHE_ROOT="${MODEL_CACHE_ROOT:-${SCD_STORAGE_ROOT}/model_cache}"

STAMP="$(date +%Y%m%d-%H%M%S)"
choose_default_run_parent() {
  if [ -n "${RUN_PARENT:-}" ]; then
    echo "${RUN_PARENT}"
    return
  fi
  local candidate
  for candidate in \
    "${SCD_STORAGE_ROOT}/runs" \
    "/nas/mql/SCD-ICML/runs" \
    "/data/SCD-ICML/runs"
  do
    if mkdir -p "${candidate}" 2>/dev/null && [ -w "${candidate}" ]; then
      echo "${candidate}"
      return
    fi
  done
  echo "No writable storage run directory found. Set RUN_PARENT or SCD_STORAGE_ROOT to a storage/NAS path." >&2
  exit 2
}
DEFAULT_RUN_PARENT="$(choose_default_run_parent)"
RUN_ROOT="${RUN_ROOT:-${DEFAULT_RUN_PARENT}/scd_icml_qwen32_repro_${STAMP}}"

SEED="${SEED:-42}"
TRAIN_LIMIT="${TRAIN_LIMIT:-200}"
EVAL_LIMIT="${EVAL_LIMIT:-200}"
TRAIN_CMRC_SPLIT="${TRAIN_CMRC_SPLIT:-train}"
EVAL_CMRC_SPLIT="${EVAL_CMRC_SPLIT:-dev}"
DATASETS="${DATASETS:-cmrc2018}"
MAX_LENGTH="${MAX_LENGTH:-0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"
BANDWIDTH_GBPS="${BANDWIDTH_GBPS:-200}"
QUANT_GROUP_SIZE="${QUANT_GROUP_SIZE:-128}"

REQUIRED_GPUS="${REQUIRED_GPUS:-2}"
if [ -z "${REQUIRED_FREE_MB+x}" ] && [ "${REQUIRED_GPUS}" = "1" ]; then
  REQUIRED_FREE_MB=78000
else
  REQUIRED_FREE_MB="${REQUIRED_FREE_MB:-70000}"
fi
GPU_MONITOR_SECONDS="${GPU_MONITOR_SECONDS:-60}"
GPU_SAMPLE_INTERVAL="${GPU_SAMPLE_INTERVAL:-10}"

CAPTURE_A_DEVICE="${CAPTURE_A_DEVICE:-cuda:0}"
CAPTURE_B_DEVICE="${CAPTURE_B_DEVICE:-cuda:1}"
TRAIN_DEVICE="${TRAIN_DEVICE:-cuda:0}"
PRODUCER_DEVICE="${PRODUCER_DEVICE:-cuda:0}"
CONSUMER_DEVICE="${CONSUMER_DEVICE:-cuda:1}"

REUSE_TRAIN_CONFIG="${REUSE_TRAIN_CONFIG:-configs/train_reuse_qwen32_layerwise_20260614.json}"
PATCH_TRAIN_CONFIG="${PATCH_TRAIN_CONFIG:-configs/qwen32_patch_xnorm_nativekv_shared_residual_mlp_alllayers_e1_h2048_r512_b1_20260614.json}"
EVAL_CONFIG_TEMPLATE="${EVAL_CONFIG_TEMPLATE:-configs/eval_scd_icml_qwen32_no_fallback_no_recompute.json}"
PATCH_BUDGET="${PATCH_BUDGET:-6}"
ROUTE_PATCH_LAYERS="${ROUTE_PATCH_LAYERS:-}"

mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/tmp_cfg" "${RUN_ROOT}/prompts" "${RUN_ROOT}/route"

cd "${ROOT}"
if [ -n "${VENV}" ] && [ -f "${VENV}/bin/activate" ]; then
  source "${VENV}/bin/activate"
elif command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate "${CONDA_ENV}"
elif [ -x /opt/miniconda3/bin/conda ]; then
  eval "$(/opt/miniconda3/bin/conda shell.bash hook)"
  conda activate "${CONDA_ENV}"
fi

python -m pip install -e .
export PYTHONPATH="${ROOT}/src"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export SCD_CAPTURE_STORE_DEVICE="${SCD_CAPTURE_STORE_DEVICE:-cpu}"
mkdir -p "${MODEL_CACHE_ROOT}/hf_home" \
  "${MODEL_CACHE_ROOT}/hf_hub" \
  "${MODEL_CACHE_ROOT}/hf_assets" \
  "${MODEL_CACHE_ROOT}/transformers" \
  "${MODEL_CACHE_ROOT}/modelscope" \
  "${MODEL_CACHE_ROOT}/torch" \
  "${MODEL_CACHE_ROOT}/xdg" \
  "${MODEL_CACHE_ROOT}/vllm"
export HF_HOME="${HF_HOME:-${MODEL_CACHE_ROOT}/hf_home}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${MODEL_CACHE_ROOT}/hf_hub}"
export HF_ASSETS_CACHE="${HF_ASSETS_CACHE:-${MODEL_CACHE_ROOT}/hf_assets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${MODEL_CACHE_ROOT}/transformers}"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-${MODEL_CACHE_ROOT}/modelscope}"
export TORCH_HOME="${TORCH_HOME:-${MODEL_CACHE_ROOT}/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${MODEL_CACHE_ROOT}/xdg}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-${MODEL_CACHE_ROOT}/vllm}"

python scripts/preflight_qwen32_a800.py \
  --models-config "${MODELS_CONFIG}" \
  --data-root "${DATA_ROOT}" \
  --datasets "${DATASETS}" \
  --required-gpus "${REQUIRED_GPUS}" \
  --required-free-mb "${REQUIRED_FREE_MB}" \
  --monitor-seconds "${GPU_MONITOR_SECONDS}" \
  --sample-interval "${GPU_SAMPLE_INTERVAL}" \
  --out "${RUN_ROOT}/preflight_qwen32_a800.json" \
  --write-env "${RUN_ROOT}/selected_gpus.env"

source "${RUN_ROOT}/selected_gpus.env"
echo "selected CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

cat > "${RUN_ROOT}/run_config.json" <<EOF
{
  "root": "${ROOT}",
  "data_root": "${DATA_ROOT}",
  "storage_root": "${SCD_STORAGE_ROOT}",
  "run_parent": "${DEFAULT_RUN_PARENT}",
  "run_root": "${RUN_ROOT}",
  "models_config": "${MODELS_CONFIG}",
  "model_cache_root": "${MODEL_CACHE_ROOT}",
  "hf_home": "${HF_HOME}",
  "hf_hub_cache": "${HF_HUB_CACHE}",
  "hf_assets_cache": "${HF_ASSETS_CACHE}",
  "transformers_cache": "${TRANSFORMERS_CACHE}",
  "modelscope_cache": "${MODELSCOPE_CACHE}",
  "torch_home": "${TORCH_HOME}",
  "xdg_cache_home": "${XDG_CACHE_HOME}",
  "vllm_cache_root": "${VLLM_CACHE_ROOT}",
  "scd_capture_store_device": "${SCD_CAPTURE_STORE_DEVICE}",
  "seed": ${SEED},
  "train_limit": ${TRAIN_LIMIT},
  "eval_limit": ${EVAL_LIMIT},
  "train_cmrc_split": "${TRAIN_CMRC_SPLIT}",
  "eval_cmrc_split": "${EVAL_CMRC_SPLIT}",
  "datasets": "${DATASETS}",
  "max_length": ${MAX_LENGTH},
  "max_new_tokens": ${MAX_NEW_TOKENS},
  "bandwidth_gbps": ${BANDWIDTH_GBPS},
  "patch_budget": ${PATCH_BUDGET},
  "route_patch_layers": "${ROUTE_PATCH_LAYERS}",
  "cuda_visible_devices": "${CUDA_VISIBLE_DEVICES}",
  "capture_a_device": "${CAPTURE_A_DEVICE}",
  "capture_b_device": "${CAPTURE_B_DEVICE}",
  "train_device": "${TRAIN_DEVICE}",
  "producer_device": "${PRODUCER_DEVICE}",
  "consumer_device": "${CONSUMER_DEVICE}"
}
EOF

run_logged() {
  local name="$1"
  shift
  local stage_dir="${RUN_ROOT}/logs/${name}"
  local status_path="${stage_dir}/status.json"
  rm -rf "${stage_dir}"
  mkdir -p "${stage_dir}"
  local start_ts end_ts exit_code
  start_ts="$(date +%s)"
  printf '%q ' "$@" > "${stage_dir}/command.txt"
  printf '\n' >> "${stage_dir}/command.txt"
  if [ -x /usr/bin/time ]; then
    if /usr/bin/time -v -o "${stage_dir}/time.txt" "$@" >"${stage_dir}/stdout.log" 2>"${stage_dir}/stderr.log"; then
      exit_code=0
    else
      exit_code=$?
    fi
  else
    echo "/usr/bin/time unavailable; running without resource-time wrapper" > "${stage_dir}/time.txt"
    if "$@" >"${stage_dir}/stdout.log" 2>"${stage_dir}/stderr.log"; then
      exit_code=0
    else
      exit_code=$?
    fi
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

make_eval_config() {
  local out_path="$1"
  python - "${EVAL_CONFIG_TEMPLATE}" "${out_path}" "${EVAL_LIMIT}" "${BANDWIDTH_GBPS}" "${EVAL_CMRC_SPLIT}" <<'PY'
import json, sys
template, out_path, eval_limit, bandwidth, cmrc_split = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4]), sys.argv[5]
cfg = json.load(open(template, encoding="utf-8-sig"))
cfg.setdefault("network", {})["bandwidth_gbps"] = bandwidth
cfg.setdefault("runtime_eval", {})["max_eval_samples"] = eval_limit
cfg.setdefault("runtime_eval", {})["cmrc_split"] = cmrc_split or "dev"
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
out.with_suffix(out.suffix + ".meta.json").write_text(
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
print(f"wrote_prompts={len(samples)} split={train_cmrc_split} path={out_path}")
PY
}

PAIR_DIR="${RUN_ROOT}/pairs/cmrc2018_train_seed${SEED}_n${TRAIN_LIMIT}"
REUSE_DIR="${RUN_ROOT}/ckpt_reuse/cmrc2018_seed${SEED}"
PROFILE_PATH="${RUN_ROOT}/route/profile_cmrc2018_seed${SEED}.json"
PATCH_DIR="${RUN_ROOT}/ckpt_patch/patch_cmrc2018_seed${SEED}"
ROUTE_PATH="${RUN_ROOT}/route/qwen32_route_k${PATCH_BUDGET}_seed${SEED}.json"
PROMPTS_JSON="${RUN_ROOT}/prompts/cmrc2018_train_prompts_seed${SEED}_n${TRAIN_LIMIT}.json"
EVAL_CONFIG="${RUN_ROOT}/tmp_cfg/eval_qwen32.json"

make_eval_config "${EVAL_CONFIG}"
make_train_prompts "${PROMPTS_JSON}"

run_logged "capture_pair" \
  python -m scd_repro.cli.capture_pair \
    --backend transformers \
    --models-config "${MODELS_CONFIG}" \
    --data-config configs/data_a800.yaml \
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

run_logged "train_reuse" \
  python -m scd_repro.cli.train_reuse \
    --pairs "${PAIR_DIR}" \
    --models-config "${MODELS_CONFIG}" \
    --train-config "${REUSE_TRAIN_CONFIG}" \
    --out "${REUSE_DIR}" \
    --seed "${SEED}" \
    --train-device "${TRAIN_DEVICE}"

run_logged "profile_layers" \
  python -m scd_repro.cli.profile_layers \
    --pairs "${PAIR_DIR}" \
    --reuse-ckpt "${REUSE_DIR}" \
    --out "${PROFILE_PATH}" \
    --train-device "${TRAIN_DEVICE}"

run_logged "train_patch" \
  python -m scd_repro.cli.train_patch \
    --pairs "${PAIR_DIR}" \
    --profile "${PROFILE_PATH}" \
    --train-config "${PATCH_TRAIN_CONFIG}" \
    --models-config "${MODELS_CONFIG}" \
    --out "${PATCH_DIR}" \
    --seed "${SEED}" \
    --train-device "${TRAIN_DEVICE}"

_patch_layer_args=()
if [ -n "${ROUTE_PATCH_LAYERS}" ]; then
  _patch_layer_args+=(--patch-layers "${ROUTE_PATCH_LAYERS}")
fi
run_logged "make_route" \
  python scripts/make_paper_scd_route.py \
    --profile "${PROFILE_PATH}" \
    --patch-dir "${PATCH_DIR}" \
    --models-config "${MODELS_CONFIG}" \
    --eval-config "${EVAL_CONFIG}" \
    --patch-budget "${PATCH_BUDGET}" \
    "${_patch_layer_args[@]}" \
    --out "${ROUTE_PATH}"

run_logged "validate_route" \
  python scripts/validate_scd_route.py \
    --route "${ROUTE_PATH}"

IFS=',' read -r -a DATASET_LIST <<< "${DATASETS}"
for dataset in "${DATASET_LIST[@]}"; do
  dataset="$(echo "${dataset}" | xargs)"
  [ -n "${dataset}" ] || continue
  OUT_DIR="${RUN_ROOT}/strict_eval/${dataset}"
  run_logged "eval_${dataset}" \
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

echo "SCD-ICML Qwen32 reproduction finished."
echo "RUN_ROOT=${RUN_ROOT}"
