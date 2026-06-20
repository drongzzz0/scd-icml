#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-/home/c101-800-mql/.conda/envs/cache/bin/python}"
CONDA_ENV="${CONDA_ENV:-cache}"
DATA_ROOT="${DATA_ROOT:-/home/c101-800-mql/nas/SCD-ICML/data}"
MODELS_CONFIG="${MODELS_CONFIG:-configs/models_a800_qwen25_32b_base_to_instruct_template.json}"
SCD_STORAGE_ROOT="${SCD_STORAGE_ROOT:-/home/c101-800-mql/nas/SCD-ICML}"
MODEL_CACHE_ROOT="${MODEL_CACHE_ROOT:-${SCD_STORAGE_ROOT}/model_cache}"
SOURCE_RUN_ROOT="${SOURCE_RUN_ROOT:-/home/c101-800-mql/nas/SCD-ICML/runs/qwen32_cmrc200_nasdata_maxlen512_free65000_fix1_20260618_192206}"

SEED="${SEED:-42}"
TRAIN_LIMIT="${TRAIN_LIMIT:-200}"
EVAL_LIMIT="${EVAL_LIMIT:-200}"
TRAIN_CMRC_SPLIT="${TRAIN_CMRC_SPLIT:-train}"
EVAL_CMRC_SPLIT="${EVAL_CMRC_SPLIT:-dev}"
DATASETS="${DATASETS:-cmrc2018}"
MAX_LENGTH="${MAX_LENGTH:-512}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"
BANDWIDTH_GBPS="${BANDWIDTH_GBPS:-200}"
QUANT_GROUP_SIZE="${QUANT_GROUP_SIZE:-128}"
REUSE_RANK="${REUSE_RANK:-64}"
PATCH_BUDGET="${PATCH_BUDGET:-6}"
ROUTE_PATCH_LAYERS="${ROUTE_PATCH_LAYERS:-}"

REUSE_TRAIN_CONFIG_BASE="${REUSE_TRAIN_CONFIG_BASE:-configs/train_reuse_qwen32_layerwise_20260614.json}"
PATCH_TRAIN_CONFIG="${PATCH_TRAIN_CONFIG:-configs/qwen32_patch_xnorm_nativekv_shared_residual_mlp_alllayers_e1_h2048_r512_b1_20260614.json}"
EVAL_CONFIG_TEMPLATE="${EVAL_CONFIG_TEMPLATE:-configs/eval_scd_icml_qwen32_no_fallback_no_recompute.json}"

PAIR_DIR="${PAIR_DIR:-${SOURCE_RUN_ROOT}/pairs/cmrc2018_train_seed${SEED}_n${TRAIN_LIMIT}}"
STAMP="$(date +%Y%m%d-%H%M%S)"
RUN_PARENT="${RUN_PARENT:-${SCD_STORAGE_ROOT}/runs}"
RUN_ROOT="${RUN_ROOT:-${RUN_PARENT}/qwen32_cmrc${TRAIN_LIMIT}_rank${REUSE_RANK}_frompairs_${STAMP}}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
CAPTURE_A_DEVICE="${CAPTURE_A_DEVICE:-cuda:0}"
CAPTURE_B_DEVICE="${CAPTURE_B_DEVICE:-cuda:0}"
TRAIN_DEVICE="${TRAIN_DEVICE:-cuda:0}"
PRODUCER_DEVICE="${PRODUCER_DEVICE:-cuda:0}"
CONSUMER_DEVICE="${CONSUMER_DEVICE:-cuda:0}"

mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/tmp_cfg" "${RUN_ROOT}/route"
cd "${ROOT}"

if [ ! -x "${PYTHON_BIN}" ]; then
  if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate "${CONDA_ENV}"
    PYTHON_BIN="$(command -v python)"
  elif [ -x /opt/miniconda3/bin/conda ]; then
    eval "$(/opt/miniconda3/bin/conda shell.bash hook)"
    conda activate "${CONDA_ENV}"
    PYTHON_BIN="$(command -v python)"
  else
    PYTHON_BIN="$(command -v python)"
  fi
fi

export PYTHONPATH="${ROOT}/src"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
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

if [ ! -f "${PAIR_DIR}/manifest.json" ]; then
  echo "missing pair manifest: ${PAIR_DIR}/manifest.json" >&2
  exit 2
fi

nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits > "${RUN_ROOT}/gpu_preflight.csv" || true

REUSE_TRAIN_CONFIG="${RUN_ROOT}/tmp_cfg/train_reuse_rank${REUSE_RANK}.json"
EVAL_CONFIG="${RUN_ROOT}/tmp_cfg/eval_qwen32.json"
REUSE_DIR="${RUN_ROOT}/ckpt_reuse/cmrc2018_seed${SEED}_rank${REUSE_RANK}"
PROFILE_PATH="${RUN_ROOT}/route/profile_cmrc2018_seed${SEED}_rank${REUSE_RANK}.json"
PATCH_DIR="${RUN_ROOT}/ckpt_patch/patch_cmrc2018_seed${SEED}_rank${REUSE_RANK}"
ROUTE_PATH="${RUN_ROOT}/route/qwen32_route_k${PATCH_BUDGET}_rank${REUSE_RANK}_seed${SEED}.json"

"${PYTHON_BIN}" - "${REUSE_TRAIN_CONFIG_BASE}" "${REUSE_TRAIN_CONFIG}" "${REUSE_RANK}" <<'PY'
import json, sys
src, dst, rank = sys.argv[1], sys.argv[2], int(sys.argv[3])
cfg = json.load(open(src, encoding="utf-8-sig"))
cfg.setdefault("xkv_group", {})["rank_k"] = rank
cfg.setdefault("xkv_group", {})["rank_v"] = rank
cfg.setdefault("compute", {})["train_device"] = "cuda:0"
cfg.setdefault("compute", {})["group_block_size"] = 1
cfg.setdefault("runtime", {})["cpu_fallback"] = False
open(dst, "w", encoding="utf-8").write(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n")
PY

"${PYTHON_BIN}" - "${EVAL_CONFIG_TEMPLATE}" "${EVAL_CONFIG}" "${EVAL_LIMIT}" "${BANDWIDTH_GBPS}" "${EVAL_CMRC_SPLIT}" <<'PY'
import json, sys
template, out_path, eval_limit, bandwidth, cmrc_split = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4]), sys.argv[5]
cfg = json.load(open(template, encoding="utf-8-sig"))
cfg.setdefault("network", {})["bandwidth_gbps"] = bandwidth
cfg.setdefault("runtime_eval", {})["max_eval_samples"] = eval_limit
cfg.setdefault("runtime_eval", {})["cmrc_split"] = cmrc_split or "dev"
open(out_path, "w", encoding="utf-8").write(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n")
PY

cat > "${RUN_ROOT}/run_config.json" <<EOF
{
  "root": "${ROOT}",
  "data_root": "${DATA_ROOT}",
  "storage_root": "${SCD_STORAGE_ROOT}",
  "source_run_root": "${SOURCE_RUN_ROOT}",
  "pair_dir": "${PAIR_DIR}",
  "run_root": "${RUN_ROOT}",
  "models_config": "${MODELS_CONFIG}",
  "reuse_train_config_base": "${REUSE_TRAIN_CONFIG_BASE}",
  "reuse_train_config": "${REUSE_TRAIN_CONFIG}",
  "patch_train_config": "${PATCH_TRAIN_CONFIG}",
  "eval_config_template": "${EVAL_CONFIG_TEMPLATE}",
  "eval_config": "${EVAL_CONFIG}",
  "model_cache_root": "${MODEL_CACHE_ROOT}",
  "python_bin": "${PYTHON_BIN}",
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
  "reuse_rank": ${REUSE_RANK},
  "route_patch_layers": "${ROUTE_PATCH_LAYERS}",
  "cuda_visible_devices": "${CUDA_VISIBLE_DEVICES}",
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
    if "$@" >"${stage_dir}/stdout.log" 2>"${stage_dir}/stderr.log"; then
      exit_code=0
    else
      exit_code=$?
    fi
  fi
  end_ts="$(date +%s)"
  "${PYTHON_BIN}" - "${status_path}" "${name}" "${exit_code}" "${start_ts}" "${end_ts}" <<'PY'
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

run_logged "train_reuse" \
  "${PYTHON_BIN}" -m scd_repro.cli.train_reuse \
    --pairs "${PAIR_DIR}" \
    --models-config "${MODELS_CONFIG}" \
    --train-config "${REUSE_TRAIN_CONFIG}" \
    --out "${REUSE_DIR}" \
    --seed "${SEED}" \
    --train-device "${TRAIN_DEVICE}"

run_logged "profile_layers" \
  "${PYTHON_BIN}" -m scd_repro.cli.profile_layers \
    --pairs "${PAIR_DIR}" \
    --reuse-ckpt "${REUSE_DIR}" \
    --models-config "${MODELS_CONFIG}" \
    --out "${PROFILE_PATH}" \
    --train-device "${TRAIN_DEVICE}"

run_logged "train_patch" \
  "${PYTHON_BIN}" -m scd_repro.cli.train_patch \
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
  "${PYTHON_BIN}" scripts/make_paper_scd_route.py \
    --profile "${PROFILE_PATH}" \
    --patch-dir "${PATCH_DIR}" \
    --models-config "${MODELS_CONFIG}" \
    --eval-config "${EVAL_CONFIG}" \
    --patch-budget "${PATCH_BUDGET}" \
    "${_patch_layer_args[@]}" \
    --out "${ROUTE_PATH}"

run_logged "validate_route" \
  "${PYTHON_BIN}" scripts/validate_scd_route.py \
    --route "${ROUTE_PATH}"

IFS=',' read -r -a DATASET_LIST <<< "${DATASETS}"
for dataset in "${DATASET_LIST[@]}"; do
  dataset="$(echo "${dataset}" | xargs)"
  [ -n "${dataset}" ] || continue
  OUT_DIR="${RUN_ROOT}/strict_eval/${dataset}"
  run_logged "eval_${dataset}" \
    "${PYTHON_BIN}" -m scd_repro.cli.eval_text_strict \
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

echo "SCD-ICML Qwen32 from-pairs reproduction finished."
echo "RUN_ROOT=${RUN_ROOT}"
