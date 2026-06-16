#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/disk_n/c101-3-mql/projects/SCD-ICML/repro_current_from_codex_20260607-154030/runtime_repo}"
VENV="${VENV:-/disk_n/c101-3-mql/venvs/A800}"
DATA_ROOT="${DATA_ROOT:-/disk_n/c101-3-mql/projects/SCD-ICML/kv/snapshots/scd-github-f7d7854/data}"
RUN_PARENT="${RUN_PARENT:-/disk_n/c101-3-mql/projects/SCD-ICML/runs}"

SEED="${SEED:-42}"
TRAIN_SAMPLES="${TRAIN_SAMPLES:-20}"
EVAL_LIMIT="${EVAL_LIMIT:-20}"
MAX_LENGTH="${MAX_LENGTH:-1024}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"
BANDWIDTH_GBPS="${BANDWIDTH_GBPS:-200}"
PRODUCER_DEVICE="${PRODUCER_DEVICE:-cuda:0}"
CONSUMER_DEVICE="${CONSUMER_DEVICE:-cuda:1}"
TRAIN_DEVICE="${TRAIN_DEVICE:-cuda:1}"
WAIT="${WAIT:-1}"

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="${RUN_ROOT:-${RUN_PARENT}/extra_model_qwen17_bf16_cmrc_smoke_${STAMP}}"
CFG_DIR="${RUN_ROOT}/configs"
LOG_DIR="${RUN_ROOT}/logs"
PAIR_DIR="${RUN_ROOT}/pairs_cmrc_train${TRAIN_SAMPLES}_seed${SEED}"
REUSE_DIR="${RUN_ROOT}/ckpt_reuse_layerwise_seed${SEED}"
PROFILE_PATH="${RUN_ROOT}/profile_cmrc_train${TRAIN_SAMPLES}_seed${SEED}.json"
PATCH_DIR="${RUN_ROOT}/ckpt_patch_directkv_seed${SEED}"
ROUTE_PATH="${RUN_ROOT}/route_qwen17_bf16_directkv_seed${SEED}.json"
STRICT_OUT="${RUN_ROOT}/strict_cmrc_dev_n${EVAL_LIMIT}"

MODELS_CONFIG="${CFG_DIR}/models_qwen17_bf16_mql.json"
REUSE_CONFIG="${CFG_DIR}/train_reuse_layerwise_qwen17_smoke.json"
PATCH_CONFIG="${CFG_DIR}/train_patch_directkv_qwen17_smoke.json"
EVAL_CONFIG="${CFG_DIR}/eval_qwen17_smoke.json"
PROMPTS_JSON="${CFG_DIR}/cmrc_train_prompts_${TRAIN_SAMPLES}.json"

mkdir -p "${CFG_DIR}" "${LOG_DIR}"
cd "${ROOT}"
source "${VENV}/bin/activate"
export PYTHONPATH="${ROOT}/src"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

python - <<PY
import json
from pathlib import Path

cfg_dir = Path("${CFG_DIR}")
cfg_dir.mkdir(parents=True, exist_ok=True)

models = {
    "model_a": {
        "name": "Qwen3-1.7B",
        "path": "/disk_n/c101-3-mql/models/Qwen3-1.7B",
    },
    "model_b": {
        "name": "Qwen3-1.7B-bf16",
        "path": "/disk_n/c101-3-mql/models/Qwen3-1.7B-bf16",
    },
    "architecture": {
        "num_layers": 28,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "hidden_dim": 2048,
        "rope_theta": 1000000.0,
        "sliding_window": None,
    },
    "runtime": {"dtype": "bfloat16", "device": "${CONSUMER_DEVICE}", "cpu_fallback": True},
    "compute": {
        "model_a_device": "${PRODUCER_DEVICE}",
        "model_b_device": "${CONSUMER_DEVICE}",
        "accel_mode": "balanced",
        "layer_block_size": 4,
        "group_block_size": 2,
    },
}

reuse = {
    "mode": "xkv_group",
    "translator": {"default_path": "shared_z", "shared_z": {"hidden_dim": 0, "activation": "identity"}},
    "xkv_group": {
        "group_size": 1,
        "rank_k": 128,
        "rank_v": 128,
        "center_before_svd": False,
        "svd_backend": "moments_eigh",
        "max_groups": 128,
        "ridge_lambda": 0.0001,
    },
    "compute": {"train_device": "${TRAIN_DEVICE}", "accel_mode": "balanced", "group_block_size": 2},
    "runtime": {"cpu_fallback": True},
}

patch = {
    "mode": "xkv_group",
    "translator": {"default_path": "shared_z", "shared_z": {"hidden_dim": 0, "activation": "identity"}},
    "xkv_group": {
        "group_size": 4,
        "rank_k": 128,
        "rank_v": 128,
        "center_before_svd": False,
        "svd_backend": "moments_eigh",
        "max_groups": 128,
        "ridge_lambda": 0.0001,
    },
    "patch": {"fit_mode": "direct_kv_v1", "patch_budget": 6, "ridge_lambda": 0.0001},
    "compute": {"train_device": "${TRAIN_DEVICE}", "accel_mode": "balanced", "layer_block_size": 2, "group_block_size": 2},
    "runtime": {"cpu_fallback": True},
}

eval_cfg = {
    "network": {"bandwidth_gbps": float("${BANDWIDTH_GBPS}"), "scan_gbps": [100, 150, 200, 400, 800, 1000]},
    "route": {
        "enable_task_level_profile": False,
        "threshold_mode": "adaptive_p20_p80",
        "reuse_threshold": 0.002,
        "fallback_threshold": 0.02,
        "patch_budget": 6,
        "quality_metric": "delta_logp",
        "cost_metric": "delta_ttft_ms",
        "route_objective": "gain_per_cost",
        "top_m_candidates": 12,
        "latency_budget_ms": None,
        "min_positive_gain": 0.0,
        "enable_short_recompute": False,
        "recompute_segment_lengths": [1],
        "recompute_eval_mode": "best_of_lengths",
        "allow_patch_only": False,
        "allow_patch_plus_recompute": False,
        "cache_search_results": False,
        "max_evals": 0,
        "restore_one_batch_size": 0,
        "greedy_batch_size": 0,
        "recompute_cost_per_layer_ms": 1.0,
        "patch_compute_cost_ms_per_layer": 0.0,
        "weighted_cost_ttft_coeff": 1.0,
        "weighted_cost_bytes_coeff": 0.0,
    },
    "acceptance": {"f1_abs_gap_le": 0.03, "kl_rel_gap_le": 0.10, "tv_rel_gap_le": 0.10, "ppl_rel_gap_le": 0.10, "ttft_rel_gap_le": 0.10},
    "compute": {"infer_device": "${CONSUMER_DEVICE}", "accel_mode": "balanced", "layer_block_size": 4, "group_block_size": 2},
    "runtime": {"cpu_fallback": True},
    "runtime_eval": {
        "run_full_consumer_baseline": True,
        "run_reuse_only": True,
        "run_reuse_fallback": True,
        "run_reuse_patch": True,
        "run_reuse_patch_recompute": False,
        "align_full_consumer_first_decode": True,
        "measure_ttft": True,
        "measure_transfer_bytes": True,
        "measure_teacher_forced_metrics": True,
        "measure_generation_metrics": True,
        "max_eval_samples": int("${EVAL_LIMIT}"),
        "save_per_sample_metrics": True,
        "save_generation_outputs": True,
        "output_dir": "",
        "default_recompute_segment_length": 0,
        "override_route_recompute_length": None,
        "strict_mode": True,
        "fail_on_missing_route_fields": False,
        "fail_on_missing_patch_fields": False,
    },
}

Path("${MODELS_CONFIG}").write_text(json.dumps(models, ensure_ascii=False, indent=2) + "\\n", encoding="utf-8")
Path("${REUSE_CONFIG}").write_text(json.dumps(reuse, ensure_ascii=False, indent=2) + "\\n", encoding="utf-8")
Path("${PATCH_CONFIG}").write_text(json.dumps(patch, ensure_ascii=False, indent=2) + "\\n", encoding="utf-8")
Path("${EVAL_CONFIG}").write_text(json.dumps(eval_cfg, ensure_ascii=False, indent=2) + "\\n", encoding="utf-8")
PY

python - <<PY
import json
from pathlib import Path
from scd_repro.cmrc import load_cmrc2018_samples

rows = load_cmrc2018_samples("${DATA_ROOT}", limit=int("${TRAIN_SAMPLES}"), split="train")
Path("${PROMPTS_JSON}").write_text(
    json.dumps([x.prompt for x in rows], ensure_ascii=False, indent=2) + "\\n",
    encoding="utf-8",
)
PY

run_pipeline() {
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
    --log "${LOG_DIR}/capture_pair.log"

  python -m scd_repro.cli.train_reuse \
    --pairs "${PAIR_DIR}" \
    --train-config "${REUSE_CONFIG}" \
    --out "${REUSE_DIR}" \
    --seed "${SEED}" \
    --train-device "${TRAIN_DEVICE}" \
    --log "${LOG_DIR}/train_reuse.log"

  python -m scd_repro.cli.profile_layers \
    --pairs "${PAIR_DIR}" \
    --reuse-ckpt "${REUSE_DIR}" \
    --out "${PROFILE_PATH}" \
    --train-device "${TRAIN_DEVICE}" \
    --log "${LOG_DIR}/profile_layers.log"

  python -m scd_repro.cli.train_patch \
    --pairs "${PAIR_DIR}" \
    --profile "${PROFILE_PATH}" \
    --models-config "${MODELS_CONFIG}" \
    --train-config "${PATCH_CONFIG}" \
    --out "${PATCH_DIR}" \
    --seed "${SEED}" \
    --train-device "${TRAIN_DEVICE}" \
    --log "${LOG_DIR}/train_patch.log"

  python -m scd_repro.cli.build_route \
    --profile "${PROFILE_PATH}" \
    --eval-config "${EVAL_CONFIG}" \
    --out "${ROUTE_PATH}" \
    --log "${LOG_DIR}/build_route.log"

  python -m scd_repro.cli.eval_text_strict \
    --dataset cmrc2018 \
    --models-config "${MODELS_CONFIG}" \
    --eval-config "${EVAL_CONFIG}" \
    --data-root "${DATA_ROOT}" \
    --route "${ROUTE_PATH}" \
    --reuse-ckpt "${REUSE_DIR}" \
    --patch-ckpt "${PATCH_DIR}" \
    --out "${STRICT_OUT}" \
    --limit "${EVAL_LIMIT}" \
    --max-length "${MAX_LENGTH}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --bandwidth-gbps "${BANDWIDTH_GBPS}" \
    --quant-group-size 128 \
    --producer-device "${PRODUCER_DEVICE}" \
    --consumer-device "${CONSUMER_DEVICE}" \
    --dtype bfloat16 \
    --seed "${SEED}" \
    --log "${LOG_DIR}/eval_text_strict.log"
}

cat > "${RUN_ROOT}/run_config.json" <<JSON
{
  "run_root": "${RUN_ROOT}",
  "model_pair": "Qwen3-1.7B_to_Qwen3-1.7B-bf16",
  "dataset": "CMRC2018",
  "train_split": "train",
  "eval_split": "dev",
  "train_samples": ${TRAIN_SAMPLES},
  "eval_limit": ${EVAL_LIMIT},
  "producer_device": "${PRODUCER_DEVICE}",
  "consumer_device": "${CONSUMER_DEVICE}",
  "train_device": "${TRAIN_DEVICE}",
  "patch_fit_mode": "direct_kv_v1"
}
JSON

if [ "${WAIT}" = "0" ]; then
  nohup env \
    ROOT="${ROOT}" \
    VENV="${VENV}" \
    DATA_ROOT="${DATA_ROOT}" \
    RUN_PARENT="${RUN_PARENT}" \
    RUN_ROOT="${RUN_ROOT}" \
    SEED="${SEED}" \
    TRAIN_SAMPLES="${TRAIN_SAMPLES}" \
    EVAL_LIMIT="${EVAL_LIMIT}" \
    MAX_LENGTH="${MAX_LENGTH}" \
    MAX_NEW_TOKENS="${MAX_NEW_TOKENS}" \
    BANDWIDTH_GBPS="${BANDWIDTH_GBPS}" \
    PRODUCER_DEVICE="${PRODUCER_DEVICE}" \
    CONSUMER_DEVICE="${CONSUMER_DEVICE}" \
    TRAIN_DEVICE="${TRAIN_DEVICE}" \
    WAIT=1 \
    OMP_NUM_THREADS="${OMP_NUM_THREADS}" \
    bash "${BASH_SOURCE[0]}" > "${LOG_DIR}/pipeline.nohup.log" 2>&1 &
  echo "$!" > "${RUN_ROOT}/pid.txt"
  echo "RUN_ROOT=${RUN_ROOT}"
  echo "PID=$(cat "${RUN_ROOT}/pid.txt")"
else
  run_pipeline
  echo "RUN_ROOT=${RUN_ROOT}"
fi
