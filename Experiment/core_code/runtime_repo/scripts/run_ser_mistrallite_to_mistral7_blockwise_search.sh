#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
VENV="/disk_n/c101-3-mql/venvs/A800"
DATA_SOURCE_DIR="/disk_n/c101-3-mql/projects/scd_main/legacy/no4/data/cmrc2018"

SEED="${SEED:-42}"
TRAIN_LIMIT="${TRAIN_LIMIT:-200}"
SEARCH_LIMIT="${SEARCH_LIMIT:-200}"
BLOCK_SIZE="${BLOCK_SIZE:-4}"
MAX_LENGTH="${MAX_LENGTH:-1024}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"
LAYERS_PER_SHARD="${LAYERS_PER_SHARD:-4}"
PRODUCER_DEVICE="${PRODUCER_DEVICE:-cuda:0}"
CONSUMER_DEVICE="${CONSUMER_DEVICE:-cuda:1}"
TRAIN_DEVICE="${TRAIN_DEVICE:-cuda:2}"
WORKER_PAIRS="${WORKER_PAIRS:-cuda:0,cuda:1;cuda:2,cuda:3}"
RUN_TAG="${RUN_TAG:-train200_blockwise_20260405}"
MAX_F1_DROP="${MAX_F1_DROP:-1.0}"
MAX_EM_DROP="${MAX_EM_DROP:-0.10}"
MIN_TTFT_GAIN_MS="${MIN_TTFT_GAIN_MS:-1.0}"
MAX_ACCEPTED_BLOCKS="${MAX_ACCEPTED_BLOCKS:-0}"
REPLAY_TOP_K="${REPLAY_TOP_K:-0}"
REPLAY_SELECTION_MODE="${REPLAY_SELECTION_MODE:-quality_threshold_ttft_first}"
REPLAY_DECISION_MAX_F1_DROP="${REPLAY_DECISION_MAX_F1_DROP:--1.0}"
REPLAY_DECISION_MAX_F1_DROP_RATIO="${REPLAY_DECISION_MAX_F1_DROP_RATIO:-0.05}"
REPLAY_DECISION_MAX_EM_DROP="${REPLAY_DECISION_MAX_EM_DROP:--1.0}"
CAPTURE_SKIP_VALIDATE="${CAPTURE_SKIP_VALIDATE:-1}"

MODELS_CONFIG="${MODELS_CONFIG:-configs/models_ser_mistrallite_to_mistral7_20260405.json}"
REUSE_TRAIN_CONFIG="${REUSE_TRAIN_CONFIG:-configs/train_reuse_mistral_self_layerwise_20260404.json}"
PATCH_TRAIN_CONFIG="${PATCH_TRAIN_CONFIG:-configs/mistral_self_patch_xnorm_nativekv_shared_residual_mlp_alllayers_e1_h2048_r512_b2_20260404.json}"
ROUTE_TEMPLATE="${ROUTE_TEMPLATE:-configs/route_reuse_patchonly_all32_20260404.json}"
PATCH_EVAL_CONFIG="${PATCH_EVAL_CONFIG:-configs/eval_patch_only_train50_20260405.json}"

RUN_ROOT="${RUN_ROOT:-${ROOT}/runs/mistrallite_to_mistral7_cmrc_locked_20260405}"
PAIR_DIR="${RUN_ROOT}/pairs_cmrc_train${TRAIN_LIMIT}_seed${SEED}"
REUSE_DIR="${RUN_ROOT}/ckpt_reuse_layerwise_seed${SEED}"
PROFILE_PATH="${RUN_ROOT}/profile_cmrc_train${TRAIN_LIMIT}_seed${SEED}.json"
PATCH_DIR="${RUN_ROOT}/ckpt_patch_xnorm_nativekv_shared_residual_mlp_all32_e1_h2048_r512_b2_seed${SEED}"
PATCH_DIR_SHARDED="${PATCH_DIR}_sharded_w${LAYERS_PER_SHARD}"
TRAIN_ONLY_ROOT="${RUN_ROOT}/tmp_cmrc_train_only_20260405"
LOCAL_CMRC_DIR="${TRAIN_ONLY_ROOT}/cmrc2018"
EVAL_ROOT="${RUN_ROOT}/data_eval_dev"
EVAL_CMRC_DIR="${EVAL_ROOT}/cmrc2018"
PROMPTS_JSON="${RUN_ROOT}/cmrc_train${TRAIN_LIMIT}_prompts_seed${SEED}.json"
SEARCH_OUT="${RUN_ROOT}/blockwise_search_${RUN_TAG}"
BASE_REUSE_OUT="${RUN_ROOT}/strict_baseline_reuse_only_${RUN_TAG}"
HYBRID_OUT="${RUN_ROOT}/strict_hybrid_${RUN_TAG}"
BASE_REUSE_CFG="${SEARCH_OUT}/generated_eval_baseline_reuse_only.json"

cd "${ROOT}"
source "${VENV}/bin/activate"
export PYTHONPATH="${ROOT}/src"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-16}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-16}"

mkdir -p "${RUN_ROOT}" "${SEARCH_OUT}" "${LOCAL_CMRC_DIR}" "${EVAL_CMRC_DIR}"

cp "${DATA_SOURCE_DIR}/cmrc2018_train.json" "${LOCAL_CMRC_DIR}/cmrc2018_train.json"
cp "${DATA_SOURCE_DIR}/cmrc2018_dev.json" "${EVAL_CMRC_DIR}/cmrc2018_dev.json"
cp "${DATA_SOURCE_DIR}/cmrc2018_train.json" "${EVAL_CMRC_DIR}/cmrc2018_train.json"
if [ -f "${DATA_SOURCE_DIR}/README.md" ]; then
  cp "${DATA_SOURCE_DIR}/README.md" "${LOCAL_CMRC_DIR}/README.md"
  cp "${DATA_SOURCE_DIR}/README.md" "${EVAL_CMRC_DIR}/README.md"
fi

python - <<PY
import json
from pathlib import Path
from scd_repro.cmrc import load_cmrc2018_samples

rows = load_cmrc2018_samples("${TRAIN_ONLY_ROOT}", limit=int("${TRAIN_LIMIT}"), split="train")
Path("${PROMPTS_JSON}").write_text(
    json.dumps([x.prompt for x in rows], ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
print(f"prepared_train_prompts={len(rows)}")
PY

if [ ! -f "${PAIR_DIR}/manifest.json" ]; then
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
fi

if [ ! -f "${REUSE_DIR}/meta.json" ]; then
  python -m scd_repro.cli.train_reuse \
    --pairs "${PAIR_DIR}" \
    --train-config "${REUSE_TRAIN_CONFIG}" \
    --out "${REUSE_DIR}" \
    --seed "${SEED}" \
    --train-device "${TRAIN_DEVICE}" \
    --log "${RUN_ROOT}/train_reuse_layerwise_seed${SEED}.log"
fi

if [ ! -f "${PROFILE_PATH}" ]; then
  python -m scd_repro.cli.profile_layers \
    --pairs "${PAIR_DIR}" \
    --reuse-ckpt "${REUSE_DIR}" \
    --out "${PROFILE_PATH}" \
    --train-device "${TRAIN_DEVICE}"
fi

if [ ! -f "${PATCH_DIR}/meta.json" ]; then
  python -m scd_repro.cli.train_patch \
    --pairs "${PAIR_DIR}" \
    --profile "${PROFILE_PATH}" \
    --route "${ROUTE_TEMPLATE}" \
    --train-config "${PATCH_TRAIN_CONFIG}" \
    --models-config "${MODELS_CONFIG}" \
    --out "${PATCH_DIR}" \
    --seed "${SEED}" \
    --train-device "${TRAIN_DEVICE}" \
    --log "${RUN_ROOT}/train_patch_shared_seed${SEED}.log"
fi

if [ ! -f "${PATCH_DIR_SHARDED}/meta.json" ]; then
  python - <<PY
from scd_repro.pipeline import reshard_patch_ckpt

reshard_patch_ckpt(
    "${PATCH_DIR}",
    "${PATCH_DIR_SHARDED}",
    layers_per_shard=int("${LAYERS_PER_SHARD}"),
    drop_identity_residual=False,
)
PY
fi

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
  --seed "${SEED}" \
  --log "${SEARCH_OUT}/search.log"

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
Path("${BASE_REUSE_CFG}").write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
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
  --log "${BASE_REUSE_OUT}.log"

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
  --log "${HYBRID_OUT}.log"

python - <<PY
import json
from pathlib import Path

def mode_metrics(path: str, key: str):
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    row = obj["modes"][key]
    return {
        "f1": row.get("f1"),
        "em": row.get("em"),
        "ttft_ms": row.get("ttft_ms"),
        "encode_ms": row.get("encode_ms"),
        "transfer_ms": row.get("transfer_ms"),
        "rebuild_ms": row.get("rebuild_ms"),
        "patch_infer_ms": row.get("patch_infer_ms"),
        "decode_first_ms": row.get("decode_first_ms"),
    }

summary = {
    "baseline": mode_metrics("${BASE_REUSE_OUT}/final_comparison.json", "full_consumer"),
    "reuse_only": mode_metrics("${BASE_REUSE_OUT}/final_comparison.json", "reuse_only"),
    "patch_only": mode_metrics("${SEARCH_OUT}/eval_anchor_patch_only/final_comparison.json", "patch_only"),
    "hybrid_best": mode_metrics("${HYBRID_OUT}/final_comparison.json", "reuse_patch"),
    "selected_hybrid_patch_layers": json.loads(Path("${SEARCH_OUT}/search_report.json").read_text(encoding="utf-8")).get("selected_hybrid_patch_layers", []),
    "accepted_block_ids": json.loads(Path("${SEARCH_OUT}/search_report.json").read_text(encoding="utf-8")).get("accepted_block_ids", []),
}
out = Path("${SEARCH_OUT}/final_summary.json")
out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY
