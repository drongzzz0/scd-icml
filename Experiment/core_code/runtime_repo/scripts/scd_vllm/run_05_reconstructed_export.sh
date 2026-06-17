#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${SCD_RUNTIME_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

: "${SCD_VLLM_PRODUCER_MODEL:?set SCD_VLLM_PRODUCER_MODEL}"
: "${SCD_VLLM_CONSUMER_MODEL:?set SCD_VLLM_CONSUMER_MODEL}"
: "${SCD_VLLM_REUSE_CKPT:?set SCD_VLLM_REUSE_CKPT}"
: "${SCD_VLLM_PATCH_CKPT:?set SCD_VLLM_PATCH_CKPT}"
: "${SCD_VLLM_ROUTE:?set SCD_VLLM_ROUTE}"

SCD_RECONSTRUCTED_OUT="${SCD_RECONSTRUCTED_OUT:-runs/scd_vllm/reconstructed_reuse_patch_bundle}"
SCD_VLLM_DTYPE="${SCD_VLLM_DTYPE:-bfloat16}"
SCD_VLLM_TRANSFER_DTYPE="${SCD_VLLM_TRANSFER_DTYPE:-$SCD_VLLM_DTYPE}"
SCD_VLLM_PRODUCER_DEVICE="${SCD_VLLM_PRODUCER_DEVICE:-cuda:0}"
SCD_VLLM_CONSUMER_DEVICE="${SCD_VLLM_CONSUMER_DEVICE:-cuda:1}"
SCD_VLLM_PATCH_CKPT_DEVICE="${SCD_VLLM_PATCH_CKPT_DEVICE:-cpu}"
SCD_VLLM_MAX_PREFIX_TOKENS="${SCD_VLLM_MAX_PREFIX_TOKENS:-256}"
SCD_VLLM_BANDWIDTH_GBPS="${SCD_VLLM_BANDWIDTH_GBPS:-200}"
SCD_VLLM_QUANT_GROUP_SIZE="${SCD_VLLM_QUANT_GROUP_SIZE:-128}"
SCD_VLLM_RUNTIME_PATCH_WINDOW_SIZE="${SCD_VLLM_RUNTIME_PATCH_WINDOW_SIZE:-0}"

PROMPT_ARGS=()
if [[ -n "${SCD_VLLM_PREFIX_FILE:-}" ]]; then
  PROMPT_ARGS+=(--prefix-file "$SCD_VLLM_PREFIX_FILE")
elif [[ -n "${SCD_VLLM_PROMPT:-}" ]]; then
  PROMPT_ARGS+=(--prompt "$SCD_VLLM_PROMPT")
else
  PROMPT_ARGS+=(--prompt "Semantic Cache Distillation transfers approximate semantic key value caches between compatible agents.")
fi

EXTRA_ARGS=()
if [[ -n "${SCD_VLLM_TOKENIZER:-}" ]]; then
  EXTRA_ARGS+=(--tokenizer "$SCD_VLLM_TOKENIZER")
fi
if [[ "${SCD_VLLM_RUNTIME_PATCH_PERSIST_SHARDS:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--runtime-patch-persist-shards)
fi
if [[ "${SCD_VLLM_RUNTIME_PATCH_PERSIST_MODULES:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--runtime-patch-persist-modules)
fi

python -m scd_vllm.reconstructed_export \
  --producer-model "$SCD_VLLM_PRODUCER_MODEL" \
  --consumer-model "$SCD_VLLM_CONSUMER_MODEL" \
  --reuse-ckpt "$SCD_VLLM_REUSE_CKPT" \
  --patch-ckpt "$SCD_VLLM_PATCH_CKPT" \
  --route "$SCD_VLLM_ROUTE" \
  --dtype "$SCD_VLLM_DTYPE" \
  --transfer-dtype "$SCD_VLLM_TRANSFER_DTYPE" \
  --producer-device "$SCD_VLLM_PRODUCER_DEVICE" \
  --consumer-device "$SCD_VLLM_CONSUMER_DEVICE" \
  --patch-ckpt-device "$SCD_VLLM_PATCH_CKPT_DEVICE" \
  --max-prefix-tokens "$SCD_VLLM_MAX_PREFIX_TOKENS" \
  --bandwidth-gbps "$SCD_VLLM_BANDWIDTH_GBPS" \
  --quant-group-size "$SCD_VLLM_QUANT_GROUP_SIZE" \
  --runtime-patch-window-size "$SCD_VLLM_RUNTIME_PATCH_WINDOW_SIZE" \
  --out "$SCD_RECONSTRUCTED_OUT" \
  "${PROMPT_ARGS[@]}" \
  "${EXTRA_ARGS[@]}"
