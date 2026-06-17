#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${SCD_RUNTIME_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

: "${SCD_VLLM_CONSUMER_MODEL:?set SCD_VLLM_CONSUMER_MODEL}"
SCD_VLLM_HF_EXPORT_OUT="${SCD_VLLM_HF_EXPORT_OUT:-runs/scd_vllm/hf_full_consumer/full_consumer}"
SCD_VLLM_DTYPE="${SCD_VLLM_DTYPE:-bfloat16}"
SCD_VLLM_DEVICE="${SCD_VLLM_DEVICE:-cuda:0}"
SCD_VLLM_MAX_PREFIX_TOKENS="${SCD_VLLM_MAX_PREFIX_TOKENS:-128}"

PROMPT_ARGS=()
if [[ -n "${SCD_VLLM_PREFIX_FILE:-}" ]]; then
  PROMPT_ARGS+=(--prefix-file "$SCD_VLLM_PREFIX_FILE")
elif [[ -n "${SCD_VLLM_PROMPT:-}" ]]; then
  PROMPT_ARGS+=(--prompt "$SCD_VLLM_PROMPT")
else
  PROMPT_ARGS+=(--prompt "Summarize Semantic Cache Distillation in one sentence.")
fi

EXTRA_ARGS=()
if [[ -n "${SCD_VLLM_PRODUCER_MODEL:-}" ]]; then
  EXTRA_ARGS+=(--producer-model "$SCD_VLLM_PRODUCER_MODEL")
fi
if [[ -n "${SCD_VLLM_TOKENIZER:-}" ]]; then
  EXTRA_ARGS+=(--tokenizer "$SCD_VLLM_TOKENIZER")
fi
if [[ "${SCD_VLLM_TRUST_REMOTE_CODE:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--trust-remote-code)
fi

python -m scd_vllm.hf_export \
  --consumer-model "$SCD_VLLM_CONSUMER_MODEL" \
  --mode full_consumer \
  --dtype "$SCD_VLLM_DTYPE" \
  --device "$SCD_VLLM_DEVICE" \
  --max-prefix-tokens "$SCD_VLLM_MAX_PREFIX_TOKENS" \
  --out "$SCD_VLLM_HF_EXPORT_OUT" \
  "${PROMPT_ARGS[@]}" \
  "${EXTRA_ARGS[@]}"
