#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${SCD_RUNTIME_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

: "${SCD_VLLM_MODEL:?set SCD_VLLM_MODEL}"
: "${SCD_VLLM_BUNDLE:?set SCD_VLLM_BUNDLE}"
SCD_INJECTION_OUT="${SCD_INJECTION_OUT:-runs/scd_vllm/scd_injection_smoke}"
SCD_INJECTION_STORAGE_DIR="${SCD_INJECTION_STORAGE_DIR:-${SCD_INJECTION_OUT}/example_connector_storage}"
SCD_VLLM_DTYPE="${SCD_VLLM_DTYPE:-bfloat16}"
SCD_VLLM_BLOCK_SIZE="${SCD_VLLM_BLOCK_SIZE:-16}"
SCD_VLLM_MAX_MODEL_LEN="${SCD_VLLM_MAX_MODEL_LEN:-512}"
SCD_VLLM_MAX_NEW_TOKENS="${SCD_VLLM_MAX_NEW_TOKENS:-16}"
SCD_VLLM_GPU_MEMORY_UTILIZATION="${SCD_VLLM_GPU_MEMORY_UTILIZATION:-0.85}"
SCD_VLLM_SEED="${SCD_VLLM_SEED:-0}"
SCD_VLLM_TEMPERATURE="${SCD_VLLM_TEMPERATURE:-0.0}"
SCD_VLLM_TOP_P="${SCD_VLLM_TOP_P:-1.0}"

PROMPT_ARGS=()
if [[ -n "${SCD_VLLM_PREFIX_FILE:-}" ]]; then
  PROMPT_ARGS+=(--prefix-file "$SCD_VLLM_PREFIX_FILE")
elif [[ -n "${SCD_VLLM_PROMPT:-}" ]]; then
  PROMPT_ARGS+=(--prompt "$SCD_VLLM_PROMPT")
fi

EXTRA_ARGS=()
if [[ -n "${SCD_VLLM_TOKENIZER:-}" ]]; then
  EXTRA_ARGS+=(--tokenizer "$SCD_VLLM_TOKENIZER")
fi
if [[ -n "${SCD_VLLM_LAYER_NAME_TEMPLATE:-}" ]]; then
  EXTRA_ARGS+=(--layer-name-template "$SCD_VLLM_LAYER_NAME_TEMPLATE")
fi
if [[ "${SCD_VLLM_ENFORCE_EAGER:-1}" == "1" ]]; then
  EXTRA_ARGS+=(--enforce-eager)
fi
if [[ "${SCD_VLLM_ENABLE_PREFIX_CACHING:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--enable-prefix-caching)
fi
if [[ "${SCD_VLLM_ENABLE_CHUNKED_PREFILL:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--enable-chunked-prefill)
fi
if [[ "${SCD_VLLM_DISABLE_HYBRID_KV_CACHE_MANAGER:-1}" == "1" ]]; then
  EXTRA_ARGS+=(--disable-hybrid-kv-cache-manager)
fi

python -m scd_vllm.injection run \
  --bundle-dir "$SCD_VLLM_BUNDLE" \
  --storage-dir "$SCD_INJECTION_STORAGE_DIR" \
  --block-size "$SCD_VLLM_BLOCK_SIZE" \
  --overwrite \
  --model "$SCD_VLLM_MODEL" \
  --out "$SCD_INJECTION_OUT" \
  --dtype "$SCD_VLLM_DTYPE" \
  --max-model-len "$SCD_VLLM_MAX_MODEL_LEN" \
  --max-new-tokens "$SCD_VLLM_MAX_NEW_TOKENS" \
  --gpu-memory-utilization "$SCD_VLLM_GPU_MEMORY_UTILIZATION" \
  --seed "$SCD_VLLM_SEED" \
  --temperature "$SCD_VLLM_TEMPERATURE" \
  --top-p "$SCD_VLLM_TOP_P" \
  "${PROMPT_ARGS[@]}" \
  "${EXTRA_ARGS[@]}"
