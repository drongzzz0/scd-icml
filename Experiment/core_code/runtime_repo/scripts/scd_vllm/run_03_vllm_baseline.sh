#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${SCD_RUNTIME_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

: "${SCD_VLLM_MODEL:?set SCD_VLLM_MODEL}"
SCD_VLLM_BASELINE_OUT="${SCD_VLLM_BASELINE_OUT:-runs/scd_vllm/vllm_plain_baseline}"
SCD_VLLM_DTYPE="${SCD_VLLM_DTYPE:-bfloat16}"
SCD_VLLM_MAX_MODEL_LEN="${SCD_VLLM_MAX_MODEL_LEN:-512}"
SCD_VLLM_MAX_NEW_TOKENS="${SCD_VLLM_MAX_NEW_TOKENS:-16}"
SCD_VLLM_TEMPERATURE="${SCD_VLLM_TEMPERATURE:-0.0}"
SCD_VLLM_TOP_P="${SCD_VLLM_TOP_P:-1.0}"
SCD_VLLM_TOP_K="${SCD_VLLM_TOP_K:-0}"
SCD_VLLM_SEED="${SCD_VLLM_SEED:-0}"
SCD_VLLM_WARMUP="${SCD_VLLM_WARMUP:-1}"
SCD_VLLM_REPEAT="${SCD_VLLM_REPEAT:-3}"
SCD_VLLM_TTFT_PROXY_REPEAT="${SCD_VLLM_TTFT_PROXY_REPEAT:-3}"
SCD_VLLM_GPU_MEMORY_UTILIZATION="${SCD_VLLM_GPU_MEMORY_UTILIZATION:-0.85}"

PROMPT_ARGS=()
if [[ -n "${SCD_VLLM_PREFIX_FILE:-}" ]]; then
  PROMPT_ARGS+=(--prefix-file "$SCD_VLLM_PREFIX_FILE")
elif [[ -n "${SCD_VLLM_PROMPT:-}" ]]; then
  PROMPT_ARGS+=(--prompt "$SCD_VLLM_PROMPT")
else
  PROMPT_ARGS+=(--prompt "Summarize Semantic Cache Distillation in one sentence.")
fi

EXTRA_ARGS=()
if [[ -n "${SCD_VLLM_TOKENIZER:-}" ]]; then
  EXTRA_ARGS+=(--tokenizer "$SCD_VLLM_TOKENIZER")
fi
if [[ -n "${SCD_VLLM_HF_BUNDLE:-}" ]]; then
  EXTRA_ARGS+=(--hf-bundle "$SCD_VLLM_HF_BUNDLE")
fi
if [[ "${SCD_VLLM_TRUST_REMOTE_CODE:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--trust-remote-code)
fi
if [[ "${SCD_VLLM_ENFORCE_EAGER:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--enforce-eager)
fi
if [[ "${SCD_VLLM_ENABLE_PREFIX_CACHING:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--enable-prefix-caching)
fi
if [[ "${SCD_VLLM_ALLOW_TOKEN_MISMATCH:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--allow-token-mismatch)
fi

python -m scd_vllm.vllm_baseline \
  --model "$SCD_VLLM_MODEL" \
  --out "$SCD_VLLM_BASELINE_OUT" \
  --dtype "$SCD_VLLM_DTYPE" \
  --max-model-len "$SCD_VLLM_MAX_MODEL_LEN" \
  --max-new-tokens "$SCD_VLLM_MAX_NEW_TOKENS" \
  --temperature "$SCD_VLLM_TEMPERATURE" \
  --top-p "$SCD_VLLM_TOP_P" \
  --top-k "$SCD_VLLM_TOP_K" \
  --seed "$SCD_VLLM_SEED" \
  --warmup "$SCD_VLLM_WARMUP" \
  --repeat "$SCD_VLLM_REPEAT" \
  --ttft-proxy-repeat "$SCD_VLLM_TTFT_PROXY_REPEAT" \
  --gpu-memory-utilization "$SCD_VLLM_GPU_MEMORY_UTILIZATION" \
  "${PROMPT_ARGS[@]}" \
  "${EXTRA_ARGS[@]}"
