#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${SCD_RUNTIME_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

SCD_TRUEKV_MODE="${SCD_TRUEKV_MODE:-inspect}"
SCD_TRUEKV_OUT="${SCD_TRUEKV_OUT:-runs/scd_vllm/truekv_oracle_inspection/metrics.json}"
SCD_TRUEKV_REPORT_OUT="${SCD_TRUEKV_REPORT_OUT:-reports/scd_vllm_truekv_api_inspection.md}"

case "$SCD_TRUEKV_MODE" in
  inspect)
    python -m scd_vllm.truekv_oracle inspect \
      --out "$SCD_TRUEKV_OUT" \
      --report-out "$SCD_TRUEKV_REPORT_OUT"
    ;;
  run)
    : "${SCD_VLLM_MODEL:?set SCD_VLLM_MODEL}"
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
    if [[ -n "${SCD_TRUEKV_STORAGE_DIR:-}" ]]; then
      EXTRA_ARGS+=(--storage-dir "$SCD_TRUEKV_STORAGE_DIR")
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
    python -m scd_vllm.truekv_oracle run \
      --model "$SCD_VLLM_MODEL" \
      --out "$SCD_TRUEKV_OUT" \
      --dtype "${SCD_VLLM_DTYPE:-bfloat16}" \
      --max-model-len "${SCD_VLLM_MAX_MODEL_LEN:-256}" \
      --max-new-tokens "${SCD_VLLM_MAX_NEW_TOKENS:-16}" \
      --gpu-memory-utilization "${SCD_VLLM_GPU_MEMORY_UTILIZATION:-0.85}" \
      --seed "${SCD_VLLM_SEED:-0}" \
      --temperature "${SCD_VLLM_TEMPERATURE:-0.0}" \
      --top-p "${SCD_VLLM_TOP_P:-1.0}" \
      "${PROMPT_ARGS[@]}" \
      "${EXTRA_ARGS[@]}"
    ;;
  save|load)
    python -m scd_vllm.truekv_oracle "$SCD_TRUEKV_MODE" --out "$SCD_TRUEKV_OUT"
    ;;
  *)
    echo "unknown SCD_TRUEKV_MODE: $SCD_TRUEKV_MODE" >&2
    exit 2
    ;;
esac
