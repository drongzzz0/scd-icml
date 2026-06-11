# Revised Model Pair Strategy for Achieving SCD Claims

**Date**: 2026-03-30
**Status**: Active strategy after research goal pivot

## Strategic Shift

**Old goal**: Validate whether original SCD method works on similar Qwen3.5 pairs
**New goal**: Achieve SCD's claimed TTFT-quality trade-offs through any effective method + pair combination

## Current Situation

### What We Know
- **Qwen3-1.7B → Qwen3-1.7B-bf16** (near-pair):
  - Original SCD: Failed (reuse_fallback F1 0.66 @ 1.21x TTFT vs full consumer)
  - Route B (direct KV): Failed (F1 0.50 @ 1.19x TTFT)
  - This is already a very similar pair (same model, bf16 conversion)

- **Qwen3.5-4B pairs**: Blocked by hybrid cache architecture (24 linear attention + 8 full attention layers)

### Key Insight
Higher similarity alone may not be sufficient. We need to explore:
1. Better method variants (Route A, Route C, hybrids)
2. Alternative model families with standard KV cache

## Two-Path Strategy

### Path 1: Method Exploration (Priority 1)
Test remaining methods on existing Qwen3-1.7B pair:

**Next to test**:
1. **Route A** (State→State→native Wk/Wv→KV) - HIGHEST PRIORITY
   - Rationale: May preserve semantic structure better than Route B's direct mapping
   - Expected advantage: Native projection through consumer's Wk/Wv matrices

2. **Route C** (State→State→suffix recompute)
   - Rationale: Trades compute for accuracy
   - May achieve better quality retention

3. **Hybrid variants**:
   - Route A with selective Route C fallback
   - Layer-wise method selection based on similarity scores

### Path 2: Pair Exploration (Parallel)
Identify alternative high-similarity pairs with standard KV cache:

**Tier 1 - Same model variants** (easiest):
- Llama-3.1-8B → Llama-3.1-8B-Instruct
- Mistral-7B-v0.1 → Mistral-7B-Instruct-v0.1
- Gemma-2-9B → Gemma-2-9B-it

**Tier 2 - Quantization pairs** (very high similarity):
- Same model: fp16 → int8/int4 quantized versions
- Example: Llama-3.1-8B-fp16 → Llama-3.1-8B-GPTQ-4bit

**Tier 3 - LoRA/Adapter pairs** (highest similarity):
- Base model → Base + LoRA adapter
- Shared weights except adapter layers
- Example: Qwen3.5-4B-Base → Qwen3.5-4B-Base + prompt-tuning adapter

**Tier 4 - Checkpoint-adjacent fine-tunes**:
- Same base, different fine-tuning steps
- Example: checkpoint-1000 → checkpoint-5000

## Decision Tree

```
Start: Test Route A on Qwen3-1.7B pair
  ├─ SUCCESS → Document, test on 1-2 more pairs for robustness → Paper
  └─ FAIL → Test Route C on same pair
      ├─ SUCCESS → Document, validate → Paper
      └─ FAIL → Test hybrid (A+C) on same pair
          ├─ SUCCESS → Document, validate → Paper
          └─ FAIL → Conclude methods insufficient, pivot to Tier 1 pairs
              └─ Test best method on Llama/Mistral/Gemma pairs
                  ├─ SUCCESS → Paper story: "requires higher similarity"
                  └─ FAIL → Test Tier 2 (quantization pairs)
                      └─ Continue until success or exhaustion
```

## Execution Priority

**Week 1** (Current):
1. Implement Route A on Qwen3-1.7B pair
2. Run evaluation on CMRC dataset
3. If fails, immediately test Route C

**Week 2**:
- If methods succeed: Validate on 1-2 additional pairs
- If methods fail: Acquire and test Tier 1 pairs (Llama/Mistral/Gemma)

**Week 3**:
- Finalize best method + pair combination
- Run full benchmark suite
- Begin paper drafting

## Success Criteria (Revised)

**Minimum viable result**:
- At least ONE method achieves paper's claimed TTFT-quality trade-off on at least ONE pair
- Clear documentation of what conditions enable the effect

**Strong result**:
- Method works on multiple pairs across different similarity levels
- Clear characterization of similarity threshold required

**Ideal result**:
- Method works robustly across diverse pairs
- Practical guidance for when SCD-style transfer is viable

## Model Pair Acquisition Plan

### Immediate (Available on HuggingFace/ModelScope):
- Llama-3.1-8B / Llama-3.1-8B-Instruct
- Mistral-7B-v0.1 / Mistral-7B-Instruct-v0.1
- Gemma-2-9B / Gemma-2-9B-it

### Quick (Can generate locally):
- Quantization pairs: Use AutoGPTQ or llama.cpp to create int4/int8 versions
- LoRA pairs: Fine-tune small adapters on base models

### Requires effort:
- Checkpoint-adjacent pairs: Need access to training checkpoints or run custom fine-tuning

## Risk Mitigation

**Risk**: All methods fail on all pairs
**Mitigation**: Negative result is still publishable - "SCD claims not achievable under tested conditions"

**Risk**: Only works on impractically similar pairs (e.g., quantization variants)
**Mitigation**: Still valuable - defines boundary conditions for cache transfer

**Risk**: Time runs out before finding success
**Mitigation**: Prioritize quick tests, document all attempts, pivot to "boundary study" narrative

## Next Immediate Actions

1. Implement Route A in existing codebase
2. Run on Qwen3-1.7B pair (200 samples, CMRC)
3. Analyze results within 24-48 hours
4. Make go/no-go decision on method vs pair exploration
