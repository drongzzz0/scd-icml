# Qwen3-1.7B -> Qwen3-1.7B-bf16 Route A State-Native Result

## Scope

This note records the Route A follow-up run on the runnable near-pair substrate:

- pair: `Qwen3-1.7B -> Qwen3-1.7B-bf16`
- task: `CMRC`, `200` samples
- route: `A = State -> State -> native Wk/Wv -> KV`
- patch layers: `[21, 22, 24, 25, 26, 27]`
- reuse layers: `[6, 7, 8, 11, 13, 15]`
- fallback layers: `[0, 1, 2, 3, 4, 5, 9, 10, 12, 14, 16, 17, 18, 19, 20, 23]`

This run was configured as the intended pure Route A approximation in the current repo:

- `fit_mode = xnorm_semantic_v1`
- `state_target_space = xnorm`
- `patch_apply_mode = overwrite`
- loss weights: `lambda_h = 1.0`, `lambda_k = 0.0`, `lambda_v = 0.0`, `lambda_sem = 0.0`

The run finished on `2026-03-30`.

## Local artifacts

- strict eval summary: `Experiment/analysis/remote_artifacts/qwen17_to_bf16_cmrc_clean_routea_state_native_main200/summary.json`
- strict eval comparison: `Experiment/analysis/remote_artifacts/qwen17_to_bf16_cmrc_clean_routea_state_native_main200/final_comparison.json`
- route-A patch meta: `Experiment/analysis/remote_artifacts/qwen17_to_bf16_cmrc_clean_routea_state_native_main200/patch_meta.json`
- reference clean original run: `Experiment/analysis/remote_artifacts/qwen17_to_bf16_cmrc_clean_orig_s42/summary.json`
- reference Route B run: `Experiment/analysis/remote_artifacts/qwen17_to_bf16_cmrc_clean_routeb_directkv_main200/summary.json`

## Headline result

Pure Route A does **not** rescue the paper-style operating point on this near-pair. It fails more severely than both the clean original semantic patch and Route B.

Within the Route A run itself:

| method | F1 | EM | TTFT(ms) | TTFT vs full | F1 retention vs full |
| --- | ---: | ---: | ---: | ---: | ---: |
| `full_consumer` | `0.6945` | `0.305` | `39.94` | `1.000x` | `100.0%` |
| `reuse_only` | `0.0011` | `0.000` | `40.54` | `1.015x` | `0.15%` |
| `reuse_fallback` | `0.6605` | `0.380` | `57.04` | `1.428x` | `95.1%` |
| `reuse_patch` | `0.0000` | `0.000` | `54.55` | `1.366x` | `0.0%` |

So Route A remains slower than `full_consumer`, and its `reuse_patch` quality collapses completely.

## Comparison against the clean original patch run

Compared against the clean original-style semantic patch rerun:

- clean semantic patch retention: `72.1%`
- clean semantic patch TTFT ratio: `1.290x`
- Route A retention: `0.0%`
- Route A TTFT ratio: `1.366x`

This means Route A is worse on both axes. It is slower in normalized TTFT and loses essentially all task quality.

## Comparison against Route B

Compared against Route B direct-KV:

- Route B retention: `71.3%`
- Route B TTFT ratio: `1.194x`
- Route A retention: `0.0%`
- Route A TTFT ratio: `1.366x`

Route A is about `0.33 ms` slower than Route B in absolute `reuse_patch` TTFT, but the important difference is quality: Route A loses `0.4951` absolute F1 relative to Route B.

## Comparison against reuse_fallback

Inside the Route A run:

- Route A is `2.49 ms` faster than `reuse_fallback`
- but Route A loses `0.6605` absolute F1 relative to `reuse_fallback`

This is not a plausible ICML story. The latency gain is tiny and the quality collapse is total.

## TTFT decomposition

For `reuse_patch` under Route A:

- `encode_ms = 1.58`
- `transfer_ms = 0.80`
- `rebuild_ms = 2.47`
- `patch_infer_ms = 4.24`
- `recompute_ms = 22.13`
- `decode_first_ms = 23.32`

Key reading:

1. Route A does remove the old heavy semantic-patch inference burden relative to the clean original run.
2. The main latency still sits in `recompute_ms` and `decode_first_ms`, so Route A does not produce a fast operating point.
3. More importantly, the state-only mapping is not behavior-preserving on this pair. The extremely low `avg_logp` and zero `F1` indicate a functional collapse rather than a mild quality trade-off.

## Practical conclusion

The near-pair evidence now looks like this:

1. The clean original semantic patch did not recover the claim.
2. Route B improved normalized TTFT ratio but still failed the claim.
3. Pure Route A failed catastrophically: slower than `full_consumer`, worse than Route B, and effectively unusable in quality terms.

Therefore Route A should be treated as a negative result on `Qwen3-1.7B -> Qwen3-1.7B-bf16`, not as a promising rescue path. The next useful branch is either a tightly scoped Route C diagnostic or a shift toward more compatible higher-similarity pairs with standard KV cache semantics.
