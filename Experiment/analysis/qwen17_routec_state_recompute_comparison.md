# Qwen3-1.7B -> Qwen3-1.7B-bf16 Route C State-Recompute Result

## Scope

This note records the Route C follow-up run on the runnable near-pair substrate:

- pair: `Qwen3-1.7B -> Qwen3-1.7B-bf16`
- task: `CMRC`, `200` samples
- route: `C = State -> State -> suffix recompute on B`
- patch layers: `[21, 22, 24, 25, 26, 27]`
- reuse layers: `[6, 7, 8, 11, 13, 15]`
- fallback layers: `[0, 1, 2, 3, 4, 5, 9, 10, 12, 14, 16, 17, 18, 19, 20, 23]`

The run reused the Route A state-only patch checkpoint:

- `fit_mode = xnorm_semantic_v1`
- `state_target_space = xnorm`
- `patch_apply_mode = overwrite`
- loss weights: `lambda_h = 1.0`, `lambda_k = 0.0`, `lambda_v = 0.0`, `lambda_sem = 0.0`

Runtime evaluation enabled `reuse_patch_recompute` with `default_recompute_segment_length = 1`. From the current `strict_runtime.py` logic, this implies a one-layer suffix recompute after each patch layer when the route file has no explicit recompute map. That means the effective recompute coverage is inferred to be the immediate next layers after each patch breakpoint.

The run finished on `2026-03-30`.

## Local artifacts

- strict eval summary: `Experiment/analysis/remote_artifacts/qwen17_to_bf16_cmrc_clean_routec_state_recompute_main200/summary.json`
- strict eval comparison: `Experiment/analysis/remote_artifacts/qwen17_to_bf16_cmrc_clean_routec_state_recompute_main200/final_comparison.json`
- reused patch meta: `Experiment/analysis/remote_artifacts/qwen17_to_bf16_cmrc_clean_routec_state_recompute_main200/patch_meta.json`
- reference clean original run: `Experiment/analysis/remote_artifacts/qwen17_to_bf16_cmrc_clean_orig_s42/summary.json`
- reference Route B run: `Experiment/analysis/remote_artifacts/qwen17_to_bf16_cmrc_clean_routeb_directkv_main200/summary.json`
- reference Route A run: `Experiment/analysis/remote_artifacts/qwen17_to_bf16_cmrc_clean_routea_state_native_main200/summary.json`

## Headline result

Route C does recover useful quality relative to the collapsed pure Route A path, but it still does **not** rescue the paper-style operating point on this near-pair.

Within the Route C run itself:

| method | F1 | EM | TTFT(ms) | TTFT vs full | F1 retention vs full |
| --- | ---: | ---: | ---: | ---: | ---: |
| `full_consumer` | `0.6945` | `0.305` | `37.09` | `1.000x` | `100.0%` |
| `reuse_only` | `0.0011` | `0.000` | `36.45` | `0.983x` | `0.15%` |
| `reuse_fallback` | `0.6605` | `0.380` | `54.10` | `1.458x` | `95.1%` |
| `reuse_patch` | `0.0000` | `0.000` | `51.40` | `1.386x` | `0.0%` |
| `reuse_patch_recompute` | `0.6195` | `0.315` | `64.71` | `1.745x` | `89.2%` |

So Route C is behaviorally much better than Route A, but it is still slower than `full_consumer` by a large margin.

## Comparison against the clean original patch run

Compared against the clean original-style semantic patch rerun:

- clean semantic patch retention: `72.1%`
- clean semantic patch TTFT ratio: `1.290x`
- Route C retention: `89.2%`
- Route C TTFT ratio: `1.745x`

Route C improves quality substantially over the clean original patch path, but normalized latency is much worse in its own run because the native full-consumer baseline in this run is faster.

## Comparison against Route B

Compared against Route B direct-KV:

- Route B retention: `71.3%`
- Route B TTFT ratio: `1.194x`
- Route C retention: `89.2%`
- Route C TTFT ratio: `1.745x`

Route C gains `0.1244` absolute F1 over Route B, but costs `10.49 ms` more absolute `reuse_patch`-path TTFT and remains much worse in normalized latency ratio.

## Comparison against Route A

Compared against pure Route A:

- Route A `reuse_patch`: `F1 = 0.0000`, `TTFT = 54.55 ms`
- Route C `reuse_patch_recompute`: `F1 = 0.6195`, `TTFT = 64.71 ms`

So the suffix recompute branch does rescue the catastrophic quality collapse of Route A. But the rescue is expensive and still not enough to beat the stronger baselines.

## Comparison against reuse_fallback

Inside the Route C run:

- Route C is `10.61 ms` slower than `reuse_fallback`
- Route C loses `0.0411` absolute F1 relative to `reuse_fallback`

This is the most important practical comparison. `reuse_fallback` dominates Route C on this pair: it is both faster and more accurate.

## TTFT decomposition

For `reuse_patch_recompute` under Route C:

- `encode_ms = 1.55`
- `transfer_ms = 0.80`
- `rebuild_ms = 2.46`
- `patch_infer_ms = 11.85`
- `recompute_ms = 24.98`
- `decode_first_ms = 23.07`

Key reading:

1. The extra quality of Route C comes with meaningful additional compute, especially `patch_infer_ms` and `recompute_ms`.
2. The dominant latency terms are still structural: `recompute_ms` plus `decode_first_ms`.
3. Route C is therefore a partial semantic rescue, not a latency-quality win.

## Practical conclusion

The near-pair evidence now looks like this:

1. The clean original semantic patch did not recover the claim.
2. Route B improved quality-latency trade relative to the clean original patch, but still failed the claim.
3. Pure Route A failed catastrophically.
4. Route C repaired most of Route A's quality loss, reaching `89.2%` F1 retention, but it remained slower than both `full_consumer` and `reuse_fallback`.

Therefore Route C is a useful diagnostic result, not a publishable rescue of the claimed operating point. On the runnable Qwen3-1.7B near-pair, the best practical transfer path is still `reuse_fallback`, and even that path remains slower than native `full_consumer`. The main remaining research branch is now more compatible higher-similarity pairs with standard KV cache semantics, unless a materially different patch/recompute design is introduced.
