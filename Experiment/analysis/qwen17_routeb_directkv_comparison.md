# Qwen3-1.7B -> Qwen3-1.7B-bf16 Route B Direct-KV Result

## Scope

This note records the first follow-up route run on the runnable near-pair substrate:

- pair: `Qwen3-1.7B -> Qwen3-1.7B-bf16`
- task: `CMRC`, `200` samples
- route: `B = State -> direct KV`
- patch fit mode: `direct_kv_v1`
- patch layers: `[21, 22, 24, 25, 26, 27]`
- reuse layers: `[6, 7, 8, 11, 13, 15]`
- fallback layers: `[0, 1, 2, 3, 4, 5, 9, 10, 12, 14, 16, 17, 18, 19, 20, 23]`

The run finished on `2026-03-30 10:47:21 CST`.

## Local artifacts

- strict eval summary: `Experiment/analysis/remote_artifacts/qwen17_to_bf16_cmrc_clean_routeb_directkv_main200/summary.json`
- strict eval comparison: `Experiment/analysis/remote_artifacts/qwen17_to_bf16_cmrc_clean_routeb_directkv_main200/final_comparison.json`
- direct-KV patch meta: `Experiment/analysis/remote_artifacts/qwen17_to_bf16_cmrc_clean_routeb_directkv_main200/patch_meta.json`
- reference clean original run: `Experiment/analysis/remote_artifacts/qwen17_to_bf16_cmrc_clean_orig_s42/summary.json`

## Headline result

Route B does **not** recover the paper-style operating point on this near-pair.

Within the Route B run itself:

| method | F1 | EM | TTFT(ms) | TTFT vs full | F1 retention vs full |
| --- | ---: | ---: | ---: | ---: | ---: |
| `full_consumer` | `0.6945` | `0.305` | `45.42` | `1.000x` | `100.0%` |
| `reuse_only` | `0.0011` | `0.000` | `42.02` | `0.925x` | `0.15%` |
| `reuse_fallback` | `0.6605` | `0.380` | `62.04` | `1.366x` | `95.1%` |
| `reuse_patch` | `0.4951` | `0.275` | `54.22` | `1.194x` | `71.3%` |

So Route B is still slower than `full_consumer` while losing a large amount of task quality.

## Comparison against the clean original patch run

The fairest cross-run comparison is the normalized TTFT-quality trade-off, because the direct-KV run used a different producer/consumer GPU assignment than the clean semantic-patch rerun.

Compared against the clean original-style `reuse_patch` run:

- clean semantic patch retention: `72.1%`
- clean semantic patch TTFT ratio: `1.290x`
- Route B retention: `71.3%`
- Route B TTFT ratio: `1.194x`

This means Route B improves the normalized latency ratio by about `0.096x`, but it does **not** improve quality. In fact it is slightly worse in F1 retention.

## Comparison against reuse_fallback

Inside the Route B run:

- Route B is `7.82 ms` faster than `reuse_fallback`
- but Route B loses `0.1655` absolute F1 relative to `reuse_fallback`

So the speed gain comes at a quality cost that is too large for the ICML claim story.

## TTFT decomposition

For `reuse_patch` under Route B:

- `encode_ms = 1.59`
- `transfer_ms = 0.80`
- `rebuild_ms = 2.49`
- `patch_infer_ms = 3.08`
- `recompute_ms = 22.71`
- `decode_first_ms = 23.55`

Key reading:

1. The direct-KV route successfully removes most of the old patch-side compute burden. `patch_infer_ms` is now only `3.08 ms`.
2. The dominant remaining costs are still structural: `recompute_ms` and `decode_first_ms`.
3. Because `recompute_ms` is still substantial and quality remains low, Route B is not a rescue of the original claim. It is only a modest latency-quality trade shift.

## Practical conclusion

The near-pair evidence now looks like this:

1. The original clean semantic patch did not recover the claim.
2. Route B direct-KV made the route faster in normalized terms.
3. Route B still failed the core criterion: it remained slower than native `full_consumer` and retained only about `71%` of full-consumer F1.

Therefore the next mainline follow-up should be **Route A**, not more optimization of Route B.
