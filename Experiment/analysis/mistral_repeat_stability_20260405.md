# Mistral Repeat Stability Check

Date: `2026-04-05`

## Scope

This note re-checks the `Mistral-7B-v0.1 -> Mistral-7B-v0.1` and `MistralLite -> MistralLite` self-self results under the locked strict routes:

- `baseline`
- `reuse-only`
- `patch-only`

All repeated runs are:

- `no fallback`
- `no recompute`
- default healthy `patch-only` path with `runtime_shared_nativekv_fastpath=false`

Local mirrored repeat artifacts:

- `/Users/weimingming/dr-claw/scd-new/Experiment/analysis/remote_artifacts/mistral_repeat_20260405/mistral7`
- `/Users/weimingming/dr-claw/scd-new/Experiment/analysis/remote_artifacts/mistral_repeat_20260405/mistrallite`

Each pair was rerun `3` times on the same 50-sample CMRC eval set.

## Aggregate TTFT Summary

The table below reports:

- `mean TTFT`: mean of per-run TTFT means
- `mean median TTFT`: mean of per-run TTFT medians
- `mean p90 TTFT`: mean of per-run TTFT p90 values
- `mean speedup`: `baseline mean TTFT / route mean TTFT`
- `median speedup`: `baseline mean-median TTFT / route mean-median TTFT`

| pair | method | mean TTFT ms | mean median TTFT ms | mean p90 TTFT ms | mean speedup vs baseline | median speedup vs baseline |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `Mistral-7B` | `baseline` | 123.65 | 96.15 | 202.65 | 1.00x | 1.00x |
| `Mistral-7B` | `reuse-only` | 40.87 | 34.83 | 44.09 | 3.03x | 2.76x |
| `Mistral-7B` | `patch-only` | 77.88 | 32.30 | 35.39 | 1.59x | 2.98x |
| `MistralLite` | `baseline` | 116.24 | 90.68 | 157.69 | 1.00x | 1.00x |
| `MistralLite` | `reuse-only` | 46.00 | 33.09 | 44.51 | 2.53x | 2.74x |
| `MistralLite` | `patch-only` | 84.50 | 30.84 | 150.47 | 1.38x | 2.94x |

## Main Findings

### 1. `reuse-only` gap is mostly a mean-statistic artifact

- The headline mean-speedup gap is visible: `Mistral-7B` `3.03x` versus `MistralLite` `2.53x`.
- But the median story is almost the same: `2.76x` versus `2.74x`.
- The tail is also nearly identical: `mean p90 TTFT` is `44.09 ms` versus `44.51 ms`.
- This means `MistralLite reuse-only` is not showing a stable per-sample slowdown. The difference is mainly a small number of rebuild spikes that inflate the mean.

Recurring rebuild spikes over `100 ms`:

- `Mistral-7B`: only `DEV_8_QUERY_2` appears once across all repeats.
- `MistralLite`: `DEV_14_QUERY_0` appears twice; other rebuild spikes are isolated one-offs.

Interpretation:

- For `reuse-only`, the smaller `MistralLite` headline speedup should not be read as an architecture-level slowdown.
- The current evidence is more consistent with occasional runtime jitter or isolated rebuild stalls.

### 2. `patch-only` on `MistralLite` has a real heavy-tail problem

- Median performance is still close across the two pairs: `32.30 ms` on `Mistral-7B` versus `30.84 ms` on `MistralLite`.
- So the steady-state central tendency of `patch-only` is not worse on `MistralLite`.
- The failure is in the tail:
  - `Mistral-7B patch-only mean p90 TTFT`: `35.39 ms`
  - `MistralLite patch-only mean p90 TTFT`: `150.47 ms`

This is the main reason `MistralLite patch-only` looks worse in the headline mean-speedup table.

There is a first-sample warm-start outlier on both pairs:

- `Mistral-7B`: `DEV_0_QUERY_0` patch-infer `4359.22`, `534.47`, `514.53 ms`
- `MistralLite`: `DEV_0_QUERY_0` patch-infer `3575.33`, `369.43`, `512.09 ms`

But warm-start is not the whole story. After removing only `DEV_0_QUERY_0` from all three repeats:

- `Mistral-7B patch-only` mean TTFT becomes `42.14 ms`, `p90 = 35.67 ms`
- `MistralLite patch-only` mean TTFT still stays at `55.45 ms`, `p90 = 148.96 ms`

So `MistralLite patch-only` still has a genuine non-warmup heavy tail.

### 3. The heavy tail is partially prompt-dependent

Hot `qid`s are defined here as `patch_infer > 100 ms`.

Recurring hot `qid`s on `MistralLite patch-only`:

| qid | hot repeats | prefix tokens | teacher tokens | observed patch_infer ms |
| --- | ---: | ---: | ---: | --- |
| `DEV_2_QUERY_2` | `3/3` | 525 | 6 | `119.90`, `122.75`, `122.11` |
| `DEV_7_QUERY_2` | `3/3` | 1023 | 17 | `295.33`, `308.52`, `276.62` |
| `DEV_7_QUERY_1` | `2/3` | 1023 | 20 | `129.79`, `12.43`, `130.35` |
| `DEV_11_QUERY_2` | `2/3` | 600 | 13 | `121.77`, `15.26`, `286.72` |
| `DEV_1_QUERY_3` | `2/3` | 752 | 14 | `11.88`, `130.27`, `125.27` |
| `DEV_13_QUERY_1` | `2/3` | 867 | 3 | `12.13`, `268.24`, `289.72` |

By contrast, `Mistral-7B patch-only` has only one clearly recurring hot sample after warm-start:

- `DEV_0_QUERY_0` appears in all three repeats
- every other hot `qid` is sparse and isolated

Interpretation:

- `MistralLite patch-only` is not suffering only from random system noise.
- A subset of prompts repeatedly triggers slow `patch_infer` behavior.
- Some hot prompts are long (`1023` tokens), but others are medium-length (`525`, `600`, `752`) or even have very short answers (`teacher_token_count = 3`), so simple prompt length is not a sufficient explanation.

### 4. Prefix length is not the driver

Across all repeated `patch-only` samples, the correlation between `prefix_tokens` and `patch_infer` is negligible:

- `Mistral-7B`: `corr = -0.0094`
- `MistralLite`: `corr = 0.0422`

This rules out the simplest explanation that `MistralLite` is slower only because its hot samples have longer prefixes.

## Conclusion

The original one-run diagnosis needs to be split by route:

- For `reuse-only`, the smaller `MistralLite` speedup is mostly an outlier-driven mean artifact. Median and p90 behavior are effectively the same as `Mistral-7B`.
- For `patch-only`, the smaller `MistralLite` speedup is a real runtime stability issue. The median is still healthy, but the tail is much worse and repeats on a partially fixed set of `qid`s.

So the correct interpretation is:

- do **not** treat `MistralLite` as globally slower for the repaired `reuse-only` route
- do treat `MistralLite patch-only` as still having an unresolved heavy-tail bug or prompt-dependent slow path inside `patch_infer`

## Next Debug Target

The next useful diagnosis is no longer another mean-level comparison. It should directly instrument `MistralLite patch-only` on the recurring hot prompts above and split `patch_infer` into finer runtime sub-terms so we can see whether the repeat stalls are coming from:

- consumer-native `k_proj/v_proj`
- `k_norm`
- rotary / layout materialization
- prompt-shape-dependent graph or kernel warm paths
- some hidden synchronization or allocator behavior

Until that is resolved, the locked default should remain unchanged:

- keep the repaired `reuse-only`
- keep the healthy pre-optimization `patch-only`
- keep `runtime_shared_nativekv_fastpath=false` by default
