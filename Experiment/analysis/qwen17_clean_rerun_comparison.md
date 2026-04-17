# Qwen3-1.7B -> Qwen3-1.7B-bf16 Clean Rerun Comparison

## Scope

This note records a cleaner rerun of the near-pair baseline on `CMRC` with `200` samples for `Qwen3-1.7B -> Qwen3-1.7B-bf16`, then compares it against the previously stored `strict_main` result.

The rerun finished on `2026-03-29 18:10:47 CST`.

## Clean rerun configuration

- producer model: `Qwen3-1.7B`
- consumer model: `Qwen3-1.7B-bf16`
- producer device: `cuda:2`
- consumer device: `cuda:0`
- train device: `cuda:0`
- methods evaluated: `full_consumer`, `reuse_only`, `reuse_fallback`, `reuse_patch`
- disabled for this rerun: `reuse_patch_recompute`, short recompute
- `max_length = 1024`
- `max_new_tokens = 32`
- `bandwidth_gbps = 200`
- route counts: `reuse=6`, `patch=6`, `fallback=16`, `recompute=0`

## Local artifact copies

- clean summary: `Experiment/analysis/remote_artifacts/qwen17_to_bf16_cmrc_clean_orig_s42/summary.json`
- clean profile: `Experiment/analysis/remote_artifacts/qwen17_to_bf16_cmrc_clean_orig_s42/profile_cmrc_seed42.json`
- clean route: `Experiment/analysis/remote_artifacts/qwen17_to_bf16_cmrc_clean_orig_s42/route_cmrc_seed42.json`
- old summary: `Experiment/analysis/remote_artifacts/qwen17_to_bf16_cmrc_strict_main/summary.json`

## Old vs clean rerun

| method | old F1 | clean F1 | delta F1 | old TTFT(ms) | clean TTFT(ms) | delta TTFT(ms) | old EM | clean EM | delta EM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `full_consumer` | `0.6945` | `0.6945` | `+0.0000` | `57.57` | `58.81` | `+1.24` | `0.305` | `0.305` | `+0.000` |
| `reuse_only` | `0.0014` | `0.0011` | `-0.0003` | `51.95` | `38.58` | `-13.37` | `0.000` | `0.000` | `+0.000` |
| `reuse_fallback` | `0.4088` | `0.6605` | `+0.2518` | `82.41` | `70.98` | `-11.42` | `0.065` | `0.380` | `+0.315` |
| `reuse_patch` | `0.0258` | `0.5010` | `+0.4751` | `84.58` | `75.84` | `-8.74` | `0.000` | `0.245` | `+0.245` |

## Claim-oriented reading

The clean rerun materially improves the transfer-side quality relative to the old run, especially for `reuse_fallback` and `reuse_patch`. This means the old result was too pessimistic about how broken those two modes were under a cleaner, tighter setup.

However, the headline conclusion does not flip. None of the transfer modes achieves a paper-style TTFT-quality sweet spot relative to `full_consumer`:

- `reuse_only` is still unusable in quality terms. Its `TTFT` is only `0.656x` of full consumer, but `F1` retention is approximately `0.15%`.
- `reuse_fallback` is the strongest quality-preserving transfer mode in this rerun, with `F1 = 0.6605` versus `0.6945` for full consumer. That is about `95.1%` F1 retention, but `TTFT = 70.98 ms`, which is still `1.207x` slower than the `58.81 ms` full-consumer baseline.
- `reuse_patch`, which is closer to the original "reuse + selective patch" story, improves sharply over the old run but still lands at `F1 = 0.5010` and `TTFT = 75.84 ms`. That is only about `72.1%` F1 retention and `1.290x` the full-consumer `TTFT`.

Therefore, on this similar Qwen-family near-pair, higher pair similarity plus cleaner execution improves the apparent stability of the transfer pipeline, but it still does not recover the paper's claimed operating point if the claim requires both strong quality retention and lower-than-full-consumer `TTFT`.

## Route diagnostics

The clean rerun assigned the patch budget to late layers with the largest patch gain estimates:

- reuse layers: `[6, 7, 8, 11, 13, 15]`
- patch layers: `[21, 22, 24, 25, 26, 27]`
- fallback layers: `[0, 1, 2, 3, 4, 5, 9, 10, 12, 14, 16, 17, 18, 19, 20, 23]`

This is consistent with the profile showing that the hardest errors are concentrated in later layers. Even after focusing patch capacity there, the resulting `reuse_patch` path still fails to beat the native full-consumer baseline on the joint TTFT-quality objective.

## Practical implication

This rerun updates the project interpretation as follows:

1. The old `qwen17_to_bf16` run should not be treated as numerically trustworthy for the absolute quality of `reuse_fallback` and `reuse_patch`.
2. The old run was still directionally correct about the main question: a more similar Qwen-family pair does not by itself reproduce the claimed original-SCD trade-off.
3. The next evidence branch should be either:
   - implement and test the three route improvements on a matched near-pair setup, or
   - unblock faithful cache-state support for the strict `Qwen3.5-4B-Base -> Qwen3.5-4B` pair.
