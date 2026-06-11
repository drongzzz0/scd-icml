# Qwen3-1.7B -> Qwen3-1.7B-gray Clean Comparison

Superseded for default-route use by:

- `/Users/weimingming/dr-claw/scd-new/Experiment/analysis/qwen17_to_gray_patch_runtime_fused_knorm_20260403.md`
- `/Users/weimingming/dr-claw/scd-new/Experiment/analysis/transfer_route_lock_20260404.md`

This note records the pre-optimization shared `patch-only` runtime and should now be treated as archived runtime history rather than the current default patch result.

Date: `2026-04-03`

## Setup

Pair:

- producer: `Qwen3-1.7B`
- consumer: `Qwen3-1.7B-gray`

Protocol:

- methods: `baseline`, repaired `reuse-only`, shared `patch-only`
- no fallback
- no recompute
- aligned `decode_first`
- route: all `28` layers in `reuse_layers` and all `28` layers in `patch_layers`
- patch fit mode: `xnorm_nativekv_shared_residual_mlp_v1`
- patch training subset: `12` capture chunks, `1` epoch
- eval set: `CMRC`, `50` samples

Artifacts mirrored locally:

- `/Users/weimingming/dr-claw/scd-new/Experiment/analysis/remote_artifacts/qwen17_to_gray_shared_patchonly_all28_subset12e1_50_20260403/summary.json`
- `/Users/weimingming/dr-claw/scd-new/Experiment/analysis/remote_artifacts/qwen17_to_gray_shared_patchonly_all28_subset12e1_50_20260403/per_mode_summary.json`
- `/Users/weimingming/dr-claw/scd-new/Experiment/analysis/remote_artifacts/qwen17_to_gray_shared_patchonly_all28_subset12e1_50_20260403/final_comparison.json`

## Main Result

| method | F1 | EM | TTFT ms | delta TTFT vs baseline |
| --- | ---: | ---: | ---: | ---: |
| `baseline` | 0.7358 | 0.44 | 140.18 | 0.00 |
| `reuse-only` | 0.7775 | 0.46 | 45.48 | -94.71 |
| `patch-only` | 0.7667 | 0.44 | 159.47 | +19.28 |

## Breakdown

### `reuse-only`

| encode | transfer | rebuild | decode_first | payload |
| ---: | ---: | ---: | ---: | ---: |
| 1.77 ms | 9.85 ms | 6.07 ms | 27.78 ms | 41.32 MB |

Rebuild split:

- `rebuild_prep_ms = 1.85`
- `rebuild_decode_core_ms = 2.05`
- `rebuild_materialize_ms = 2.17`
- `rebuild_rerope_ms = 0.00`

### `patch-only`

| patch_infer | shard_load | shard_gpu_copy | kproj/vproj | knorm | rotary | decode_first | payload |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 132.93 ms | 3.72 ms | 0.76 ms | 43.83 ms | 41.47 ms | 43.15 ms | 26.53 ms | 41.32 MB |

## Interpretation

This pair does not support the current shared `patch-only` route as a TTFT improvement method. `patch-only` keeps task quality above `baseline`, but it is slower than `baseline` by about `19.3 ms` and much slower than repaired `reuse-only`.

The repaired clean `reuse-only` route is the clear winner on this `gray` pair under the user's no-fallback definition. On this `50`-sample slice it is both faster and better on task F1 than same-run `baseline`, with:

- `+0.0417` F1 over `baseline`
- `+0.02` EM over `baseline`
- `-94.71 ms` TTFT versus `baseline`

One caution is that the average token-level likelihood moves in the opposite direction:

- `baseline avg_nll = 0.4621`
- `reuse-only avg_nll = 0.5349`
- `patch-only avg_nll = 0.5794`

So on this run, answer-level CMRC metrics favor transfer methods, especially repaired `reuse-only`, while token-level likelihood favors native `baseline`. This should be treated as a real observation from the current `50`-sample slice, not yet as a generalized claim.

## Direct Conclusion For This Pair

- repaired `reuse-only`: works well and currently dominates
- shared `patch-only`: quality is acceptable, but TTFT is not competitive
- old reuse family: remains deprecated; only the repaired clean reuse route is valid
