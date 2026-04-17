# Qwen3-1.7B -> Qwen3-1.7B-gray Patch Runtime Optimization Update

Date: `2026-04-03`

## Scope

This note records the runtime-only optimization pass for the strict no-fallback shared `patch_only` route on:

- producer: `Qwen3-1.7B`
- consumer: `Qwen3-1.7B-gray`
- task slice: `CMRC`, `50` samples

The optimization target was the patch compute chain, not the learned mapping. The user-requested constraint was:

- `V` should not go through `k_norm`
- `k_norm` should be fused into the fast path instead of executed as a separate stage

That is now the implemented behavior.

## Runtime Change

The final runtime path does three things:

1. Rotary is applied to `K` only.
2. `k_proj` and `v_proj` are fused into one batched projection.
3. `k_norm` is fused into that path on the `K` half only.

So the current patch chain is:

- hidden -> fused `[k_proj | v_proj]`
- apply `k_norm` only to the `K` slice
- apply native rotary only to `K`
- return native `K/V`

`V` is never normalized.

## Patch-Only Optimization Trace

| variant | F1 | EM | TTFT ms | patch_infer ms | kproj/vproj ms | knorm ms | rotary ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| original shared runtime | 0.7667 | 0.44 | 159.47 | 132.93 | 43.83 | 41.47 | 43.15 |
| `+` key-only rotary | 0.7547 | 0.42 | 119.63 | 90.20 | 43.82 | 1.40 | 2.48 |
| `+` fused `k_proj/v_proj` | 0.7547 | 0.42 | 68.64 | 39.76 | 31.43 | 1.05 | 2.21 |
| `+` fused K-only `k_norm` | 0.7547 | 0.42 | 63.96 | 36.29 | 26.26 | 0.00 | 1.63 |

Compared with the original runtime, the final optimized path reduces:

- `TTFT`: `159.47 -> 63.96 ms`
- `patch_infer`: `132.93 -> 36.29 ms`
- patch compute core (`kproj/vproj + knorm + rotary`): `128.45 -> 27.89 ms`

## Final Same-Run Comparison

Final optimized run artifact:

- `/Users/weimingming/dr-claw/scd-new/Experiment/analysis/remote_artifacts/qwen17_to_gray_shared_patchonly_all28_subset12e1_50_20260403_keyonlyrot_fusedkvknorm/summary.json`

| method | F1 | EM | TTFT ms | delta TTFT vs baseline |
| --- | ---: | ---: | ---: | ---: |
| `baseline` | 0.7358 | 0.44 | 99.05 | 0.00 |
| `reuse-only` | 0.7775 | 0.46 | 58.11 | -40.94 |
| `patch-only` | 0.7547 | 0.42 | 63.96 | -35.09 |

`patch-only` is now faster than `baseline` on this pair and still above `baseline` in F1, but it remains slower and weaker than repaired `reuse-only`.

## Final Patch Breakdown

For the final optimized `patch-only` run:

| shard_load | shard_gpu_copy | patch_infer | kproj/vproj | knorm | rotary | decode_first |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4.05 ms | 4.35 ms | 36.29 ms | 26.26 ms | 0.00 ms | 1.63 ms | 27.67 ms |

Interpretation:

- `patch_knorm_ms = 0.00` means standalone `k_norm` work has been removed from the timed path.
- The remaining dominant patch compute is the fused native `k_proj/v_proj`.
- The next bottleneck is shard residency / copy variance, not `k_norm`.
