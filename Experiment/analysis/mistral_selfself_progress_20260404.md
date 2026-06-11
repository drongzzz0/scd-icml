# Mistral Self-Self Progress

Date: `2026-04-04`

## Scope

Canonical routes only:

- `baseline`
- `reuse-only`
- `patch-only`

All runs are strict:

- `no fallback`
- `no recompute`
- current locked `patch-only`

## Completed: `Mistral-7B-v0.1 -> Mistral-7B-v0.1`

Remote run:

- `/disk_n/c101-3-1/projects/scd-repro-codex-mistral-instruct-20260323/runs/mistral7_self_cmrc_locked_20260404`

Local mirrored artifacts:

- `/Users/weimingming/dr-claw/scd-new/Experiment/analysis/remote_artifacts/mistral7_self_strict_baseline_reuse_patchonly_all32_50_20260404`

Summary on 50 CMRC samples:

| method | F1 | EM | TTFT ms | encode ms | transfer ms | rebuild ms | patch_infer ms | decode_first ms | payload MB | compression |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseline` | 0.6185 | 0.36 | 129.91 | 0.00 | 0.00 | 0.00 | 0.00 | 24.37 | 0.00 | - |
| `reuse-only` | 0.6486 | 0.36 | 35.01 | 2.01 | 7.87 | 1.57 | 0.00 | 23.56 | 73.10 | 1.0 |
| `patch-only` | 0.6385 | 0.38 | 47.10 | 0.00 | 0.00 | 0.00 | 24.96 | 22.14 | 146.21 | 0.5 |

Immediate read:

- `reuse-only` preserved quality well on this self-self pair and was the fastest route.
- `patch-only` also stayed close to baseline quality while keeping TTFT far below `baseline`.
- This run does not show the self-self collapse that earlier broken routes showed on older Qwen experiments.

## Completed: `MistralLite -> MistralLite` (original launch)

Remote run:

- `/disk_n/c101-3-1/projects/scd-repro-codex-mistral-instruct-20260323/runs/mistrallite_self_cmrc_locked_20260404`

Local mirrored artifacts:

- `/Users/weimingming/dr-claw/scd-new/Experiment/analysis/remote_artifacts/mistrallite_self_strict_baseline_reuse_patchonly_all32_50_20260404`

Summary on 50 CMRC samples:

| method | F1 | EM | TTFT ms | encode ms | transfer ms | rebuild ms | patch_infer ms | decode_first ms | payload MB | compression |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseline` | 0.7829 | 0.60 | 116.18 | 0.00 | 0.00 | 0.00 | 0.00 | 28.34 | 0.00 | - |
| `reuse-only` | 0.7686 | 0.58 | 52.65 | 3.14 | 10.47 | 16.40 | 0.00 | 22.63 | 73.10 | 1.0 |
| `patch-only` | 0.7875 | 0.60 | 68.96 | 0.00 | 0.00 | 0.00 | 45.49 | 23.47 | 146.21 | 0.5 |

Immediate read:

- `reuse-only` remained the fastest route and preserved most of `baseline` quality.
- `patch-only` slightly exceeded same-run `baseline` F1 while still cutting TTFT substantially versus `baseline`.
- the main remaining `patch-only` bottleneck on this pair is still `patch_kproj_vproj_ms` inside `patch_infer_ms`.

Stability note after sample-level inspection:

- the smaller mean speedup on `MistralLite` is driven mostly by a handful of large runtime spikes rather than by a stable per-sample slowdown.
- on `reuse-only`, the median TTFT is essentially the same across the two self-self runs: `Mistral-7B` `31.71 ms` versus `MistralLite` `31.14 ms`.
- on `patch-only`, the median TTFT is also very close: `Mistral-7B` `26.09 ms` versus `MistralLite` `26.89 ms`.
- the same pattern appears in the preparation subterms. `reuse-only.rebuild_ms` medians are `1.31` versus `1.28 ms`, and `patch-only.patch_infer_ms` medians are `4.88` versus `6.29 ms`.
- the mean gap comes from a few large stalls on the `MistralLite` run. For example, `reuse-only` has three rebuild spikes of about `279.55`, `276.42`, and `195.07 ms`, while `patch-only` has several patch spikes in the `123` to `189 ms` range plus a `426.03 ms` first-sample spike.
- these spikes correlate only weakly with prefix length, and the offending `qid`s are not consistently shared across `reuse-only` and `patch-only`, so the current evidence points more to runtime jitter or intermittent system-side stalls than to a clean model-architecture effect.

Runtime note:

- this queued run was launched with `OMP_NUM_THREADS=16`
- that reduced the early `capture/profile/stats` wall time relative to the first `Mistral-7B` run
- the long pole is still the per-layer linear-base solve inside `train_patch`

## Completed: `MistralLite -> MistralLite` optimized rerun

Remote run:

- `/disk_n/c101-3-1/projects/scd-repro-codex-mistral-instruct-20260323/runs/mistrallite_self_cmrc_locked_opt_20260404`

What changed relative to the original launch:

- reused the finished `pairs`, `reuse`, and `profile` artifacts from the original `MistralLite` run
- retrain only the shared `patch-only` checkpoint
- enabled cached block-level consumer-native projection weights in the patch training path
- raised `patch_batch_size` from `128` to `256`
- raised `layer_block_size` from `4` to `8`
- launch environment now uses `OMP_NUM_THREADS=32`, `MKL_NUM_THREADS=32`, `OPENBLAS_NUM_THREADS=32`

Local mirrored artifacts:

- `/Users/weimingming/dr-claw/scd-new/Experiment/analysis/remote_artifacts/mistrallite_self_strict_baseline_reuse_patchonly_all32_50_20260404_opt`

Summary on 50 CMRC samples:

| method | F1 | EM | TTFT ms | encode ms | transfer ms | rebuild ms | patch_infer ms | patch_kproj_vproj ms | decode_first ms | payload MB | compression |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseline` | 0.7829 | 0.60 | 114.94 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 22.73 | 0.00 | - |
| `reuse-only` | 0.7686 | 0.58 | 42.48 | 5.23 | 8.24 | 7.25 | 0.00 | 0.00 | 21.77 | 73.10 | 1.0 |
| `patch-only` | 0.7875 | 0.60 | 75.99 | 0.00 | 0.00 | 0.00 | 53.10 | 41.70 | 22.89 | 146.21 | 0.5 |

Direct comparison against the original launch:

| method | old TTFT ms | new TTFT ms | delta TTFT ms | old F1 | new F1 | delta F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseline` | 116.18 | 114.94 | -1.23 | 0.7829 | 0.7829 | 0.0000 |
| `reuse-only` | 52.65 | 42.48 | -10.16 | 0.7686 | 0.7686 | 0.0000 |
| `patch-only` | 68.96 | 75.99 | +7.03 | 0.7875 | 0.7875 | 0.0000 |

Immediate read:

- this optimization pass did not improve `patch-only` runtime on `MistralLite`
- `patch-only` quality stayed exactly unchanged, so the method behavior is stable
- the regression is concentrated in `patch_infer_ms`, especially `patch_kproj_vproj_ms` (`35.76 -> 41.70 ms`)
- `reuse-only` did improve materially in the optimized rerun, but that is likely launch/runtime variance rather than a method change because its algorithm was unchanged
- default `patch-only` training config is therefore locked back to the pre-optimization version; the two April 4 training-side tweaks are kept only as failed optimization history

## Completed: `MistralLite -> MistralLite` runtime-side three-opt rerun

Remote run:

- `/disk_n/c101-3-1/projects/scd-repro-codex-mistral-instruct-20260323/runs/mistrallite_self_cmrc_locked_20260404/strict_baseline_reuse_patchonly_all32_limit50_20260404_runtime3opt_v3`

What changed relative to the original launch:

- kept the locked strict methods unchanged: `baseline`, repaired `reuse-only`, strict `patch-only`
- enabled runtime-side prepacked consumer-native projection weights
- enabled fused / grouped batched `k_proj/v_proj` GEMM via `torch.baddbmm`
- enabled CUDA Graph on the `patch_kproj_vproj` path with a single-slot reusable graph cache instead of accumulating one graph per prompt shape
- moved graph creation into the timed projection stage and removed the previous per-sample out-of-band shape prewarm

Local mirrored artifacts:

- `/Users/weimingming/dr-claw/scd-new/Experiment/analysis/remote_artifacts/mistrallite_self_strict_baseline_reuse_patchonly_all32_50_20260404_runtime3opt_v3`

Summary on 50 CMRC samples:

| method | F1 | EM | TTFT ms | patch_infer ms | patch_kproj_vproj ms | patch_rotary ms | decode_first ms | payload MB | compression |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseline` | 0.7829 | 0.60 | 107.05 | 0.00 | 0.00 | 0.00 | 23.60 | 0.00 | - |
| `reuse-only` | 0.7686 | 0.58 | 43.34 | 0.00 | 0.00 | 0.00 | 23.38 | 73.10 | 1.0 |
| `patch-only` | 0.7875 | 0.60 | 87.34 | 63.11 | 34.12 | 22.43 | 24.23 | 146.21 | 0.5 |

Direct comparison against the healthy original launch:

| method | old TTFT ms | new TTFT ms | delta TTFT ms | old F1 | new F1 | delta F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseline` | 116.18 | 107.05 | -9.13 | 0.7829 | 0.7829 | 0.0000 |
| `reuse-only` | 52.65 | 43.34 | -9.31 | 0.7686 | 0.7686 | 0.0000 |
| `patch-only` | 68.96 | 87.34 | +18.38 | 0.7875 | 0.7875 | 0.0000 |

Direct comparison against the earlier failed `runtime3opt` rerun:

| method | bad rerun TTFT ms | stable rerun TTFT ms | delta TTFT ms |
| --- | ---: | ---: | ---: |
| `baseline` | 141.99 | 107.05 | -34.94 |
| `reuse-only` | 53.03 | 43.34 | -9.69 |
| `patch-only` | 149.58 | 87.34 | -62.25 |

Immediate read:

- the single-slot CUDA Graph strategy fixed the earlier OOM and completed all 50 samples
- the fused projection core itself did get slightly faster on `patch-only`, with `patch_kproj_vproj_ms` moving from `35.76` to `34.12`
- the three runtime-side optimizations as a bundle still do not improve end-to-end `patch-only` TTFT on `MistralLite`
- the regression now concentrates after projection: `patch_rotary_ms` jumped from `1.63` to `22.43`, so the current graph-based path is likely pushing or exposing downstream K/V layout materialization cost rather than removing it end-to-end
- the bundle is therefore stable but not yet a net serving win for the locked `patch-only` route

## Thread Backend Check

Checked on `ser`:

- `PyTorch` uses `OpenMP + Intel oneAPI MKL`
- `NumPy/SciPy` use `OpenBLAS`
- current Mistral code path is dominated by `torch`, not direct `numpy/scipy` linear algebra

Practical implication:

- the main tuning knobs for this route are `OMP_NUM_THREADS` and `MKL_NUM_THREADS`
- `OPENBLAS_NUM_THREADS` is still worth setting for consistency, but it is not the main limiter here

Default for future runs has been updated to:

```bash
OMP_NUM_THREADS=32
MKL_NUM_THREADS=32
OPENBLAS_NUM_THREADS=32
```

This default was written into:

- `/Users/weimingming/dr-claw/scd-new/Experiment/core_code/run_ser_mistral_self_locked_routes.sh`

The completed original `MistralLite` run used the earlier `OMP_NUM_THREADS=16` launch environment. The new optimized rerun uses the updated `32/32/32` thread defaults.

## Next Action

Next action:

1. instrument the post-projection K/V layout path on `patch-only` and split the current `patch_rotary_ms` bucket into true rotary versus layout/materialization cost
2. decide whether the CUDA Graph path should stay enabled only for projection benchmarking or be kept off by default until the downstream layout regression is removed
3. only after the breakdown is clean, try a second runtime pass that fuses or reorders the K/V reshape-materialize path instead of touching the locked training setup again
