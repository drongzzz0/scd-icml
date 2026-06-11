# Qwen3-1.7B -> Qwen3-1.7B-bf16 Strict Baseline / Reuse-only / Patch-only (2026-04-03)

Run source:
- Remote run: `ser:/disk_n/c101-3-1/projects/scd-repro-codex-mistral-instruct-20260323/runs/qwen17_to_bf16_cmrc/strict_baseline_reuse_patchonly_all28_subset12e1_50_20260403`
- Local mirror:
  - `/Users/weimingming/dr-claw/scd-new/Experiment/analysis/remote_artifacts/qwen17_to_bf16_strict_baseline_reuse_patchonly_all28_subset12e1_50_20260403/summary.json`
  - `/Users/weimingming/dr-claw/scd-new/Experiment/analysis/remote_artifacts/qwen17_to_bf16_strict_baseline_reuse_patchonly_all28_subset12e1_50_20260403/per_mode_summary.json`

Setup:
- Pair: `Qwen3-1.7B -> Qwen3-1.7B-bf16`
- Data: CMRC dev, `50` samples
- Strict setting: no fallback, no recompute
- Route:
  - `reuse_layers = 28`
  - `patch_layers = 28`
  - `fallback_layers = 0`
- Patch checkpoint:
  - fit mode: `xnorm_nativekv_residual_mlp_v1`
  - trained on `subset12`
  - all `28` residual layers
  - hidden `2048`
  - `2` blocks
  - `1` epoch
  - sharded runtime format: `layers_per_shard = 4`

Main result:

| Method | F1 | EM | TTFT (ms) | Payload (MB) |
| --- | ---: | ---: | ---: | ---: |
| baseline | 0.7408 | 0.38 | 119.97 | 0.00 |
| reuse-only | 0.0055 | 0.00 | 138.88 | 10.33 |
| patch-only | 0.7457 | 0.38 | 176.10 | 41.32 |

Patch-only breakdown:

| patch_infer | patch_kproj_vproj | patch_knorm | patch_rotary | decode_first |
| ---: | ---: | ---: | ---: | ---: |
| 147.61 | 20.01 | 20.72 | 22.47 | 28.49 |

Reuse-only breakdown:

| encode | transfer | staging_copy | rebuild | rebuild_decode_core | rebuild_rerope | rebuild_materialize | decode_first |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 21.17 | 17.51 | 17.51 | 74.08 | 24.67 | 28.94 | 20.46 | 26.12 |

Key takeaways:
- `reuse-only` still collapses in quality on this pair: `F1 0.0055`.
- `patch-only` fully recovers quality to baseline level and is slightly above baseline on this `50`-sample slice: `0.7457` vs `0.7408` F1.
- But the current all-layer patch runtime is much slower than baseline:
  - `+56.13 ms` TTFT vs baseline
  - `+37.22 ms` TTFT vs reuse-only
- The bottleneck is overwhelmingly `patch_infer_ms = 147.61`, not transfer or rebuild.
- `patch_shard_load_ms` and `patch_shard_gpu_copy_ms` are both `0.0 ms` in this run, so the extra latency is compute-bound rather than I/O-bound.

Interpretation:
- On the close non-self pair `Qwen3-1.7B -> Qwen3-1.7B-bf16`, the all-layer `xnorm_nativekv_residual_mlp_v1` route is quality-effective but latency-inefficient.
- This is the first clean evidence in the current workflow that a non-self near pair can support strict `patch-only` quality recovery without fallback.
- However, in its current all-layer form, this route does **not** deliver the SCD paper's claimed TTFT-quality trade-off, because TTFT is worse than the baseline.
