# Qwen3-1.7B Self-Self Aligned Comparison (2026-04-02)

Run source:
- Remote run: `ser:/disk_n/c101-3-1/projects/scd-repro-codex-mistral-instruct-20260323/runs/qwen17_self_cmrc/strict_patch_only_reuse_aligned_20260402_v2`
- Local mirror: `/Users/weimingming/dr-claw/scd-new/Experiment/analysis/remote_artifacts/qwen17_self_patchonly_reuse_aligned_20260402_v2`

Settings:
- Dataset: CMRC dev, 50 samples
- Pair: Qwen3-1.7B -> Qwen3-1.7B self-self
- Strict mode: yes
- Fallback: none
- Recompute: none
- Route: `reuse=[]`, `patch=[0..27]`, `fallback=[]`
- Patch fit mode: `xnorm_nativekv_residual_mlp_v1`
- Baseline decode boundary: `align_full_consumer_first_decode = true`
- Devices: producer `cuda:2`, consumer `cuda:3`

## Main Table

| Method | F1 | EM | TTFT (ms) | decode_first (ms) | Payload (MB) |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 0.7462 | 0.38 | 62.38 | 25.22 | 0.00 |
| reuse_only | 0.7372 | 0.38 | 32.38 | 23.92 | 41.32 |
| patch_only | 0.7700 | 0.46 | 56.25 | 23.81 | 41.32 |

## Patch-Only Breakdown

| Method | encode (ms) | transfer (ms) | rebuild (ms) | patch_infer (ms) | patch_kproj_vproj (ms) | patch_knorm (ms) | patch_rotary (ms) | decode_first (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| patch_only | 0.00 | 0.00 | 0.00 | 32.44 | 0.18 | 4.36 | 15.02 | 23.81 |

## Reuse-Only Breakdown

| Method | encode (ms) | transfer (ms) | rebuild (ms) | rebuild_prep (ms) | rebuild_decode_core (ms) | rebuild_rerope (ms) | rebuild_materialize (ms) | decode_first (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| reuse_only | 0.41 | 5.88 | 2.16 | 0.10 | 0.42 | 0.98 | 0.66 | 23.92 |

## Notes

- This run uses the aligned baseline decode-first definition, so `baseline.decode_first` is directly comparable to the transfer routes.
- `patch_only` is strictly patch-only here: no reuse scaffold, no fallback, no recompute.
- Relative to aligned baseline, `reuse_only` saves `30.01 ms` TTFT but loses `0.0090` F1.
- Relative to aligned baseline, `patch_only` saves `6.14 ms` TTFT and gains `0.0239` F1.
