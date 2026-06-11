# Qwen3-1.7B Self-Self Patch-Only All-Layer Identity Residual on `ser` (2026-04-03)

Run source:
- Remote run: `ser:/disk_n/c101-3-1/projects/scd-repro-codex-mistral-instruct-20260323/runs/qwen17_self_cmrc/strict_patch_only_alllayers_identity_sersharded_20260403`
- Local mirror: `/Users/weimingming/dr-claw/scd-new/Experiment/analysis/remote_artifacts/qwen17_self_patchonly_alllayers_identity_sersharded_20260403`

Setup:
- Pair: Qwen3-1.7B -> Qwen3-1.7B self-self
- Data: CMRC dev, 50 samples
- Method: `patch_only`
- Route: all 28 layers patched, no reuse, no fallback, no recompute
- Patch fit mode: `xnorm_nativekv_residual_mlp_v1`
- Synthetic checkpoint: rebuild from the healthy mixed all-28 checkpoint, but rewrite metadata to `residual_layers=[0..27]` with all-layer identity hints
- Checkpoint format: `patch_ckpt_sharded_v1`, `layers_per_shard=4`, `drop_identity_residual=true`
- Sharded checkpoint size: `448.24 MB`

Key result:

| Method | F1 | EM | TTFT (ms) | Payload (MB) |
| --- | ---: | ---: | ---: | ---: |
| patch_only all-layer identity residual | 0.7462 | 0.38 | 48.20 | 41.32 |

Patch breakdown:

| patch_infer | patch_shard_load | patch_shard_gpu_copy | patch_residual_forward | patch_kproj_vproj | patch_knorm | patch_rotary | decode_first |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 23.81 | 0.00 | 0.00 | 0.00 | 3.82 | 4.14 | 6.18 | 24.39 |

Notes:
- This run is a runtime/latency probe for the "all layers use layer-0 style residual/native-kv patch" route. It is not a trained all-layer residual checkpoint.
- Because the synthetic checkpoint uses all-layer identity hints and the shard bundle is small enough to stay resident, timed `patch_shard_load_ms` and `patch_shard_gpu_copy_ms` both dropped to `0.00 ms`.
- Quality landed at baseline level (`F1 0.7462`, `EM 0.38`), which is consistent with an all-layer identity residual route rather than a learned all-layer residual mapper.
- Two runtime fixes were needed to make this route executable on `ser`: prewarm must skip loading residual modules for identity-hinted layers, and residual layers must not eagerly fetch `direct_wk/direct_wv` keys.
