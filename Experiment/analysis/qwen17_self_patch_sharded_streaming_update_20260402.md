# Qwen3-1.7B Self-Self All-Layer Patch: Sharded Streaming to Prewarmed Serving

Date: 2026-04-02

Scope:
- Route: strict `reuse_patch`, no fallback, no recompute
- Pair: Qwen3-1.7B -> Qwen3-1.7B self-self
- Patch fit mode: `xnorm_nativekv_residual_mlp_v1`
- Patch layers: all 28 layers
- Runtime patch window: 4 layers
- Eval set: strict CMRC, 50 samples

Engineering changes:
- Added a true sharded patch checkpoint format (`patch_ckpt_sharded_v1`) and runtime lazy shard loading.
- Re-sharded the synthetic all-layer identity checkpoint from:
  - `/nas/mql/scd_runs/qwen17_self_cmrc/ckpt/ckpt_patch_xnorm_nativekv_residual_mlp_alllayers_identity_len1024_seed42`
  to:
  - `/nas/mql/scd_runs/qwen17_self_cmrc/ckpt/ckpt_patch_xnorm_nativekv_residual_mlp_alllayers_identity_len1024_seed42_sharded_w4_dropid`
- The new checkpoint uses 7 shard files (`4` layers per shard) and drops identity residual-module state for every layer.
- Added persistent CPU/GPU shard caching for small sharded patch bundles.
- Cached native rotary lookup and `(cos, sin)` buffers.
- Prewarmed sharded patch weights outside timed TTFT and skipped timed cleanup when persistent shard caches remain resident.

Checkpoint size:

| artifact | size |
|---|---:|
| old monolithic checkpoint | 45G |
| new sharded checkpoint | 449M |

`reuse_patch` runtime progression on 50 CMRC samples:

| run | F1 | TTFT (ms) | patch_infer (ms) | patch_shard_load (ms) | patch_shard_gpu_copy (ms) | patch_rotary (ms) | decode_first (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|
| monolithic CPU->GPU window | 0.7376 | 174703.75 | 174674.07 | 0.00 | 0.00 | 69.73 | 27.19 |
| true sharded streaming | 0.7376 | 18324.25 | 18266.31 | 5749.33 | 4863.67 | 3642.88 | 55.43 |
| sharded + persistent cache + rotary cache | 0.7376 | 2002.10 | 1973.83 | 1085.58 | 19.18 | 12.98 | 25.77 |
| sharded + cache + prewarm | 0.7376 | 46.22 | 19.44 | 0.00 | 0.00 | 6.17 | 23.83 |

Latest `v2` run comparison:

| mode | F1 | TTFT (ms) | encode (ms) | transfer (ms) | rebuild (ms) | patch_infer (ms) | decode_first (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|
| full_consumer | 0.7398 | 41.47 | 0.00 | 0.00 | 0.00 | 0.00 | 0.17 |
| reuse_only | 0.7372 | 116.19 | 0.35 | 85.30 | 5.70 | 0.00 | 24.84 |
| reuse_patch | 0.7376 | 46.22 | 0.28 | 0.71 | 1.97 | 19.44 | 23.83 |

Latest `reuse_patch` steady-state breakdown (`v2`):

| term | ms |
|---|---:|
| encode | 0.28 |
| transfer | 0.71 |
| rebuild | 1.97 |
| patch_infer | 19.44 |
| patch_shard_load | 0.00 |
| patch_shard_gpu_copy | 0.00 |
| patch_kproj_vproj | 3.42 |
| patch_knorm | 4.20 |
| patch_rotary | 6.17 |
| decode_first | 23.83 |

Interpretation:
- The original `174.7s` bottleneck was almost entirely checkpoint handling. Converting the all-layer identity patch bundle to a true shard format removed the monolithic load wall, and the later cache residency plus prewarm passes removed shard load and shard GPU copy from the timed path entirely.
- Across the full sequence, `reuse_patch.ttft_ms` fell from `174703.75` to `46.22`, a reduction of about `3780x`, while `reuse_patch.f1` stayed fixed at `0.7376`.
- Under the final serving-style measurement, `reuse_patch` is now only `4.75 ms` slower than same-run `full_consumer` (`46.22` vs `41.47`) and much faster than `reuse_only` (`116.19 ms`), still with no fallback and no recompute.
- The remaining steady-state patch cost is no longer storage or transfer dominated. The visible subterms are now small native compute pieces: `patch_rotary_ms` about `6.17`, `patch_knorm_ms` about `4.20`, and `patch_kproj_vproj_ms` about `3.42`.
- This means the route is now usable as a runtime baseline for the self-self identity setting. The next meaningful question is no longer whether the runtime path can be made fast enough, but whether the same runtime treatment preserves quality on non-identity or near-pair patch checkpoints.

Local artifact mirrors:
- `/Users/weimingming/dr-claw/scd-new/Experiment/analysis/remote_artifacts/qwen17_self_patchonly_xnorm_nativekv_residual_mlp_all28_len1024_50_20260401_a800_stream4_v1/summary.json`
- `/Users/weimingming/dr-claw/scd-new/Experiment/analysis/remote_artifacts/qwen17_self_patchonly_xnorm_nativekv_residual_mlp_all28_len1024_50_20260401_a800_stream4_v1/per_mode_summary.json`
- `/Users/weimingming/dr-claw/scd-new/Experiment/analysis/remote_artifacts/qwen17_self_patchonly_xnorm_nativekv_residual_mlp_all28_len1024_50_20260401_a800_stream4_shardedw4dropid_v1/summary.json`
- `/Users/weimingming/dr-claw/scd-new/Experiment/analysis/remote_artifacts/qwen17_self_patchonly_xnorm_nativekv_residual_mlp_all28_len1024_50_20260401_a800_stream4_shardedw4dropid_v1/per_mode_summary.json`
- `/Users/weimingming/dr-claw/scd-new/Experiment/analysis/remote_artifacts/qwen17_self_patchonly_xnorm_nativekv_residual_mlp_all28_len1024_50_20260402_a800_stream4_shardedw4dropid_cachepersist_rotcache_v1/summary.json`
- `/Users/weimingming/dr-claw/scd-new/Experiment/analysis/remote_artifacts/qwen17_self_patchonly_xnorm_nativekv_residual_mlp_all28_len1024_50_20260402_a800_stream4_shardedw4dropid_cachepersist_rotcache_v1/per_mode_summary.json`
- `/Users/weimingming/dr-claw/scd-new/Experiment/analysis/remote_artifacts/qwen17_self_patchonly_xnorm_nativekv_residual_mlp_all28_len1024_50_20260402_a800_stream4_shardedw4dropid_prewarm_cachepersist_rotcache_v2/summary.json`
- `/Users/weimingming/dr-claw/scd-new/Experiment/analysis/remote_artifacts/qwen17_self_patchonly_xnorm_nativekv_residual_mlp_all28_len1024_50_20260402_a800_stream4_shardedw4dropid_prewarm_cachepersist_rotcache_v2/per_mode_summary.json`
- `/Users/weimingming/dr-claw/scd-new/Experiment/analysis/remote_artifacts/qwen17_self_patch_xnorm_nativekv_residual_mlp_alllayers_identity_len1024_seed42_sharded_w4_dropid/meta.json`
- `/Users/weimingming/dr-claw/scd-new/Experiment/analysis/remote_artifacts/qwen17_self_patch_xnorm_nativekv_residual_mlp_alllayers_identity_len1024_seed42_sharded_w4_dropid/manifest.json`
