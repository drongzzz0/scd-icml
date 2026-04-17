# Qwen3-1.7B Self-Self All-Layer Residual Approximation on `ser` (2026-04-03)

Run source:
- Remote run: `ser:/disk_n/c101-3-1/projects/scd-repro-codex-mistral-instruct-20260323/runs/qwen17_self_cmrc/strict_patch_only_alllayers_approx_subset12_e2_h2048_b2_20260403`
- Local mirror: `/Users/weimingming/dr-claw/scd-new/Experiment/analysis/remote_artifacts/qwen17_self_patchonly_alllayers_approx_subset12_e2_h2048_b2_20260403`

Approximation setup:
- Pair: Qwen3-1.7B -> Qwen3-1.7B self-self
- Data: CMRC layerwise pairs subset, first 12 chunks out of 50
- Method family: all-layer `xnorm_nativekv_residual_mlp_v1`
- Residual coverage: 28/28 layers
- Residual MLP size: hidden `2048`, `2` blocks
- Epochs: `2`
- Checkpoint size after training: `6.56 GB`
- Sharded checkpoint size: `6.56 GB`, `7` shards, `4` layers per shard

Main result:

| Method | F1 | EM | TTFT (ms) | Payload (MB) |
| --- | ---: | ---: | ---: | ---: |
| patch_only all-layer residual approx | 0.7462 | 0.38 | 70.52 | 41.32 |

Patch breakdown:

| patch_infer | patch_shard_load | patch_shard_gpu_copy | patch_residual_forward | patch_kproj_vproj | patch_knorm | patch_rotary | decode_first |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 38.98 | 0.00 | 0.00 | 0.00 | 4.14 | 4.45 | 6.70 | 31.54 |

Why this did not become non-identity in practice:
- This is a self-self run, so the hidden-state target for every residual layer is exactly `xb_target = xa`.
- Training metrics confirm that the initialization is already optimal for the hidden mapping and native KV target: sampled layers ended with `base_h_rel_mse = h_rel_mse = 0`, `base_k_rel_mse = k_rel_mse`, `base_v_rel_mse = v_rel_mse`, and `delta_reg = 0`.
- As a result, the best checkpoint stayed at the zero-delta identity solution on every layer even though the model class and optimizer were nontrivial.

Implication:
- On self-self, "train a non-identity all-layer residual mapper" is not a meaningful target: the optimum is identity.
- The runtime cost still went up because the checkpoint stayed materialized as a nontrivial all-layer residual bundle (`6.56 GB`) instead of the compact synthetic identity-hinted bundle (`448 MB`), even though functionally it implemented the same mapping.
