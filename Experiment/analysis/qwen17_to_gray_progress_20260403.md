# Qwen3-1.7B -> Qwen3-1.7B-gray Progress

Date: `2026-04-03`

## Scope

This note records the status of the new-pair rerun requested after the reuse-route reset.

Target pair:

- producer: `Qwen3-1.7B`
- consumer: `Qwen3-1.7B-gray`

Requested comparison:

- `baseline`
- repaired `reuse-only`
- shared `patch-only`

Default policy:

- no fallback
- repaired reuse route only

## Completed Before The SSH Tunnel Broke

### 1. Local and remote configs

Prepared and synced:

- `Experiment/core_code/ser_mirror/configs/models_ser_qwen17_gray.yaml`
- `Experiment/core_code/ser_mirror/configs/qwen17_to_gray_patch_xnorm_nativekv_shared_residual_mlp_alllayers_subset12_e1_h2048_r512_b2_20260403.json`

### 2. Full pair capture on `ser`

A full real `transformers` capture was rerun after discarding an earlier invalid synthetic one-sample capture.

Confirmed properties of the corrected capture:

- output root: `runs/qwen17_to_gray_cmrc/pairs_cmrc_seed42`
- `num_chunks = 200`
- `eval_samples = 200`
- `batch_size = 200`
- `prefix_tokens = 734`
- first chunk: `chunks/chunk_00000.pt`
- last chunk: `chunks/chunk_00199.pt`

Important note:

- the first capture attempt was invalid because it forgot `--backend transformers` and `--prompt-file .../cmrc_prompts_200.json`, so it only produced one synthetic prompt
- that invalid asset was deleted and replaced by the corrected 200-prompt capture

### 3. Clean layerwise reuse training

The repaired reuse route was trained successfully on the corrected 200-chunk capture:

- output: `runs/qwen17_to_gray_cmrc/ckpt_reuse_layerwise_seed42`
- route family: `layerwise`
- `group_size = 1`
- `rank_k = 128`
- `rank_v = 128`
- all 28 groups solved with `compression_ratio = 1.000`

This is the only reuse checkpoint that should be used for this pair.

### 4. Layer profiling

The profile run completed and wrote:

- `runs/qwen17_to_gray_cmrc/profile_cmrc_seed42.json`

Notable logged layer diagnostics from the completed profile:

| layer | reuse_error | patch_xnorm_error | patch_kv_error |
| --- | ---: | ---: | ---: |
| 0 | 30.198241 | 0.000001 | 39.172938 |
| 1 | 0.015108 | 0.000003 | 3.032827 |
| 10 | 0.041657 | 0.004398 | 1.393191 |
| 16 | 0.175617 | 0.063745 | 1.576275 |
| 21 | 0.515171 | 0.417783 | 0.689858 |
| 24 | 1.215283 | 0.839884 | 1.256763 |
| 25 | 1.906650 | 1.385069 | 1.439722 |
| 26 | 3.038430 | 1.759874 | 2.055930 |
| 27 | 3.944412 | 2.039967 | 2.330645 |

Interpretation:

- this pair is not a trivial bf16-style near-identity patch case
- deep-layer reuse error rises strongly
- deep-layer patch KV error also stays nontrivial

## Patch Training Attempts

### Attempt 1: invalid subset packaging

The first patch attempt failed for two fixable reasons:

1. `pairs_cmrc_seed42_subset12` initially only had a manifest, not accessible chunk files.
2. `--route` was not passed, so selection fell back to profile-recommended layers instead of the requested all-28 route.

Observed failure:

- `FileNotFoundError: runs/qwen17_to_gray_cmrc/pairs_cmrc_seed42_subset12/chunks/chunk_00000.pt`

### Attempt 2: fixed subset plus all-28 route

The second patch attempt repaired both issues:

- rebuilt `pairs_cmrc_seed42_subset12`
- added `chunks -> ../pairs_cmrc_seed42/chunks` symlink
- passed `--route runs/qwen17_to_gray_cmrc/route_reuse_patchonly_all28_20260403.json`

This successfully started true all-28 shared patch training.

Confirmed start state:

- selected layers: `0..27`
- residual layers: `0..27`

Observed patch-training logs before the SSH tunnel failed:

- linear-base solve reached at least layer `20`
- several mid/deep layers showed extremely large linear-base relative MSE

Representative logged values:

| layer | linear_base_kv_rel_mse |
| --- | ---: |
| 0 | 23.321392 |
| 1 | 7375.179037 |
| 11 | 232829.951643 |
| 12 | 33126104.585198 |
| 16 | 1028660188.532756 |
| 19 | 110003103.319796 |
| 20 | 1701751.900394 |

These numbers strongly suggest the `gray` pair is substantially harder for shared patch than the `bf16` pair.

## Temporary Blocker History

The remaining problem is external to the experiment code:

- the SSH path to `ser` started failing with
  `Connection closed by 127.0.0.1 port 7897`

After that point, new SSH sessions could not be established, so it was not possible to verify whether the second patch run finished, whether the checkpoint was reshared successfully, or whether the strict final evaluation produced results.

Current verification on this workspace shows a cleaner root cause on the public jump path itself:

- `frp-jump` at `frp.bnuzh.top:31005` is now `Connection refused`
- `A800` at `frp.bnuzh.top:28010` is also `Connection refused`

So the present blocker is not inside the SCD code or the `gray` run state. It is currently an external FRP/SSH availability problem that blocks both the `ser` route and the backup `A800` route.

## Completion After Connectivity Was Restored

SSH connectivity to `ser` was later restored and the remote run was checked directly.

Confirmed completed outputs:

- patch checkpoint:
  `runs/qwen17_to_gray_cmrc/ckpt_patch_xnorm_nativekv_shared_residual_mlp_alllayers_subset12_e1_h2048_r512_b2_20260403`
- sharded patch checkpoint:
  `runs/qwen17_to_gray_cmrc/ckpt_patch_xnorm_nativekv_shared_residual_mlp_alllayers_subset12_e1_h2048_r512_b2_20260403_sharded_w4`
- strict eval output:
  `runs/qwen17_to_gray_cmrc/strict_baseline_reuse_patchonly_all28_shared_subset12e1_50_20260403`

The final result note is:

- `/Users/weimingming/dr-claw/scd-new/Experiment/analysis/qwen17_to_gray_shared_patchonly_all28_subset12e1_50_20260403.md`

Final 50-sample comparison:

| method | F1 | EM | TTFT ms |
| --- | ---: | ---: | ---: |
| `baseline` | 0.7358 | 0.44 | 140.18 |
| `reuse-only` | 0.7775 | 0.46 | 45.48 |
| `patch-only` | 0.7667 | 0.44 | 159.47 |

Patch-side timing breakdown:

- `patch_infer_ms = 132.93`
- `patch_shard_load_ms = 3.72`
- `patch_shard_gpu_copy_ms = 0.76`
- `patch_kproj_vproj_ms = 43.83`
- `patch_knorm_ms = 41.47`
- `patch_rotary_ms = 43.15`

Interpretation:

- repaired `reuse-only` clearly wins on this pair
- shared `patch-only` keeps reasonable quality but fails the TTFT target
- the main patch bottleneck is steady-state native projection plus `k_norm` plus rotary, not shard loading

## Current Status

Completed:

- reuse-route documentation reset
- new-pair config sync
- corrected 200-sample capture
- clean layerwise reuse training
- completed layer profile
- repaired subset packaging and all-28 patch rerun launch

Finalized:

- final shared patch checkpoint for `gray`
- reshared patch bundle
- strict 50-sample final comparison
- final `baseline / reuse-only / patch-only` F1 and TTFT table for `gray`

## Next Resume Point

Completed resume steps:

1. reconnected to `ser`
2. verified patch training and resharing had finished
3. verified strict eval had finished
4. mirrored final artifacts into `Experiment/analysis/remote_artifacts/`
