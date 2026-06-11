Archived-default note:

- This file records an earlier shared `patch-only` runtime before the April 4 route lock.
- Default method definitions are now controlled by:
  `/Users/weimingming/dr-claw/scd-new/Experiment/analysis/transfer_route_lock_20260404.md`

Qwen3-1.7B -> Qwen3-1.7B-bf16 strict no-fallback rerun for option 3 completed on April 3, 2026. This route kept all 28 patch layers active, but replaced the previous per-layer `xnorm_nativekv_residual_mlp_v1` stack with a shared all-layer variant `xnorm_nativekv_shared_residual_mlp_v1` plus a batched runtime path for native `k_proj/v_proj`, `k_norm`, and rotary.

The training result is important: on the subset12, one-epoch run, the best checkpoint stayed at the identity initialization. In other words, the shared residual module did not learn a useful hidden-state correction beyond `xb_hat = xa`, but the route still remained quality-preserving on the close non-self pair because the consumer-native projection path is already sufficiently compatible.

Strict 50-sample CMRC results:

| method | F1 | EM | TTFT ms |
| --- | ---: | ---: | ---: |
| baseline | 0.7408 | 0.38 | 103.26 |
| reuse-only | 0.0055 | 0.00 | 152.41 |
| patch-only shared | 0.7457 | 0.38 | 112.05 |

Compared with the previous all-layer per-layer-MLP patch-only run on the same pair and same 50-sample strict setup, patch-only quality stayed effectively unchanged while latency improved materially:

| route | F1 | TTFT ms | patch_infer ms |
| --- | ---: | ---: | ---: |
| old per-layer residual MLP | 0.7457 | 176.10 | 147.61 |
| new shared all-layer patch | 0.7457 | 112.05 | 86.06 |

Patch-only latency breakdown for the new shared route:

| term | ms |
| --- | ---: |
| patch_infer | 86.06 |
| patch_shard_load | 3.69 |
| patch_shard_gpu_copy | 4.64 |
| patch_residual_forward | 0.00 |
| patch_kproj_vproj | 26.20 |
| patch_knorm | 24.43 |
| patch_rotary | 27.09 |
| decode_first | 26.00 |

Interpretation: option 3 works as a runtime optimization, not as a better learned transport map. The speedup came from eliminating the heavy per-layer residual forward path and batching the consumer-native reconstruction steps. The remaining gap to the same-run baseline is now about 8.79 ms TTFT, and most of that residual gap is split between batched native projection, k_norm, rotary, and the one-time shared patch shard load/copy.
