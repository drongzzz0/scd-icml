# Canonical Transfer Route Lock

Superseded for current default-route policy by:

- `/Users/weimingming/dr-claw/scd-new/Experiment/analysis/transfer_route_lock_20260405.md`

This note remains the original April 4 lock record.

Date: `2026-04-04`

## Decision

As of April 4, 2026, the transfer-method defaults are locked.

For all future experiments, tables, and claim-facing comparisons, only these three methods are valid:

- `baseline`
- repaired `reuse-only`
- current strict `patch-only`

Everything else is archived-only and must not be used as a default comparison route.

## Canonical Methods

### `baseline`

Definition:

- native full-consumer run
- same aligned `decode_first` boundary as transfer routes
- use the label `baseline` in notes and tables

### repaired `reuse-only`

Definition:

- `layerwise`
- `group_size = 1`
- batched encode/decode fast path
- actual measured `transfer_ms`
- pure consumer-side `rebuild_ms`
- `no fallback`
- `no recompute`

This is the only valid reuse route.

### current strict `patch-only`

Definition:

- dedicated `patch_only` runtime branch
- no reuse scaffold
- `no fallback`
- `no recompute`
- current fit family: `xnorm_nativekv_shared_residual_mlp_v1`
- all active patch layers are residual/shared-patch layers
- consumer-native `Wk/Wv` reconstruction
- `K-only` `k_norm`
- `K-only` rotary
- fused batched `k_proj/v_proj`
- fused `k_norm` inside the fast path

Serving semantics:

- hidden transport path: `xnorm_a -> shared residual patch -> xb_hat`
- native KV path: `xb_hat -> consumer Wk/Wv -> K-only knorm -> K-only rotary`
- in the latest `gray` run, the checkpoint effectively stayed near identity at serving time, so the runtime behaved close to native hidden reprojection rather than a materially active nonlinear transport map

This is the only valid patch route.

Training/runtime default lock:

- keep the pre-optimization Mistral shared-patch training recipe
- do not use the April 4 training-side experimental tweaks by default:
  block-level cached consumer-native `Wk/Wv` stacking in patch training
  `patch_batch_size=256`
  `layer_block_size=8`
- these two tweaks were measured on `MistralLite -> MistralLite` and did not improve the target `patch-only` runtime; they are archived as failed optimizations

## Explicitly Deprecated Routes

The following are no longer valid for headline use, default comparisons, or claim judgment:

- grouped/shared-z reuse
- any reuse result whose speed story depends on grouped compression
- any old TTFT accounting that mixed modeled transfer with measured staging copy
- any result with implicit or explicit fallback unless the user explicitly requested fallback
- `reuse_fallback`
- `reuse_patch`
- `reuse_patch_recompute`
- any patch route that still runs inside the old reuse scaffold
- old direct-KV patch families such as `direct_kv_v1` and `direct_kv_knorm_v1`
- old residual patch families such as `direct_kv_knorm_residual_mlp_v1`
- old semantic patch families such as `xnorm_semantic_v1`
- old non-shared hidden-residual patch families such as `xnorm_nativekv_residual_mlp_v1`
- older patch runtime reports before the `K-only rotary + fused KV + fused K-only knorm` path

These artifacts remain useful only as debugging evidence, ablation history, or failure analysis.

## Default Comparison Contract

For any new pair, the default serving comparison is:

| method | allowed | notes |
| --- | --- | --- |
| `baseline` | yes | canonical native reference |
| `reuse-only` | yes | repaired layerwise reuse only |
| `patch-only` | yes | current strict patch-only only |
| `reuse_fallback` | no | archived only |
| `reuse_patch` | no | archived only |
| `reuse_patch_recompute` | no | archived only |

If a run does not match this contract, it is not a default result.

## Current Default Reference Run

Current default non-self reference pair:

- producer: `Qwen3-1.7B`
- consumer: `Qwen3-1.7B-gray`
- artifact:
  `/Users/weimingming/dr-claw/scd-new/Experiment/analysis/remote_artifacts/qwen17_to_gray_shared_patchonly_all28_subset12e1_50_20260403_keyonlyrot_fusedkvknorm/summary.json`

| method | F1 | EM | TTFT ms |
| --- | ---: | ---: | ---: |
| `baseline` | 0.7358 | 0.44 | 99.05 |
| `reuse-only` | 0.7775 | 0.46 | 58.11 |
| `patch-only` | 0.7547 | 0.42 | 63.96 |

Interpretation:

- repaired `reuse-only` is the current best default route
- current strict `patch-only` is now latency-competitive and faster than `baseline`
- old patch and reuse families must not be mixed back into the default story

## What To Do Going Forward

For any new model pair:

1. Run `baseline`.
2. Train and evaluate repaired `reuse-only`.
3. Train and evaluate current strict `patch-only`.
4. Report only these three in the default comparison table.

If an archived route is needed for diagnosis, label it explicitly as archived and keep it out of the main table.
