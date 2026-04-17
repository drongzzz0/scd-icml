# Canonical Transfer Route Lock

Date: `2026-04-05`

## Decision

As of April 5, 2026, the current default transfer routes are frozen.

For all future experiments, tables, and claim-facing comparisons, only these four methods are valid:

- `baseline`
- repaired `reuse-only`
- current strict `patch-only`
- current strict `hybrid`

Within this project, the phrase `SCD method` now has a single claim-facing meaning:

- the locked strict method family above

Historical fallback-based, recompute-based, scaffold-mixed, or older patch-fit variants are not valid `SCD methods` for default use, even if they remain useful as debugging history.

Everything else is archived-only and must not be used as a default comparison route.

This note supersedes the earlier lock note from April 4 by tightening two points:

- `patch-only` benchmark semantics now explicitly require strict TTFT accounting as `encode + transfer + wall-clock patch_infer + decode_first`
- on Mistral-family runs where shared-process evaluation perturbs `patch_kproj_vproj`, `patch-only` must be benchmarked in an isolated fresh process and merged back afterward
- `hybrid` is now fixed as a first-class canonical route, but only in its strict no-fallback / no-recompute form

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

TTFT accounting:

- `reuse-only` TTFT must be charged as `encode_ms + transfer_ms + rebuild_ms + decode_first_ms`
- `transfer_ms` is the measured transport term, not a modeled bandwidth placeholder
- `rebuild_ms` must stay restricted to consumer-side reconstruction/materialization compute

This is the only valid reuse route.

### current strict `patch-only`

Definition:

- dedicated `patch_only` runtime branch
- no reuse scaffold
- `no fallback`
- `no recompute`
- fit family `xnorm_nativekv_shared_residual_mlp_v1`
- hidden transport path: `xnorm_a -> shared residual patch -> xb_hat`
- native KV reconstruction path: `xb_hat -> consumer Wk/Wv -> native post-projection ops`

Native post-projection ops are consumer-architecture-specific:

- apply `K-only k_norm` only when the consumer architecture actually has it
- apply native rotary only where the consumer architecture requires it
- do not add extra normalization or rotary work that is not part of the consumer model's own KV path

TTFT accounting:

- `patch-only` TTFT must be charged as `encode_ms + transfer_ms + patch_infer_ms + decode_first_ms`
- `patch_infer_ms` is the end-to-end patch compute wall-clock, not a loose sum of sub-buckets
- unless the user explicitly asks for untimed prewarm, timed `patch_infer_ms` still owns timed shard load/copy overhead

Benchmarking rule:

- on Mistral-family runs, benchmark `patch-only` in an isolated fresh process by default and merge afterward if shared-process evaluation inflates `patch_kproj_vproj`
- if `patch-only` is measured inside a shared multi-method process, that result is diagnostic-only unless it has been checked against the isolated path

This is the only valid patch route.

### current strict `hybrid`

Definition:

- `patch-only` anchor plus greedy reuse search
- use the repaired clean `reuse-only` path as the only reuse primitive
- `no fallback`
- `no recompute`
- replay selector: quality first, TTFT second
- stop immediately when replay quality-pass set becomes empty

Quality gate:

- by default, gate against `patch-only`
- use the explicitly reported gate for the run, such as relative `F1 drop <= 5%` versus `patch-only`

Reporting rule:

- if a pair has no validated strict hybrid artifact, omit `hybrid` from the result table
- do not substitute any fallback-based or recompute-based archived route in its place

This is the only valid hybrid route.

## Explicitly Deprecated Routes

The following are no longer valid for headline use, default comparisons, or claim judgment:

- grouped/shared-z reuse
- any reuse result whose speed story depends on grouped compression
- any old TTFT accounting that mixed modeled transfer with measured staging copy
- any result with implicit or explicit fallback unless the user explicitly requested fallback
- `reuse_fallback`
- `reuse_patch`
- `reuse_patch_recompute`
- any fallback-based or recompute-based artifact that is later reused as if it were a canonical `hybrid`
- any patch route that still runs inside the old reuse scaffold
- old direct-KV patch families such as `direct_kv_v1` and `direct_kv_knorm_v1`
- old residual patch families such as `direct_kv_knorm_residual_mlp_v1`
- old semantic patch families such as `xnorm_semantic_v1`
- old non-shared hidden-residual patch families such as `xnorm_nativekv_residual_mlp_v1`
- older patch runtime reports before the current strict TTFT accounting and isolated-process rule

These artifacts remain useful only as debugging evidence, ablation history, or failure analysis.

## Default Comparison Contract

For any new pair, the default serving comparison is:

| method | allowed | notes |
| --- | --- | --- |
| `baseline` | yes | canonical native reference |
| `reuse-only` | yes | repaired layerwise reuse only |
| `patch-only` | yes | current strict patch-only only |
| `hybrid` | yes | strict no-fallback / no-recompute hybrid only |
| `reuse_fallback` | no | archived only |
| `reuse_patch` | no | archived only |
| `reuse_patch_recompute` | no | archived only |

If a run does not match this contract, it is not a default result.

Equivalently:

- if a table row is presented as an `SCD method`, it must satisfy this contract
- otherwise it must be labeled explicitly as archived, diagnostic, or failure-analysis only

## Current Reference Runs

Current self-self reference table:

- `/Users/weimingming/dr-claw/scd-new/Experiment/analysis/mistral_self_default_comparison_20260405.md`

Current cross-pair full-dev reference:

- `/Users/weimingming/dr-claw/scd-new/Experiment/analysis/mistrallite_to_mistral7_fulldev_locked_routes_20260405.md`

Pair-specific reverse-route lock for `Mistral-7B -> MistralLite`:

- `/Users/weimingming/dr-claw/scd-new/Experiment/analysis/mistral7_to_mistrallite_final_route_lock_20260409.md`

Full-dev `MistralLite -> Mistral-7B` headline on `CMRC dev = 3219`:

| method | F1 | EM | TTFT ms |
| --- | ---: | ---: | ---: |
| `baseline` | 0.6911 | 0.4427 | 133.39 |
| `reuse-only` | 0.0109 | 0.0000 | 38.49 |
| `patch-only` | 0.7898 | 0.5769 | 67.37 |

Interpretation:

- `reuse-only` remains broken on the current Mistral cross-pair branch
- `patch-only` remains the only healthy cross-model route
- the route definitions are now fixed independently of which pair happens to favor reuse or patch more strongly

## What To Do Going Forward

For any new model pair:

1. Run `baseline`.
2. Train and evaluate repaired `reuse-only`.
3. Train and evaluate current strict `patch-only`.
4. If a validated strict no-fallback / no-recompute hybrid exists, report `hybrid`.
5. Otherwise, omit `hybrid` instead of substituting an archived route.

If an archived route is needed for diagnosis, label it explicitly as archived and keep it out of the main table.
