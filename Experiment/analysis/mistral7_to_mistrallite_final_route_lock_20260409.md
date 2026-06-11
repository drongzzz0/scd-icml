# Mistral-7B -> MistralLite Final Route Lock

Date: `2026-04-09`

## Decision

For the reverse Mistral pair

- `Mistral-7B -> MistralLite`

the final canonical route family is now frozen.

The source-of-truth artifact for the accepted strict reverse-pair baseline / reuse / patch comparison is:

- [summary.md](/Users/weimingming/dr-claw/scd-new/Experiment/analysis/remote_artifacts/mistral7_to_mistrallite_cmrc_locked_20260405_gpu23/strict_baseline_reuse_patchonly_all32_limit50_20260404/summary.md)
- [summary.json](/Users/weimingming/dr-claw/scd-new/Experiment/analysis/remote_artifacts/mistral7_to_mistrallite_cmrc_locked_20260405_gpu23/strict_baseline_reuse_patchonly_all32_limit50_20260404/summary.json)

For this reverse pair, that locked strict artifact is now the only accepted pair-specific realization of the `SCD method`.

Concretely:

- when a note, table, or rebuttal cites the `Mistral-7B -> MistralLite` `SCD method`, it must cite this locked artifact
- the accepted reverse-pair `SCD method` rows are currently only `baseline`, repaired `reuse-only`, and current strict `patch-only`
- `hybrid` remains part of the global `SCD method` definition, but on this reverse pair it is definition-only until a same-contract strict artifact is accepted

All older reverse-direction historical schemes are now archived-only and must not be used as default routes, headline comparisons, or rebuttal-facing evidence.

In particular, the later `Q2` reverse chain with `fallback` and `reuse_patch_recompute` is explicitly deprecated for route selection.

## Canonical Reverse-Pair Routes

The only valid default routes for `Mistral-7B -> MistralLite` are:

- `baseline`
- repaired `reuse-only`
- current strict `patch-only`
- current strict `hybrid`

### `baseline`

Definition:

- native `full_consumer`
- aligned `decode_first` boundary
- no transfer-side shortcuts

### repaired `reuse-only`

Definition:

- repaired clean layerwise reuse path
- `group_size = 1`
- actual measured `transfer_ms`
- pure consumer-side `rebuild_ms`
- `no fallback`
- `no recompute`

Canonical reverse-pair reference result from the locked strict artifact:

- `F1 = 0.0212`
- `EM = 0.0000`
- `TTFT = 33.60 ms`

Interpretation:

- this is the final reverse-pair reuse baseline
- it stays archived as a canonical comparison row, even though quality is effectively broken

### current strict `patch-only`

Definition:

- dedicated `patch_only` runtime branch
- isolated fresh-process evaluation
- `no reuse scaffold`
- `no fallback`
- `no recompute`
- fit family `xnorm_nativekv_shared_residual_mlp_v1`
- strict TTFT accounting:
  - `encode + transfer + wall-clock patch_infer + decode_first`

Canonical reverse-pair reference result from the locked strict artifact:

- `F1 = 0.7528`
- `EM = 0.5000`
- `TTFT = 75.94 ms`

Interpretation:

- this is the final reverse-pair patch route
- it preserves most baseline quality and materially reduces TTFT

### current strict `hybrid`

Definition:

- `patch-only` as the quality anchor
- greedy blockwise reuse search on top of the repaired reuse and strict patch routes
- `no fallback`
- `no recompute`
- replay selector: quality first, TTFT second
- stop immediately when replay quality-pass set becomes empty

Important scope note:

- the reverse-pair route contract is frozen now
- but there is no accepted reverse-pair hybrid metric in this note, because the locked strict source artifact named above only validates `baseline / reuse-only / patch-only`
- therefore for `Mistral-7B -> MistralLite`, `hybrid` is fixed as the only valid future hybrid route definition, but older fallback-based reverse hybrid numbers are discarded rather than carried forward

## Accepted Reverse-Pair Numbers

These are the currently accepted reverse-pair strict numbers.

| method | F1 | EM | TTFT ms | total ms |
| --- | ---: | ---: | ---: | ---: |
| `baseline` | 0.7829 | 0.6000 | 122.50 | 295.08 |
| `reuse-only` | 0.0212 | 0.0000 | 33.60 | 684.10 |
| `patch-only` | 0.7528 | 0.5000 | 75.94 | 608.83 |

## Explicitly Discarded Reverse-Pair History

The following reverse-pair artifacts are no longer valid for default use:

- any reverse result with `fallback_layers != 0`
- any reverse result with `reuse_fallback`
- any reverse result with `reuse_patch`
- any reverse result with `reuse_patch_recompute`
- the `Reviewer Q2 Fresh Rerun: Mistral-7B -> MistralLite` route as a source for final route selection

Those artifacts remain useful only as debugging or failure-analysis history.

## Practical Rule Going Forward

For `Mistral-7B -> MistralLite`, all future tables, scripts, and narration should follow this rule:

1. Use the locked strict `baseline / reuse-only / patch-only` artifact above as the canonical reverse reference.
2. Treat the strict no-fallback / no-recompute `hybrid` definition as the only valid future hybrid route.
3. Do not quote any reverse fallback-based route as if it were canonical evidence.

## Canonical Files

- [summary.md](/Users/weimingming/dr-claw/scd-new/Experiment/analysis/remote_artifacts/mistral7_to_mistrallite_cmrc_locked_20260405_gpu23/strict_baseline_reuse_patchonly_all32_limit50_20260404/summary.md)
- [summary.json](/Users/weimingming/dr-claw/scd-new/Experiment/analysis/remote_artifacts/mistral7_to_mistrallite_cmrc_locked_20260405_gpu23/strict_baseline_reuse_patchonly_all32_limit50_20260404/summary.json)
- [final_comparison.json](/Users/weimingming/dr-claw/scd-new/Experiment/analysis/remote_artifacts/mistral7_to_mistrallite_cmrc_locked_20260405_gpu23/strict_baseline_reuse_patchonly_all32_limit50_20260404/final_comparison.json)
- [per_mode_summary.json](/Users/weimingming/dr-claw/scd-new/Experiment/analysis/remote_artifacts/mistral7_to_mistrallite_cmrc_locked_20260405_gpu23/strict_baseline_reuse_patchonly_all32_limit50_20260404/per_mode_summary.json)
