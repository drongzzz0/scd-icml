# MistralLite To Mistral-7B Full-Dev Locked-Route Comparison

Date: `2026-04-05`

> Note (2026-07-02): the original per-sample strict-eval artifacts under
> `Experiment/analysis/remote_artifacts/` referenced by this document are not
> kept on this branch. They are preserved on the `legacy-main` branch.

## Scope

This note records the full-dev strict no-fallback comparison for the locked transfer routes on:

- `MistralLite -> Mistral-7B`
- dataset split: `CMRC2018 dev`
- total evaluated QA pairs: `3219`

The table keeps only the three canonical methods:

- `baseline` (`full_consumer`)
- `reuse-only`
- `patch-only`

Measurement convention:

- `baseline + reuse-only` are run together
- `patch-only` is benchmarked in an isolated fresh process and merged back afterward
- `fallback_layers = 0`
- `selected_recompute_layers = 0`
- `patch-only` TTFT is measured strictly as `encode_ms + transfer_ms + patch_infer_ms + decode_first_ms`

## Raw Results

| method | F1 | EM | TTFT ms |
| --- | ---: | ---: | ---: |
| `baseline` | 0.6911 | 0.4427 | 133.39 |
| `reuse-only` | 0.0109 | 0.0000 | 38.49 |
| `patch-only` | 0.7898 | 0.5769 | 67.37 |

## TTFT Breakdown

| method | encode ms | transfer ms | rebuild/patch compute ms | decode first ms | TTFT ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| `baseline` | 0.00 | 0.00 | 0.00 | 23.00 | 133.39 |
| `reuse-only` | 0.69 | 9.55 | 5.87 | 22.38 | 38.49 |
| `patch-only` | 2.07 | 17.21 | 24.22 | 23.86 | 67.37 |

`patch-only` sub-breakdown:

| sub-stage | ms |
| --- | ---: |
| `patch_shard_load` | 0.09 |
| `patch_shard_gpu_copy` | 0.02 |
| `patch_kproj_vproj` | 9.00 |
| `patch_knorm` | 0.00 |
| `patch_rotary` | 5.57 |

## Normalized Against Baseline

| method | TTFT speedup vs baseline | F1 retention vs baseline |
| --- | ---: | ---: |
| `reuse-only` | 3.47x | 0.0158 |
| `patch-only` | 1.98x | 1.1428 |

## Reading

- `reuse-only` remains unusable on this cross-model direction. It is fast, but quality collapses to near zero.
- `patch-only` remains healthy at full-dev scale and is the only viable cross-model route on the current Mistral branch.
- On this direction, `patch-only` is not merely latency-competitive. It improves both axes relative to `baseline`: TTFT falls from `133.39 ms` to `67.37 ms`, while F1 rises from `0.6911` to `0.7898`.
- The earlier 50-sample `MistralLite -> Mistral-7B` strict cross-pair headline is now only a small-sample diagnostic. The full-dev result is the canonical reference for this direction.

## Artifact Sources

- merged local artifact:
  `/Users/weimingming/dr-claw/scd-new/Experiment/analysis/remote_artifacts/mistrallite_to_mistral7_cmrc_locked_20260405_parallel/final_comparison.json`
- merged local per-mode summary:
  `/Users/weimingming/dr-claw/scd-new/Experiment/analysis/remote_artifacts/mistrallite_to_mistral7_cmrc_locked_20260405_parallel/per_mode_summary.json`
- merged local sample-level rows:
  `/Users/weimingming/dr-claw/scd-new/Experiment/analysis/remote_artifacts/mistrallite_to_mistral7_cmrc_locked_20260405_parallel/per_sample_metrics.jsonl`
