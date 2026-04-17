# Mistral Cross-Pair Locked-Route Comparison

Date: `2026-04-05`

## Scope

This note records the strict no-fallback cross-model comparison for the locked Mistral-family routes on:

- `MistralLite -> Mistral-7B`
- `Mistral-7B -> MistralLite`

The table keeps only the three canonical methods:

- `baseline` (`full_consumer`)
- `reuse-only`
- `patch-only`

Measurement convention:

- `patch-only` is benchmarked in an isolated fresh process and then merged back into the final comparison artifact
- both directions use `fallback_layers=0` and `selected_recompute_layers=0`
- `patch-only` TTFT is now measured strictly as `encode_ms + transfer_ms + patch_infer_ms + decode_first_ms`
- `patch_infer_ms` is the end-to-end patch compute wall-clock, not a loose sum of sub-stage buckets

This file supersedes the earlier same-day cross-pair headline that undercounted `patch-only` TTFT.

Reverse-pair status update:

- for `Mistral-7B -> MistralLite`, the accepted route source-of-truth is now the locked strict reverse artifact recorded in:
  - `/Users/weimingming/dr-claw/scd-new/Experiment/analysis/mistral7_to_mistrallite_final_route_lock_20260409.md`
- later reverse fallback-based `Q2` artifacts are archived-only and must not override this note

## Raw Results

| model pair | method | F1 | EM | TTFT ms |
| --- | --- | ---: | ---: | ---: |
| `MistralLite -> Mistral-7B` | `baseline` | 0.6185 | 0.36 | 127.86 |
| `MistralLite -> Mistral-7B` | `reuse-only` | 0.0117 | 0.00 | 33.39 |
| `MistralLite -> Mistral-7B` | `patch-only` | 0.8072 | 0.54 | 121.48 |
| `Mistral-7B -> MistralLite` | `baseline` | 0.7829 | 0.60 | 122.50 |
| `Mistral-7B -> MistralLite` | `reuse-only` | 0.0212 | 0.00 | 33.60 |
| `Mistral-7B -> MistralLite` | `patch-only` | 0.7528 | 0.50 | 75.94 |

## Patch-Only Strict TTFT Breakdown

| model pair | encode ms | transfer ms | patch compute ms | decode first ms | TTFT ms | shard load ms | shard gpu copy ms | kproj/vproj ms | rotary ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `MistralLite -> Mistral-7B` | 0.86 | 9.08 | 87.17 | 24.37 | 121.48 | 62.61 | 1.57 | 11.23 | 5.48 |
| `Mistral-7B -> MistralLite` | 0.83 | 15.92 | 35.04 | 24.15 | 75.94 | 5.03 | 1.10 | 5.98 | 12.81 |

## Normalized Against Baseline

| model pair | method | TTFT speedup vs baseline | F1 retention vs baseline |
| --- | --- | ---: | ---: |
| `MistralLite -> Mistral-7B` | `reuse-only` | 3.83x | 0.0190 |
| `MistralLite -> Mistral-7B` | `patch-only` | 1.05x | 1.3050 |
| `Mistral-7B -> MistralLite` | `reuse-only` | 3.65x | 0.0271 |
| `Mistral-7B -> MistralLite` | `patch-only` | 1.61x | 0.9614 |

## Reading

- `reuse-only` is effectively broken in both cross-model directions: TTFT improves sharply, but F1 collapses to near zero.
- `patch-only` is still the only healthy cross-model transfer route on the current locked Mistral branch, but the strict TTFT story is much narrower than the earlier loose-accounting result suggested.
- On `MistralLite -> Mistral-7B`, `patch-only` still improves quality from `0.6185` to `0.8072`, but TTFT only falls slightly from `127.86 ms` to `121.48 ms`. The main bottleneck is patch-path compute, especially `patch_shard_load_ms = 62.61 ms`.
- On `Mistral-7B -> MistralLite`, `patch-only` keeps most of baseline quality at `96.1%` F1 retention while still reducing TTFT from `122.50 ms` to `75.94 ms`.
- The earlier `patch-only` cross-pair numbers of `49.18 ms` and `40.48 ms` should be treated as superseded failure-analysis artifacts, because they did not strictly charge the patch-path `encode + transfer + wall-clock patch compute`.

## Artifact Sources

- `/Users/weimingming/dr-claw/scd-new/Experiment/analysis/remote_artifacts/mistrallite_to_mistral7_cmrc_locked_20260405/strict_baseline_reuse_patchonly_all32_limit50_20260404/final_comparison.json`
- `/Users/weimingming/dr-claw/scd-new/Experiment/analysis/remote_artifacts/mistral7_to_mistrallite_cmrc_locked_20260405_gpu23/strict_baseline_reuse_patchonly_all32_limit50_20260404/final_comparison.json`
