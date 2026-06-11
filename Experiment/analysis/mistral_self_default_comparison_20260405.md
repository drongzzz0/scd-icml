# Mistral Self-Self Default Comparison

Date: `2026-04-05`

## Scope

This note consolidates the current default serving-style comparison for the locked self-self routes on:

- `Mistral-7B`
- `MistralLite`

The comparison table keeps only the three canonical methods:

- `baseline` (`full_consumer`)
- `reuse-only`
- `patch-only`

Important measurement convention:

- `Mistral-7B` uses the existing canonical strict run
- `MistralLite` uses the new corrected default path where `patch-only` is benchmarked in an isolated fresh process and merged back into the final comparison artifact

## Raw Results

| model pair | method | F1 | EM | TTFT ms |
| --- | --- | ---: | ---: | ---: |
| `Mistral-7B -> Mistral-7B` | `baseline` | 0.6185 | 0.36 | 129.91 |
| `Mistral-7B -> Mistral-7B` | `reuse-only` | 0.6486 | 0.36 | 35.01 |
| `Mistral-7B -> Mistral-7B` | `patch-only` | 0.6385 | 0.38 | 47.10 |
| `MistralLite -> MistralLite` | `baseline` | 0.7829 | 0.60 | 122.95 |
| `MistralLite -> MistralLite` | `reuse-only` | 0.7686 | 0.58 | 48.70 |
| `MistralLite -> MistralLite` | `patch-only` | 0.7829 | 0.60 | 39.88 |

## Normalized Against Baseline

| model pair | method | TTFT speedup vs baseline | F1 retention vs baseline |
| --- | --- | ---: | ---: |
| `Mistral-7B -> Mistral-7B` | `reuse-only` | 3.71x | 1.0487 |
| `Mistral-7B -> Mistral-7B` | `patch-only` | 2.76x | 1.0323 |
| `MistralLite -> MistralLite` | `reuse-only` | 2.52x | 0.9816 |
| `MistralLite -> MistralLite` | `patch-only` | 3.08x | 1.0000 |

## Reading

The current default-route picture is:

- on `Mistral-7B`, both `reuse-only` and `patch-only` beat the baseline in TTFT, and both slightly exceed the same-run baseline F1
- on `MistralLite`, `patch-only` is now the strongest operating point under the corrected isolated-process benchmark: it matches baseline quality exactly and improves TTFT from `122.95 ms` to `39.88 ms`
- on `MistralLite`, `reuse-only` is still faster than baseline, but it loses a small amount of quality relative to both `baseline` and `patch-only`

## Artifact Sources

- `Mistral-7B`: `/Users/weimingming/dr-claw/scd-new/Experiment/analysis/remote_artifacts/mistral7_self_strict_baseline_reuse_patchonly_all32_50_20260404/per_mode_summary.json`
- `MistralLite`: `/Users/weimingming/dr-claw/scd-new/Experiment/analysis/remote_artifacts/mistrallite_self_strict_baseline_reuse_patchonly_all32_50_20260405_isolated/per_mode_summary.json`
