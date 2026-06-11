# MistralLite To Mistral-7B Train50 Route Comparison

Date: `2026-04-05`

## Scope

This note records the strict no-fallback `train50` comparison on:

- pair: `MistralLite -> Mistral-7B`
- data split: `CMRC2018 train`
- evaluated QA pairs: `50`

The current numbers combine:

- new four-GPU parallel run for `baseline` and `reuse-only`
- earlier isolated run for `patch-only`

## Headline Results

| method | F1 | EM | TTFT ms |
| --- | ---: | ---: | ---: |
| `baseline` | 0.6364 | 0.2200 | 128.56 |
| `reuse-only` | 0.0161 | 0.0000 | 52.77 |
| `patch-only` | 0.6676 | 0.2800 | 76.44 |

## TTFT Breakdown

| method | encode ms | transfer ms | rebuild / patch ms | decode first ms | TTFT ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| `baseline` | 0.00 | 0.00 | 0.00 | 25.43 | 128.56 |
| `reuse-only` | 0.73 | 11.66 | 14.14 | 26.24 | 52.77 |
| `patch-only` | 0.98 | 9.68 | 39.45 | 26.34 | 76.44 |

`patch-only` sub-breakdown:

| sub-stage | ms |
| --- | ---: |
| `patch_shard_load` | 5.09 |
| `patch_shard_gpu_copy` | 6.19 |
| `patch_kproj_vproj` | 7.01 |
| `patch_rotary` | 8.84 |

`reuse-only` sub-breakdown:

| sub-stage | ms |
| --- | ---: |
| `rebuild_prep` | 0.12 |
| `rebuild_decode_core` | 11.27 |
| `rebuild_rerope` | 1.92 |
| `rebuild_materialize` | 0.83 |

## Reading

- `reuse-only` remains broken on `train50`. It keeps the TTFT gain, but quality collapses.
- `patch-only` still works on `train50`, but the result is materially weaker than the old full-dev number. This is consistent with the fact that the current patch checkpoint was not retrained on `train`; it was reused from the earlier `dev-50` training run.
- On this split, `patch-only` is the only healthy transfer route among the three canonical methods.

## Artifact Sources

- `baseline + reuse-only` merged artifact:
  `/Users/weimingming/dr-claw/scd-new/Experiment/analysis/remote_artifacts/strict_baseline_reuse_only_train50_20260405_parallel`
- `patch-only` artifact:
  `/Users/weimingming/dr-claw/scd-new/Experiment/analysis/remote_artifacts/mistrallite_to_mistral7_patchonly_train50_20260405`
