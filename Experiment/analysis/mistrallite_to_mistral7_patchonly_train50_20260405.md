# MistralLite To Mistral-7B Patch-Only Train50 Online Eval

Date: `2026-04-05`

## Setup

- model pair: `MistralLite -> Mistral-7B`
- method: `patch-only`
- route lock: current canonical strict `patch-only`
- dataset split: `CMRC2018 train`
- online evaluated samples: `50`
- sample source: a temporary train-only data root was used so the loader fell through from missing `dev` to `cmrc2018_train.json`

## Result

| method | samples | F1 | EM | TTFT ms |
| --- | ---: | ---: | ---: | ---: |
| `patch-only` | 50 | 0.6676 | 0.2800 | 76.44 |

## TTFT Breakdown

| encode ms | transfer ms | patch_infer ms | decode_first ms |
| ---: | ---: | ---: | ---: |
| 0.98 | 9.68 | 39.45 | 26.34 |

`patch_infer` sub-breakdown:

| shard load ms | shard gpu copy ms | kproj/vproj ms | rotary ms |
| ---: | ---: | ---: | ---: |
| 5.09 | 6.19 | 7.01 | 8.84 |

## Artifact

- local synced artifact:
  `/Users/weimingming/dr-claw/scd-new/Experiment/analysis/remote_artifacts/mistrallite_to_mistral7_patchonly_train50_20260405/per_mode_summary.json`
