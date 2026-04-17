# Reviewer Q2 Fresh Rerun: Mistral-7B -> MistralLite

## Deprecation Notice

- This report is retained only as historical debugging evidence.
- It must not be used as the canonical reverse-pair route selection source.
- The final reverse-pair route lock now points instead to:
  - `/Users/weimingming/dr-claw/scd-new/Experiment/analysis/mistral7_to_mistrallite_final_route_lock_20260409.md`
- Reason:
  - this Q2 chain includes fallback-based route structure and therefore does not match the later accepted canonical reverse-pair route contract.

Date: 2026-04-07

## Status

- Fresh run root: `/disk_n/c101-3-1/projects/scd-repro-codex-mistral-instruct-20260323/runs/scd-q2-m7_to_lite-freshfix-20260407-175900`
- Host: `ser`
- Actual GPU assignment:
  - producer: `GPU 0`
  - consumer: `GPU 1`
  - train/profile/patch: `GPU 2`
- Execution chain status: `capture_pair -> train_reuse -> profile -> build_route -> train_patch -> strict_eval` all succeeded.
- Execution code version: `dafeb4bdd1adbbef31a724001b805252228c7933bd99880765b203e674064aa0`
- Runtime repair applied before this rerun:
  - `strict_runtime.py` now only applies `residual_blend` when the target patch layer already has a valid base KV cache entry.
  - If the layer has no base KV entry, strict runtime now writes patch KV directly instead of incorrectly unpacking `None`.
- Reporting note:
  - The run itself finished successfully.
  - Post-run summary aggregation initially failed only because the backbone parameter counter assumed `safetensors`.
  - The summary script was patched afterward to support `pytorch_model-*.bin` checkpoints via `transformers` meta-device instantiation.
  - This reporting-only patch did not change any stage outputs, route, ckpt, or strict-eval metrics.

## Calibration / Data Cost

| item | value |
|---|---:|
| calibration samples | 200 |
| prefix tokens per sample | 256 |
| total calibration prefix tokens | 51,200 |
| pairs dir size | 67,126,066,950 bytes |
| pairs dir size (GiB) | 62.52 |

## Offline Stage Cost

| stage | success | wall-clock sec | wall-clock min | peak GPU MB | max CPU RSS KB | GPU ids |
|---|---|---:|---:|---:|---:|---|
| capture_pair | true | 401.184 | 6.686 | 14997 | 15193308 | 0,1 |
| train_reuse | true | 383.051 | 6.384 | 1179 | 1307176 | 2 |
| profile | true | 828.255 | 13.804 | 7479 | 1457980 | 2 |
| build_route | true | 4.369 | 0.073 |  | 595432 |  |
| train_patch | true | 647.622 | 10.794 | 767 | 43118360 | 2 |
| strict_eval | true | 1074.693 | 17.912 | 14807 | 10780100 | 0,1 |

## Module Overhead

| item | value |
|---|---:|
| reuse ckpt dir size | 8,407,635 bytes |
| reuse ckpt dir size (MiB) | 8.02 |
| patch ckpt dir size | 105,112,020 bytes |
| patch ckpt dir size (MiB) | 100.24 |
| reuse module params | 2,097,152 |
| patch module params | 26,275,200 |
| total extra params | 28,372,352 |
| producer backbone params | 7,241,732,096 |
| consumer backbone params | 7,241,756,672 |
| total extra params / consumer backbone | 0.0039178825 |

## Route / Engineering Configuration

| item | value |
|---|---|
| producer model | `Mistral-7B-v0.1` |
| consumer model | `MistralLite` |
| dataset | `CMRC2018` |
| group_size | 4 |
| rank_k | 128 |
| rank_v | 128 |
| patch_budget | 6 |
| patch_num_epochs | 5 |
| patch_batch_size | 64 |
| patch_mlp_hidden_dim | 512 |
| backbone frozen | true |
| semantic probe shared | true |
| semantic probe frozen | true |
| patch fit mode | `xnorm_semantic_v1` |
| patch apply mode | `residual_blend` |
| state target space | `residual_pre_ln` |
| reuse layers | `[0, 1, 2, 3, 4, 5, 9]` |
| patch layers | `[12, 20, 28, 29, 30, 31]` |
| fallback layers | `[6, 7, 8, 10, 11, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23, 24, 25, 26, 27]` |
| selected recompute layers | `[]` |

## Online Metrics

Definition notes:

- `ttft_ratio_method_over_full = method_ttft_ms / full_consumer_ttft_ms`; smaller is better.
- `ttft_speedup_full_over_method = full_consumer_ttft_ms / method_ttft_ms`; larger than `1` means faster than `full_consumer`.
- `total_latency_ratio_method_over_full = method_total_ms / full_consumer_total_ms`; smaller is better.
- `payload_ratio_vs_raw_cache = bytes_transferred / raw_cache_bytes`; smaller is better.

| mode | F1 | EM | TTFT ms | total ms | payload MB | bytes transferred | compression ratio | encode ms | transfer ms | rebuild ms | patch infer ms | recompute ms | F1 retention vs full | F1 abs drop vs full | TTFT ratio method/full | TTFT delta ms vs full | total ratio method/full | payload ratio vs raw |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| full_consumer | 0.063023 | 0.000000 | 47.841829 | 527.969215 | 0.000000 | 0 |  | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0.000000 | 1.000000 | 0.000000 | 1.000000 | 0.000000 |
| reuse_only | 0.010224 | 0.000000 | 45.018838 | 701.621711 | 7.968750 | 8355840 | 4.000000 | 10.539522 | 2.363992 | 10.593257 | 0.000000 | 0.000000 | 0.162220 | 0.052800 | 0.940993 | -2.822991 | 1.328906 | 0.250000 |
| reuse_fallback | 0.047237 | 0.000000 | 64.484974 | 454.485331 | 2.988281 | 3133440 | 10.666667 | 3.820779 | 0.854313 | 3.812914 | 0.000000 | 35.161869 | 0.749510 | 0.015787 | 1.347879 | 16.643145 | 0.860818 | 0.093750 |
| reuse_patch | 0.025270 | 0.000000 | 68.335126 | 708.906562 | 14.941406 | 15667200 | 2.133333 | 3.821252 | 0.855806 | 3.796006 | 12.247988 | 26.602399 | 0.400960 | 0.037754 | 1.428355 | 20.493296 | 1.342704 | 0.468750 |
| reuse_patch_recompute | 0.030227 | 0.000000 | 74.991940 | 716.718386 | 14.941406 | 15667200 | 2.133333 | 3.817844 | 0.854482 | 3.794786 | 12.143555 | 33.534666 | 0.479610 | 0.032797 | 1.567497 | 27.150111 | 1.357500 | 0.468750 |

## Key Remote Paths

- pairs dir:
  - `/disk_n/c101-3-1/projects/scd-repro-codex-mistral-instruct-20260323/runs/scd-q2-m7_to_lite-freshfix-20260407-175900/pairs/cmrc2018_seed42`
- manifest:
  - `/disk_n/c101-3-1/projects/scd-repro-codex-mistral-instruct-20260323/runs/scd-q2-m7_to_lite-freshfix-20260407-175900/pairs/cmrc2018_seed42/manifest.json`
- measure root:
  - `/disk_n/c101-3-1/projects/scd-repro-codex-mistral-instruct-20260323/runs/scd-q2-m7_to_lite-freshfix-20260407-175900/measure`
- measure/capture_pair:
  - `/disk_n/c101-3-1/projects/scd-repro-codex-mistral-instruct-20260323/runs/scd-q2-m7_to_lite-freshfix-20260407-175900/measure/capture_pair`
- measure/train_reuse:
  - `/disk_n/c101-3-1/projects/scd-repro-codex-mistral-instruct-20260323/runs/scd-q2-m7_to_lite-freshfix-20260407-175900/measure/train_reuse`
- measure/profile:
  - `/disk_n/c101-3-1/projects/scd-repro-codex-mistral-instruct-20260323/runs/scd-q2-m7_to_lite-freshfix-20260407-175900/measure/profile`
- measure/build_route:
  - `/disk_n/c101-3-1/projects/scd-repro-codex-mistral-instruct-20260323/runs/scd-q2-m7_to_lite-freshfix-20260407-175900/measure/build_route`
- measure/train_patch:
  - `/disk_n/c101-3-1/projects/scd-repro-codex-mistral-instruct-20260323/runs/scd-q2-m7_to_lite-freshfix-20260407-175900/measure/train_patch`
- measure/strict_eval:
  - `/disk_n/c101-3-1/projects/scd-repro-codex-mistral-instruct-20260323/runs/scd-q2-m7_to_lite-freshfix-20260407-175900/measure/strict_eval`
- reuse ckpt dir:
  - `/disk_n/c101-3-1/projects/scd-repro-codex-mistral-instruct-20260323/runs/scd-q2-m7_to_lite-freshfix-20260407-175900/ckpt_reuse/xkvgroup_g4_r128_gpu2_seed42`
- patch ckpt dir:
  - `/disk_n/c101-3-1/projects/scd-repro-codex-mistral-instruct-20260323/runs/scd-q2-m7_to_lite-freshfix-20260407-175900/ckpt_patch/semantic_fix7_gpu2_seed42`
- profile output:
  - `/disk_n/c101-3-1/projects/scd-repro-codex-mistral-instruct-20260323/runs/scd-q2-m7_to_lite-freshfix-20260407-175900/profile/profile_cmrc2018_seed42.json`
- route output:
  - `/disk_n/c101-3-1/projects/scd-repro-codex-mistral-instruct-20260323/runs/scd-q2-m7_to_lite-freshfix-20260407-175900/route/route_cmrc2018_seed42_fix7_xkvgroup_g4_r128_gpu2.json`
- strict eval dir:
  - `/disk_n/c101-3-1/projects/scd-repro-codex-mistral-instruct-20260323/runs/scd-q2-m7_to_lite-freshfix-20260407-175900/strict_eval/eval_stepc_fix7_semantic_rllm_gpu01`
- summary.json:
  - `/disk_n/c101-3-1/projects/scd-repro-codex-mistral-instruct-20260323/runs/scd-q2-m7_to_lite-freshfix-20260407-175900/strict_eval/eval_stepc_fix7_semantic_rllm_gpu01/summary.json`
- per_mode_summary.json:
  - `/disk_n/c101-3-1/projects/scd-repro-codex-mistral-instruct-20260323/runs/scd-q2-m7_to_lite-freshfix-20260407-175900/strict_eval/eval_stepc_fix7_semantic_rllm_gpu01/per_mode_summary.json`
- rebuttal summary:
  - `/disk_n/c101-3-1/projects/scd-repro-codex-mistral-instruct-20260323/runs/scd-q2-m7_to_lite-freshfix-20260407-175900/measure/rebuttal_summary.json`
- local mirrored artifact bundle:
  - `/Users/weimingming/dr-claw/scd-new/Experiment/analysis/remote_artifacts/reviewer_q2_m7_to_mistrallite_freshfix_20260407`
