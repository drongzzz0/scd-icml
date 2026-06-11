# MistralLite Patch-Only Breakdown

Date: `2026-04-05`

## Scope

This note extends the earlier Mistral repeat-stability diagnosis by adding sample-level patch breakdown logging for:

- `patch_shard_load`
- `patch_shard_gpu_copy`
- `patch_kproj_vproj`
- `patch_knorm`
- `patch_rotary`

Instrumentation change:

- `/Users/weimingming/dr-claw/scd-new/Experiment/core_code/ser_mirror/src/scd_repro/cli/eval_cmrc_strict.py`
- `/Users/weimingming/dr-claw/scd-new/Experiment/core_code/ser_mirror/src/scd_repro/cli_eval_cmrc_strict.py`

The CLI now also supports `qid` subsetting through `--qids` / `--qid-file`.

## Run 1: Hot-qid subset only

Targeted qids:

- `DEV_0_QUERY_0`
- `DEV_2_QUERY_2`
- `DEV_7_QUERY_2`
- `DEV_7_QUERY_1`
- `DEV_11_QUERY_2`
- `DEV_1_QUERY_3`
- `DEV_13_QUERY_1`

Artifacts:

- `/Users/weimingming/dr-claw/scd-new/Experiment/analysis/remote_artifacts/mistrallite_patch_hotqids_20260405`

Observed result:

- only `DEV_0_QUERY_0` remained hot
- the other previously recurring hot qids all fell back to normal `patch_infer` values around `11-14 ms`

`DEV_0_QUERY_0` breakdown:

- `patch_infer = 351.26 ms`
- `patch_shard_load = 282.24 ms`
- `patch_shard_gpu_copy = 58.05 ms`
- `patch_kproj_vproj = 5.08 ms`
- `patch_rotary = 5.89 ms`

Interpretation:

- the targeted qids do not remain intrinsically slow when isolated
- the hot-qid-only run mainly exposes first-sample shard load / copy warm-start cost

## Run 2: Full 50-sample patch-only-only rerun

Artifact:

- `/Users/weimingming/dr-claw/scd-new/Experiment/analysis/remote_artifacts/mistrallite_patchonly_full50_breakdown_20260405`

This rerun keeps the same MistralLite model pair and patch checkpoint, but runs only `patch-only` rather than `baseline + reuse-only + patch-only` in one process.

Per-mode result:

| method | F1 | TTFT ms | patch_infer ms | patch_shard_load ms | patch_shard_gpu_copy ms | patch_kproj_vproj ms | patch_rotary ms | decode_first ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `patch-only` | 0.7829 | 39.66 | 17.47 | 4.89 | 1.11 | 6.15 | 5.33 | 22.20 |

Sample-level result:

- only one sample had `patch_infer > 100 ms`
- that sample was again `DEV_0_QUERY_0`

`DEV_0_QUERY_0` breakdown:

- `patch_infer = 311.07 ms`
- `patch_shard_load = 244.31 ms`
- `patch_shard_gpu_copy = 55.48 ms`
- `patch_kproj_vproj = 5.24 ms`
- `patch_rotary = 6.04 ms`

All other samples stayed in the normal regime:

- typical `patch_kproj_vproj`: about `8-9 ms`
- typical `patch_rotary`: about `4.8-5.3 ms`
- no sample showed the earlier `120-300 ms` non-warmup patch spikes

## Run 3: Full 50-sample shared-process rerun with all three methods

Artifact:

- `/Users/weimingming/dr-claw/scd-new/Experiment/analysis/remote_artifacts/mistrallite_strict_baseline_reuse_patchonly_20260405_breakdown`

This rerun uses the same model pair, same route, same checkpoints, and the same sample-level logging, but restores the original three-method structure:

- `baseline`
- `reuse-only`
- `patch-only`

Per-mode result for `patch-only`:

| method | F1 | TTFT ms | patch_infer ms | patch_shard_load ms | patch_shard_gpu_copy ms | patch_kproj_vproj ms | patch_rotary ms | decode_first ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `patch-only` | 0.7829 | 69.35 | 46.59 | 4.91 | 4.78 | 31.56 | 5.34 | 22.75 |

Compared with the isolated `patch-only-only` rerun:

| metric | isolated patch-only | shared-process patch-only | delta |
| --- | ---: | ---: | ---: |
| `TTFT ms` | 39.66 | 69.35 | +29.69 |
| `patch_infer ms` | 17.47 | 46.59 | +29.13 |
| `patch_shard_load ms` | 4.89 | 4.91 | +0.03 |
| `patch_shard_gpu_copy ms` | 1.11 | 4.78 | +3.67 |
| `patch_kproj_vproj ms` | 6.15 | 31.56 | +25.41 |
| `patch_rotary ms` | 5.33 | 5.34 | +0.02 |
| `decode_first ms` | 22.20 | 22.75 | +0.56 |

Recurring hot samples in the shared-process rerun (`patch_infer > 100 ms`):

- `DEV_0_QUERY_0`
- `DEV_1_QUERY_3`
- `DEV_2_QUERY_2`
- `DEV_7_QUERY_1`
- `DEV_7_QUERY_2`
- `DEV_11_QUERY_2`
- `DEV_13_QUERY_0`

Representative breakdowns:

- `DEV_0_QUERY_0`
  - `patch_infer = 495.23 ms`
  - `patch_shard_load = 245.74 ms`
  - `patch_shard_gpu_copy = 238.85 ms`
  - `patch_kproj_vproj = 5.12 ms`
  - this is still the first-sample shard warm path
- `DEV_13_QUERY_0`
  - `patch_infer = 309.08 ms`
  - `patch_kproj_vproj = 301.28 ms`
  - `patch_rotary = 7.80 ms`
- `DEV_7_QUERY_2`
  - `patch_infer = 270.91 ms`
  - `patch_kproj_vproj = 263.40 ms`
  - `patch_rotary = 7.51 ms`
- `DEV_2_QUERY_2`
  - `patch_infer = 200.38 ms`
  - `patch_kproj_vproj = 194.62 ms`
  - `patch_rotary = 5.75 ms`

So once the three methods share a process again, the old heavy tail reappears, and for the non-warmup hot samples it is almost entirely a `patch_kproj_vproj` explosion rather than a rotary or shard-load issue.

## Main Finding

The earlier `MistralLite patch-only` heavy tail does **not** reproduce when `patch-only` is evaluated alone, but it **does** return when the original three-method shared-process evaluation structure is restored.

The isolated full-50 rerun gives:

- `F1 = 0.7829`
- `TTFT = 39.66 ms`

The shared-process rerun gives:

- `F1 = 0.7829`
- `TTFT = 69.35 ms`

This is much closer to the earlier three-method repeated runs and confirms that the shared-process setting is the trigger condition.

So the current best interpretation is:

- the earlier heavy tail is created or amplified by the shared-process multi-method evaluation setup
- within that setup, the concrete runtime bucket that blows up is `patch_kproj_vproj`
- the issue is therefore not a generic `patch-only` prompt pathology, and not mainly a rotary bottleneck either
- likely candidates are allocator pressure, GEMM workspace/state interaction, or another same-process effect that specifically hurts the consumer-native projection stage

## Updated Diagnosis

What still holds:

- first-sample warm-start cost is real and is dominated by `patch_shard_load + patch_shard_gpu_copy`

What no longer holds as the leading explanation:

- that the previously recurring MistralLite hot qids are inherently slow under isolated `patch-only`

## Run 4: End-to-End Isolated Wrapper Validation

Artifact:

- `/Users/weimingming/dr-claw/scd-new/Experiment/analysis/remote_artifacts/mistrallite_self_strict_baseline_reuse_patchonly_all32_50_20260405_isolated`

The wrapper was then changed to make isolated-process `patch-only` the default benchmarking path:

- phase 1: run `full_consumer + reuse_only`
- phase 2: run `patch_only` alone in a fresh process
- phase 3: merge the two outputs back into the canonical comparison directory

The merged artifact confirms that the default path now preserves the isolated `patch-only` result instead of the contaminated shared-process value:

| method | F1 | TTFT ms |
| --- | ---: | ---: |
| `full_consumer` | 0.7829 | 122.95 |
| `reuse_only` | 0.7686 | 48.70 |
| `patch_only` | 0.7829 | 39.88 |

Merged metadata:

- `meta.isolated_patch_only = true`
- `meta.methods = [full_consumer, reuse_only, patch_only]`

Merged `patch-only` breakdown:

- `patch_infer = 17.33 ms`
- `patch_shard_load = 4.90 ms`
- `patch_shard_gpu_copy = 1.11 ms`
- `patch_kproj_vproj = 6.06 ms`
- `patch_rotary = 5.26 ms`
- `decode_first = 22.55 ms`

Sample-level stability from the merged isolated artifact:

- only the first sample still shows warm-start inflation
- `patch_kproj_vproj` stays bounded, with `mean 6.06 ms` and `max 11.17 ms`
- no non-warmup sample reproduces the earlier shared-process `patch_kproj_vproj` explosion

## Decision

For MistralLite self-self patch benchmarking, the fair default is now:

- `baseline`: same run as `reuse_only`
- `reuse_only`: same run as `baseline`
- `patch_only`: isolated fresh process, then merged back into the final table

The old three-method shared-process `patch_only` number remains useful only as a failure-analysis artifact for allocator / projection-stage interference. It should not be used as the headline latency result.

## Next Step

For the current locked-route benchmark, no more reruns are needed. The trigger condition is already clear, and the default fair-measurement path is in place.

If this interference path is revisited later as a systems optimization problem, the direct target should be the shared-process `patch_kproj_vproj` path itself, for example by checking:

- whether separate worker processes remove the regression without changing method semantics
- whether explicit cache clearing or allocator reset between methods suppresses the `patch_kproj_vproj` explosion
- whether `patch-only` should be benchmarked in an isolated process by default for fair serving-time measurement
