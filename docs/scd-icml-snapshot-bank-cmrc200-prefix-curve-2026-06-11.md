# SCD-ICML Snapshot-bank CMRC200 Prefix Curve - 2026-06-11

## Purpose

Find the maximum number of Reuse layers that preserves CMRC2018 F1/EM when
starting from the recovered GitHub/new-main all-Patch anchor and replacing
prefix-selected Patch layers with Reuse.

This run uses the current `runtime_repo` strict runtime, but the dataset-specific
Reuse/Patch banks come from the `scd-github-f7d7854` snapshot.

## Run

- Run root:
  `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_snapshot_bank_prefix_curve_cmrc200_20260611_001`
- Wrapper:
  `scripts/run_ser_scd_icml_conditional_route_screen.sh`
- Work queue:
  `GPU_PAIRS=0,1;2,3`
- Dataset:
  `cmrc2018`
- Evaluation:
  `EVAL_LIMIT=200`, `MAX_LENGTH=1024`, `MAX_NEW_TOKENS=32`
- Base route:
  `configs/route_reuse_patchonly_all32_20260404.json`
- Reuse bank:
  `/disk_n/c101-3-mql/projects/SCD-ICML/kv/snapshots/scd-github-f7d7854/runs/mistral_strict_default_gpu1_200_b2_20260421/ckpt_reuse/cmrc2018_seed42`
- Patch bank:
  `/disk_n/c101-3-mql/projects/SCD-ICML/kv/snapshots/scd-github-f7d7854/runs/mistral_strict_default_gpu1_200_b2_20260421/ckpt_patch/patch_cmrc2018_seed42`

Candidate layer order:

```text
3,5,6,8,9,11,13,14,17,20,21,22,23,24,25,27,28,29,30,26,10,4,7,12,15,16,18,19,31,0,1,2
```

## Results

Full Consumer reference is stable across rows: F1 `0.657307`, EM `0.365000`,
avg NLL `0.569284`, TTFT about `100 ms`.

| route | reuse layers | patch layers | F1 | EM | F1 retention | avg NLL | PPL | TTFT ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `reuse08_patch24` | 8 | 24 | 0.668018 | 0.360000 | 1.0163 | 0.677726 | 1.969394 | 43.052900 |
| `reuse12_patch20` | 12 | 20 | 0.637175 | 0.355000 | 0.9694 | 0.759787 | 2.137821 | 42.009192 |
| `reuse16_patch16` | 16 | 16 | 0.697745 | 0.425000 | 1.0615 | 0.777772 | 2.176618 | 43.041855 |
| `reuse19_patch13` | 19 | 13 | 0.636821 | 0.340000 | 0.9688 | 0.901801 | 2.464036 | 37.930758 |
| `reuse20_patch12` | 20 | 12 | 0.623676 | 0.360000 | 0.9488 | 0.991149 | 2.694328 | 37.312358 |
| `reuse21_patch11` | 21 | 11 | 0.571623 | 0.335000 | 0.8696 | 1.268747 | 3.556393 | 40.572355 |
| `reuse22_patch10` | 22 | 10 | 0.504905 | 0.245000 | 0.7681 | 1.929602 | 6.886768 | 36.732076 |
| `reuse23_patch09` | 23 | 9 | 0.485349 | 0.250000 | 0.7384 | 2.053342 | 7.793904 | 39.598155 |
| `reuse24_patch08` | 24 | 8 | 0.387813 | 0.185000 | 0.5900 | 2.788695 | 16.259786 | 36.317053 |

## Interpretation

- The work-queue scheduler behaved correctly: workers on GPU pairs `0,1` and
  `2,3` dequeued routes independently as soon as their current route completed.
- The quality cliff begins after `reuse20_patch12`.
- If the CMRC F1 gate is at least `0.95` retention, `reuse19_patch13` is the
  conservative maximum and `reuse20_patch12` is just below the gate.
- If a small additional F1 loss is acceptable, `reuse20_patch12` gives one more
  Reuse layer while preserving EM almost exactly (`0.360` vs Full `0.365`).
- `reuse21_patch11` and above should be rejected for CMRC2018 because F1 drops
  too far.

## Next

Validate the candidate boundary routes (`reuse19_patch13` and `reuse20_patch12`)
on HotpotQA-compressed and WikiText-2 using dataset-specific snapshot banks.
The final paper-facing route should not be selected from CMRC alone.
