# MistralLite -> Mistral-7B True Single-Layer Search Report (2026-04-06)

## Scope

- Host: `ser`
- Run root: `/disk_n/c101-3-1/projects/scd-repro-codex-mistral-instruct-20260323/runs/mistrallite_to_mistral7_cmrc_blockwise_serrerun_20260406`
- Search setup:
  - `block_size=1`
  - `replay_top_k=3`
  - replay selector: `quality_threshold_ttft_first`
  - replay quality gate: relative `F1 drop <= 5%` versus `patch-only`
  - `EM` gate disabled
  - `fallback=[]`
  - `recompute=[]`

## Execution Note

The first full rerun on `ser` completed the entire step-1 single-layer scan and the replay-top3 stage, but it was launched with `--max-accepted-blocks 0`, so after the single-layer decision it incorrectly continued into step 2 of multi-layer greedy search.

That run is still valid for the question we actually care about, because:

1. the full step-1 scan over all 32 single-layer candidates was completed on `ser`
2. the replay-top3 stage was also completed on `ser`
3. the selected single-layer winner was already determined before step 2 began

So the final single-layer report below uses:

- the completed full-rerun step-1 and replay artifacts from `blockwise_search_serrerun_20260406_b1_replayk3_f1pct5`
- plus fresh strict finalization runs for:
  - `baseline + reuse-only`
  - `hybrid` with the selected single-layer route

This avoids repeating the identical 32-candidate scan while keeping the result tied to the true full rerun rather than the earlier lightweight replay-only approximation.

## Final Single-Layer Route

- Selected reuse layer: `14`
- Selected patch layers: all layers except `14`
- Fallback layers: `[]`
- Recompute layers: `[]`
- Decision source: `replay_quality_threshold_ttft_first`
- First-pass eligible count: `16`
- Replay quality gate:
  - allowed `F1 drop <= 0.03142694` (`patch-only F1 * 5%`)
  - `EM` not gated
  - quality-passing replay blocks:
    - `block_14_0014_0014`
    - `block_12_0012_0012`
    - `block_04_0004_0004`

The winner is `layer 14` because all three replayed candidates passed the F1 gate, and `layer 14` had the smallest replay TTFT.

## First-Pass Top-3

| rank | layer | first-pass F1 | first-pass EM | first-pass TTFT ms |
| --- | ---: | ---: | ---: | ---: |
| 1 | 12 | 0.6167 | 0.290 | 50.12 |
| 2 | 14 | 0.6216 | 0.300 | 50.24 |
| 3 | 4 | 0.6310 | 0.280 | 50.37 |

This is the main difference from the earlier lightweight replay-only approximation: the true full rerun did not rank `layer 10` in the top-3 at step 1.

## Replay-Top3 Results

| layer | replay F1 | replay EM | replay TTFT ms | F1 drop vs patch-only | EM drop vs patch-only |
| --- | ---: | ---: | ---: | ---: | ---: |
| 14 | 0.6216 | 0.300 | 51.45 | 0.0070 | -0.015 |
| 12 | 0.6167 | 0.290 | 51.64 | 0.0118 | -0.005 |
| 4 | 0.6310 | 0.280 | 54.90 | -0.0025 | 0.005 |

Interpretation:

- `layer 4` has the best replay F1, but it is slower than `patch-only`
- `layer 12` and `layer 14` both pass the `F1 <= 5% drop` gate and are faster than `patch-only`
- `layer 14` is the fastest point inside the quality-passing set

## Final Strict Comparison

These are the final numbers to use going forward for this true single-layer rerun.

| method | F1 | EM | TTFT ms | encode | transfer | rebuild | patch_infer | decode_first |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 0.5969 | 0.250 | 103.78 | 0.00 | 0.00 | 0.00 | 0.00 | 22.33 |
| reuse-only | 0.0141 | 0.000 | 33.41 | 0.70 | 8.14 | 2.91 | 0.00 | 21.66 |
| patch-only | 0.6285 | 0.285 | 54.01 | 0.81 | 12.85 | 0.00 | 18.51 | 21.83 |
| hybrid (reuse layer 14) | 0.6216 | 0.300 | 51.60 | 0.89 | 9.48 | 0.51 | 18.18 | 22.53 |

## Takeaways

- The true full rerun no longer supports the earlier lightweight conclusion that `layer 10` is the default single-layer winner.
- Under the true full-rerun ranking plus replay gate, the single-layer winner is `layer 14`.
- Relative to `patch-only`, the final single-layer hybrid gives:
  - `TTFT -2.41 ms`
  - `F1 -0.00698`
  - `EM +0.015`
- So this route is a real latency-quality trade-off point, but it is not a clean Pareto win over `patch-only`:
  - faster TTFT
  - slightly lower F1
  - slightly higher EM

## Local Artifact Mirror

- Raw synced artifacts:
  - `/Users/weimingming/dr-claw/scd-new/Experiment/analysis/remote_artifacts/mistrallite_to_mistral7_singlelayer_true_rerun_20260406`
- The key subdirectories are:
  - `blockwise_search_serrerun_20260406_b1_replayk3_f1pct5`
  - `blockwise_search_serrerun_20260406_b1_replayk3_f1pct5_singlelayer_manual`
  - `strict_baseline_reuse_only_serrerun_20260406_b1_replayk3_f1pct5_singlelayer_manual`
  - `strict_hybrid_serrerun_20260406_b1_replayk3_f1pct5_singlelayer_manual`
