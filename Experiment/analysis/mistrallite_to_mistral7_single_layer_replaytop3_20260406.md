# MistralLite -> Mistral-7B Single-Layer Replay-Top3 Check (2026-04-06)

## Setup

- Host: `ser`
- Base run root: `/disk_n/c101-3-1/projects/scd-repro-codex-mistral-instruct-20260323/runs/mistrallite_to_mistral7_cmrc_blockwise_serrerun_20260406`
- Base single-layer search: `blockwise_search_serrerun_20260406_b1`
- Replay-only artifact: `blockwise_search_serrerun_20260406_b1_replaytop3_fast`
- Policy tested here:
  - reuse the existing single-layer first-pass results from `b1`
  - take first-pass top-3 eligible blocks by the current search sort
  - rerun strict replay for those 3 candidates only
  - compare replay metrics against `patch-only`

This is equivalent to replacing the single-pass block decision with a replay-aware decision for the first single-layer step, without re-running all 32 first-pass candidates.

## First-Pass Top-3

The original `b1` single-layer search ranked these three blocks highest before replay:

| block | reused layer | first-pass F1 | first-pass EM | first-pass TTFT ms |
| --- | ---: | ---: | ---: | ---: |
| `block_04_0004_0004` | 4 | 0.6310 | 0.280 | 60.93 |
| `block_18_0018_0018` | 18 | 0.5836 | 0.250 | 61.17 |
| `block_10_0010_0010` | 10 | 0.6482 | 0.305 | 61.62 |

## Strict Replay Results

| method | F1 | EM | TTFT ms | encode | transfer | rebuild | patch_infer | decode_first |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| patch-only | 0.6285 | 0.285 | 68.79 | 4.68 | 14.18 | 0.00 | 26.22 | 23.71 |
| old selected layer 4 | 0.6310 | 0.280 | 68.84 | 3.80 | 14.77 | 0.55 | 26.12 | 23.60 |
| replay layer 18 | 0.5836 | 0.250 | 61.71 | 2.55 | 13.43 | 0.54 | 21.56 | 23.63 |
| replay layer 10 | 0.6482 | 0.305 | 66.05 | 2.86 | 14.99 | 0.54 | 23.70 | 23.96 |

## What Replay Changed

- The old single-layer winner `layer 4` was a search-time optimism artifact for TTFT:
  - first-pass `60.93 ms`
  - strict replay `70.73 ms`
- After replay, if we keep the **old decision rule** that prioritizes TTFT gain first, the selected layer becomes `18`, not `4`.
- But that replay-selected `layer 18` is clearly worse than `patch-only` in quality:
  - `F1 -0.0450`
  - `EM -0.035`
  - `TTFT -7.08 ms`

## Main Conclusion

- `top-k replay` by itself is not enough.
- It fixes the unstable single-pass estimate, but under the current search ranking it still selects the wrong layer.
- The important replay result is actually `layer 10`:
  - `F1 0.6482 > 0.6285` (`patch-only`)
  - `EM 0.305 > 0.285`
  - `TTFT 66.05 ms < 68.79 ms`

So on this Mistral cross-pair, the hybrid route is **not** obviously dead. The current failure is more specifically a **search objective problem**:

- current rule: maximize TTFT gain first
- observed better operating point: `layer 10`, which improves both quality and TTFT over `patch-only`

The next logical change is therefore not another replay pass, but a new decision rule such as:

1. keep only replay candidates that do not underperform `patch-only` on quality
2. among them, maximize TTFT gain

## Selector Update

That selector change has now been implemented in the searcher.

- replay selection mode default: `quality_threshold_ttft_first`
- replay decision gate default:
  - relative `F1 drop <= 5%` versus `patch-only`
  - `EM` gate disabled by default

Under this new default rule, the current replay-top3 artifact selects:

- `layer 10`
- because within the replay-top3 set, both `layer 10` and `layer 4` satisfy the new `F1 <= 5% drop` rule, but `layer 10` has the better TTFT

So for the current verified single-layer evidence, the default replay-aware hybrid choice is now:

- reuse layer `10`
- patch all other layers
