# MistralLite -> Mistral-7B Single-Layer Blockwise Search (2026-04-06)

## Setup

- Host: `ser`
- Run root: `/disk_n/c101-3-1/projects/scd-repro-codex-mistral-instruct-20260323/runs/mistrallite_to_mistral7_cmrc_blockwise_serrerun_20260406`
- Search tag: `serrerun_20260406_b1`
- Search policy: `block_size=1`, `max_f1_drop=1.0`, `max_em_drop=0.10`, `min_ttft_gain_ms=1.0`
- Runtime change before this rerun: `reuse_patch` was changed from full-layer reuse decode to route-aware selective reuse decode, so the online path only encodes/transfers/rebuilds the final `reuse_layers`.

## Final Route

- Accepted reuse block ids: `[4]`
- Accepted reuse layers: `[4]`
- Final patch layers: all layers except `4`
- Final fallback layers: `[]`
- Final recompute layers: `[]`

## Final Verified Metrics

These are the final strict eval outputs and should be treated as the canonical numbers.

| Method | F1 | EM | TTFT ms | encode | transfer | rebuild | patch_infer | decode_first |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 0.5969 | 0.250 | 125.87 | 0.00 | 0.00 | 0.00 | 0.00 | 24.16 |
| reuse-only | 0.0141 | 0.000 | 40.69 | 0.70 | 12.69 | 4.31 | 0.00 | 22.99 |
| patch-only | 0.6285 | 0.285 | 68.79 | 4.68 | 14.18 | 0.00 | 26.22 | 23.71 |
| hybrid (reuse layer 4) | 0.6310 | 0.280 | 68.84 | 3.80 | 14.77 | 0.55 | 26.12 | 23.60 |

## Interpretation

- Single-layer search found one reusable layer: layer `4`.
- Search-time candidate metrics for layer `4` were optimistic (`TTFT 60.93 ms`), but the final strict replay for the same selected route landed at `68.84 ms`.
- Therefore the final verified outcome is:
  - quality is slightly higher than patch-only in F1 (`+0.0025`)
  - EM is slightly lower than patch-only (`-0.005`)
  - TTFT is effectively unchanged relative to patch-only (`+0.05 ms`, slightly worse)
- Practical conclusion: single-layer selective reuse is now technically viable, but the current hybrid route still does not beat patch-only under the final unified strict replay.
