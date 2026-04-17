# MistralLite -> Mistral-7B Single-Layer Selector-Locked Route (2026-04-06)

This note freezes the current lightweight single-layer hybrid decision without re-running a full new search.

## Locked Selector

- replay source: `blockwise_search_serrerun_20260406_b1_replaytop3_fast`
- selector: `quality_threshold_ttft_first`
- quality gate:
  - relative `F1 drop <= 5%` versus `patch-only`
  - `EM` ignored
- decision rule inside the quality-passing set:
  - choose the smallest `TTFT`

## Locked Route

- selected reuse layer: `10`
- selected patch layers: all layers except `10`
- route file:
  - `/Users/weimingming/dr-claw/scd-new/Experiment/analysis/route_variants/mistrallite_to_mistral7_single_layer10_f1drop5pct_route_20260406.json`

## Verified Metrics

| method | F1 | EM | TTFT ms |
| --- | ---: | ---: | ---: |
| patch-only | 0.6285 | 0.285 | 68.79 |
| locked hybrid (reuse layer 10) | 0.6482 | 0.305 | 66.05 |

## Delta vs Patch-Only

- `TTFT`: `-2.75 ms`
- `F1`: `+0.0197`
- `EM`: `+0.020`

## Why This Route

- `layer 18` is faster but fails the `F1 <= 5% drop` gate.
- `layer 4` passes the `F1` gate, but it is slower than `patch-only`, so it loses to `layer 10`.
- `layer 10` is the best verified point inside the quality-passing set.

