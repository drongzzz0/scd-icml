# A1 Method Freeze

Date: 2026-06-29

This document freezes the current working SCD configuration as method `A1`.

## Definition

`A1` is the current Qwen32 CMRC2018 strict-runtime SCD profile:

- Model pair: `Qwen2.5-32B Base -> Qwen2.5-32B Instruct`.
- Dataset split: train on CMRC2018 train samples; evaluate on held-out CMRC2018 dev.
- Route: k8 mixed SCD route.
  - Reuse layers: `0-42, 47-59`.
  - Patch layers: `43-46, 60-63`.
- Runtime policy: no fallback and no recompute for claim-facing SCD evaluation.
- Decode calibration: `runtime_eval.generation_stop.stop_token_logit_bias=3.0`.
- Bias start step: `runtime_eval.generation_stop.stop_token_bias_start_step=2`.

Bias `4.0` is retained only as a latency-leaning candidate. Bias `5.0` is not
the default because it crossed the preset F1 gate against no-bias SCD.

## Current Evidence

CMRC2018 full-dev evaluation, 3219 samples:

| Method | F1 | EM | Avg gen tokens | Max-new rate | TTFT ms | Total ms |
|---|---:|---:|---:|---:|---:|---:|
| Full consumer | 0.716842 | 0.579062 | 11.702 | 0.189500 | 231.124 | 830.057 |
| SCD k8 no bias | 0.701681 | 0.545200 | 13.299 | 0.232992 | 70.619 | 726.261 |
| A1: SCD k8 bias 3.0 start2 | 0.702013 | 0.549860 | 12.618 | 0.223361 | 70.457 | 712.605 |

Against full consumer, A1 preserves the main TTFT speedup while keeping the
quality gap bounded for the current CMRC2018 validation setting. Against no-bias
SCD, A1 improves F1 and EM while reducing generated tokens and total latency.

## Scope Boundary

The A1 numbers above are strict-runtime results. The repository contains the
`scd_vllm` and `scd_vllm_plugin` prototype, but this stop-token logit-bias
profile has not yet been validated as a vLLM serving result.

The A1 branch should therefore be cited as the current strict-runtime release
candidate, not as a completed vLLM integration claim.
