# SCD Open Release Evidence Audit - 2026-06-18

This audit records the current evidence boundary for the open-source SCD
release branch. It is narrower than the camera-ready paper and should be used
as the release-facing claim boundary.

## Release-Facing Scope

- Main model pair: MistralLite producer -> Mistral-7B consumer.
- vLLM integration: external `scd-vllm` adapter and ExampleConnector-compatible
  storage, not a forked vLLM engine.
- Strict route policy:
  - reuse+patch covers all 32 layers.
  - `fallback_layers=[]`.
  - `selected_recompute_layers=[]`.
  - `recompute_segment_length_by_layer={}`.
- Calibration/training bank: CMRC2018 train-200.
- Evaluation: CMRC2018 dev, HotpotQA dev, and WikiText-2 test are separate
  from the CMRC train-200 bank.

## Current Route Roles

| Route | Role | Evidence Boundary |
| --- | --- | --- |
| `reuse19_patch13` | Historical maximum-Reuse reference | Raw-bank plugin path passes, but greedy continuation does not match. Do not use it as the vLLM demo route. |
| `patch20_reuse12` | Current vLLM demo route | Raw-bank-to-vLLM smoke passes with `greedy_1tok_match=True` and `greedy_ntok_match=True`; full CMRC2018, HotpotQA, and WikiText-2 evidence exists. |
| `patch26_reuse6` | Quality-first reference route | Better CMRC2018 F1/EM and better WikiText-2 PPL than `patch20_reuse12`, but lower Reuse and no separate passing raw-bank vLLM greedy smoke has been recorded for this route. |

## Dataset Evidence

| Dataset | Samples | Primary Metric | Current Stronger Route | Release Note |
| --- | ---: | --- | --- | --- |
| CMRC2018 dev | `3219` | F1/EM | `patch26_reuse6` for quality, `patch20_reuse12` for speed/reuse | Both strict routes improve F1/EM versus Full Consumer. |
| HotpotQA dev | `7405` | F1/EM | `patch20_reuse12` | Both strict routes improve F1/EM, but NLL/PPL worsen and must be reported as risk signals. |
| WikiText-2 test | `763` | NLL/PPL | `patch26_reuse6` | Both strict routes improve TTFT but worsen PPL versus Full Consumer. |

## vLLM Evidence

Current passing end-to-end plugin evidence:

- Artifact:
  `Experiment/results/a800_scd_vllm_raw_bank_patch20_run_20260618_032217/`.
- Route: `patch20_reuse12`.
- Runtime: A800mql, `conda activate cache`, vLLM `0.18.0`.
- Result:
  `doctor_pass=True`, `plugin_pass=True`, `adapter_smoke_pass=True`,
  `scd_injection_pass=True`, `bundle_token_match=True`,
  `same_token_ids=True`, `greedy_1tok_match=True`,
  `greedy_ntok_match=True`.

This is enough to advertise a plug-and-play experimental vLLM integration for
the checked Mistral route. It is not enough to claim production-grade
multi-tenant safety, privacy isolation, dynamic routing, or general vLLM
throughput improvement under concurrent serving load.

## Release Risk Notes

- Do not present Qwen3-1.7B gray as a speedup result. It is additional-model
  portability/risk evidence only.
- Do not present one universal route. Route quality is dataset- and metric-
  dependent.
- Do not use F1/EM as the WikiText-2 acceptance metric.
- Do not claim privacy or multi-tenant safety for reused KV caches without a
  separate isolation design and evaluation.
- Keep `patch20_reuse12` as the current vLLM demo route until a higher-Reuse or
  quality-first route passes the same raw-bank-to-vLLM greedy smoke.
