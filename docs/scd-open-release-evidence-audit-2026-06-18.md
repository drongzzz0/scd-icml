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
| `patch26_reuse6` | Quality-first vLLM-validated route | Better CMRC2018 F1/EM and better WikiText-2 PPL than `patch20_reuse12`; raw-bank-to-vLLM smoke also passes with greedy continuation equality, but it uses fewer Reuse layers. |

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
- Artifact:
  `Experiment/results/a800_scd_vllm_raw_bank_patch26_run_20260618_044521/`.
- Route: `patch26_reuse6`.
- Runtime: A800mql, `conda activate cache`, vLLM `0.18.0`.
- Result:
  `doctor_pass=True`, `plugin_pass=True`, `adapter_smoke_pass=True`,
  `scd_injection_pass=True`, `bundle_token_match=True`,
  `same_token_ids=True`, `greedy_1tok_match=True`,
  `greedy_ntok_match=True`.

These runs are enough to advertise a plug-and-play experimental vLLM
integration for the checked Mistral routes. They are not enough to claim
production-grade
multi-tenant safety, privacy isolation, dynamic routing, or general vLLM
throughput improvement under concurrent serving load.

Additional model-family vLLM baseline evidence:

- Artifact:
  `Experiment/results/a800_qwen17_vllm_baseline_20260618_0530/`.
- Model: Qwen3-1.7B self -> self.
- Runtime: A800mql, `conda activate cache`, vLLM `0.18.0`,
  `CUDA_VISIBLE_DEVICES=0`.
- GPU selection: GPUs `0,1,2,5` were sampled for 60 seconds and all stayed at
  utilization `0` before launch.
- Result: KV compatibility `True`; plain vLLM baseline passed with prompt
  tokens `11`, mean 16-token generation time `0.207544s`,
  mean throughput `77.092046` tokens/s, and 1-token proxy time `0.013509s`.

This Qwen run is plain-vLLM compatibility evidence for a second model family.
It is not a Qwen SCD Reuse/Patch route-quality or speedup claim.

## Release Risk Notes

- Do not present Qwen3-1.7B gray as a speedup result. It is additional-model
  portability/risk evidence only.
- Do not present the Qwen3-1.7B plain-vLLM baseline as an SCD result. It only
  proves the current A800 vLLM environment can load and generate with that
  model family.
- Do not present one universal route. Route quality is dataset- and metric-
  dependent.
- Do not use F1/EM as the WikiText-2 acceptance metric.
- Do not claim privacy or multi-tenant safety for reused KV caches without a
  separate isolation design and evaluation.
- Use `patch20_reuse12` for the higher-Reuse/speed demo and `patch26_reuse6`
  for quality-first CMRC2018 and WikiText-2 demonstrations.
