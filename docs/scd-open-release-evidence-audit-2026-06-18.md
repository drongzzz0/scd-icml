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

Mistral repeated-prefix batch benchmark evidence:

- Batch benchmark artifacts:
  `Experiment/results/a800_mistral_patch20_scd_vllm_preloadedstats_20260618_cache/`
  and
  `Experiment/results/a800_mistral_patch26_scd_vllm_preloadedstats_20260618_cache/`.
- Model pair: MistralLite producer -> Mistral-7B consumer.
- Runtime: A800mql,
  `source /opt/miniconda3/etc/profile.d/conda.sh && conda activate cache`,
  vLLM `0.18.0`.
- GPU selection: GPUs `0` and `1` were sampled for 60 seconds and stayed at
  utilization `0`; `patch20_reuse12` used GPU `0` and `patch26_reuse6` used
  GPU `1`. Both cards had resident low-priority python server memory but
  sufficient free memory.
- Batch shape: `batch_size=128`, `num_batches=3`, expected requests `384`.
- Result:
  both routes report `benchmark_pass=True`,
  `load_generation_ok=True`, `bundle_token_match=True`, `storage_ready=True`,
  `prefix_cache_generation_ok=True`; connector `SCDPreloadedConnector` from
  module `scd_vllm.preloaded_connector`.
- `patch20_reuse12` throughput: plain vLLM `969.093202` tokens/s; vLLM
  prefix-cache `4103.775535` tokens/s; SCDPreloaded `2773.109399` tokens/s.
  Connector stats: `cache_hits=16352`, `cache_misses=32`,
  `cache_hit_rate=0.998047`.
- `patch26_reuse6` throughput: plain vLLM `988.035039` tokens/s; vLLM
  prefix-cache `4233.686187` tokens/s; SCDPreloaded `2657.446443` tokens/s.
  Connector stats: `cache_hits=16352`, `cache_misses=32`,
  `cache_hit_rate=0.998047`.

This is the main Mistral batch-command evidence. It supports an external
plug-in connector, worker-side preloaded-cache reuse, and a repeated-prefix
gain over plain vLLM recomputation, but native vLLM prefix caching remains
faster on this static-prefix microbenchmark.

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

Additional model-family SCD-vLLM plugin evidence:

- Doctor artifact:
  `Experiment/results/a800_qwen17_scd_vllm_doctor_20260618_cache/`.
- Plugin smoke artifact:
  `Experiment/results/a800_qwen17_scd_vllm_plugin_smoke_20260618_cache/`.
- Model pair: Qwen3-1.7B producer -> Qwen3-1.7B-gray consumer.
- Runtime: A800mql, `conda activate cache`, vLLM `0.18.0`,
  `CUDA_VISIBLE_DEVICES=0,1`.
- GPU selection: all A800 GPUs were sampled for 60 seconds and stayed at
  utilization `0` before launch.
- Route: `paper_scd_reuse_patch_k6`, patch/reuse `6/22`, fallback/recompute
  `0/0`.
- Result:
  `doctor_pass=True`, `plugin_pass=True`, `adapter_smoke_pass=True`,
  `scd_injection_pass=True`, `bundle_token_match=True`,
  `same_token_ids=True`, `loaded_generation_ok=True`.
- Connector evidence: vLLM reported `External Cache Hit!` and injected
  `16` aligned tokens through `ExampleConnector`.

This improves portability evidence for the plug-and-play adapter, but it is
still not a release-facing Qwen speedup claim. The Qwen strict runtime table
remains cautionary because it does not support Qwen acceleration.

Additional Qwen batched SCD-vLLM evidence:

- Batch benchmark artifact:
  `Experiment/results/a800_qwen17_scd_vllm_preloadedstats_20260618_cache/`.
- Model pair: Qwen3-1.7B producer -> Qwen3-1.7B-gray consumer.
- Runtime: A800mql, `conda activate cache`, vLLM `0.18.0`,
  `CUDA_VISIBLE_DEVICES=2`.
- GPU selection: A800 GPUs `3/4/6/7` were busy; GPUs `0/1/2` stayed at `0`
  utilization during the 60-second idle check and GPU `2` was used.
- Route: `paper_scd_reuse_patch_k6`, patch/reuse `6/22`, fallback/recompute
  `0/0`.
- Batch shape: `batch_size=128`, `num_batches=3`, expected requests `384`.
- Result:
  `benchmark_pass=True`, `load_generation_ok=True`,
  `bundle_token_match=True`, `storage_ready=True`,
  `prefix_cache_generation_ok=True`; connector `SCDPreloadedConnector` from
  module `scd_vllm.preloaded_connector`.
- Throughput: plain vLLM `11069.329059` tokens/s; vLLM prefix-cache baseline
  `12135.789531` tokens/s; SCDPreloaded load `5930.711074` tokens/s.
  Prefix-cache/plain ratio is `1.096344`; SCDPreloaded/plain ratio is
  `0.535779`; SCDPreloaded/prefix-cache ratio is `0.488696`.
- Worker-side connector stats: `cache_hits=14308`, `cache_misses=28`,
  `cache_hit_rate=0.998047`, `cached_files=28`, `cached_bytes=1835008`.

This proves the repeated-prefix `scd-vllm batch-benchmark` command and the
repository's external connector module path work for the Qwen bundle. The
preloaded connector reduces the disk-backed ExampleConnector overhead but is
still slower than both plain vLLM and vLLM prefix caching for this
microbenchmark, so it should be reported as integration evidence and overhead
risk, not a speedup result.

## Release Risk Notes

- Do not present Qwen3-1.7B gray as a speedup result. It is additional-model
  portability/risk evidence only, even though the SCD-vLLM plugin smoke now
  passes.
- Do not present the Qwen3-1.7B plain-vLLM baseline as an SCD result. It only
  proves the current A800 vLLM environment can load and generate with that
  model family.
- Do not present one universal route. Route quality is dataset- and metric-
  dependent.
- Do not use F1/EM as the WikiText-2 acceptance metric.
- Do not claim privacy or multi-tenant safety for reused KV caches without a
  separate isolation design and evaluation.
- Do not present the Qwen batched SCDPreloadedConnector run as a throughput
  improvement; it passed functionally and improves over the disk-backed
  ExampleConnector path, but it is still slower than both plain vLLM and vLLM
  prefix caching in the checked repeated-prefix microbenchmark.
- Do not present the Mistral SCDPreloadedConnector batch benchmark as faster
  than vLLM's native prefix caching. It is faster than plain vLLM in the
  checked repeated-prefix shape, but prefix caching remains the stronger
  built-in static-prefix baseline.
- Use `patch20_reuse12` for the higher-Reuse/speed demo and `patch26_reuse6`
  for quality-first CMRC2018 and WikiText-2 demonstrations.
