# SCD vLLM Open Release Status - 2026-06-17

This note records the current release-facing evidence for the open-source
MistralLite -> Mistral-7B SCD path. It is intentionally narrower than the
camera-ready paper: it tracks what the current branch and current runtime runs
can support.

## Current Scope

- Model path: MistralLite producer -> Mistral-7B consumer.
- Qwen32B is not release-facing in this branch.
- Qwen3-1.7B -> Qwen3-1.7B-gray is tracked only as additional-model risk
  evidence, not as release-facing speedup evidence.
- Routes are dataset- and objective-specific. Do not claim one universal route.
- Strict SCD evidence must use reuse+patch only:
  - `fallback_layers=[]`
  - `selected_recompute_layers=[]`
  - `recompute_segment_length_by_layer={}`
  - reuse and patch layers cover all 32 transformer layers without overlap.
- F1/EM are the primary QA quality metrics. NLL/PPL remain auxiliary risk
  signals.
- Branch-local result artifacts:
  `Experiment/results/scd_vllm_multidataset_full_strict_20260617/`.

## vLLM Integration Evidence

The current vLLM plugin path is implemented as an external SCD KV connector
adapter, not as a forked vLLM codebase. The checked A800mql environment used:

- Python: `/home/c101-800-mql/.conda/envs/cache/bin/python`
- Activation: `source /opt/miniconda3/etc/profile.d/conda.sh && conda activate cache`
- vLLM: `0.18.0`
- PyTorch: `2.10.0+cu128`

The initial passing vLLM route was `patch20_reuse12`:

- Export run:
  `/home/c101-800-mql/nas/mql/SCD-ICML/runtime_repo_open_release_20260617/runs/scd_vllm/reconstructed_mistrallite_to_mistral7_patch20_reuse12_gpu34`
- Injection run:
  `/home/c101-800-mql/nas/mql/SCD-ICML/runtime_repo_open_release_20260617/runs/scd_vllm/injection_reconstructed_mistrallite_to_mistral7_patch20_reuse12_gpu3`
- Adapter smoke: pass.
- SCD injection: pass.
- Reconstructed bundle type: `reuse_patch`.
- `greedy_1tok_match=True`.
- `greedy_ntok_match=True`.
- ExampleConnector payload bytes: `29,362,944`.

The more aggressive `reuse19_patch13` route remains useful as a paper-aligned
maximum-reuse reference, but the reconstructed vLLM greedy check did not match
on the long-prefix test. Treat that as route-quality risk, not adapter failure.

### A800 Raw-Bank Plugin E2E Smoke

Artifacts:

`Experiment/results/a800_scd_vllm_raw_bank_run_20260618_025602/`

This run validates the source-to-plugin path rather than only a prebuilt bundle:

- Raw reuse bank:
  `open_build_native_fixedroute_cmrc200_20260616_133222/layerwise_seed42`.
- Raw patch bank:
  `open_build_native_fixedroute_cmrc200_20260616_133222/native_alllayers_seed42`.
- Route: `reuse19_patch13`, strict reuse+patch only.
- `scd-vllm doctor`: pass, `failure_count=0`, `warning_count=0`.
- Route strictness: patch/reuse `13/19`, fallback `0`, recompute `0`,
  `num_layers=32`.
- Exported bundle: `bundle_type=reuse_patch`, `num_tokens=229`.
- ExampleConnector conversion: `aligned_tokens=224`, payload bytes
  `29,362,944`.
- vLLM plugin smoke: `pass=True`, `adapter_smoke_pass=True`,
  `scd_injection_pass=True`, `reconstructed_scd=True`,
  `loaded_generation_ok=True`, `bundle_token_match=True`,
  `same_token_ids=True`.

The greedy continuation check is still false for this aggressive route, so this
is integration evidence for the raw-bank workflow. Quality claims should still
come from the full dataset F1/EM/PPL runs below.

### A800 Raw-Bank Plugin E2E Smoke - patch20/reuse12

Artifacts:

`Experiment/results/a800_scd_vllm_raw_bank_patch20_run_20260618_032217/`

This run repeats the source-to-plugin validation for the current vLLM demo
route selected from the CMRC200 bank:

- Raw reuse bank:
  `open_build_native_fixedroute_cmrc200_20260616_133222/layerwise_seed42`.
- Raw patch bank:
  `open_build_native_fixedroute_cmrc200_20260616_133222/native_alllayers_seed42`.
- Route:
  `configs/routes/scd_icml_mistrallite_to_mistral7_patch20_reuse12_cmrc200.json`.
- Runtime:
  A800mql, `conda activate cache`, vLLM `0.18.0`,
  `CUDA_VISIBLE_DEVICES=0,1`.
- `scd-vllm doctor`: pass, `failure_count=0`, `warning_count=0`.
- Route strictness: patch/reuse `20/12`, fallback `0`, recompute `0`,
  `num_layers=32`.
- Reconstructed bundle: `bundle_type=reuse_patch`, `num_tokens=229`,
  `num_layers=32`.
- ExampleConnector conversion: `aligned_tokens=224`, payload bytes
  `29,362,944`.
- vLLM plugin smoke: `pass=True`, `adapter_smoke_pass=True`,
  `scd_injection_pass=True`, `reconstructed_scd=True`,
  `loaded_generation_ok=True`, `bundle_token_match=True`,
  `same_token_ids=True`.
- Greedy output equality:
  `greedy_1tok_match=True`, `greedy_ntok_match=True`.

This is the strongest current plug-and-play integration evidence because it
starts from raw trained reuse/patch banks and still reaches the reconstructed
vLLM injection path with deterministic greedy agreement on the checked prompt.

### A800 Raw-Bank Plugin E2E Smoke - patch26/reuse6

Artifacts:

`Experiment/results/a800_scd_vllm_raw_bank_patch26_run_20260618_044521/`

This run repeats the same raw-bank-to-vLLM validation for the current
quality-first route:

- Raw reuse bank:
  `open_build_native_fixedroute_cmrc200_20260616_133222/layerwise_seed42`.
- Raw patch bank:
  `open_build_native_fixedroute_cmrc200_20260616_133222/native_alllayers_seed42`.
- Route:
  `configs/routes/scd_icml_mistrallite_to_mistral7_patch26_reuse6_cmrc200.json`.
- Runtime:
  A800mql, `conda activate cache`, vLLM `0.18.0`,
  `CUDA_VISIBLE_DEVICES=0,1`.
- GPU selection:
  GPUs `0,1` were sampled for 60 seconds before launch and stayed at
  utilization `0`.
- `scd-vllm doctor`: pass, `failure_count=0`, `warning_count=0`.
- Route strictness: patch/reuse `26/6`, fallback `0`, recompute `0`,
  `num_layers=32`.
- Reconstructed bundle: `bundle_type=reuse_patch`, `num_tokens=229`,
  `num_layers=32`.
- ExampleConnector conversion: `aligned_tokens=224`, payload bytes
  `29,362,944`.
- vLLM plugin smoke: `pass=True`, `adapter_smoke_pass=True`,
  `scd_injection_pass=True`, `reconstructed_scd=True`,
  `bundle_token_match=True`, `same_token_ids=True`.
- Greedy output equality:
  `greedy_1tok_match=True`, `greedy_ntok_match=True`.

This closes the previous integration gap for the quality-first route. The
release can now present both `patch20_reuse12` and `patch26_reuse6` as
raw-bank-to-vLLM validated routes, with different quality/speed trade-offs.

### A800 Mistral SCDPreloadedConnector Batch Benchmark

Artifacts:

- Current open-release snapshot:
  - `Experiment/results/a800_mistral_bc03cd4_b256_gpu7_20260618_104914/patch20/`
  - `Experiment/results/a800_mistral_bc03cd4_b256_gpu7_20260618_104914/patch26/`
- Prior implementation snapshot:
- `Experiment/results/a800_mistral_d1ce508_b256_shm_20260618_0938/patch20/`
- `Experiment/results/a800_mistral_d1ce508_b256_shm_20260618_0938/patch26/`

The current `bc03cd4` run validates the public `scd-vllm batch-benchmark`
command on the two main MistralLite -> Mistral-7B routes after the release
path-hygiene fixes. It compares plain vLLM, official vLLM prefix caching, and
the repository-provided external
`SCDPreloadedConnector` through vLLM's `kv_connector_module_path` mechanism.

Runtime boundary:

- Host: A800mql.
- Environment:
  `source /opt/miniconda3/etc/profile.d/conda.sh && conda activate cache`.
- Code snapshot: `runtime_repo_codex_bc03cd4_20260618_1045`.
- vLLM: `0.18.0`.
- Batch shape: `batch_size=256`, `num_batches=5`, expected requests `1280`.
- GPU selection: A800 GPU `7` was sampled for about 70 seconds and stayed at
  utilization `0` with about `437 MiB` used before launch. Both route
  benchmarks used `CUDA_VISIBLE_DEVICES=7`.
- Model-loading optimization: the Mistral-7B model was served from
  `/dev/shm/scd_models/Mistral-7B-v0.1` before launch to avoid repeated slow
  storage reads across the plain, prefix-cache, and SCDPreloaded stages.
- The benchmark code now automatically disables vLLM HMA for
  `SCDPreloadedConnector`, matching the vLLM `0.18.0` connector constraint.

| Route | Plain vLLM tok/s | vLLM prefix-cache tok/s | SCDPreloaded tok/s | SCD/Plain | SCD/Prefix | Hits | Misses | Hit Rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `patch20_reuse12` | `803.700215` | `5516.152993` | `3185.841785` | `3.963968` | `0.577548` | `49120` | `32` | `0.999349` |
| `patch26_reuse6` | `1005.013664` | `5471.840579` | `2864.393751` | `2.850104` | `0.523479` | `49120` | `32` | `0.999349` |

Interpretation:

- Both Mistral routes pass the batch benchmark with
  `kv_connector=SCDPreloadedConnector`,
  `kv_connector_module_path=scd_vllm.preloaded_connector`,
  `load_generation_ok=True`, `prefix_cache_generation_ok=True`,
  `bundle_token_match=True`, and `storage_ready=True`.
- The stats artifact confirms actual preloaded tensor reuse in the worker:
  one connector worker per route, `32` first-touch layer misses, and `49120`
  subsequent cache hits.
- SCDPreloaded is faster than plain vLLM on this repeated-prefix benchmark
  because it loads the already reconstructed prefix KV instead of recomputing
  the full prefix for every request.
- vLLM native prefix caching remains faster for this static-prefix workload.
  Therefore, this is plug-in integration and overhead-boundary evidence, not a
  claim that SCD beats vLLM's built-in prefix-cache serving path.

## Full CMRC2018 Dev Result

Run root:

`/disk_n/c101-3-mql/projects/SCD-ICML/runs/multidataset_patch20_patch26_20260617_1815`

Data:

- CMRC2018 full dev, `3219` samples.
- Reuse/patch banks trained from the CMRC train-200 bank at
  `open_build_native_fixedroute_cmrc200_20260616_133222`.
- Train/eval split is separated: train bank from train prompts, evaluation on
  dev.
- Methods evaluated: Full Consumer and strict `reuse_patch` only.

| Route | Full F1 / EM | SCD F1 / EM | Delta F1 / EM | Full TTFT | SCD TTFT | TTFT Speedup | Total Speedup | Compression |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `patch20_reuse12` | `0.744253 / 0.477788` | `0.791814 / 0.505436` | `+0.047561 / +0.027648` | `106.853 ms` | `45.562 ms` | `2.345x` | `2.189x` | `0.615385` |
| `patch26_reuse6` | `0.744253 / 0.477788` | `0.837965 / 0.585275` | `+0.093713 / +0.107487` | `107.564 ms` | `52.169 ms` | `2.062x` | `2.422x` | `0.551724` |

Interpretation:

- Both strict routes preserve or improve F1/EM on this full CMRC dev run.
- `patch26_reuse6` is the stronger quality route on CMRC and now also passes
  the raw-bank-to-vLLM greedy smoke.
- `patch20_reuse12` remains the better speed/reuse demo because it keeps higher
  Reuse and lower TTFT.

## HotpotQA Full Dev Result

Run root:

`/home/c101-800-mql/nas/mql/SCD-ICML/runs/hotpotqa_patch20_patch26_a800_20260617_1820`

Data:

- HotpotQA full dev file `hotpot_dev_distractor_v1.json`, loaded as `7405`
  samples.
- Reuse/patch banks are still the CMRC train-200 banks. This is intentionally
  a cross-dataset generalization test, not HotpotQA-trained patch/reuse.

Completed results:

- `hotpotqa_patch20_reuse12`: completed on full HotpotQA dev.
- `hotpotqa_patch26_reuse6`: completed on full HotpotQA dev.

| Route | Full F1 / EM | SCD F1 / EM | Delta F1 / EM | Full TTFT | SCD TTFT | TTFT Speedup | Total Speedup | Compression |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `patch20_reuse12` | `0.286109 / 0.208103` | `0.310008 / 0.231330` | `+0.023900 / +0.023228` | `81.379 ms` | `42.646 ms` | `1.908x` | `5.602x` | `0.615385` |
| `patch26_reuse6` | `0.286109 / 0.208103` | `0.292555 / 0.223768` | `+0.006447 / +0.015665` | `89.031 ms` | `60.723 ms` | `1.466x` | `6.109x` | `0.551724` |

Auxiliary risk:

- Full NLL/PPL for both route runs: `1.600161 / 4.953832`.
- `patch20_reuse12` SCD NLL/PPL: `1.854967 / 6.391488`.
- `patch26_reuse6` SCD NLL/PPL: `1.845791 / 6.333107`.
- Both routes improve F1/EM but worsen NLL/PPL. Treat NLL/PPL as risk signals
  for generation stability, not as the primary QA acceptance metric.

Interpretation:

- `patch20_reuse12` is the stronger current HotpotQA route under the CMRC200
  bank: it has higher F1/EM and lower TTFT than `patch26_reuse6`.
- `patch26_reuse6` remains useful for quality-first CMRC and WikiText evidence,
  but it is not the best current HotpotQA route in this cross-dataset test.
- GPU `7` was selected for the parallel patch26 run only after a 60-second idle
  check, and both routes validated strict reuse+patch-only before launch.

## WikiText-2 Full Test Result

Run root:

`/disk_n/c101-3-mql/projects/SCD-ICML/runs/wikitext2_patch20_patch26_full_current_20260618_034812`

Branch-local artifacts:

`Experiment/results/wikitext2_patch20_patch26_full_current_20260618_034812/`

Data:

- WikiText-2 `wiki.test.txt`, loaded as `763` samples.
- Reuse/patch banks are the CMRC train-200 banks. This is a cross-dataset
  language-modeling generalization test, not WikiText-trained reuse/patch.
- Full sample evaluation used `EVAL_LIMIT=0`.
- Methods evaluated: Full Consumer and strict `reuse_patch` only.
- The two routes were launched in parallel on ser GPUs `0,1` and `2,3` after a
  60-second zero-utilization check.

For WikiText-2, NLL/PPL are the primary quality metrics; F1/EM are not the
acceptance metrics for language modeling.

| Route | Full NLL / PPL | SCD NLL / PPL | Delta NLL | Full TTFT | SCD TTFT | TTFT Speedup | Total Speedup | Compression |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `patch20_reuse12` | `1.983874 / 7.270854` | `2.381655 / 10.822796` | `+0.397781` | `62.404 ms` | `35.840 ms` | `1.741x` | `1.072x` | `0.615385` |
| `patch26_reuse6` | `1.983874 / 7.270854` | `2.206730 / 9.085953` | `+0.222856` | `61.302 ms` | `37.343 ms` | `1.642x` | `1.076x` | `0.551724` |

Interpretation:

- Both routes speed up TTFT on full WikiText-2 but degrade PPL versus Full
  Consumer.
- `patch26_reuse6` is the safer quality route for WikiText-2 because it has a
  lower SCD NLL/PPL than `patch20_reuse12`.
- `patch20_reuse12` remains faster and uses more Reuse, so it is still the
  current vLLM plugin demo route rather than the quality-first WikiText route.

## Release Guidance

- Keep `patch20_reuse12` as the higher-Reuse/speed vLLM plugin demo. It has
  full CMRC, full HotpotQA, full WikiText-2, and raw-bank-to-vLLM greedy smoke
  evidence.
- For quality-first CMRC claims, prefer `patch26_reuse6`; it now also has
  raw-bank-to-vLLM greedy smoke evidence.
- For quality-first WikiText-2 claims, prefer `patch26_reuse6`, while clearly
  reporting that both strict routes increase PPL versus Full Consumer.
- For speed/reuse-first CMRC claims, `patch20_reuse12` is a better trade-off.
- Do not claim privacy or multi-tenant safety for reused KV caches. The release
  should frame SCD-vLLM as suitable for trusted or controlled deployments unless
  isolation and policy checks are added.
- The current launch-spec path is fail-closed for this boundary:
  `launch-smoke` rejects specs that do not declare
  `multi_tenant_safe=false`, `dynamic_request_routing=false`, and
  `trusted_bundle_required=true`; release readiness also checks the stored
  Mistral and Qwen launch-spec artifacts.

## Additional Qwen3-1.7B vLLM Baseline Check

Artifacts:

`Experiment/results/a800_qwen17_vllm_baseline_20260618_0530/`

This is a second-model-family vLLM compatibility and plain-generation check,
not an SCD Reuse/Patch result. It used Qwen3-1.7B self -> self as the plain
vLLM baseline before the Qwen gray plugin smoke below.

Runtime boundary:

- Host: A800mql.
- Environment: `conda activate cache`.
- Code snapshot: `runtime_repo_codex_f9f07b5`.
- Model:
  `/home/c101-800-mql/maqianli/2/models/Qwen/Qwen3-1___7B`.
- GPU selection: GPUs `0,1,2,5` were sampled for 60 seconds and all stayed at
  utilization `0`; the run used `CUDA_VISIBLE_DEVICES=0`.
- vLLM: `0.18.0`.

Result:

| Check | Value |
| --- | ---: |
| KV compatibility | `True` |
| Layers / KV heads / head dim | `28 / 8 / 128` |
| Prompt tokens | `11` |
| Mean 16-token generation time | `0.207544s` |
| Mean tokens/s | `77.092046` |
| 1-token proxy time | `0.013509s` |

Interpretation: this closes a small part of the multi-model release risk by
showing that the current A800 vLLM environment can load and generate with a
second model family. It does not support a Qwen SCD speedup or route-quality
claim.

## Additional Qwen3-1.7B Gray SCD-vLLM Plugin Smoke

Artifacts:

- `Experiment/results/a800_qwen17_scd_vllm_doctor_20260618_cache/`
- `Experiment/results/a800_qwen17_scd_vllm_plugin_smoke_20260618_cache/`

This run validates the raw-bank-to-vLLM adapter path for the previously checked
Qwen3-1.7B -> Qwen3-1.7B-gray strict route:

- Host: A800mql.
- Environment: `conda activate cache`.
- Code snapshot: `runtime_repo_codex_f9f07b5`.
- vLLM: `0.18.0`.
- Producer:
  `/home/c101-800-mql/maqianli/2/models/Qwen/Qwen3-1___7B`.
- Consumer:
  `/home/c101-800-mql/maqianli/SCD-ICML/assets/qwen/Qwen3-1.7B-gray`.
- Route: `paper_scd_reuse_patch_k6`, patch/reuse `6/22`.
- Fallback/recompute: `0/0`.
- GPU selection: all A800 GPUs were sampled for 60 seconds and stayed at
  utilization `0`; the run used `CUDA_VISIBLE_DEVICES=0,1`.
- vLLM load stage required `--disable-hybrid-kv-cache-manager` because
  `ExampleConnector` does not support HMA in vLLM `0.18.0`.

Result:

| Check | Value |
| --- | ---: |
| `scd-vllm doctor` | `pass` |
| Plugin pipeline | `pass` |
| Adapter smoke / SCD injection | `true / true` |
| Bundle token match / same token ids | `true / true` |
| Loaded generation OK | `true` |
| Exported layers | `28` |
| Export payload bytes | `2,981,888` |
| ExampleConnector aligned tokens | `16` |
| ExampleConnector payload bytes | `1,837,472` |

Interpretation: this improves the multi-model integration story because Qwen
now has an actual SCD-vLLM raw-bank plugin smoke, not only a plain vLLM
baseline. It still must not be reported as a Qwen speedup result; Qwen quality
and speed remain bounded by the strict runtime table below.

## Additional Qwen3-1.7B Gray Batched SCD-vLLM Benchmark

Artifact:

- `Experiment/results/a800_qwen17_scd_vllm_preloadedstats_20260618_cache/`

This run validates the public `scd-vllm batch-benchmark` command on the same
Qwen3-1.7B -> Qwen3-1.7B-gray bundle as a repeated-prefix trusted
microbenchmark with plain vLLM, vLLM prefix caching, and SCDPreloaded load
stages:

- Host: A800mql.
- Environment: `conda activate cache`.
- Code snapshot: `runtime_repo_codex_stats_20260618_0717`.
- vLLM: `0.18.0`.
- Route: `paper_scd_reuse_patch_k6`, patch/reuse `6/22`.
- Batch shape: `batch_size=128`, `num_batches=3`, expected requests `384`.
- GPU selection: A800 GPUs `3/4/6/7` were busy; GPUs `0/1/2` stayed at `0`
  utilization during a 60-second check, so the run used
  `CUDA_VISIBLE_DEVICES=2`.

Result:

| Check | Value |
| --- | ---: |
| Batch benchmark pass | `true` |
| Connector | `SCDPreloadedConnector` |
| Connector module | `scd_vllm.preloaded_connector` |
| Load generation OK | `true` |
| Bundle token match | `true` |
| Storage ready | `true` |
| Plain vLLM tokens/s | `11069.329059` |
| vLLM prefix-cache tokens/s | `12135.789531` |
| SCDPreloaded load tokens/s | `5930.711074` |
| Prefix-cache/Plain tokens/s ratio | `1.096344` |
| SCDPreloaded/Plain tokens/s ratio | `0.535779` |
| SCDPreloaded/Prefix-cache tokens/s ratio | `0.488696` |
| Cache hits / misses | `14308 / 28` |
| Cache hit rate | `0.998047` |
| Cached files / bytes | `28 / 1835008` |

Interpretation: this closes the missing batched-command evidence gap for the
plugin and validates vLLM's external `kv_connector_module_path` mechanism for
the repository connector. It does not support a Qwen speedup claim.
SCDPreloaded is materially faster than the earlier disk-backed
`ExampleConnector` path, but it remains slower than both plain vLLM and vLLM
prefix caching, so this result should be framed as integration evidence and an
overhead warning.

Current snapshot high-batch rerun:

- Artifact:
  `Experiment/results/a800_qwen17_gray_c925185_b256_gpu6_20260618_110715/`.
- Code snapshot: `c925185` local release branch runtime pushed to
  `runtime_repo_codex_c925185_20260618_1105`.
- Runtime: A800mql, `conda activate cache`, vLLM `0.18.0`,
  `CUDA_VISIBLE_DEVICES=6`.
- GPU selection: GPU `6` stayed at `0` utilization for the 70-second idle
  check but already had `35844 MiB` resident memory, so the run used
  `gpu_memory_utilization=0.45`.
- Batch shape: `batch_size=256`, `num_batches=5`, expected requests `1280`.
- Result:

| Check | Value |
| --- | ---: |
| Batch benchmark pass | `true` |
| Connector | `SCDPreloadedConnector` |
| Connector module | `scd_vllm.preloaded_connector` |
| Plain vLLM tokens/s | `13218.147026` |
| vLLM prefix-cache tokens/s | `16298.884031` |
| SCDPreloaded load tokens/s | `5944.215648` |
| SCDPreloaded/Plain tokens/s ratio | `0.449701` |
| SCDPreloaded/Prefix-cache tokens/s ratio | `0.364701` |
| Cache hits / misses | `42980 / 28` |
| Cache hit rate | `0.999349` |
| Cached files / bytes | `28 / 1835008` |

Interpretation: the current branch still runs the Qwen gray external connector
at larger batch size, but this is negative throughput-boundary evidence. It
confirms portability of the SCDPreloaded path on a second model family and
should not be used as a Qwen acceleration claim.

## Additional Qwen3-1.7B Gray Risk Check

Artifacts:

`Experiment/results/qwen17_gray_full_strict_microsharded_20260617/`

This strict no-fallback/no-recompute route uses Qwen3-1.7B ->
Qwen3-1.7B-gray with patch layers `[0, 22, 24, 25, 26, 27]` and reuse layers
covering the remaining 22 layers. Calibration/training used CMRC2018 train
samples; evaluation used full CMRC2018 dev, HotpotQA dev, and WikiText-2 test.

| Dataset | Samples | Full F1 / EM | SCD F1 / EM | Delta F1 / EM | Total Speedup | TTFT Speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CMRC2018 | 3219 | `0.748358 / 0.440509` | `0.782770 / 0.504815` | `+0.034411 / +0.064306` | `0.950x` | `0.592x` |
| HotpotQA | 7405 | `0.033071 / 0.007292` | `0.038549 / 0.010533` | `+0.005479 / +0.003241` | `0.802x` | `0.268x` |
| WikiText-2 | 763 | `0.245261 / 0.000000` | `0.236718 / 0.000000` | `-0.008543 / +0.000000` | `1.007x` | `1.123x` |

Interpretation: this result supports strict route coverage and split discipline
on an additional model pair, but it does not support a Qwen speedup claim.

## Additional Qwen2.5-32B vLLM Full Consumer Baseline

This is a Qwen2.5-32B-Instruct official-vLLM tensor-parallel Full Consumer
baseline on CMRC2018 dev. It is not an SCD reuse/patch result and must not be
reported as Qwen32B SCD speedup evidence.

Environment:

- Host: A800 current account, `conda activate cache`.
- vLLM: `0.18.0`.
- Model: `/home/c101-800-mql/nas/models/Qwen2.5-32B-Instruct`.
- Storage/cache: `/home/c101-800-mql/nas/SCD-ICML`, not the `/home`
  development disk.
- GPUs: `CUDA_VISIBLE_DEVICES=0,1,2,3`, `tensor_parallel_size=4`.

Two prompt formats were checked:

| Prompt format | Samples | F1 | EM | Notes |
|---|---:|---:|---:|---|
| Plain completion prompt | 3219 | `0.4156507535` | `0.0043491768` | Diagnostic only; generated full-sentence answers and is not a fair Qwen-Instruct baseline. |
| Chat template + short-answer instruction | 3219 | `0.9098067394` | `0.7086051569` | Current Qwen32B Full Consumer baseline for CMRC2018 dev. |

Artifacts:

- Plain prompt:
  `Experiment/results/a800_qwen32_vllm_cmrc_full_tp4_20260618/summary.json`.
- Chat template:
  `Experiment/results/a800_qwen32_vllm_cmrc_full_tp4_chat_20260618/summary.json`.

Interpretation: Qwen2.5-32B-Instruct quality evaluation must use the tokenizer
chat template and task instruction. The current branch can load and evaluate
Qwen32B through official vLLM TP4, but Qwen32B SCD reuse/patch quality remains
unverified in this release branch.
Keep it as cautionary evidence for route portability.
