# SCD vLLM Integration Status

This report records the current release-facing vLLM evidence for the
MistralLite producer to Mistral-7B consumer SCD path, plus an additional
Qwen3-1.7B plain-vLLM compatibility baseline. The integration is an
experimental external adapter around vLLM connector storage, not a forked vLLM
engine and not a production multi-tenant serving connector.

## Environment

- Host used for current runtime checks: `A800mql`.
- Activation: `conda activate cache`.
- Python: `/home/c101-800-mql/.conda/envs/cache/bin/python`.
- Python version: `3.12.12`.
- PyTorch: `2.10.0+cu128`.
- Transformers: `5.3.0`.
- vLLM: `0.18.0`.
- CUDA: available, 8 x `NVIDIA A800-SXM4-80GB`.
- Env probe artifact:
  `../../Experiment/results/a800_env_probe_mistrallite_mistral7_20260618/env_probe.json`.

## Model Pair And KVSpec

- Producer: `MistralLite`.
- Consumer: `Mistral-7B-v0.1`.
- Tokenizer for vLLM decode: consumer tokenizer.
- Dtype: `bfloat16`.
- KV compatibility probe: `compatible=True`.
- Shared KV shape:
  - `num_layers=32`
  - `num_attention_heads=32`
  - `num_kv_heads=8`
  - `head_dim=128`
  - `hidden_size=4096`

The producer and consumer have different RoPE theta metadata in the HF config,
but the exported SCD bundles are consumer-shaped and the vLLM decode path uses
the consumer model and consumer tokenizer.

## Workload Boundary

- Single node.
- Tensor parallel size: `1`.
- Static prefix smoke.
- Deterministic decode: `temperature=0`, `top_p=1`.
- Prefix caching disabled for the smoke path.
- Chunked prefill disabled for the smoke path.
- `ExampleConnector`-compatible storage is used as the current vLLM boundary.

## Phase Results

- Phase 0 Repo Audit: PASS.
  - Artifact: `reports/scd_vllm_repo_audit.md`.
- Phase 1 Env Probe: PASS.
  - Artifact:
    `../../Experiment/results/a800_env_probe_mistrallite_mistral7_20260618/env_probe.json`.
  - vLLM `0.18.0`; KV spec compatibility `True`.
- Phase 2 HF/SCD Bundle Export: PASS.
  - Reconstructed `reuse_patch` bundles are exported from raw Reuse/Patch banks
    for both current release routes.
- Phase 3 vLLM Baseline: PASS.
  - Plain vLLM runtime was verified on the same A800 `cache` environment before
    SCD injection smoke runs.
- Phase 4 TrueKV Oracle: PASS.
  - Prior `ExampleConnector` save/load oracle passed deterministic greedy
    output checks for vLLM-native KV. Logits are not exposed by this offline
    vLLM path, so logit drift is reported as `N/A`.
- Phase 5 SCD Injection: PASS.
  - `patch20_reuse12` and `patch26_reuse6` both pass raw-bank-to-vLLM smoke
    from trained Reuse/Patch banks.

## Current vLLM Route Evidence

| Route | Role | Doctor | SCD Injection | Greedy 1tok | Greedy ntok | Aligned Tokens | Payload Bytes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `patch20_reuse12` | higher-Reuse/speed demo | pass | pass | true | true | 224 | 29362944 |
| `patch26_reuse6` | quality-first route | pass | pass | true | true | 224 | 29362944 |

The corresponding raw result keys are `greedy_1tok_match=True` and
`greedy_ntok_match=True` for both routes.

Artifacts:

- `../../Experiment/results/a800_scd_vllm_raw_bank_patch20_run_20260618_032217/`
- `../../Experiment/results/a800_scd_vllm_raw_bank_patch26_run_20260618_044521/`

Both runs used:

- Raw Reuse bank:
  `open_build_native_fixedroute_cmrc200_20260616_133222/layerwise_seed42`.
- Raw Patch bank:
  `open_build_native_fixedroute_cmrc200_20260616_133222/native_alllayers_seed42`.
- Runtime: A800mql, `conda activate cache`, vLLM `0.18.0`.
- `CUDA_VISIBLE_DEVICES=0,1`.
- For `patch26_reuse6`, GPUs `0,1` were sampled for 60 seconds before launch
  and stayed at utilization `0`.

## Mistral Batched SCDPreloadedConnector Evidence

This is the release-facing repeated-prefix trusted microbenchmark for the two
current MistralLite -> Mistral-7B routes. It checks plain vLLM, official vLLM
prefix caching, and the repository-provided external
`SCDPreloadedConnector`. It is not a production serving, multi-tenant,
privacy, or arbitrary-routing claim.

Artifacts:

- `../../Experiment/results/a800_mistral_d1ce508_b256_shm_20260618_0938/patch20/`
- `../../Experiment/results/a800_mistral_d1ce508_b256_shm_20260618_0938/patch26/`

Run boundary:

- Host: A800mql, `conda activate cache`, vLLM `0.18.0`.
- Code snapshot: `runtime_repo_codex_d1ce508_20260618_0915`.
- Batch shape: `batch_size=256`, `num_batches=5`, expected requests `1280`.
- GPU selection: A800 GPUs `0` and `1` were sampled for 60 seconds and
  stayed at utilization `0`; the final runs used `CUDA_VISIBLE_DEVICES=0`
  for `patch20_reuse12` and `CUDA_VISIBLE_DEVICES=1` for `patch26_reuse6`.
- Model-loading optimization: the Mistral-7B model was copied from NAS to
  `/dev/shm/scd_models/Mistral-7B-v0.1` before launch to avoid repeated NAS
  reads across the plain, prefix-cache, and SCDPreloaded stages.
- The benchmark code now automatically disables the vLLM hybrid KV cache
  manager for `SCDPreloadedConnector`, because that connector path does not
  support HMA in vLLM `0.18.0`.

| Route | Batch Pass | Plain vLLM tok/s | vLLM prefix-cache tok/s | SCDPreloaded tok/s | SCD/Plain | SCD/Prefix | Cache Hits | Cache Misses | Hit Rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `patch20_reuse12` | true | `995.460475` | `5236.793824` | `3286.296158` | `3.301282` | `0.627540` | `49120` | `32` | `0.999349` |
| `patch26_reuse6` | true | `1016.189398` | `5522.382560` | `3168.681346` | `3.118200` | `0.573789` | `49120` | `32` | `0.999349` |

Interpretation: for Mistral repeated-prefix batches, SCDPreloaded is faster
than plain vLLM because it injects the cached prefix KV instead of recomputing
the full prefix every request. Native vLLM prefix caching remains faster than
SCDPreloaded for this static-prefix microbenchmark. The connector stats confirm
actual worker-side tensor reuse after the first per-layer miss: each run saw
`32` misses, `49120` hits, and one connector worker. The release should claim
functional plug-in integration, cache-reuse evidence, and measured overhead
boundaries, not a general serving speedup over vLLM's built-in prefix cache.

## Mistral Official vLLM Launch-Smoke Evidence

This checks that the `launch-spec` JSON is directly consumable by official
vLLM through `vllm.LLM(**llm_kwargs)`. It uses the Mistral `patch26_reuse6`
trusted bundle and `SCDPreloadedConnector`. It is a static-prefix integration
smoke test, not a production serving benchmark or dataset-quality result.

Artifact:

- `../../Experiment/results/a800_mistral_patch26_scd_vllm_launchsmoke_20260618_cache/`

Run boundary:

- Host: A800, `conda activate cache`, vLLM `0.18.0`.
- Code snapshot:
  `/home/student_t/maqianli/SCD-ICML/runtime_repo_codex_launchsmoke_20260618_081014`.
- GPU selection: GPUs `3/4/6/7` were busy with unrelated jobs. GPUs `0/1/2`
  were sampled for 60 seconds and stayed at utilization `0`; the final run
  used `CUDA_VISIBLE_DEVICES=2`.
- Shape: `batch_size=8`, `warmup_batches=1`, `max_new_tokens=16`.

Result: `launch_smoke_pass=True`, `generation_ok=True`,
`bundle_token_match=True`, `request_count=8`, `output_tokens=128`,
`tokens_per_second=596.609039`, and
`disable_hybrid_kv_cache_manager=True`. Final connector stats were collected
after the vLLM child process exited: `cache_hits=480`, `cache_misses=32`,
`cache_events=512`, `cache_hit_rate=0.9375`, `cached_files=32`, and
`final_workers=1`.

## Additional Model vLLM Baseline Evidence

This is a plain official-vLLM compatibility and generation baseline for a
second model family. It does not use SCD Reuse/Patch artifacts and must not be
reported as a Qwen SCD speedup result.

| Model | Pair | KV Compatible | vLLM Baseline | Prompt Tokens | Mean Gen Seconds | Tokens/s | 1-token Proxy |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `Qwen3-1.7B` | self -> self | true | pass | 11 | 0.207544 | 77.092046 | 0.013509s |

Artifacts:

- `../../Experiment/results/a800_qwen17_vllm_baseline_20260618_0530/`

Run boundary:

- Host: A800mql, `conda activate cache`, vLLM `0.18.0`.
- Model:
  `/home/c101-800-mql/maqianli/2/models/Qwen/Qwen3-1___7B`.
- `CUDA_VISIBLE_DEVICES=0`.
- GPUs `0,1,2,5` were sampled for 60 seconds before launch and all reported
  utilization `0`.
- The env probe reports Qwen KV compatibility with `num_layers=28`,
  `num_attention_heads=16`, `num_kv_heads=8`, and `head_dim=128`.

## Additional Model SCD-vLLM Smoke Evidence

This is Qwen raw-bank-to-vLLM connector evidence for the previously evaluated
Qwen3-1.7B -> Qwen3-1.7B-gray strict route. It proves that the external
SCD-vLLM adapter can export and load a reconstructed `reuse_patch` bundle for a
second model family. It is not a Qwen speedup claim.

Artifacts:

- Doctor:
  `../../Experiment/results/a800_qwen17_scd_vllm_doctor_20260618_cache/`
- Plugin smoke:
  `../../Experiment/results/a800_qwen17_scd_vllm_plugin_smoke_20260618_cache/`

Run boundary:

- Host: A800mql, `conda activate cache`, vLLM `0.18.0`.
- Code snapshot: `runtime_repo_codex_f9f07b5`.
- Producer:
  `/home/c101-800-mql/maqianli/2/models/Qwen/Qwen3-1___7B`.
- Consumer:
  `/home/c101-800-mql/maqianli/SCD-ICML/assets/qwen/Qwen3-1.7B-gray`.
- Reuse/Patch assets were copied from the ser strict Qwen run to A800 local
  disk under `/home/c101-800-mql/maqianli/SCD-ICML/assets/qwen/`.
- GPU selection: all A800 GPUs were sampled for 60 seconds and stayed at
  utilization `0`; the run used `CUDA_VISIBLE_DEVICES=0,1`.
- The vLLM load stage required `--disable-hybrid-kv-cache-manager` because
  vLLM `ExampleConnector` does not support HMA.

| Check | Value |
| --- | ---: |
| `scd-vllm doctor` | `pass` |
| Route | `paper_scd_reuse_patch_k6` |
| Patch / Reuse layers | `6 / 22` |
| Fallback / Recompute layers | `0 / 0` |
| Exported layers | `28` |
| Export payload bytes | `2,981,888` |
| ExampleConnector aligned tokens | `16` |
| ExampleConnector payload bytes | `1,837,472` |
| Adapter smoke / SCD injection | `true / true` |
| Token hash match / same token ids | `true / true` |

The vLLM log reported `External Cache Hit!` and
`Inject KV cache of 16 tokens to the paged memory` during the load stage.
Greedy equality is not required for reconstructed SCD bundles in this smoke
gate; full-dataset quality for Qwen remains the strict runtime result below.

## Additional Model Batched vLLM Benchmark Evidence

This is a repeated-prefix trusted microbenchmark for the same Qwen3-1.7B ->
Qwen3-1.7B-gray `paper_scd_reuse_patch_k6` bundle. It checks the public
`scd-vllm batch-benchmark` command, a plain vLLM baseline, an official vLLM
prefix-cache baseline, and the external `SCDPreloadedConnector` load path under a
batched request shape. It is not a general serving-speed, multi-tenant,
privacy, or dataset-quality claim.

Artifact:

- `../../Experiment/results/a800_qwen17_scd_vllm_preloadedstats_20260618_cache/`

Run boundary:

- Host: A800mql, `conda activate cache`, vLLM `0.18.0`.
- Code snapshot: `runtime_repo_codex_stats_20260618_0717`.
- GPU selection: A800 GPUs `3/4/6/7` were busy; GPUs `0/1/2` were sampled for
  60 seconds and stayed at utilization `0`, so the run used
  `CUDA_VISIBLE_DEVICES=2`.
- vLLM load stage used `--disable-hybrid-kv-cache-manager` because
  `ExampleConnector` does not support HMA.

| Check | Value |
| --- | ---: |
| Batch benchmark | `pass` |
| Route | `paper_scd_reuse_patch_k6` |
| Patch / Reuse layers | `6 / 22` |
| Connector | `SCDPreloadedConnector` |
| Connector module | `scd_vllm.preloaded_connector` |
| Batch size / batches | `128 / 3` |
| Expected requests | `384` |
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

Interpretation: batched SCD-vLLM loading works and reports positive throughput,
and the external preloaded connector is materially faster than the disk-backed
`ExampleConnector` path measured earlier. It remains slower than both plain
vLLM and vLLM's native prefix-cache baseline for this Qwen repeated-prefix
microbenchmark. This strengthens integration evidence and reduces a known
overhead source; it is not a serving speedup claim.

## Additional Model Official vLLM Launch-Smoke Evidence

This checks the same launch-spec consumption path on the Qwen3-1.7B ->
Qwen3-1.7B-gray route. It proves that the JSON emitted by `scd-vllm
launch-spec` can be consumed by official vLLM for a second model family. This
is still a trusted static-prefix integration smoke test, not a production
serving benchmark or dataset-quality result.

Artifact:

- `../../Experiment/results/a800_qwen17_scd_vllm_launchsmoke_20260618_cache/`

Run boundary:

- Host: A800, `conda activate cache`, vLLM `0.18.0`.
- Code snapshot:
  `/home/student_t/maqianli/SCD-ICML/runtime_repo_codex_launchsmoke_20260618_081014`.
- GPU selection: A800 GPUs `3/4/6/7` were sampled for 60 seconds and stayed
  at utilization `0`; the final run used `CUDA_VISIBLE_DEVICES=7`.
- Shape: `batch_size=128`, `warmup_batches=1`, `max_new_tokens=16`.

Result: `launch_smoke_pass=True`, `generation_ok=True`,
`bundle_token_match=True`, `request_count=128`, `output_tokens=2048`,
`tokens_per_second=6291.929930`, and
`disable_hybrid_kv_cache_manager=True`. Final connector stats were collected
after the vLLM child process exited: `cache_hits=7140`, `cache_misses=28`,
`cache_events=7168`, `cache_hit_rate=0.996094`, `cached_files=28`, and
`final_workers=1`.

## Dataset Quality Evidence

Dataset quality is measured by the strict HF/SCD runtime, not by the vLLM
single-prefix smoke test. The vLLM smoke proves connector loading and
deterministic decode from reconstructed SCD KV; dataset quality claims use
full-dataset strict evaluations with train/eval separation.

| Dataset | Samples | Primary Metric | Stronger Current Route | Release Interpretation |
| --- | ---: | --- | --- | --- |
| CMRC2018 dev | 3219 | F1/EM | `patch26_reuse6` | Both strict routes improve F1/EM versus Full Consumer. |
| HotpotQA dev | 7405 | F1/EM | `patch20_reuse12` | Both strict routes improve F1/EM; NLL/PPL worsens and is a risk signal. |
| WikiText-2 test | 763 | NLL/PPL | `patch26_reuse6` | Both strict routes improve TTFT but worsen PPL versus Full Consumer. |

The calibration/training bank uses CMRC2018 train-200. The listed evaluations
use CMRC2018 dev, HotpotQA dev, and WikiText-2 test, so the evidence is not a
train/eval leakage artifact.

## Gate Results

- Strict route gate: PASS for both `patch20_reuse12` and `patch26_reuse6`.
  Patch plus Reuse covers all 32 layers; fallback and recompute are disabled.
- vLLM adapter gate: PASS for both current routes.
- Token hash gate: PASS for both current routes.
- Deterministic greedy smoke gate: PASS for both current routes.
- Logit drift gate: `N/A`; the checked vLLM offline path does not expose logits.
- Dataset quality gate: dataset-specific.
  Use F1/EM for QA datasets and NLL/PPL for WikiText-2.
- Release readiness now reads the branch-local full-dataset JSON artifacts
  when they are present: Mistral CMRC2018 dev (`3219`), HotpotQA dev (`7405`),
  WikiText-2 test (`763`), and the Qwen3-1.7B gray three-dataset risk
  aggregate. This prevents the publishable branch from relying only on
  prose-level dataset-quality claims.
- Release readiness also gates train/eval split evidence: release routes must
  declare the CMRC2018 train-200 calibration bank, CMRC strict summaries must
  use full dev evaluation with an empty qid allowlist, WikiText-2 must use the
  full test artifact, and Qwen must document CMRC2018 train calibration with
  CMRC2018 dev, HotpotQA dev, and WikiText-2 test evaluation.

## Blockers And Limits

- This is not a production-grade vLLM connector. KV lifecycle, batching,
  concurrency, multi-tenant isolation, and request-level routing are not yet
  implemented.
- Do not claim privacy or multi-tenant safety for reused KV caches.
- The launch path is fail-closed for this boundary: `launch-smoke` rejects a
  launch-spec unless it explicitly declares `multi_tenant_safe=false`,
  `dynamic_request_routing=false`, and `trusted_bundle_required=true`, and
  release readiness checks the stored Mistral and Qwen launch-spec artifacts.
- Do not claim a universal route across all datasets.
- Qwen3-1.7B gray evidence is portability/risk evidence only; it is not a
  speedup claim.
- Qwen3-1.7B plain-vLLM baseline and Qwen raw-bank-to-vLLM smoke evidence only
  prove current A800 vLLM compatibility and adapter portability for the second
  model family.
- Qwen3-1.7B batched benchmark evidence proves the repeated-prefix
  `SCDPreloadedConnector` path runs and reduces the previous disk-read
  overhead, but the measured load path is still slower than both plain vLLM and
  vLLM prefix caching in the checked microbenchmark.
- Model loading from NAS dominates smoke runtime and is not an SCD speed result.

## Reproduction Commands

```bash
bash scripts/scd_vllm/run_00_repo_audit.sh
bash scripts/scd_vllm/run_01_env_probe.sh
bash scripts/scd_vllm/run_02_hf_export.sh
bash scripts/scd_vllm/run_03_vllm_baseline.sh
bash scripts/scd_vllm/run_04_truekv_oracle.sh
bash scripts/scd_vllm/run_05_reconstructed_export.sh
bash scripts/scd_vllm/run_05_scd_injection.sh
bash scripts/scd_vllm/run_06_report.sh
```

For the release-facing route checks, use:

```bash
scd-vllm doctor --reuse-ckpt <reuse-bank> --patch-ckpt <patch-bank> \
  --route configs/routes/scd_icml_mistrallite_to_mistral7_patch20_reuse12_cmrc200.json

scd-vllm doctor --reuse-ckpt <reuse-bank> --patch-ckpt <patch-bank> \
  --route configs/routes/scd_icml_mistrallite_to_mistral7_patch26_reuse6_cmrc200.json

scd-vllm launch-spec \
  --bundle-dir <validated-reuse-patch-bundle> \
  --storage-dir <example-connector-storage> \
  --model <consumer-model> \
  --stats-dir <optional-stats-dir> \
  --out runs/scd_vllm/launch_spec.json

scd-vllm launch-smoke \
  --launch-spec runs/scd_vllm/launch_spec.json \
  --batch-size 8 \
  --max-new-tokens 16 \
  --out runs/scd_vllm/launch_smoke.json
```

`launch-spec` emits JSON containing official `vllm.LLM(**llm_kwargs)` inputs,
the `SCDPreloadedConnector` KV transfer config, the HMA-disable vLLM kwarg
required by vLLM `0.18.0`, and optional connector stats environment variables.
`launch-smoke` consumes the JSON through official vLLM and records a
generation artifact with token-hash matching against the bundle manifest. This
is a single-tenant trusted-bundle contract, not a dynamic request router.

## Next Steps

- Continue replacing disk-backed `ExampleConnector` storage with a dedicated
  SCD connector. The current `SCDPreloadedConnector` removes repeated
  per-request disk reads inside a worker and `launch-spec` makes it easier to
  consume from official vLLM, but it still performs connector-side KV injection
  for every request.
- Add a true serving benchmark only if the release wants throughput claims.
- Keep dataset-quality reporting tied to strict full-dataset evaluations, not
  single-prefix vLLM smoke tests.
