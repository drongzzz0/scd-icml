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
`scd-vllm batch-benchmark` command and the vLLM `ExampleConnector` load path
under a batched request shape. It is not a general serving-speed,
multi-tenant, privacy, or dataset-quality claim.

Artifact:

- `../../Experiment/results/a800_qwen17_scd_vllm_batchbench_20260618_cache/`

Run boundary:

- Host: A800mql, `conda activate cache`, vLLM `0.18.0`.
- Code snapshot: `runtime_repo_codex_batchbench_20260618`.
- GPU selection: A800 GPUs `3,4,6,7` were sampled for 60 seconds and stayed
  busy; GPUs `0,1,2,5` stayed at `0` utilization, so the run used
  `CUDA_VISIBLE_DEVICES=0`.
- vLLM load stage used `--disable-hybrid-kv-cache-manager` because
  `ExampleConnector` does not support HMA.

| Check | Value |
| --- | ---: |
| Batch benchmark | `pass` |
| Route | `paper_scd_reuse_patch_k6` |
| Patch / Reuse layers | `6 / 22` |
| Batch size / batches | `128 / 3` |
| Expected requests | `384` |
| Load generation OK | `true` |
| Bundle token match | `true` |
| Storage ready | `true` |
| Load tokens/s | `2000.185735` |
| Baseline tokens/s | `6514.906148` |
| Load/Baseline tokens/s ratio | `0.307017` |

Interpretation: batched SCD-vLLM loading works and reports positive throughput,
but the current disk-backed `ExampleConnector` path is slower than plain vLLM
for this Qwen repeated-prefix microbenchmark. This strengthens integration
evidence, not a speedup claim.

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

## Blockers And Limits

- This is not a production-grade vLLM connector. KV lifecycle, batching,
  concurrency, multi-tenant isolation, and request-level routing are not yet
  implemented.
- Do not claim privacy or multi-tenant safety for reused KV caches.
- Do not claim a universal route across all datasets.
- Qwen3-1.7B gray evidence is portability/risk evidence only; it is not a
  speedup claim.
- Qwen3-1.7B plain-vLLM baseline and Qwen raw-bank-to-vLLM smoke evidence only
  prove current A800 vLLM compatibility and adapter portability for the second
  model family.
- Qwen3-1.7B batched benchmark evidence proves the repeated-prefix
  `ExampleConnector` path runs, but the measured load path is slower than
  plain vLLM in the checked microbenchmark.
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
```

## Next Steps

- Replace disk-backed `ExampleConnector` storage with a dedicated SCD connector
  only after keeping the current smoke path as a regression gate.
- Add a batched vLLM serving benchmark if the release wants throughput claims.
- Keep dataset-quality reporting tied to strict full-dataset evaluations, not
  single-prefix vLLM smoke tests.
