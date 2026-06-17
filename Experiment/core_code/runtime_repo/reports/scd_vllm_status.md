# SCD vLLM Integration Status

This report records the current release-facing vLLM evidence for the
MistralLite producer to Mistral-7B consumer path. The integration is an
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
