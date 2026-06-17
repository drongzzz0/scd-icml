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

The current passing vLLM route is `patch20_reuse12`:

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
- `patch26_reuse6` is the stronger quality route on CMRC.
- `patch20_reuse12` is the better vLLM demo route today because it passes the
  reconstructed vLLM greedy checks and keeps higher reuse / lower TTFT.

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

## Release Guidance

- Keep `patch20_reuse12` as the current vLLM plugin demo route until a stricter
  or higher-reuse route passes the same reconstructed vLLM greedy checks. It
  now has full CMRC and full HotpotQA F1/EM evidence.
- For quality-first CMRC claims, prefer `patch26_reuse6`.
- For speed/reuse-first CMRC claims, `patch20_reuse12` is a better trade-off.
- Do not claim privacy or multi-tenant safety for reused KV caches. The release
  should frame SCD-vLLM as suitable for trusted or controlled deployments unless
  isolation and policy checks are added.

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
Keep it as cautionary evidence for route portability.
