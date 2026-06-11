# Remote `ser` Asset Audit and Existing Qwen Results

## Purpose

This note records what is currently available on remote host `ser` under `/disk_n/c101-3-1`, what can already be reused for SCD validation, and what the existing Qwen-family results already imply for the similar-pair question.

This is not a paper claim. It is an execution-state audit grounded in files already present on `ser`.

## Remote Runtime Status

Remote Python environment:

- activation path: `/disk_n/c101-3-1/venvs/A800/bin/activate`
- Python: `3.12.3`

Observed package status from the active environment:

- `torch = 2.9.0+cu128`
- `transformers = 5.3.0`
- `datasets = 2.21.0`
- `accelerate = 1.11.0`
- `safetensors = 0.6.2`
- `sentencepiece = missing`

Implication:

- the remote environment is usable for Hugging Face style causal-LM capture and evaluation
- unlike the local workspace, `ser` is already capable of running the existing SCD reproduction code

## Reusable Remote Codebase

The most relevant existing remote repo is:

- `/disk_n/c101-3-1/projects/scd-repro-codex-mistral-instruct-20260323`

Why this repo is the current best execution base:

- it already contains `capture_pair`, `train_reuse`, `profile_layers`, `train_patch`, `build_route`, and `eval_cmrc_strict`
- it already has Qwen-specific model configs and completed run directories
- it already stores route/profile artifacts and strict CMRC summaries that can be inspected directly

The simpler upstream-style repo also exists:

- `/disk_n/c101-3-1/projects/scd-repro`

However, the codex branch is currently more useful because it already includes Qwen CMRC runs and stricter evaluation outputs.

## Remote Model Inventory Relevant to This Project

Observed model directories under `/disk_n/c101-3-1/models`:

- `Qwen3-1.7B`
- `Qwen3-1.7B-bf16`
- `Qwen3-1.7B-gray`
- `Qwen3-Doc`
- `Qwen3.5-4B`
- plus non-Qwen families such as Mistral, ToolACE, and EvolCode

### What is immediately pairable

The remote repo already has explicit configs for:

- `Qwen3-1.7B -> Qwen3-1.7B`
- `Qwen3-1.7B -> Qwen3-1.7B-bf16`

These are the currently lowest-friction Qwen-family similar pairs available on `ser`.

### What is not yet available

Only one explicit Qwen3.5 checkpoint was found:

- `/disk_n/c101-3-1/models/Qwen3.5-4B`

No second Qwen3.5-family checkpoint was found under the searched remote paths, so a true Qwen3.5-vs-Qwen3.5 similar-pair experiment is not yet ready from existing assets alone.

## Important Qwen3.5 Caveat

`Qwen3.5-4B/config.json` indicates:

- top-level architecture: `Qwen3_5ForConditionalGeneration`
- top-level model type: `qwen3_5`
- text details live inside `text_config`

This means the currently available Qwen3.5 checkpoint is not a drop-in match for the existing causal-LM code path used in the `qwen17` runs. It likely needs an adapter or text-submodule-aware loading path before reuse with the current SCD reproduction scripts.

## Existing Similar-Qwen Results Already Present on `ser`

### 1. `Qwen3-1.7B -> Qwen3-1.7B-bf16`

Source files:

- route: `/disk_n/c101-3-1/projects/scd-repro-codex-mistral-instruct-20260323/runs/qwen17_to_bf16_cmrc/route_cmrc_seed42.json`
- profile: `/disk_n/c101-3-1/projects/scd-repro-codex-mistral-instruct-20260323/runs/qwen17_to_bf16_cmrc/profile_cmrc_seed42.json`
- strict summary: `/disk_n/c101-3-1/projects/scd-repro-codex-mistral-instruct-20260323/runs/qwen17_to_bf16_cmrc/strict_main/summary.json`

Key summary values on `200` CMRC samples:

| method | F1 | TTFT(ms) | payload(MB) |
| --- | ---: | ---: | ---: |
| `full_consumer` | `0.6945` | `57.57` | `0.00` |
| `reuse_only` | `0.00137` | `51.95` | `10.32` |
| `reuse_fallback` | `0.4088` | `82.41` | `10.32` |
| `reuse_patch` | `0.0258` | `84.58` | `19.17` |
| `reuse_patch_recompute` | `0.0701` | `88.33` | `19.17` |

Interpretation:

- this pair is much more similar than the original heterogeneous Mistral-style pair, but the current SCD-like path still does not recover full-consumer quality
- the best quality among the transfer modes in this run is `reuse_fallback`, not `reuse_patch`
- the `reuse_patch` family is substantially slower than full consumer TTFT in this existing run, not faster

Therefore, the currently stored `qwen17_to_bf16` result does **not** support the paper-style claim that the full SCD path reaches a strong TTFT-quality sweet spot on this similar Qwen-family pair.

### 2. `Qwen3-1.7B -> Qwen3-1.7B` self-pair diagnostics

Source files include:

- `/disk_n/c101-3-1/projects/scd-repro-codex-mistral-instruct-20260323/runs/qwen17_self_cmrc/strict_diag_self_all_reuse_layerwise_50/summary.json`
- `/disk_n/c101-3-1/projects/scd-repro-codex-mistral-instruct-20260323/runs/qwen17_self_cmrc/strict_diag_self_all_reuse_group4_xkvstyle_20/summary.json`

Two useful anchors:

#### Layerwise self-pair reuse

On `50` CMRC samples:

| method | F1 | TTFT(ms) | payload(MB) |
| --- | ---: | ---: | ---: |
| `full_consumer` | `0.7375` | `55.99` | `0.00` |
| `reuse_only` | `0.7372` | `64.76` | `41.32` |

Interpretation:

- on an identical self-pair, the layerwise path can nearly preserve quality
- but it does not deliver a TTFT advantage and has raw-KV-scale payload (`compression_ratio = 1.0`)

This is evidence that similarity alone can help fidelity, but not enough to produce the claimed TTFT-quality operating point.

#### Grouped xKV-style self-pair reuse

On `20` CMRC samples:

| method | F1 | TTFT(ms) | payload(MB) |
| --- | ---: | ---: | ---: |
| `full_consumer` | `0.7375` | `62.24` | `0.00` |
| `reuse_only` | `0.1705` | `46.28` | `38.73` |

Interpretation:

- grouped low-rank compression recovers some TTFT advantage
- but quality is still far below the full-consumer reference even on the self-pair

This is further evidence that the quality-speed trade-off remains difficult even in a maximally similar setting.

## Implications for the Current Project

1. `ser` is immediately usable for real runs.
2. The best existing execution base is the remote `scd-repro-codex-mistral-instruct-20260323` repo, not the local scaffold alone.
3. A true Qwen3.5-vs-Qwen3.5 experiment is currently blocked by model availability: only one Qwen3.5 checkpoint was found.
4. Existing Qwen-family results already suggest that merely moving to a more similar pair does not automatically recover the paper's claimed effect.
5. The next highest-value experiment is not another blind full run; it is a controlled near-pair experiment using the remote Qwen assets that best approximate the intended story while explicitly documenting that the pair is `Qwen3`, not yet `Qwen3.5`.

## Recommended Immediate Execution Order

1. Treat `Qwen3-1.7B -> Qwen3-1.7B-bf16` as the current remote near-pair baseline anchor because it already has full artifacts and strict summaries.
2. Use those existing artifacts to backfill our local claim-gap machinery before launching new long runs.
3. If the user still wants strict Qwen3.5 priority, obtain a second Qwen3.5-family checkpoint before claiming a true Qwen3.5 similar-pair result.
4. Only after the pair question is settled should we spend A800 time on route A/B/C follow-up runs.
