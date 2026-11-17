# SCD-ICML: Semantic Cache Distillation & High-Performance vLLM V1 Connector

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](./LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](./pyproject.toml)
[![vLLM V1](https://img.shields.io/badge/vLLM-V1%20KVConnector-green.svg)](./upstream_pr/README_SUBMISSION.md)

Official open-source repository for **Semantic Cache Distillation (SCD-ICML)** — cross-model prefix KV cache transfer (`MistralLite -> Mistral-7B-Instruct-v0.2` via hybrid **Layerwise Reuse** + **Learned Shared-Residual Patch**) integrated directly into **vLLM V1** (`KVConnectorBase_V1`).

> **Branch Note**: The historical full research workspace snapshot and the Qwen2.5-32B `A1` profile remain archived under the [`archive/workspace-snapshot-20260702`](https://github.com/drongzzz0/scd-icml/tree/archive/workspace-snapshot-20260702) and [`A1`](https://github.com/drongzzz0/scd-icml/tree/A1) branches.

---

## Key Capabilities

1. **Complete Offline Reproduction & Route Construction (`src/scd_repro/`)**
   - Paired activation capture (`capture-pair`), pre-RoPE low-rank Reuse training (`train-reuse`), shared-trunk SwiGLU residual Patch training (`train-patch`), quality-gated route selection (`profile-layers`, `build-route`), and strict evaluation (`eval-text-strict`).
   - Ships the three canonical 32-layer Mistral routes in [`configs/routes/`](./configs/routes/):
     - `mistral_patch20_reuse12.json` (Primary quality-speed Pareto route)
     - `mistral_patch26_reuse6.json` (High-capacity Patch26 ablation)
     - `mistral_reuse19_patch13.json` (Lightweight Reuse19 ablation)

2. **Deeply Optimized vLLM V1 Serving Runtime (`src/scd_runtime/` & `src/scd_vllm/`)**
   - **Pre-Compiled GPU Execution (`CompiledICMLPlan`)**: Pre-stages stacked FP32 Reuse projection banks, BF16 target `k_proj`/`v_proj` weights, shared SwiGLU Patch trunk weights, and precomputed source/target RoPE tables up to `max_model_len` on GPU once at engine initialization.
   - **Vectorized 32-Layer Conversion & Zero-Copy Packed Snapshot**: Vectorizes RoPE rotation across all Patch layers simultaneously and packs `[L, N, H, 2*D]` BF16 KV tensors via 2 batched `index_copy_` kernels with zero extra snapshot allocation (`adopt_packed=True`).
   - **Stream-Ordered Page-Level Scatter (`write_packed_layers_fast`)**: Replaces per-layer `torch.cat` + 2D advanced-indexing scatters + `cudaDeviceSynchronize()` stalls with single-pass block-level `dst.index_copy_` on the worker's active compute stream.
   - **CUDA Graphs Enabled (`enforce_eager=False`)**: Installs zero target model hooks (`reader.kind == "none"`), allowing vLLM V1 to replay full/piecewise CUDA graphs for both the 1-token suffix prefill step (`import_policy="fullprefix"`) and all decode steps.
   - **Fused 2-RPC Serial & Batch Lifecycle (`scd_gpu_prepare_and_admit[_batch]` / `scd_gpu_finalize[_batch]`)**: Collapses per-request and per-batch IPC round-trips down to 2 RPCs total, with automatic within-batch shared-prefix deduplication, optional bounded LRU GPU prefix memoization (`memoize_prefix_entries`), and automatic FlashInfer JIT sampler fallback when `ninja`/`nvcc` are absent (`ensure_sampler_jit_safe`).

3. **Zero-Fork vLLM Plugin + Companion Upstream vLLM PR (`upstream_pr/`)**
   - **Out-of-the-box on unmodified vLLM**: Installing `scd-icml` registers `[project.entry-points."vllm.general_plugins"]` (`scd_icml = "scd_vllm.plugin:register"`), automatically registering `SCDConnectorV1`, `SCDGPUConnectorV1`, and `SCDPreloadedConnector` across frontend, `EngineCore`, and GPU `Worker` processes.
   - **Ready-to-submit upstream vLLM PR**: [`upstream_pr/`](./upstream_pr/README_SUBMISSION.md) provides the thin `vllm/distributed/kv_transfer/kv_connector/v1/scd_connector.py` wrapper (following `LMCacheConnectorV1` and `FlexKVConnectorV1`), factory registration, docs, example, unit test, and verified git patch (`scd_connector_v1.patch`).

---

## Verified Performance (`NVIDIA A800-SXM4-80GB`, vLLM V1 + FlashAttention-2)

Measured on `MistralLite -> Mistral-7B-v0.1` (`mistral_patch20_reuse12`, BF16, `block_size=16`, CUDA Graphs) with vLLM `2b1f75c80`.

- **Conversion parity.** In the earlier microbenchmark, all 64 converted BF16 K/V tensors per case matched the uncompiled ICML reference on four CMRC inputs. The largest standalone conversion speedup was **1.64x** (`4.28 ms -> 2.61 ms`).

### Single-Request Prefix Scaling (`import_policy="fullprefix"`, 16-token greedy generation)

H-mode starts with source KV already available. E-mode includes source prefill. The native single-request baseline has automatic prefix caching (APC) disabled.

| Prefix Tokens | Native TTFT (ms) | SCD H-mode TTFT (ms) | SCD E-mode TTFT (ms) | H-mode TTFT Speedup | Native 16-tok (ms) | SCD H-mode 16-tok (ms) |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | 52.99 | 29.35 | 89.58 | 1.81x | 249.57 | 225.31 |
| 1024 | 80.79 | 30.27 | 116.53 | 2.67x | 283.92 | 228.28 |
| 2048 | 149.24 | 34.74 | 202.82 | 4.30x | 363.74 | 242.50 |
| 3584 | 253.79 | 49.45 | 347.40 | 5.13x | 488.57 | 271.85 |

### Shared-Prefix Continuous Batch Throughput (`2048` prefix tokens, `32` new tokens/req)

Each cell reports mean request TTFT and total batch wall time in milliseconds, using the median of three trials. Both measurements start at the outer batch call, so SCD TTFT includes prefix preparation and admission. Before each cold trial, native APC and SCD memoization are cleared for the respective arm. The same prefix is reused in the following warm trial.

| Batch Size | Native APC Off | Native APC Cold | Native APC Warm | SCD Cold | SCD Warm |
|---:|---:|---:|---:|---:|---:|
| 1 | 149.7 / 588.4 | 153.6 / 595.1 | 38.9 / 485.2 | 200.9 / 637.0 | 50.2 / 486.0 |
| 4 | 531.4 / 956.4 | 178.6 / 629.7 | 62.6 / 515.5 | 225.4 / 689.3 | 109.4 / 578.4 |
| 8 | 1074.4 / 1511.5 | 213.3 / 706.8 | 95.3 / 587.7 | 289.2 / 778.4 | 201.5 / 703.7 |
| 16 | 2117.0 / 2669.3 | 280.6 / 834.4 | 154.2 / 712.4 | 387.3 / 945.6 | 326.1 / 884.4 |

At batch size 16, target APC reused all 2,048 prefix tokens for 15 of 16 requests in each cold trial and for every request in each warm trial. Reproduce the comparison with `scripts/benchmark_a800_vllm_open.py --cudagraph-and-batch --batch-reps 3` and the local profile, assets, target model and case inputs.

The paired CMRC experiments report official F1 and exact match alongside the per-request latency breakdown. [A800 experiment records](docs/experiment-log.md)

---

## Repository Structure

```text
scd-icml-vllm-open/
├── configs/
│   └── routes/                 # Canonical Mistral Patch/Reuse layer routes
├── profiles/
│   └── icml_mistral.example.json
├── src/
│   ├── scd_repro/              # Offline pair capture, Reuse/Patch training & strict eval
│   ├── scd_runtime/            # Verified contracts, CompiledICMLPlan & GPUResidentConsumer
│   ├── scd_vllm/               # vLLM V1 KVConnectors, fast paged layout & plugin entry point
│   └── scd_vllm_plugin/        # scd-vllm CLI facade, doctor & launch-spec builder
├── scripts/                    # Offline reproduction & paper table/figure scripts
├── examples/                   # Quickstart scripts for official vLLM launch
├── tests/                      # 280+ offline, contract, bitwise-parity & plugin unit tests
└── upstream_pr/                # Ready-to-submit PR package for vllm-project/vllm
```

---

## Installation

```bash
# Install directly from GitHub (automatically registers vLLM V1 general_plugins)
pip install "git+https://github.com/drongzzz0/scd-icml.git"

# Or clone for local development & reproduction
git clone https://github.com/drongzzz0/scd-icml.git
cd scd-icml
pip install -e ".[vllm,test]"
```

Run the local CPU verification suite (no GPU required):

```bash
pytest tests -q
```

---

## Serving Quickstart with vLLM V1

### 1. Import Trained SCD-ICML Banks into Verified `safetensors` Assets

```bash
scd-runtime import-assets \
  --profile profiles/icml_mistral.json \
  --input reuse_bank=/path/to/reuse_bank.pt \
  --input patch_ckpt=/path/to/shared_patch.pt \
  --input source_model=/path/to/MistralLite \
  --input source_manifest=/path/to/source_manifest.json \
  --input target_model=/path/to/Mistral-7B-Instruct-v0.2 \
  --input target_manifest=/path/to/target_manifest.json \
  --trusted-legacy \
  --out ./imported_assets
```

### 2. Same-Worker GPU-Resident Serving (CUDA Graphs Enabled by Default)

```python
import json
from pathlib import Path
from scd_runtime.contracts.types import PrefixContract, profile_from_dict
from scd_runtime.resident_runner import GPUResidentConsumer

profile = profile_from_dict(json.loads(Path("profiles/icml_mistral.json").read_text()))
consumer = GPUResidentConsumer(
    profile,
    "./imported_assets",
    engine_config={
        "model": "/path/to/Mistral-7B-Instruct-v0.2",
        "model_manifest": "/path/to/target_manifest.json",
        "block_size": 16,
        "max_model_len": 4096,
        "gpu_memory_utilization": 0.45,
        "enforce_eager": False,          # CUDA Graphs enabled!
        "memoize_prefix_entries": 64,    # Optional LRU GPU prefix cache for repeated contexts
        "attention_config": {"backend": "FLASH_ATTN"},
    },
)

prefix = PrefixContract(
    tuple(prefix_token_ids),
    0,
    profile.source.tokenizer_sha256,
    profile.source.chat_template_sha256,
)
receipt = consumer.run(
    prefix,
    prompt_token_ids,
    {"max_new_tokens": 32, "temperature": 0.0},
    mode="E",
    import_policy="fullprefix",
    full_prompt_ids=prompt_token_ids,
)
print("Generated tokens:", receipt.generation.token_ids)
print("TTFT (ms):", receipt.metrics["pipeline_first_token_ms"])
consumer.close()
```

### 3. Direct Official `vllm.LLM` / CLI Usage via `SCDConnectorV1`

Once `scd-icml` is installed (`pip install -e .`), `SCDConnectorV1` is automatically registered in `KVConnectorFactory` via `vllm.general_plugins`:

```bash
vllm serve /path/to/Mistral-7B-Instruct-v0.2 \
  --dtype bfloat16 \
  --block-size 16 \
  --no-enable-chunked-prefill \
  --disable-hybrid-kv-cache-manager \
  --kv-transfer-config '{"kv_connector":"SCDConnectorV1","kv_role":"kv_consumer","kv_load_failure_policy":"fail","kv_connector_extra_config":{"mode":"preloaded","shared_storage_path":"/path/to/scd_storage","cache_device":"cuda"}}'
```

---

## Submitting Upstream PR to `vllm-project/vllm`

See [`upstream_pr/README_SUBMISSION.md`](./upstream_pr/README_SUBMISSION.md) for the verified patch (`scd_connector_v1.patch`), checklist, and DCO sign-off (`git commit -s`) workflow.

---

## License

Released under the [MIT License](./LICENSE). Files inside [`upstream_pr/files/`](./upstream_pr/files/) intended for contribution to `vllm-project/vllm` carry the `Apache-2.0` SPDX license header matching upstream vLLM.
