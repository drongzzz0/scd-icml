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

Measured on `MistralLite -> Mistral-7B-v0.1` (`mistral_patch20_reuse12`, BF16, `block_size=16`, `enforce_eager=False` with CUDA Graphs):

- **Bitwise Parity**: **64/64** converted BF16 K/V tensors match the uncompiled ICML reference implementation bitwise (`max_abs_diff = 0.0`) across all CMRC validation cases, with up to **`1.64x`** faster standalone conversion (`4.28 ms -> 2.61 ms`).

### Single-Request Prefix Scaling (`import_policy="fullprefix"`, 16-token greedy generation)

| Prefix Tokens | Native TTFT (ms) | SCD H-Mode TTFT (ms) | TTFT Speedup | Native 16-tok (ms) | SCD H-Mode 16-tok (ms) | Completion Speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | 53.96 | **29.01** | **1.86x** | 251.99 | **207.57** | **1.21x** |
| 1024 | 81.93 | **29.91** | **2.74x** | 288.75 | **229.84** | **1.26x** |
| 2048 | 148.40 | **32.82** | **4.52x** | 363.10 | **241.99** | **1.50x** |
| 3584 | 260.80 | **49.57** | **5.26x** | 490.29 | **267.08** | **1.84x** |

### Shared-Prefix Continuous Batch Throughput (`2048` prefix tokens, `32` new tokens/req)

| Batch Size | Native Wall (ms) | SCD Cold E2E Wall (ms) | Cold Speedup | SCD Memoized Wall (ms) | Warm Speedup | Native Out (tok/s) | SCD Out (tok/s) | Mean TTFT Reduction |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 602.2 | 641.3 | 0.94x | **502.6** | **1.20x** | 53.1 | **63.7** | `149.4 -> 44.2 ms` (**3.38x**) |
| 4 | 1014.7 | **689.0** | **1.47x** | **587.7** | **1.73x** | 126.1 | **217.8** | `550.5 -> 86.6 ms` (**6.36x**) |
| 8 | 1593.5 | **778.3** | **2.05x** | **709.1** | **2.25x** | 160.6 | **361.0** | `1084.8 -> 140.9 ms` (**7.70x**) |
| 16 | 2725.9 | **913.1** | **2.99x** | **899.6** | **3.03x** | 187.8 | **569.1** | `2096.5 -> 243.2 ms` (**8.62x**) |

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
