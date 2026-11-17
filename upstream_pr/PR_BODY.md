## Purpose

Add **`SCDConnectorV1`**, a thin V1 `KVConnectorBase_V1` wrapper for **[Semantic Cache Distillation (`scd-icml`)](https://github.com/drongzzz0/scd-icml)** cross-model KV cache transfer, following the same lazy third-party adapter delegation pattern as `LMCacheConnectorV1` and `FlexKVConnectorV1`.

Unlike homogeneous KV connectors that require identical producer and consumer model architectures, `SCDConnectorV1` enables transferring prefix representations from a lightweight producer model (e.g., `MistralLite` without sliding-window truncation) into a stronger target model's paged KV cache (e.g., `Mistral-7B-Instruct-v0.2`) using hybrid layerwise **Reuse** projections and **Learned Shared-Residual Patch** projections.

### Design Highlights

- **Zero core / hot-path intrusion**: No changes to `ModelRunner`, `Attention`, `Scheduler`, or `ForwardContext`. When `SCDConnectorV1` is unused, zero modules are imported (`KVConnectorFactory.register_connector` is lazy).
- **Thin delegator (`vllm/distributed/kv_transfer/kv_connector/v1/scd_connector.py`)**: Lazily imports `scd_vllm.vllm_v1_adapter.SCDConnectorV1Impl` from the external `scd-icml` package and delegates all V1 scheduler and worker lifecycle hooks.
- **CUDA-Graph compatible (`enforce_eager=False`)**: Installs zero target model forward hooks; KV blocks are written via stream-ordered paged `index_copy_` inside `start_load_kv` prior to target CUDA-graph replay.
- **Supports three transport modes via `kv_connector_extra_config["mode"]`**:
  - `"resident"`: Zero-copy same-worker GPU conversion + stream-ordered paged KV scatter.
  - `"file"`: Lease-bound `safetensors` artifact store for disaggregated producer/consumer processes.
  - `"preloaded"`: Bounded LRU GPU-memoized static prefix serving.

## Test Plan

1. **Upstream unit tests (no `scd-icml` dependency required in CI)**:
   ```bash
   pytest tests/v1/kv_connector/unit/test_scd_connector_v1.py -q
   ```
   Verifies `KVConnectorFactory` registration, clear `ImportError` guidance when `scd-icml` is absent, and full V1 scheduler/worker method delegation.
2. **End-to-end GPU verification (`NVIDIA A800-SXM4-80GB`, `MistralLite -> Mistral-7B-v0.1` Patch20/Reuse12, FlashAttention-2, `enforce_eager=False` with CUDA Graphs)**:
   - **Bitwise parity**: `64/64` BF16 K/V tensors match the uncompiled ICML reference implementation bitwise (`max_abs_diff = 0.0`) across all validation prompts.
   - **Single-request prefix scaling (`512 -> 3584` tokens)**: Up to **`5.26x` TTFT speedup** (`260.80 ms -> 49.57 ms` at 3584 tokens; `3.05x` on 1273-token CMRC QA with exact 16/16 greedy token match) and **`1.84x`** 16-token full completion speedup (`490.29 ms -> 267.08 ms`).
   - **Shared-prefix continuous batching (`2048` prefix tokens, `32` generated tokens, `B=16`)**: **`2.99x` end-to-end speedup even cold** (including producer prefill + cross-model KV conversion: `2725.9 ms -> 913.1 ms`, `187.8 -> 560.7` output tok/s) and **`3.03x` warm throughput speedup** (`569.1` output tok/s, **`8.62x`** lower mean TTFT: `2096.5 ms -> 243.2 ms`).

## Checklist

- [x] SPDX-License-Identifier headers included on all new files
- [x] Lazy registration in `KVConnectorFactory` (zero import cost when unused)
- [x] Unit tests pass without third-party package installed
- [x] Documentation added to `docs/features/disagg_prefill.md` and `examples/disaggregated/scd_connector/`
