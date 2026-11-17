# SCDConnectorV1: Cross-Model Semantic Cache Distillation KV Transfer

`SCDConnectorV1` connects [Semantic Cache Distillation (`scd-icml`)](https://github.com/drongzzz0/scd-icml) to vLLM's V1 `KVConnectorBase_V1` interface.

Unlike homogeneous KV cache connectors that require identical producer and consumer model architectures, `SCDConnectorV1` transfers prefix context from a lightweight producer model (e.g., `MistralLite` with full sliding-window-free context) into a stronger target model (e.g., `Mistral-7B-Instruct-v0.2`) using hybrid layerwise **Reuse** projections and **Learned Shared-Residual Patch** projections.

## Installation

```bash
pip install "git+https://github.com/drongzzz0/scd-icml.git"
```

## Supported Transport Modes

`SCDConnectorV1` delegates to `scd_vllm.vllm_v1_adapter.SCDConnectorV1Impl` and supports three transport modes via `kv_connector_extra_config`:

1. **`resident` (`mode: "resident"`)**: Zero-copy same-worker GPU conversion + stream-ordered paged KV scatter (`enforce_eager=False` CUDA graphs supported).
2. **`file` (`mode: "file"`)**: Lease-verified `safetensors` artifact store for disaggregated producer/consumer processes.
3. **`preloaded` (`mode: "preloaded"`)**: Bounded LRU GPU-memoized static prefix serving compatible with `ExampleConnector` storage.

## Quick Example

```bash
python examples/disaggregated/scd_connector/cross_model_kv_handoff.py \
  --profile /path/to/profile.json \
  --assets /path/to/imported_assets \
  --target-model /path/to/Mistral-7B-Instruct-v0.2 \
  --target-manifest /path/to/model_manifest.json \
  --prefix-ids /path/to/prefix_ids.json \
  --prompt-ids /path/to/prompt_ids.json
```
