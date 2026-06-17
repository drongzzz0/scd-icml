# SCD-vLLM Plugin MVP

This repository exposes the current vLLM integration as an experimental
companion adapter instead of a vLLM fork.

## Status

- The plugin path is experimental.
- It uses vLLM's connector boundary, currently through `ExampleConnector`
  compatible storage.
- It does not patch vLLM core and is not a production serving backend.
- Dataset-level quality evidence remains the strict runtime evaluation; vLLM
  smoke tests prove connector loading and generation.

## CLI

```bash
scd-vllm export-bundle \
  --producer-model /path/to/producer \
  --consumer-model /path/to/consumer \
  --reuse-ckpt /path/to/reuse \
  --patch-ckpt /path/to/patch \
  --route configs/routes/scd_icml_mistrallite_to_mistral7_reuse19_patch13_cmrc200.json \
  --prefix-file prompts/prefix.txt \
  --out runs/scd_vllm/my_bundle

scd-vllm convert-bundle \
  --bundle-dir runs/scd_vllm/my_bundle \
  --storage-dir runs/scd_vllm/my_storage \
  --overwrite

scd-vllm smoke-test \
  --bundle-dir runs/scd_vllm/my_bundle \
  --storage-dir runs/scd_vllm/my_storage \
  --model /path/to/consumer \
  --out runs/scd_vllm/my_smoke \
  --enforce-eager \
  --disable-hybrid-kv-cache-manager \
  --overwrite
```

The combined smoke pipeline is:

```bash
scd-vllm run \
  --producer-model /path/to/producer \
  --consumer-model /path/to/consumer \
  --reuse-ckpt /path/to/reuse \
  --patch-ckpt /path/to/patch \
  --route configs/routes/scd_icml_mistrallite_to_mistral7_reuse19_patch13_cmrc200.json \
  --prefix-file prompts/prefix.txt \
  --out-dir runs/scd_vllm/my_plugin_run \
  --enforce-eager \
  --disable-hybrid-kv-cache-manager \
  --overwrite
```

## Python API

```python
from scd_vllm_plugin import SCDVLLMConfig, SCDVLLMEngine

engine = SCDVLLMEngine(
    SCDVLLMConfig(
        producer_model="/path/to/producer",
        consumer_model="/path/to/consumer",
        reuse_ckpt="/path/to/reuse",
        patch_ckpt="/path/to/patch",
        route="configs/routes/scd_icml_mistrallite_to_mistral7_reuse19_patch13_cmrc200.json",
    )
)

metrics = engine.run_smoke_pipeline(
    out_dir="runs/scd_vllm/my_plugin_run",
    prefix_file="prompts/prefix.txt",
    enforce_eager=True,
    disable_hybrid_kv_cache_manager=True,
    overwrite=True,
)
```

## Current Boundary

The plugin is useful as a reproducible integration layer:

1. Export a reconstructed SCD `reuse_patch` KV bundle.
2. Convert the bundle to vLLM connector storage.
3. Load it through vLLM and verify deterministic smoke generation.

The next production step is replacing disk-backed `ExampleConnector` storage
with a dedicated SCD connector that can manage in-memory KV lifecycle, batching,
and request-level routing.
