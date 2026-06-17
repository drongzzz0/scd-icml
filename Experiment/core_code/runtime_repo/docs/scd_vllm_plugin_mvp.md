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
- The current vLLM demo route is
  `configs/routes/scd_icml_mistrallite_to_mistral7_patch20_reuse12_cmrc200.json`.
  The `reuse19_patch13` route remains available as a paper-aligned
  maximum-Reuse reference.

## CLI

Run a static preflight first. This does not launch GPU work; it checks that
the package can import vLLM, required artifacts exist, and the route is a
strict no-fallback/no-recompute mixed reuse+patch route:

```bash
scd-vllm doctor \
  --producer-model /path/to/producer \
  --consumer-model /path/to/consumer \
  --reuse-ckpt /path/to/reuse \
  --patch-ckpt /path/to/patch \
  --route configs/routes/scd_icml_mistrallite_to_mistral7_patch20_reuse12_cmrc200.json \
  --out runs/scd_vllm/doctor.json
```

Use `--skip-vllm-import` only for documentation or CI environments that do not
install the optional `vllm` extra. Use `--require-local-models` when producer
and consumer model arguments must be local directories rather than model IDs.

If a reconstructed SCD bundle already exists, validate that artifact directly:

```bash
scd-vllm doctor \
  --bundle-dir runs/scd_vllm/my_bundle \
  --route runs/scd_vllm/my_bundle/route.json \
  --out runs/scd_vllm/doctor_bundle.json
```

Bundle preflight validates `manifest.json`, `token_ids.npy`, KV layer files,
and the route hash when both the bundle manifest and `--route` expose one.

```bash
scd-vllm export-bundle \
  --producer-model /path/to/producer \
  --consumer-model /path/to/consumer \
  --reuse-ckpt /path/to/reuse \
  --patch-ckpt /path/to/patch \
  --route configs/routes/scd_icml_mistrallite_to_mistral7_patch20_reuse12_cmrc200.json \
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
  --route configs/routes/scd_icml_mistrallite_to_mistral7_patch20_reuse12_cmrc200.json \
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
        route="configs/routes/scd_icml_mistrallite_to_mistral7_patch20_reuse12_cmrc200.json",
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
