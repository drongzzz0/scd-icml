# SCD-vLLM Plugin MVP

This repository exposes the current vLLM integration as an experimental
companion adapter instead of a vLLM fork.

## Status

- The plugin path is experimental.
- It uses vLLM's connector boundary through `SCDPreloadedConnector`, with
  `ExampleConnector`-compatible storage as the current on-disk format.
- It does not patch vLLM core and is not a production serving backend.
- Dataset-level quality evidence remains the strict runtime evaluation; vLLM
  smoke tests prove connector loading and generation.
- The supported claim is single-tenant trusted-bundle loading for a static
  prefix. It is not a multi-tenant cache-sharing, dynamic request-routing, or
  same-model prefix-cache replacement claim.

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

The official-vLLM launch-spec path is:

```bash
scd-vllm doctor \
  --bundle-dir runs/scd_vllm/my_bundle \
  --route runs/scd_vllm/my_bundle/route.json

scd-vllm launch-spec \
  --bundle-dir runs/scd_vllm/my_bundle \
  --storage-dir runs/scd_vllm/my_storage \
  --model /path/to/consumer \
  --out runs/scd_vllm/launch_spec.json \
  --enforce-eager

scd-vllm launch-smoke \
  --launch-spec runs/scd_vllm/launch_spec.json \
  --out runs/scd_vllm/launch_smoke.json \
  --batch-size 1 \
  --max-new-tokens 16
```

`launch-spec` emits ordinary `vllm.LLM(**llm_kwargs)` arguments plus a
`kv_transfer_config` that points to `SCDPreloadedConnector` and
`scd_vllm.preloaded_connector`. `launch-smoke` consumes that JSON and verifies
that official vLLM can instantiate the engine, load the trusted prefix KV
bundle, and generate.

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
3. Emit an official-vLLM launch spec.
4. Load it through vLLM and verify deterministic smoke generation.

Current verified evidence:

- Local vLLM-related tests: `62 passed`.
- ser `launch-spec`: `launch_spec_pass=true`, `storage_ready=true`.
- ser `doctor` under `vllm==0.18.0`: `pass=true`.
- ser `launch-smoke`: `launch_smoke_pass=true`,
  `bundle_token_match=true`, `generation_ok=true`.
- ser repeated-prefix benchmark on Mistral-7B, batch size `16`,
  `3` measured batches: plain vLLM `385.642508 tokens/s`, SCD load
  `649.831790 tokens/s`, speedup `1.685063x`, connector cache hit rate
  `0.984375`.
- The same repeated-identical-prefix setting gives official vLLM prefix cache
  `763.958763 tokens/s`, which is faster than SCD load. Therefore the release
  should position SCD-vLLM as trusted cross-model / external-bundle KV
  preloading, not as a replacement for native same-model prefix caching.

The next production step is replacing disk-backed `ExampleConnector` storage
with a dedicated SCD connector that can manage in-memory KV lifecycle, batching,
and request-level routing.
