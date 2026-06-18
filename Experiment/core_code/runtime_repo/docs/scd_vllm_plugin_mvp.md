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
- The current higher-Reuse/speed vLLM demo route is
  `configs/routes/scd_icml_mistrallite_to_mistral7_patch20_reuse12_cmrc200.json`.
  The quality-first validated vLLM route is
  `configs/routes/scd_icml_mistrallite_to_mistral7_patch26_reuse6_cmrc200.json`.
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

After a bundle has been converted to `ExampleConnector` storage, run a
repeated-prefix batch microbenchmark:

```bash
scd-vllm batch-benchmark \
  --bundle-dir runs/scd_vllm/my_plugin_run/reuse_patch_bundle \
  --storage-dir runs/scd_vllm/my_plugin_run/example_connector_storage \
  --model /path/to/consumer \
  --batch-size 64 \
  --num-batches 3 \
  --warmup-batches 1 \
  --max-new-tokens 16 \
  --out runs/scd_vllm/my_batch_bench \
  --enforce-eager \
  --disable-hybrid-kv-cache-manager \
  --include-prefix-cache-baseline
```

This benchmark intentionally repeats one validated prefix. Treat it as a
trusted, static-prefix connector throughput check. With
`--include-prefix-cache-baseline`, it reports plain vLLM, official vLLM prefix
caching, and SCD load metrics in one artifact. It is not evidence for
multi-tenant safety, arbitrary prompt routing, or dataset-level quality.

The release also includes an external vLLM connector prototype that can be
loaded through vLLM's `kv_connector_module_path` mechanism:

```bash
scd-vllm batch-benchmark \
  --bundle-dir runs/scd_vllm/my_plugin_run/reuse_patch_bundle \
  --storage-dir runs/scd_vllm/my_plugin_run/example_connector_storage \
  --model /path/to/consumer \
  --batch-size 64 \
  --num-batches 3 \
  --warmup-batches 1 \
  --max-new-tokens 16 \
  --out runs/scd_vllm/my_preloaded_connector_bench \
  --enforce-eager \
  --disable-hybrid-kv-cache-manager \
  --include-prefix-cache-baseline \
  --kv-connector SCDPreloadedConnector \
  --kv-connector-module-path scd_vllm.preloaded_connector \
  --connector-cache-device cuda
```

`SCDPreloadedConnector` keeps vLLM's connector metadata semantics but caches
loaded SCD KV tensors per worker by filename, avoiding repeated safetensors
reads for identical prefixes inside a batch. It is still an experimental
connector prototype and does not solve multi-tenant isolation or dynamic route
selection.

For direct integration with official vLLM Python code, emit a launch spec:

```bash
scd-vllm launch-spec \
  --bundle-dir runs/scd_vllm/my_plugin_run/reuse_patch_bundle \
  --storage-dir runs/scd_vllm/my_plugin_run/example_connector_storage \
  --model /path/to/consumer \
  --stats-dir runs/scd_vllm/my_preloaded_connector_bench/preloaded_connector_stats \
  --out runs/scd_vllm/launch_spec.json

scd-vllm launch-smoke \
  --launch-spec runs/scd_vllm/launch_spec.json \
  --batch-size 8 \
  --max-new-tokens 16 \
  --out runs/scd_vllm/launch_smoke.json
```

The JSON contains `llm_kwargs` for `vllm.LLM(**llm_kwargs)`, a
`kv_transfer_config` using `SCDPreloadedConnector`, and the optional
`SCD_PRELOADED_STATS_DIR` environment setting. It automatically sets
`disable_hybrid_kv_cache_manager=True` for the preloaded connector because
vLLM `0.18.0` rejects that connector when HMA is enabled. This spec is a
single-tenant trusted-bundle launch contract: verify `token_hash` and
`route_id` before routing a request to it. `launch-smoke` consumes the same
JSON, validates that the safety contract declares `multi_tenant_safe=false`,
`dynamic_request_routing=false`, and `trusted_bundle_required=true`, applies
its environment settings, calls official `vllm.LLM`, runs one trusted-prefix
generation batch, and writes an auditable `launch_smoke.json`.

The release preview also includes a small programmatic helper:

```bash
python examples/scd_vllm_official_launch.py \
  --launch-spec runs/scd_vllm/launch_spec.json \
  --dry-run
```

The dry run validates the launch spec without importing vLLM or using a GPU.
Omit `--dry-run` to execute the same trusted-prefix smoke path through
official vLLM and write generation metrics.

## Python API

```python
import os

from scd_vllm_plugin import (
    SCDVLLMConfig,
    SCDVLLMEngine,
    build_preloaded_launch_spec,
    run_launch_smoke,
)

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

launch_spec = build_preloaded_launch_spec(
    model="/path/to/consumer",
    bundle_dir="runs/scd_vllm/my_plugin_run/reuse_patch_bundle",
    storage_dir="runs/scd_vllm/my_plugin_run/example_connector_storage",
    stats_dir="runs/scd_vllm/my_preloaded_connector_bench/preloaded_connector_stats",
)

# In the same Python process:
# from vllm import LLM
# for key, value in launch_spec["environment"].items():
#     os.environ[key] = value
# llm = LLM(**launch_spec["llm_kwargs"])

launch_smoke_metrics = run_launch_smoke(
    launch_spec="runs/scd_vllm/launch_spec.json",
    batch_size=8,
    out="runs/scd_vllm/launch_smoke.json",
)
```

## Current Boundary

The plugin is useful as a reproducible integration layer:

1. Export a reconstructed SCD `reuse_patch` KV bundle.
2. Convert the bundle to vLLM connector storage.
3. Load it through vLLM and verify deterministic smoke generation.

The next production step is replacing disk-backed `ExampleConnector` storage
with a dedicated SCD connector that can manage in-memory KV lifecycle,
concurrent request isolation, and request-level routing.
