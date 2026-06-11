# Qwen3.5 ModelScope Download Handoff

## Status Update

The primary strict-Qwen3.5 producer checkpoint has now been acquired on `ser`.

- model id: `Qwen/Qwen3.5-4B-Base`
- target path on `ser`: `/disk_n/c101-3-1/models/Qwen3.5-4B-Base`
- confirmed success timestamp in the remote terminal: `2026-03-29 17:08:06`
- confirmed success line: `DOWNLOAD_DONE path=/disk_n/c101-3-1/models/Qwen3.5-4B-Base`

Both the producer checkpoint and the existing consumer checkpoint now have the expected top-level files:

- `config.json`
- `configuration.json`
- `model.safetensors.index.json`
- `model.safetensors-00001-of-00002.safetensors`
- `model.safetensors-00002-of-00002.safetensors`
- tokenizer files

## Actual Successful Path

The successful acquisition path was an inline remote ModelScope command executed on `ser` inside:

```bash
source /disk_n/c101-3-1/venvs/A800/bin/activate
```

The environment was verified at run time with:

- `modelscope 1.35.0`
- `torch 2.9.0+cu128`
- `transformers 5.3.0`
- `peft 0.18.1`

## Important Caveat

The prepared local wrapper remains a documentation artifact, not the authoritative execution path.

Prepared files:

- `/Users/weimingming/dr-claw/scd-new/Experiment/core_code/download_qwen35_4b_base_from_modelscope.py`
- `/Users/weimingming/dr-claw/scd-new/Experiment/core_code/run_ser_modelscope_qwen35_base_download.sh`

The shell wrapper mixes local macOS paths with remote `/disk_n/...` paths and should not be treated as the trusted replay command without patching it first.

## Resulting Main Pair

The intended strict pair is now materially present:

- producer: `/disk_n/c101-3-1/models/Qwen3.5-4B-Base`
- consumer: `/disk_n/c101-3-1/models/Qwen3.5-4B`

## Current Next Step

The asset-acquisition blocker is closed. The current blocker is no longer download-related: it is the Qwen3.5 hybrid cache/state semantics exposed during the real smoke capture on `ser`. See `Experiment/analysis/qwen35_hybrid_cache_blocker.md`.
