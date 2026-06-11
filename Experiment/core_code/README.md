# SCD Experiment Core

This directory is the curated experiment/code layer for the `scd-new` project.

It now follows a simple rule:

- `runtime_repo/` is the canonical runtime code snapshot to keep, maintain, and upload.
- `configs/`, `scd_validation/`, and the top-level `run_*.sh` wrappers are the active experiment orchestration layer around that runtime package.
- Previously archived `legacy/` content has been removed so this directory now keeps only the current maintained surface.

## Canonical Layout

### Active

- `runtime_repo/`
  - The final runtime/package snapshot.
  - Contains `src/scd_repro`, stable configs, and reusable scripts.
- `scd_validation/`
  - The config-driven validation scaffold used by the local research workflow.
- `configs/`
  - Templates and generated configs for screening, trace planning, and baseline aggregation.
- `build_pair_screening_config.py`
- `build_original_scd_config.py`
- `check_runtime_env.py`
- `plan_trace_capture.py`
- `run_pair_screening.py`
- `run_original_scd_baseline.py`
- `run_remote_*.sh`
- `run_ser_*.sh`

## Recommended Entry Points

Use the scaffold first:

```bash
python3 Experiment/core_code/run_pair_screening.py \
  --config Experiment/core_code/configs/smoke_pair_screening.json

python3 Experiment/core_code/run_original_scd_baseline.py \
  --config Experiment/core_code/configs/smoke_original_scd_baseline.json
```

For the runtime package itself, start from:

- `Experiment/core_code/runtime_repo/pyproject.toml`
- `Experiment/core_code/runtime_repo/src/scd_repro/`
- `Experiment/core_code/runtime_repo/scripts/`

## Notes

- The local `runtime_repo/` is newer than the inspected A800 copies.
- During the April 17, 2026 check, A800 was reachable as `gpu010`, but `/nas/mql` was unavailable with `ls: 鏃犳硶璁块棶 '/nas/mql': 涓绘満鍏抽棴`, so NAS-backed run artifacts should be treated as temporarily inaccessible.
