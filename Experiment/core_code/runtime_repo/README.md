# SCD Runtime Repository

This directory is the canonical runtime/package snapshot for the final
`scd-new` code line.

## What To Keep

- `src/scd_repro/`
  - Main Python package.
- `configs/`
  - Active training/eval/model configs still referenced by the local workflow.
- `scripts/`
  - Remote execution helpers and reusable utility scripts.
- Top-level `run_ser_*.sh`
  - Stable wrappers still useful for reproducible route runs.

## What Was Archived

The following are no longer part of the canonical runtime surface and were
moved under `legacy/`:

- temporary patch prototypes (`legacy/prototypes/`)
- backup snapshots of `strict_runtime.py` and `eval_cmrc_strict.py`
  (`legacy/backups/`)
- generated packaging metadata (`legacy/build_metadata/scd_repro.egg-info/`)

These files are preserved only for audit/debug history.

## Current Canonical Interpretation

For the claim-facing `scd-new` story, this runtime package should be read as the
implementation base for:

- baseline
- repaired reuse-only
- current strict patch-only
- strict hybrid only when separately validated

Historical grouped/shared-z defaults, fallback-heavy routes, and scratch patch
variants may still exist in configs or reports, but they should not be treated
as the default final method line.

## Quick Start

```bash
python -m pip install -e .
capture-pair --models-config configs/models.yaml --data-config configs/data.yaml --out artifacts/pairs/run_demo
train-reuse --pairs artifacts/pairs/run_demo --train-config configs/train.yaml --out artifacts/ckpt_reuse/ckpt_demo
profile-layers --pairs artifacts/pairs/run_demo --reuse-ckpt artifacts/ckpt_reuse/ckpt_demo --out artifacts/route/profile_demo.json
train-patch --pairs artifacts/pairs/run_demo --profile artifacts/route/profile_demo.json --train-config configs/train.yaml --out artifacts/ckpt_patch/patch_demo
```

## Read This Before Uploading

- `legacy/` is historical and optional for a public-facing clean repo.
- `configs/` still includes some older experiment branches for reproducibility.
- If you want a minimal GitHub repo, trim to `src/`, the subset of `configs/`
  you still use, selected `scripts/`, and this README.
