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

Previously archived runtime backups, temporary prototypes, and build metadata
have been removed so this package surface now reflects only the current kept
version.

## Current Canonical Interpretation

For the claim-facing `scd-new` story, this runtime package should be read as the
implementation base for:

- baseline
- repaired reuse-only
- current strict patch-only

Historical grouped/shared-z defaults, fallback-heavy routes, and scratch patch
variants may still exist in configs or reports, but they should not be treated
as the default final method line.

## Quick Start

```bash
python -m pip install -e .
run-mistral-strict-default --models-config configs/models_ser_mistrallite_to_mistral7_20260405.json --data-root /disk_n/c101-3-1/projects/scd_main/legacy/no4/data --output-root /disk_n/c101-3-1/projects/scd-default-mistral-strict
```

Default strict defaults:

- `configs/train.yaml`: canonical strict training defaults, using the canonical `xnorm_nativekv_shared_residual_mlp_v1` patch path for all-layer `patch-only`
- `configs/eval.yaml`: canonical `baseline + reuse-only` strict eval defaults
- `configs/eval_patch_only.yaml`: canonical isolated `patch-only` strict eval defaults
- `configs/route_patch_only_all32.json`: canonical all-layer patch-only route template

The default strict entrypoint now uses disjoint CMRC subsets by default:

- training capture/fit uses the first `--samples` examples
- eval uses the next `--eval-samples` examples
- default merge output only contains `baseline`, `reuse-only`, and `patch-only`

Legacy paper-runner defaults:

- `scripts/run_paper_eval.py`
- `scripts/run_cmrc_paper_subset.py`
- `configs/train_paper_semantic_legacy.json`
- `configs/eval_paper_semantic_legacy.json`

## Read This Before Uploading

- `configs/` still includes some older experiment branches for reproducibility.
- If you want an even more minimal GitHub repo, trim to `src/`, the subset of
  `configs/` you still use, selected `scripts/`, and this README.
