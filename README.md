# scd-new

Final curated workspace for the `scd-new` Semantic Cache Distillation validation project.

## TL;DR

- Canonical code: `Experiment/core_code/runtime_repo/`
- Canonical method family:
  - `baseline`
  - repaired `reuse-only`
  - strict `patch-only`
  - strict `hybrid` only when separately validated
- Current strongest accepted story: the strict Mistral-family line
- Not fully closed: the original Qwen3.5 similar-pair claim line

This repository is a consolidated research workspace, not a minimal paper-only artifact. It keeps the final code line plus a large amount of analysis history for auditability.

## Current Status

What is final:

- strict runtime semantics
- strict TTFT accounting
- the canonical Mistral result line
- the maintained code snapshot under `Experiment/core_code/runtime_repo/`

What is not final:

- a completed positive reproduction of the original Qwen3.5 headline claim

Reason:

- the intended Qwen3.5 close-pair line is still blocked by hybrid-cache incompatibility with the current SCD runtime assumptions

See:

- `Experiment/analysis/qwen35_hybrid_cache_blocker.md`
- `.pipeline/docs/research_brief.json`

## Canonical Result Line

The project-wide route lock is:

- `Experiment/analysis/transfer_route_lock_20260405.md`

The reverse-pair strict lock is:

- `Experiment/analysis/mistral7_to_mistrallite_final_route_lock_20260409.md`

Anything fallback-heavy, recompute-heavy, grouped/shared-z, or otherwise deprecated by those lock notes should be treated as archived history, not as the default final method line.

## Main Accepted Results

### MistralLite -> Mistral-7B

Full-dev strict no-fallback comparison on `CMRC2018 dev = 3219`:

| method | F1 | EM | TTFT ms |
| --- | ---: | ---: | ---: |
| `baseline` | 0.6911 | 0.4427 | 133.39 |
| `reuse-only` | 0.0109 | 0.0000 | 38.49 |
| `patch-only` | 0.7898 | 0.5769 | 67.37 |

Reading:

- `reuse-only` is fast but effectively broken
- `patch-only` is the only healthy default transfer route
- on this direction, `patch-only` improves both quality and TTFT versus `baseline`

Canonical note:

- `Experiment/analysis/mistrallite_to_mistral7_fulldev_locked_routes_20260405.md`

### MistralLite -> Mistral-7B Strict Hybrid

Accepted strict-stop multi-layer point:

| method | F1 | EM | TTFT ms |
| --- | ---: | ---: | ---: |
| `patch-only` | 0.6285 | 0.285 | 53.93 |
| `hybrid` | 0.6072 | 0.250 | 46.18 |

Reading:

- this is a valid quality-constrained latency trade-off
- it is faster than `patch-only`
- it is not a Pareto win over `patch-only`

Canonical note:

- `Experiment/analysis/mistrallite_to_mistral7_multilayer_strictstop_step15_20260407.md`

### Mistral-7B -> MistralLite

Accepted reverse-pair strict reference:

| method | F1 | EM | TTFT ms |
| --- | ---: | ---: | ---: |
| `baseline` | 0.7829 | 0.6000 | 122.50 |
| `reuse-only` | 0.0212 | 0.0000 | 33.60 |
| `patch-only` | 0.7528 | 0.5000 | 75.94 |

Reading:

- `reuse-only` remains unusable
- `patch-only` preserves most baseline quality and materially reduces TTFT
- this reverse direction should be cited only through the locked strict artifact

Canonical note:

- `Experiment/analysis/mistral7_to_mistrallite_final_route_lock_20260409.md`

## Qwen Line

The Qwen line is not the finished positive result of this repository.

Current state:

- `Experiment/analysis/claim_gap_summary.csv` contains a smoke `fail`
- the runnable Qwen3-1.7B near-pair line does not recover the original claim under the strict no-fallback reading
- the intended Qwen3.5 close-pair line remains blocked by cache-architecture mismatch

Use the Qwen material as:

- failure analysis
- protocol development
- claim-gap framing
- motivation for the stricter Mistral story

Do not present the Qwen line as a completed positive reproduction.

## Code Entry Points

Start here for the maintained final code line:

- `Experiment/core_code/runtime_repo/`
- `Experiment/core_code/runtime_repo/README.md`
- `Experiment/core_code/runtime_repo/src/scd_repro/`
- `Experiment/core_code/runtime_repo/configs/`
- `Experiment/core_code/runtime_repo/scripts/`

Supporting scaffold:

- `Experiment/core_code/README.md`
- `Experiment/core_code/scd_validation/`
- `Experiment/core_code/configs/`
- `docs/final-version-map.md`

## What Is Archived

These should not be treated as canonical unless you are explicitly doing debugging or historical audit:

- `Experiment/core_code/legacy/`
- `Experiment/core_code/runtime_repo/legacy/`
- grouped/shared-z reuse results
- fallback-dependent routes
- `reuse_patch_recompute`
- older patch families superseded by `xnorm_nativekv_shared_residual_mlp_v1`

## Reading Order

If you are returning to the project after time away:

1. `docs/final-version-map.md`
2. `Experiment/analysis/transfer_route_lock_20260405.md`
3. `Experiment/analysis/mistrallite_to_mistral7_fulldev_locked_routes_20260405.md`
4. `Experiment/analysis/mistrallite_to_mistral7_multilayer_strictstop_step15_20260407.md`
5. `Experiment/analysis/mistral7_to_mistrallite_final_route_lock_20260409.md`
6. `Experiment/core_code/runtime_repo/README.md`

## Notes

- Many research notes still carry historical absolute paths from the original local workspace; treat them as provenance rather than current-path documentation.
- During the April 17, 2026 audit, A800 did not contain a newer canonical runtime snapshot than the local `runtime_repo`, and `/nas/mql` was unavailable, so this repository should be treated as the primary final source.
