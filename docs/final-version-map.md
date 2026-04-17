# Final Version Map

## Canonical Code

- `Experiment/core_code/runtime_repo/`
  - Final runtime/package snapshot to preserve and upload.
- `Experiment/core_code/scd_validation/`
  - Local validation scaffold used by the research workflow.
- `Experiment/core_code/configs/`
  - Screening and aggregation config templates.
- `Experiment/core_code/run_pair_screening.py`
- `Experiment/core_code/run_original_scd_baseline.py`
- `Experiment/core_code/run_remote_*.sh`
- `Experiment/core_code/run_ser_*.sh`

## Removed Old Content

The previously archived `legacy/` content was removed during repository cleanup.
The canonical code line now keeps only the maintained runtime package and active
validation/orchestration scaffold.

## A800 Inspection Summary (2026-04-17)

- Host alias `A800` resolved to machine `gpu010`.
- Login landed in `/home/student_t`, not the previously assumed
  `/home/student/maqianli`.
- `/nas/mql` was unavailable and returned:
  - `ls: 无法访问 '/nas/mql': 主机关闭`
- Candidate SCD directories on A800 included:
  - `/home/student_t/maqianli/scd-repro`
  - `/home/student_t/maqianli/scd-repro-blockwise-20260405`
  - `/home/student_t/maqianli/scd-repro-codex-a800-mistral-instruct-20260323`
- The inspected A800 copies were not newer than the local
  `Experiment/core_code/runtime_repo/` snapshot.

## Implication For GitHub Upload

If the goal is a clean final GitHub repository, the local
`Experiment/core_code/runtime_repo/` should be treated as the source package,
without keeping the previously archived `legacy/` material.
