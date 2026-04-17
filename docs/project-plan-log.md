# Project Plan Log

## 2026-04-17 Session
- Scope: Initial inspection of the scd-new project workspace.
- Notes: Created by Codex per global workspace rule for persistent planning records.

## 2026-04-17 Repository Consolidation Update
- Scope: Reorganize the workspace around a single final code line before GitHub upload.
- Goal: Keep the canonical runtime and validation scaffold visible, archive mirrors/prototypes/backups in place, and document which version should be treated as final.
- Design choice: Preserve `Experiment/core_code/runtime_repo/` as the canonical runtime package instead of lifting another directory to the root. Archive `ser_mirror`, `remote_work`, top-level patch prototypes, runtime `.bak` files, and `egg-info` metadata under explicit `legacy/` directories rather than deleting them.
- Remote validation note: A read-only A800 inspection found no newer SCD code than the local runtime snapshot. The remote host was reachable as `gpu010`, but `/nas/mql` was unavailable during inspection.
