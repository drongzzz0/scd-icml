# Experiment Log

## 2026-04-17 Session
- Purpose: Initialize persistent experiment record for the scd-new workspace.
- Notes: Created by Codex per global workspace rule before project work proceeds.

## 2026-04-17 Remote Environment Audit
- Purpose: Check whether A800 contains a newer SCD runtime snapshot or accessible run data before consolidating the local final version.
- Method: Read-only SSH inspection of host identity, home directory layout, candidate SCD project directories, and NAS availability.
- Environment: Remote host alias `A800`, resolved machine `gpu010`, login directory `/home/student_t`.
- Findings:
  - `/nas/mql` was unavailable and returned `ls: 无法访问 '/nas/mql': 主机关闭`.
  - Candidate directories on local disk included `scd-repro`, `scd-repro-blockwise-20260405`, and `scd-repro-codex-a800-mistral-instruct-20260323`.
  - The inspected A800 copies were not newer than the local `Experiment/core_code/runtime_repo/` snapshot.
- Conclusion: Treat the local workspace as the primary final-code source for GitHub preparation. Treat NAS-backed artifacts as temporarily unavailable.
