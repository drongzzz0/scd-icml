# Project Plan Log

## 2026-04-17 Session
- Scope: Initial inspection of the scd-new project workspace.
- Notes: Created by Codex per global workspace rule for persistent planning records.

## 2026-04-17 Repository Consolidation Update
- Scope: Reorganize the workspace around a single final code line before GitHub upload.
- Goal: Keep the canonical runtime and validation scaffold visible, archive mirrors/prototypes/backups in place, and document which version should be treated as final.
- Design choice: Preserve `Experiment/core_code/runtime_repo/` as the canonical runtime package instead of lifting another directory to the root. Archive `ser_mirror`, `remote_work`, top-level patch prototypes, runtime `.bak` files, and `egg-info` metadata under explicit `legacy/` directories rather than deleting them.
- Remote validation note: A read-only A800 inspection found no newer SCD code than the local runtime snapshot. The remote host was reachable as `gpu010`, but `/nas/mql` was unavailable during inspection.

## 2026-04-17 Repository Presentation Update
- Scope: Add a root-level repository README that reflects the actual final accepted project story.
- Goal: Make the GitHub/front-page view immediately answer four questions: what code is canonical, which results are accepted, which lines are still blocked, and which historical artifacts must not be used as default evidence.
- Design choice: Present the repository as a code-level final version with a locked Mistral strict-result line, while explicitly stating that the original Qwen3.5 claim line is not yet a completed positive reproduction.

## 2026-04-17 GitHub README Compression Update
- Scope: Rework the new root README into a shorter GitHub-first landing page.
- Goal: Let a reader understand the repository within about 30 seconds by leading with `TL;DR`, final status, canonical result tables, code entry points, and archived-scope boundaries.
- Design choice: Keep the accepted result tables and final/unfinished boundary explicit, while removing most narrative phrasing that made the README read like an internal note instead of a repository homepage.

## 2026-04-17 GitHub README Chinese Localization
- Scope: Convert the root-level GitHub landing README from English to Simplified Chinese.
- Goal: Make the repository homepage align with the project's current primary communication language while preserving the same final/unfinished boundary and canonical result story.
- Design choice: Keep the existing README structure and result hierarchy stable, and translate the page into concise Chinese rather than redesigning the information architecture again.
