# SCD-ICML Current Open-Source Release

This branch contains the current checked SCD-ICML reproduction and release
candidate for the MistralLite -> Mistral-7B path.

The release-facing runtime is:

- `Experiment/core_code/runtime_repo/`

The paper-facing evidence boundary is:

- Non-Qwen only. Qwen32B is explicitly out of scope for this branch.
- Qwen3-1.7B -> Qwen3-1.7B-gray artifacts are retained only as additional
  risk/portability evidence, not as release-facing speedup claims.
- Dataset-specific routes are used instead of one universal route:
  - CMRC2018: `Reuse19/Patch13`
  - HotpotQA-compressed full7405: `Reuse06/Patch26`
  - WikiText-2: `Reuse20/Patch12`
- F1/EM are the primary QA quality metrics. NLL/PPL are auxiliary risk signals.
- Historical camera-ready numbers are not claimed byte-for-byte. Current
  generated tables and figures are used when they preserve the checked trend and
  do not contradict the current evidence boundary.

## Key Documents

- `Experiment/core_code/runtime_repo/REPRODUCING_SCD_ICML.md`
- `Experiment/core_code/runtime_repo/OPEN_SOURCE_RELEASE_MANIFEST.md`
- `docs/scd-vllm-open-release-status-2026-06-17.md`
- `docs/scd-icml-final-nonqwen-completion-audit-2026-06-11.md`
- `docs/scd-icml-full-paper-reproduction-matrix-2026-06-11.md`
- `docs/scd-icml-reproduction-coverage-audit-2026-06-11.md`
- `Experiment/results/paper_artifact_bundle_final_neighboring_checked_20260611_001/`
- `Experiment/results/qwen17_gray_full_strict_microsharded_20260617/`

## Verification Snapshot

The final local audit for this release candidate reports:

- Runtime release readiness passed.
- Unit test suite passed locally.
- Final paper artifact bundle status: `ok`.
- Bundle copied files: `51`.
- Bundle missing files: `0`.
- Paper artifact check errors: `[]`.

See:

- `Experiment/results/paper_artifact_bundle_final_neighboring_checked_20260611_001/finalize_report.json`
- `Experiment/results/paper_artifact_bundle_final_neighboring_checked_20260611_001/paper_artifact_check_report.json`

## Current Claim Boundary

The supported claim is narrower than the older camera-ready text:

- Supported: current MistralLite -> Mistral-7B non-Qwen release path with
  dataset-specific quality gates and checked release artifacts.
- Current vLLM demo route: `Patch20/Reuse12`, because it passes the
  reconstructed SCD injection greedy checks in the checked vLLM 0.18.0
  environment.
- Not supported on this branch: Qwen32B, one universal `k=6` route across all
  datasets, full-context HotpotQA quality preservation, or exact byte-identical
  regeneration of all historical figure sweeps.
