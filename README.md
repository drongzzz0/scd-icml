# SCD-ICML Current Open-Source Release

This branch contains the current checked SCD-ICML reproduction and release
candidate. The primary release path is MistralLite -> Mistral-7B; a
strict-runtime Qwen2.5-32B profile (method `A1`) is additionally supported as
documented below.

The release-facing runtime is:

- `Experiment/core_code/runtime_repo/`

The paper-facing evidence boundary is:

- Primary release path: MistralLite -> Mistral-7B (non-Qwen).
- Qwen2.5-32B Base -> Instruct is supported as the strict-runtime `A1` profile
  (see `docs/A1_METHOD.md`): CMRC2018 full dev (n=3219), no fallback and no
  recompute, F1 retention 97.9%, TTFT 231.1 ms -> 70.5 ms (3.28x).
- Qwen3-1.7B -> Qwen3-1.7B-gray artifacts are retained only as additional
  risk/portability evidence, not as release-facing speedup claims.
- Dataset-specific routes are used instead of one universal route:
  - CMRC2018 (Mistral pair): `Reuse19/Patch13`
  - HotpotQA-compressed full7405: `Reuse06/Patch26`
  - WikiText-2: `Reuse20/Patch12`
  - CMRC2018 (Qwen32B A1): reuse layers `0-42, 47-59`, patch layers
    `43-46, 60-63` (k=8), stop-token logit bias `3.0` from step `2`.
- F1/EM are the primary QA quality metrics. NLL/PPL are auxiliary risk signals.
- Historical camera-ready numbers are not claimed byte-for-byte. Current
  generated tables and figures are used when they preserve the checked trend and
  do not contradict the current evidence boundary.

## Key Documents

- `Experiment/core_code/runtime_repo/REPRODUCING_SCD_ICML.md`
- `Experiment/core_code/runtime_repo/OPEN_SOURCE_RELEASE_MANIFEST.md`
- `docs/A1_METHOD.md`
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

## Paper Alignment

Relationship between this release and the ICML camera-ready paper:

- Trend-level conclusions of the paper are fully reproduced and strengthened
  under the current strict runtime (no fallback, no recompute):
  - F1 retention 96.9%-97.9% (paper reported 95.3%-96.7%).
  - TTFT speedup 2.42x-3.28x (paper reported 1.98x-2.65x).
  - Raw KV and 4-bit quantized KV collapse; SCD dominates the
    quality-latency Pareto frontier.
- Absolute F1/TTFT/PPL values are not comparable cell-by-cell with the
  camera-ready tables because the evaluation protocol changed (prompt style,
  max length, decoding settings, strict no-fallback runtime).
- The strict runtime removes implicit fallback present in the original
  evaluation path. Two consequences:
  - All-layer Reuse-only collapses (F1 about 0.01 on CMRC2018), which
    strengthens the paper's argument that Patch is necessary.
  - The optimal patch budget is dataset-dependent (k=8-26, selected by
    per-dataset quality gates) rather than the historical universal k=6.

## Current Claim Boundary

The supported claim is narrower than the older camera-ready text:

- Supported: current MistralLite -> Mistral-7B release path with
  dataset-specific quality gates and checked release artifacts.
- Supported: Qwen2.5-32B Base -> Instruct strict-runtime `A1` profile on
  CMRC2018 full dev as defined in `docs/A1_METHOD.md`.
- Current vLLM demo route: `Patch20/Reuse12`, because it passes the
  reconstructed SCD injection greedy checks in the checked vLLM 0.18.0
  environment.
- Not supported on this branch: a vLLM-served Qwen32B profile, one universal
  `k=6` route across all datasets, full-context HotpotQA quality preservation,
  or exact byte-identical regeneration of all historical figure sweeps.

## Historical Evidence

The pre-A1 curated workspace (including the original per-sample strict-eval
artifacts under `Experiment/analysis/remote_artifacts/` referenced by the
route-lock documents) is preserved on the `legacy-main` branch.
