# SCD-ICML Reproduction Coverage Audit - 2026-06-11

This audit maps the current paper-facing tables and figures to reproducible
code/results in the workspace. Qwen32B is excluded by the current scope.

## Current Evidence Status

| Artifact | Paper location | Current evidence | Status | Action |
|---|---|---|---|---|
| CMRC selected-route table | `Publication/paper/scd-icml-current-safe-paper-2026-06-10.tex`, Table `current_cmrc_selected_route` | `Experiment/results/paper_artifact_bundle_final_neighboring_checked_20260611_001/tables/mixed_route_boundary_table.{csv,md,tex}`; remote CMRC selector report | Covered in current-safe paper | Closed |
| CMRC maximum-Reuse boundary | Table `current_reuse_boundary` | `Experiment/results/paper_artifact_bundle_final_neighboring_checked_20260611_001/tables/mixed_route_boundary_table.{csv,md,tex}` and `figures/route_curve_cmrc2018.svg` | Covered in current-safe paper and final bundle | Closed |
| Patch-only anchors across datasets | Table `patchonly_anchors` | `Experiment/results/paper_artifact_bundle_final_neighboring_checked_20260611_001/tables/patch_anchor_table.{csv,md,tex}` | Covered and wired into current-safe paper | Closed |
| HotpotQA-compressed 200-sample boundary | Represented in the current-safe paper as a screen-only boundary row | `mixed_route_boundary_table.md`, HotpotQA-compressed rows; selector report selected `reuse08_patch24` under F1 drop <= 0.03 | Covered for screen only | Superseded by full7405 for main table evidence |
| HotpotQA-compressed full7405 | Current-safe paper Table `current_non_cmrc_diagnostics` | `Experiment/results/scd_icml_result_tables_full7405_boundary_20260611_001`; `Experiment/results/paper_artifact_bundle_final_neighboring_checked_20260611_001`; selector chose `reuse06_patch26`, while `reuse07_patch25` and `reuse08_patch24` failed the full7405 F1-drop gate | Covered for the final accepted route | Closed; no-train-overlap selector also chooses `reuse06_patch26` |
| WikiText-2 boundary | Current-safe paper Table `current_non_cmrc_diagnostics` | `mixed_route_boundary_table.md`, `route_curve_wikitext2.*`; selector report selected `reuse20_patch12` under PPL +10% gate | Covered | Keep as dataset-specific PPL-gated boundary, not an F1/EM claim |
| Route-curve figures | Final bundle figures/tables, not all inserted into paper text | `Experiment/results/paper_artifact_bundle_final_neighboring_checked_20260611_001/figures/route_curve_*.svg`; `tables/route_curve_*.{csv,md}` | Covered as release artifacts | Closed as generated release evidence; insert into paper only if a longer appendix is desired |
| Patch budget / ablation | Figure `layer_selection_viz` plus final bundle table/figure | `Experiment/results/paper_artifact_bundle_final_neighboring_checked_20260611_001/tables/patch_gain_curve.{csv,md}`; `figures/patch_budget_curve.*` | Covered as current replacement | Closed under trend-compatible replacement criterion |
| Latency-quality / Pareto figures | Final bundle latency-quality/Pareto and bandwidth artifacts | `Experiment/results/paper_artifact_bundle_final_neighboring_checked_20260611_001/figures/latency_quality_pareto_*.svg`; `figures/bandwidth_sensitivity.svg`; `tables/latency_quality_pareto.{csv,md}`; `tables/bandwidth_sensitivity.{csv,md}` | Covered as current replacement | Closed under trend-compatible replacement criterion |
| TTFT breakdown table | Replacement for latency breakdown rows | `Experiment/results/paper_artifact_bundle_final_neighboring_checked_20260611_001/tables/ttft_breakdown_table.{csv,md,tex}` | Covered for final completed rows | Closed |
| Figure `fig1.pdf` / `fig2.pdf` | Conceptual overview figures | Static PDFs in `Publication/paper/current_safe_build/` | Present, not empirical | Keep as conceptual assets; no numeric reproduction required |
| Figure `patch_error1.pdf` | Layer-wise error propagation | `scripts/build_scd_icml_analysis_figures.py`; `Experiment/results/profile_analysis_figures_20260611_002/layer_error_profile.pdf`; `Publication/paper/generated_figures/layerwise_xnorm_replacement.pdf` | Covered and wired into current-safe paper | Regenerate after final route/profile changes if needed |
| Figure `layer_selection.pdf` | Layer selection policy | `scripts/build_scd_icml_analysis_figures.py`; `layer_gain_profile.pdf`; `patch_gain_curve.pdf`; `Publication/paper/generated_figures/` | Covered as current replacement panels | Exact historical interaction panel remains historical unless rerun |
| Figure `stack_viz.pdf` | t-SNE KV alignment | `Experiment/results/stack_viz_current_extraction_20260611_001/`; `scripts/extract_stack_viz_inputs.py`; `scripts/build_stack_viz.py`; `Publication/paper/generated_figures/stack_viz.pdf` | Covered and wired into current-safe paper | Use current extraction bundle; archived bundle remains provenance only |
| Offline calibration cost table | Table `offline_cost` | `Experiment/results/offline_cost_evidence_20260611_001/`; `scripts/build_offline_cost_table.py`; raw remote `/usr/bin/time -v` logs | Covered for available measured stages | Use generated table values; do not claim the empty paper-mixed `train_patch` time log |
| Neighboring compression baselines | Original appendix Table `neighboring_baselines` | `scripts/run_svdllm_style_kv_cmrc.py`; `scripts/run_slicegpt_style_kv_cmrc.py`; `scripts/run_tensorllm_style_kv_cmrc.py`; `scripts/build_neighboring_baseline_table.py`; `Experiment/results/neighboring_baselines_cmrc200_maxlen1024_20260611_001/` | Covered by same-protocol CMRC200 rerun | Full CMRC200 `max_length=1024` SVD-LLM-style, SliceGPT-style, and TensorLLM-style reruns completed and were consolidated with the selected SCD summary under `--require-compatible-protocol`; manifest warnings are empty |
| Paper artifact bundle | Release artifact bundle | `Experiment/results/paper_artifact_bundle_final_neighboring_checked_20260611_001/`; `scripts/build_paper_artifact_bundle.py`; `scripts/finalize_scd_icml_full7405_bundle.py`; manifest `status=ok`, `copied=51`, `missing=0`, `paper_artifact_check_report.json` errors `[]`, and `finalize_report.json` enforces selected paper text `Reuse06/Patch26` plus Table `neighboring_baselines` | Covered for final full7405 and neighboring artifacts | Closed for the current non-Qwen release scope |

## Important Protocol Correction

The older full-context HotpotQA diagnostic rows are not the paper-comparable
HotpotQA result path for the restored GitHub code. The current-safe paper now
uses them only as diagnostic context and keeps the comparable path as:

- Dataset: `hotpotqa_compressed`.
- Max length: `512`.
- Full validation size: `7405`.
- Train/eval split: disjoint for the current Reuse/Patch bank training and
  full-dev evaluation checks.

The current final gate is therefore not the old full-context HotpotQA table.
The full7405 compressed-protocol boundary has now completed for
`reuse04_patch28`, `reuse05_patch27`, `reuse06_patch26`, `reuse07_patch25`, and
`reuse08_patch24`.  The accepted maximum-reuse route is `reuse06_patch26`:
unfiltered Full/SCD F1 is `0.291855/0.304212`, while `reuse07_patch25` fails
with F1 drop `0.034595`.  After excluding the 100 training-overlap qids, the
selector remains stable at `reuse06_patch26` with Full/SCD F1
`0.292006/0.304488`.

## Open-Source Code Coverage

The release runtime currently covers:

- strict CMRC/HotpotQA-compressed/WikiText-2 evaluation;
- Full Consumer, Reuse-only, Patch-only, and mixed Reuse+Patch routes;
- quality-gated route selection using QA F1/EM retention, QA absolute F1/EM
  drop, or WikiText-2 PPL relative increase;
- result table generation to CSV/Markdown/LaTeX and route-curve generation from
  completed JSON artifacts;
- profile-based analysis figure generation from completed profile JSON;
- stack/t-SNE input extraction from pair artifacts plus Reuse checkpoints, and
  plotting from generated KV-distance and embedding CSV/meta artifacts;
- offline cost table generation from raw `/usr/bin/time -v` logs;
- paper artifact bundle assembly from generated tables/figures/cost artifacts
  plus the protocol-checked neighboring-baseline appendix table;
- final post-full7405 paper bundle assembly plus paper artifact gating, with a
  default guard that rejects result tables whose mixed-route sources do not
  include `full7405`;
- final paper text gating that requires a HotpotQA full7405 selected-route
  statement, requires the restored neighboring-baseline appendix table, and
  rejects stale pending-status or old unsupported-neighboring text;
- appendix neighboring-baseline rerun scripts for SVD-LLM-style, SliceGPT-style,
  and TensorLLM-style adapted KV transfer, plus a consolidator for the
  appendix table that records protocol warnings;
- non-destructive release preview and readiness checks.

Non-blocking historical-literal differences:

1. Offline calibration cost has a source artifact now, but the paper-mixed
   `train_patch` time log is empty and must not be used as a measured value.
2. Exact historical patch-budget/rank sweeps are not literal byte-for-byte
   reruns, but current generated replacements cover the release-facing claims
   under the user's trend-compatible criterion.
3. Qwen32B is intentionally excluded from this audit scope.

## Completion Rule

The current non-Qwen completion gate is satisfied only after a fresh final
verification pass confirms:

- local unit tests and release readiness pass;
- the release preview rebuild passes readiness;
- the updated paper source compiles;
- `paper_artifact_bundle_final_neighboring_checked_20260611_001`
  remains checked with `errors=[]` and selected route text `Reuse06/Patch26`.
