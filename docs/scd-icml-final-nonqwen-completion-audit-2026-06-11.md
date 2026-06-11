# SCD-ICML Final Non-Qwen Completion Audit - 2026-06-11

## Scope

User objective: reproduce the full SCD-ICML paper results except the Qwen32B
portion, finish the open-source code, and ensure reproduced numbers are
trend-compatible with the paper claims without contradicting the main
conclusions.

This audit treats Qwen32B as explicitly out of scope.  It treats exact
byte-for-byte historical reruns as unnecessary when a current generated table or
figure preserves the paper-facing trend and claim boundary.

## Requirement Audit

| Requirement | Evidence | Status |
|---|---|---|
| Open-source runtime exists for the current MistralLite to Mistral-7B release path | `Experiment/core_code/runtime_repo/`; `release_preview/scd_icml_runtime_release/`; local and remote `scripts/check_release_readiness.py` pass | Complete |
| CMRC2018 main mixed route is reproduced | `reuse19_patch13`: Full F1/EM `0.657307/0.365000`, SCD F1/EM `0.636821/0.340000`, F1 retention `0.9688`; paper Table `current_cmrc_selected_route`; final bundle `tables/mixed_route_boundary_table.*` | Complete |
| Maximum-Reuse quality boundary is reproduced | Paper Table `current_reuse_boundary`; final bundle `tables/mixed_route_boundary_table.*` and `figures/route_curve_cmrc2018.svg`; Reuse20 borderline, Reuse21+ fail | Complete |
| Patch-only and Reuse-only anchor behavior is reproduced | Paper Table `patchonly_anchors`; final bundle `tables/patch_anchor_table.{csv,md,tex}`; Patch-only quality recovers on CMRC2018, HotpotQA-compressed, and WikiText-2, while Reuse-only collapses on QA | Complete |
| HotpotQA-compressed full evaluation is reproduced without train/eval leakage as claim-facing result | Full7405 selector chooses `reuse06_patch26`; Full/SCD F1 `0.291855/0.304212`; no-train-overlap selector also chooses `reuse06_patch26` with Full/SCD F1 `0.292006/0.304488`; paper Table `current_non_cmrc_diagnostics` | Complete |
| WikiText-2 uses the correct quality metric and route gate | Paper Table `current_non_cmrc_diagnostics`; final bundle route table; selected `reuse20_patch12` has PPL `8.329555` versus Full PPL `7.584029`, within the 10% relative PPL gate | Complete |
| Neighboring compression appendix table is reproduced | Paper Table `neighboring_baselines`; final bundle `tables/neighboring_baselines.{csv,md,tex}` and `manifests/neighboring_baselines_manifest.json`; manifest warnings are empty; SCD retention/speedup `97.8%/2.47x` while adapted baselines retain `0.3%-2.8%` | Complete |
| Layer/error/profile figures are generated from current artifacts | Paper figures `layerwise_xnorm`, `layer_selection_viz`, and `tsne_kv_align`; final bundle `figures/layerwise_xnorm_replacement.*`, `figures/layer_selection_gain_profile.*`, `figures/patch_budget_curve.*`, `figures/stack_viz.*` | Complete |
| Route, latency-quality, patch-budget, bandwidth, and TTFT replacement artifacts exist | Final bundle `figures/route_curve_*.svg`, `figures/latency_quality_pareto_*.svg`, `figures/bandwidth_sensitivity.svg`; final bundle `tables/route_curve_*.{csv,md}`, `tables/latency_quality_pareto.{csv,md}`, `tables/bandwidth_sensitivity.{csv,md}`, `tables/ttft_breakdown_table.{csv,md,tex}`, `tables/patch_gain_curve.{csv,md}` | Complete |
| Offline calibration cost table is represented without fabricated timings | Paper Table `offline_cost`; final bundle `tables/offline_cost_table.{csv,md}` and `figures/offline_cost_wall_time.svg`; paper-mixed Patch training row is explicitly `no time fields` instead of imputed | Complete |
| Paper source and PDF are current and internally consistent | `Publication/paper/scd-icml-current-safe-paper-2026-06-10.tex`; compiled PDF `Publication/paper/scd-icml-current-safe-paper-2026-06-10.pdf`, 16 pages, `886613` bytes; final LaTeX log has no undefined references | Complete |
| Final paper artifact bundle is checked | `Experiment/results/paper_artifact_bundle_final_neighboring_checked_20260611_001/finalize_report.json`: `status=ok`, `bundle_status=ok`, `copied=51`, `missing=0`, `paper_check=true`; `paper_artifact_check_report.json` has `errors=[]` and requires `Reuse06/Patch26`, `tab:patchonly_anchors`, and `tab:neighboring_baselines` | Complete |
| Local release verification passes | `PYTHONPATH=src python -m unittest discover -s tests -v`: 45 tests OK; `python scripts/check_release_readiness.py`: passed; `python scripts/build_release_tree.py --out release_preview/scd_icml_runtime_release --force`: preview readiness passed | Complete |
| Remote runtime copy on `ser` is synced and passes key checks | Synced runtime scripts/tests/docs to `/disk_n/c101-3-mql/projects/SCD-ICML/repro_current_from_codex_20260607-154030/runtime_repo`; remote release readiness passed; remote targeted paper-bundle tests passed | Complete |
| Qwen32B is excluded, not silently claimed | Paper and docs explicitly state Qwen32B is outside the current release-safety audit and requested scope | Complete |

## Final Boundary

The remaining difference from a literal historical rerun is that exact old
patch-budget/rank sweeps are represented by current generated replacement
figures and tables rather than by byte-identical historical sweeps.  Under the
user's stated criterion, this is not a blocking gap because the results are
trend-compatible, do not contradict the main SCD conclusion, and are included
in the checked final artifact bundle.

Therefore, the non-Qwen SCD-ICML reproduction and open-source release state are
complete under the stated objective.
