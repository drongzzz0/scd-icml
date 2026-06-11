# SCD-ICML Full Paper Reproduction Matrix - 2026-06-11

## Current Answer

The current non-Qwen release-facing result set is now closed for the verified
MistralLite$\to$Mistral-7B scope, with dataset-specific route gates:

- Patch-only anchors are recovered for CMRC2018, HotpotQA-compressed, and
  WikiText-2 under the current runtime.
- CMRC2018 supports a high-Reuse mixed route: `reuse19_patch13` is
  conservative and `reuse20_patch12` is borderline.
- WikiText-2 supports `reuse20_patch12` under a roughly 10% relative PPL
  increase gate.
- HotpotQA-compressed is the limiting benchmark. The 200-sample screen suggested
  `reuse08_patch24`, but full7405 validation rejects both `reuse07_patch25` and
  `reuse08_patch24`; the maximum passing route is `reuse06_patch26`.
- HotpotQA-compressed full7405 is complete: Full F1/EM
  `0.291855/0.206077`, selected SCD F1/EM `0.304212/0.221742`.

## Result Evidence

| Evidence layer | Artifact | Current status |
|---|---|---|
| Current runtime release | `Experiment/core_code/runtime_repo` | Local py_compile, 45 unit tests, release readiness, and release preview checks passed on 2026-06-11 after adding the full7405 postprocess/finalization wrappers and neighboring-baseline protocol checks; release readiness validates shell wrapper LF/shebang/strict-mode and runs `bash -n` when available. The latest neighboring hard-gate scripts were synced to `ser`; remote release readiness and the neighboring-table targeted tests passed. |
| Patch-only anchor | `/disk_n/c101-3-mql/projects/SCD-ICML/runs/patchonly_anchor_cmrc200_20260611_001`; `/disk_n/c101-3-mql/projects/SCD-ICML/runs/patchonly_anchor_hotpot_wiki200_20260611_002` | Recovered on all three benchmark protocols |
| CMRC mixed prefix curve | `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_snapshot_bank_prefix_curve_cmrc200_20260611_001` | Completed through `reuse24_patch08` |
| HotpotQA-compressed 200 curve | `scd_icml_boundary_hotpotqa_compressed_low200_20260611_001`; `scd_icml_boundary_hotpotqa_compressed_mid200_20260611_001` | Completed through `reuse12_patch20`; cliff starts before `reuse09_patch23` |
| WikiText-2 mixed boundary | `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_boundary_wikitext2_200_20260611_001` | Completed through `reuse20_patch12` |
| HotpotQA-compressed full7405 | `scd_icml_hotpotqa_compressed_full7405_reuse04_20260611_001`; `scd_icml_hotpotqa_compressed_full7405_reuse05_07_20260611_001`; `scd_icml_hotpotqa_compressed_full7405_reuse08_20260611_001` | Completed. Full7405 selector chooses `reuse06_patch26`; `reuse07_patch25` and `reuse08_patch24` fail the full7405 F1-drop gate. |
| Local partial table/figure bundle | `Experiment/results/scd_icml_result_tables_partial_20260611_005` | Local copy of current partial Markdown/CSV/LaTeX/SVG outputs, now including TTFT breakdown, latency-quality/Pareto artifacts, and `bandwidth_sensitivity.csv/md/svg`; excludes full7405 |
| Current paper artifact bundle | `Experiment/results/paper_artifact_bundle_final_neighboring_checked_20260611_001` | Final boundary-refined full7405 and neighboring-baseline bundle. `finalize_report.json` reports `status=ok`, `bundle_status=ok`, selected paper text `Reuse06/Patch26`, `paper_check=true`, `copied=51`, and `missing=0`; `paper_artifact_check_report.json` has `errors=[]` and now requires Table `neighboring_baselines`. |
| Neighboring-baseline CMRC200 rerun | `Experiment/results/neighboring_baselines_cmrc200_maxlen1024_20260611_001` | Full `LIMIT=200`, `max_length=1024` current-runtime rerun completed for SVD-LLM-style, SliceGPT-style, and TensorLLM-style adapted KV transfer, then consolidated with the selected Reuse19/Patch13 SCD summary under `--require-compatible-protocol`. Manifest warnings are empty. The table reports SVD-LLM-style `2.4%` retention / `0.30x`, SliceGPT-style `2.8%` / `1.43x`, TensorLLM-style `0.3%` / `0.08x`, and SCD `97.8%` / `2.47x`. |

## Machine-Generated Route Selection Reports

| Dataset | Gate | Selected route | Report |
|---|---|---|---|
| CMRC2018 | F1 retention `>= 0.95` | `reuse19_patch13` | `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_snapshot_bank_prefix_curve_cmrc200_20260611_001/lite_to_7b/conditional_report/cmrc_f1_retention095_selection.json` |
| HotpotQA-compressed 200-screen | absolute F1 drop `<= 0.03` | `reuse08_patch24` | `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_boundary_hotpotqa_compressed_mid200_20260611_001/lite_to_7b/conditional_report/hotpotqa_f1_drop003_selection.json` |
| HotpotQA-compressed full7405 | absolute F1 drop `<= 0.03` | `reuse06_patch26` | `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_full7405_boundary_postprocess_20260611_001/hotpotqa_full7405_f1_drop003_selection.json` |
| WikiText-2 | PPL relative increase `<= 0.10` | `reuse20_patch12` | `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_boundary_wikitext2_200_20260611_001/lite_to_7b/conditional_report/wikitext2_ppl_gate_selection.json` |

## Patch-only Anchor

| Dataset | Full metric | Reuse-only metric | Patch-only metric | TTFT trend |
|---|---:|---:|---:|---|
| CMRC2018 | F1 `0.657307`, EM `0.365000` | F1 `0.012848`, EM `0.000000` | F1 `0.774237`, EM `0.515000` | `100.20 -> 52.32 ms` |
| HotpotQA-compressed | F1 `0.330552`, EM `0.250000` | F1 `0.007903`, EM `0.000000` | F1 `0.372672`, EM `0.305000` | `78.94 -> 44.77 ms` |
| WikiText-2 | PPL `7.584029` | PPL `13.631409` | PPL `7.913751` | `62.21 -> 40.66 ms` |

## Mixed Reuse+Patch Boundary

| Dataset | Route | Full reference | SCD metric | Retention / delta | Status |
|---|---|---:|---:|---:|---|
| CMRC2018 | `reuse19_patch13` | F1/EM `0.657307/0.365000` | F1/EM `0.636821/0.340000` | F1 retention `0.9688` | Pass, conservative max |
| CMRC2018 | `reuse20_patch12` | F1/EM `0.657307/0.365000` | F1/EM `0.623676/0.360000` | F1 retention `0.9488` | Borderline |
| CMRC2018 | `reuse21_patch11` | F1/EM `0.657307/0.365000` | F1/EM `0.571623/0.335000` | F1 retention `0.8696` | Fail |
| HotpotQA-compressed 200-screen | `reuse04_patch28` | F1/EM `0.330552/0.250000` | F1/EM `0.377345/0.290000` | F1 retention `1.1416` | Pass on 200-sample screen |
| HotpotQA-compressed 200-screen | `reuse08_patch24` | F1/EM `0.330552/0.250000` | F1/EM `0.305450/0.235000` | F1 delta `-0.025102` | 200-screen candidate, rejected by full7405 |
| HotpotQA-compressed full7405 | `reuse04_patch28` | F1/EM `0.291855/0.206077` | F1/EM `0.328284/0.242944` | F1 retention `1.1248` | Pass |
| HotpotQA-compressed full7405 | `reuse05_patch27` | F1/EM `0.291855/0.206077` | F1/EM `0.322962/0.237812` | F1 retention `1.1066` | Pass |
| HotpotQA-compressed full7405 | `reuse06_patch26` | F1/EM `0.291855/0.206077` | F1/EM `0.304212/0.221742` | F1 retention `1.0423` | Pass, selected full7405 boundary |
| HotpotQA-compressed full7405 | `reuse07_patch25` | F1/EM `0.291855/0.206077` | F1/EM `0.257260/0.181769` | F1 drop `0.034595` | Fail full7405 F1-drop gate |
| HotpotQA-compressed full7405 | `reuse08_patch24` | F1/EM `0.291855/0.206077` | F1/EM `0.242740/0.169885` | F1 drop `0.049115` | Fail full7405 F1-drop gate |
| HotpotQA-compressed | `reuse09_patch23` | F1/EM `0.330552/0.250000` | F1/EM `0.249338/0.190000` | F1 retention `0.7543` | Fail |
| HotpotQA-compressed | `reuse10_patch22` | F1/EM `0.330552/0.250000` | F1/EM `0.250413/0.200000` | F1 retention `0.7576` | Fail |
| HotpotQA-compressed | `reuse11_patch21` | F1/EM `0.330552/0.250000` | F1/EM `0.242163/0.190000` | F1 retention `0.7326` | Fail |
| HotpotQA-compressed | `reuse12_patch20` | F1/EM `0.330552/0.250000` | F1/EM `0.250531/0.200000` | F1 retention `0.7579` | Fail |
| WikiText-2 | `reuse16_patch16` | PPL `7.584029` | PPL `8.236739` | PPL `+8.61%` | Pass under 10% PPL gate |
| WikiText-2 | `reuse19_patch13` | PPL `7.584029` | PPL `8.335916` | PPL `+9.91%` | Pass under 10% PPL gate |
| WikiText-2 | `reuse20_patch12` | PPL `7.584029` | PPL `8.329555` | PPL `+9.83%` | Pass under 10% PPL gate |

## Paper Artifact Status

| Paper artifact | Non-Qwen reproduction status | Current action |
|---|---|---|
| Table `main_results_both`, Mistral slice | Closed for current release-facing evidence with dataset-specific routes. CMRC uses Reuse19/Patch13; HotpotQA-compressed full7405 uses Reuse06/Patch26; WikiText-2 uses Reuse20/Patch12. | Use generated full7405 boundary tables and explicitly avoid claiming one universal route. |
| Table `main_results_both`, Qwen-32B slice | Excluded by user scope. | Keep explicitly excluded. |
| Table `quality_main` | Closed for current release-facing evidence under dataset-specific gates. | Use `paper_artifact_bundle_final_neighboring_checked_20260611_001`. |
| Table `patch_ablation` | Covered as a current replacement by `tables/patch_gain_curve.{md,csv}` and `figures/patch_budget_curve.*` in `paper_artifact_bundle_final_neighboring_checked_20260611_001`. | Use the current generated patch-budget replacement for release-facing claims. |
| Figure `bw_latency_tradeoff` | Covered as a current replacement by `latency_quality_pareto_*.svg` plus `bandwidth_sensitivity.csv/md/svg` in `paper_artifact_bundle_final_neighboring_checked_20260611_001`. The bandwidth sweep is payload-based and only includes rows with real `payload_mb`; current mixed-route rows still rely on direct TTFT because their route metrics have no payload bytes. | Use the final generated replacement; do not invent mixed-route payloads unless future eval summaries record them. |
| Table `ttft_breakdown_wiki` | Covered by `ttft_breakdown_table.{csv,md,tex}` in `paper_artifact_bundle_final_neighboring_checked_20260611_001`. | Use the final generated table. |
| Figure `pareto_rank` | Covered as a current replacement by `latency_quality_pareto_*.svg` in `paper_artifact_bundle_final_neighboring_checked_20260611_001`. | Use the current generated replacement under the trend-compatible criterion. |
| Figure `patch_budget` | Covered by the current profile replacement `patch_budget_curve.*` inside `paper_artifact_bundle_final_neighboring_checked_20260611_001`, based on `patch_gain_curve.*`. | Use the final generated replacement. |
| Figure `layerwise_xnorm` | Covered and wired into the current-safe paper via `Publication/paper/generated_figures/layerwise_xnorm_replacement.pdf`, generated from `profile_analysis_figures_20260611_002`. | Regenerate after final route/profile changes if needed. |
| Figure `layer_selection_viz` | Covered as current generated replacement panels via `layer_selection_gain_profile.pdf` and `patch_budget_curve.pdf` in `Publication/paper/generated_figures/`; exact historical interaction panel remains historical. | Regenerate after final route/profile changes if needed. |
| Figure `tsne_kv_align` | Covered and wired into the current-safe paper with `Publication/paper/generated_figures/stack_viz.pdf`, generated from current-runtime sampled vectors. | Use the current extraction bundle; archived t-SNE bundle is provenance only. |
| Table `neighboring_baselines` | Covered by a current-runtime, same-protocol CMRC200 rerun. The runtime includes public SVD-LLM-style, SliceGPT-style, and TensorLLM-style adapted KV scripts plus a table consolidator with protocol warnings and a hard compatibility flag. | Use `Experiment/results/neighboring_baselines_cmrc200_maxlen1024_20260611_001/table/neighboring_baselines.{csv,md,tex}`. The earlier mixed-protocol `neighboring_baselines_cmrc200_20260611_001/table/` remains provenance only. |
| Table `offline_cost` | Covered and wired into the current-safe paper with measured rows from `offline_cost_evidence_20260611_001`; the paper-mixed `train_patch` row is explicitly `no time fields`. | Replace only if accepted final-route logs add stronger complete timing evidence. |

## Current Next Gate

The release-facing non-Qwen evidence is now closed for CMRC2018, WikiText-2,
HotpotQA-compressed full7405, patch-only anchors, route/budget/latency figures,
offline cost, and the neighboring-baseline appendix. Exact historical
patch-budget/rank sweeps are represented by current generated replacements
under the trend-compatible criterion rather than byte-for-byte historical
reruns. Qwen32B remains outside the requested scope.
