# Open-Source Release Manifest

This manifest identifies the files needed for the paper-facing SCD-ICML
Mistral reproduction path. It is not a deletion list for historical experiment
files; it is the minimal set that should remain visible and documented in a
public release.

## Required Runtime Code

```text
src/scd_repro/
  cli/capture_pair.py
  cli/train_reuse.py
  cli/profile_layers.py
  cli/train_patch.py
  cli/eval_text_strict.py
  cli/summarize_mistral_repro.py
  text_benchmarks.py
  strict_runtime.py
  pipeline.py
  patch.py
  reuse.py
  cmrc.py
```

Other package modules imported by these files are also required.

The release also includes the experimental vLLM adapter facade:

```text
src/scd_vllm/
src/scd_vllm_plugin/
docs/scd_vllm_plugin_mvp.md
```

The public plugin entry point is `scd-vllm`. Run `scd-vllm doctor` before GPU
work to verify that local artifacts exist, vLLM can import, and the selected
route is a strict no-fallback/no-recompute mixed reuse+patch route.

## Required Lightweight Tests

```text
tests/
```

These tests cover model-free loader, metric, route-serialization, and route
selection behavior. After installing the package, run
`python -m unittest discover -s tests` before launching GPU jobs.

## Required Configs

```text
configs/eval_scd_icml_no_fallback_no_recompute.json
configs/eval_scd_icml_full_scd_no_fallback_no_recompute.json
configs/eval_baseline_reuse_only_fulldev3219_20260405.json
configs/eval_patch_only_fulldev3219_20260405.json
configs/data_ser.yaml
configs/models_mistrallite_to_mistral7_template.json
configs/models_ser_mistrallite_to_mistral7_mql.json
configs/models_ser_mistrallite_to_mistral7_mql_single_gpu.json
configs/models_ser_mistral7_to_mistrallite_mql.json
configs/train_reuse_mistral_self_layerwise_20260404.json
configs/mistral_self_patch_xnorm_nativekv_shared_residual_mlp_alllayers_e1_h2048_r512_b2_20260404.json
configs/mistral_self_patch_xnorm_nativekv_shared_residual_mlp_paper_k6_e1_h2048_r512_b2_20260404.json
configs/mistral_self_patch_xnorm_nativekv_residual_mlp_parallel_e1_h2048_b2_20260404.json
configs/mistral_self_patch_xnorm_nativekv_linear_parallel_20260404.json
configs/route_selection.yaml
configs/route_reuse_patchonly_all32_20260404.json
configs/routes/scd_icml_mistrallite_to_mistral7_reuse19_patch13_cmrc200.json
```

`eval_scd_icml_no_fallback_no_recompute.json` is the release-facing aggregate
eval configuration for the 5-method table. `eval_scd_icml_full_scd_no_fallback_no_recompute.json`
is the narrower claim-boundary config for comparing only Full Consumer and SCD
on a fixed selected route. Older configs that enable selective recompute or
fallback-heavy diagnostics may remain in the working tree for traceability, but
they are not part of the default open-source release preview.

The `models_ser_*_mql.json` files document the verified local run. Users on a
new machine should copy `models_mistrallite_to_mistral7_template.json` and pass
it through `MODELS_CONFIG_LITE_TO_7B`.

## Required Scripts

```text
scripts/run_ser_scd_icml_mistral_repro.sh
scripts/run_ser_scd_icml_patchonly_anchor.sh
scripts/run_ser_scd_icml_parallel_patch_shards.sh
scripts/run_ser_neighboring_baselines_cmrc.sh
scripts/make_paper_scd_route.py
scripts/make_paper_scd_candidate_routes.py
scripts/merge_patch_ckpts.py
scripts/run_ser_scd_icml_eval_existing_route.sh
scripts/run_ser_scd_icml_conditional_route_screen.sh
scripts/select_quality_gated_scd_route.py
scripts/generate_conditional_reuse_routes.py
scripts/collect_route_eval_metrics.py
scripts/filter_strict_eval_by_qids.py
scripts/build_scd_icml_result_tables.py
scripts/build_scd_icml_analysis_figures.py
scripts/extract_stack_viz_inputs.py
scripts/build_stack_viz.py
scripts/build_offline_cost_table.py
scripts/build_paper_artifact_bundle.py
scripts/finalize_scd_icml_full7405_bundle.py
scripts/run_svdllm_style_kv_cmrc.py
scripts/run_slicegpt_style_kv_cmrc.py
scripts/run_tensorllm_style_kv_cmrc.py
scripts/build_neighboring_baseline_table.py
scripts/wait_postprocess_scd_icml_full7405_tables.sh
scripts/wait_finalize_scd_icml_full7405_bundle.sh
scripts/check_paper_artifacts.py
scripts/report_pairwise_interactions.py
scripts/estimate_layer_mapping.py
scripts/profile_routes.py
scripts/select_patch_reuse_route.py
scripts/evaluate_selected_route.py
scripts/validate_scd_route.py
scripts/check_release_readiness.py
scripts/check_release_smoke_status.py
scripts/build_release_tree.py
```

`run_ser_scd_icml_mistral_repro.sh` is the main claim-facing entry point. It
defaults to `DIRECTIONS=lite_to_7b`, `DATASETS=cmrc2018`, the selected
Reuse19/Patch13 route, and the Full Consumer plus SCD eval config. It trains
Reuse plus an all-layer Patch bank, then evaluates the fixed selected route.
The historical paper-profile `k=6` route remains available only as an explicit
diagnostic by setting `ROUTE_MODE=paper_profile` and the corresponding `k=6`
Patch train config.

`run_ser_scd_icml_patchonly_anchor.sh` is the historical GitHub/new-main trend
anchor. It evaluates Full Consumer, Reuse-only, and Patch-only with
dataset-specific Reuse/Patch banks from the `scd-github-f7d7854` snapshot. Use
it to verify the older strong Patch-only trend on CMRC2018, HotpotQA compressed
prompting, and WikiText-2 before optimizing a mixed Reuse+Patch route. It is not
the default mixed-route SCD claim.

`run_ser_scd_icml_parallel_patch_shards.sh` is the high-coverage Patch helper
for no-fallback mixed routes. It launches mergeable per-layer Patch shards,
uses `scripts/merge_patch_ckpts.py` after shard completion, and produces a
merged Patch pool. `scripts/make_paper_scd_candidate_routes.py` then emits
fixed top-k paper-SCD candidates, and
`scripts/run_ser_scd_icml_eval_existing_route.sh` evaluates any one candidate
without retraining. The final route should maximize Reuse subject to the chosen
quality gate: F1/EM retention or absolute F1/EM drop for QA, and PPL relative
increase for WikiText-2. `scripts/select_quality_gated_scd_route.py` selects
the maximum-Reuse passing route from completed candidate eval roots. The high-coverage route
itself is only an upper-bound diagnostic. The teacher-aligned route-selection
scripts implement the newer main path: layer mapping, baseline diagnostic table
schema, conditional Reuse pruning from a high-quality route, and selected-route
eval metadata. `scripts/report_pairwise_interactions.py` turns completed route
metrics into pairwise interaction tables, reporting missing combinations
explicitly instead of imputing them. They never invent missing teacher metrics;
missing trials are written as pending routes for strict evaluation.
`run_ser_neighboring_baselines_cmrc.sh` is the appendix neighboring-baseline
helper. It runs SVD-LLM-style and TensorLLM-style adapted KV baselines in
parallel on two GPU pairs, then runs SliceGPT-style adapted KV transfer. It
builds `neighboring_baselines.{csv,md,tex}` only when `SCD_SUMMARY` is set to a
same-protocol selected SCD summary; otherwise it leaves a pending-table note in
`logs/build_table.log`. When `SCD_SUMMARY` is set, the wrapper invokes
`build_neighboring_baseline_table.py --require-compatible-protocol` so a
mixed-protocol table fails instead of becoming a claim-facing artifact.
Override `LIMIT`, `RUN_ROOT`, `DATA_ROOT`, `PAIR_DIR`, and the `*_GPUS`
variables for a different host layout.
`scripts/build_scd_icml_result_tables.py` converts completed
`patchonly_anchor_summary.json` and `route_metrics.json` artifacts into
Markdown/CSV/LaTeX tables and per-dataset route-curve SVGs for paper-facing
result review. It is a postprocessing helper only; it never fabricates missing
full-dev or candidate-route rows. It also emits TTFT breakdown tables and
latency-quality/Pareto SVGs from the same completed summaries. Its
`bandwidth_sensitivity.csv/md/svg` outputs recompute transfer time only for
rows with real `payload_mb`; mixed-route rows without payload measurements are
not imputed.
`scripts/build_scd_icml_analysis_figures.py` converts completed
`profile_cmrc2018_seed42.json` artifacts into layer-profile CSV/Markdown and
SVG/PDF/PNG figures for Reuse-vs-Patch error, expected Patch gain, and
cumulative gain curves. It does not generate t-SNE plots because those require
representation samples that are not contained in the profile JSON.
`scripts/extract_stack_viz_inputs.py` converts pair artifacts plus a Reuse
checkpoint into KV-distance and t-SNE/2D embedding CSV/meta inputs for the
stack visualization. `scripts/build_stack_viz.py` then converts those stored
CSV/meta artifacts into the stack/t-SNE PDF/PNG/SVG visualization.
`scripts/build_offline_cost_table.py` converts raw `/usr/bin/time -v` logs into
offline calibration/training cost CSV/Markdown/JSON plus a wall-time SVG. It
preserves missing or malformed time logs as `status=no_time_fields` instead of
inventing stage costs.
`scripts/build_paper_artifact_bundle.py` assembles generated result tables,
profile figures, stack/t-SNE figures, offline-cost artifacts, and the
protocol-checked neighboring-baseline appendix table into a single paper-facing
bundle with `paper_artifact_manifest.json`. It copies only existing generated
artifacts and reports missing inputs; it does not compute or invent metrics.
`scripts/finalize_scd_icml_full7405_bundle.py` wraps the final post-full7405
bundle assembly and paper-artifact gate so the final result table directory can
be converted into a checked paper-facing bundle with one command. When the
HotpotQA selector report is present, it also requires the paper source to
mention the selected `ReuseXX/PatchYY` route, preventing stale route text from
passing finalization.
`scripts/run_svdllm_style_kv_cmrc.py`,
`scripts/run_slicegpt_style_kv_cmrc.py`, and
`scripts/run_tensorllm_style_kv_cmrc.py` are appendix-only neighboring
compression baseline entry points for the CMRC strict runtime. They emit
`summary.json`, `summary.md`, and `per_sample_metrics.jsonl` with Full
Consumer, Raw KV, and the adapted compression method. These scripts provide a
public rerun path for the historical appendix comparison; their rows are not
claim-facing unless the corresponding runs are completed and consolidated.
`scripts/build_neighboring_baseline_table.py` consolidates those completed
baseline summaries plus a completed SCD summary into CSV/Markdown/LaTeX
versions of the adapted-neighboring-baselines appendix table, and records
protocol warnings if sources disagree on fields such as `max_length`; pass
`--require-compatible-protocol` for final paper-facing consolidation.
`scripts/wait_postprocess_scd_icml_full7405_tables.sh` waits for one or more
full7405 route-eval summaries, generates `full7405_route_metrics.json`, runs
the HotpotQA F1-drop selector, and writes `result_tables_with_full7405/`.
`scripts/filter_strict_eval_by_qids.py` re-aggregates a strict text eval from
saved per-sample metrics after including or excluding qids. Use it for
HotpotQA leakage audits when old HotpotQA-trained checkpoints overlap with part
of the full7405 dev set; it recomputes the summary files without rerunning GPU
inference.
`scripts/wait_finalize_scd_icml_full7405_bundle.sh` waits for the
post-full7405 result table to appear, verifies that the table source includes
`full7405`, requires the neighboring-baseline table directory, and then invokes
the finalizer.
`scripts/check_paper_artifacts.py` validates that a paper source references
existing generated figures, does not accidentally fall back to forbidden stale
figure names, has an `ok` artifact-bundle manifest, and can enforce literal
claim-boundary text such as the restored neighboring-baseline table label, the
HotpotQA full7405 selected-route statement, and the current claim-facing table
labels.

## Required Documentation

```text
README.md
REPRODUCING_SCD_ICML.md
DATA_PREPARATION.md
OPEN_SOURCE_RELEASE_MANIFEST.md
LICENSE
```

`REPRODUCING_SCD_ICML.md` defines the no-fallback/no-recompute reproduction
boundary. `DATA_PREPARATION.md` explains portable data/model paths.

## Release Check

Before publishing, run:

```bash
python scripts/check_release_readiness.py
```

To build a non-destructive release preview tree:

```bash
python scripts/build_release_tree.py --out release_preview/scd_icml_runtime_release --force
```

To inspect a queued or completed release smoke:

```bash
python scripts/check_release_smoke_status.py --run-root <smoke-run-root> --pid <wrapper-pid>
```

If the verified run root is available, also run:

```bash
python scripts/check_release_readiness.py \
  --run-root /disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_selected_reuse19_5method_aggregate_20260610_034426
```

The second command verifies the aggregate Mistral summary contains the expected
15 rows for `lite_to_7b` over CMRC2018, HotpotQA, and WikiText-2:
full consumer, raw KV, 4-bit KV, reuse-only, and SCD. This is a structural
release check; it does not certify that every dataset row supports the same
quality claim.

The default SCD route must follow the paper implementation boundary:
non-patch layers are reconstructed by Reuse, selected transition layers are
corrected by Patch, Patch and Reuse jointly cover all layers, and
`fallback_layers` / recomputation are not allowed in the claim-facing
`scd_full` path. The selected route should maximize Reuse while preserving
F1/EM against the full-consumer/oracle baseline. The current
CMRC200 F1/EM-primary selected route is
`configs/routes/scd_icml_mistrallite_to_mistral7_reuse19_patch13_cmrc200.json`.
It uses 19 Reuse layers and 13 Patch layers. All 13 one-layer Reuse20
extensions from this route were tested on CMRC200 and failed to preserve F1;
avg NLL is an auxiliary risk signal rather than the primary selector. Validate the route
with:

```bash
python scripts/validate_scd_route.py --route <route.json> --require-mixed
```

The current verified aggregate run is
`/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_selected_reuse19_5method_aggregate_20260610_034426`.
It contains 15 rows for CMRC2018, HotpotQA, and WikiText-2 across
`full_consumer`, `raw_kv`, `quantized_4bit`, `reuse_only`, and `scd_full`, and
passes `scripts/check_release_readiness.py --run-root`.

Use this aggregate's CMRC2018 rows as the current selected-route quality
evidence. Do not use its HotpotQA rows for quality claims: later full-context,
no-truncation HotpotQA runs fixed the abnormal baseline and show poor
Reuse19/Patch13 retention. The current HotpotQA evidence is:

- `/disk_n/c101-3-mql/projects/SCD-ICML/runs/hotpotqa_fullcontext_full_scd_20260610_115952`
  reports Full Consumer F1 `0.4005`, EM `0.265` versus SCD F1 `0.1383`, EM
  `0.105`.
- `/disk_n/c101-3-mql/projects/SCD-ICML/runs/hotpotqa_full_banks_20260610_192156`
  reports Full Consumer F1 `0.3838`, EM `0.235` versus SCD mixed-from-banks F1
  `0.0920`, EM `0.065` on a disjoint heldout split.

The older snapshot at
`/disk_n/c101-3-mql/projects/SCD-ICML/kv/snapshots/scd-github-f7d7854` has a
stronger HotpotQA patch-only result, but that path uses question-aware evidence
compression and `max_length=512`; treat it as a separate compressed-prompt
benchmark, not as evidence for the diagnostic full-context rows. The current
runtime exposes this protocol as `--dataset hotpotqa_compressed`, and this is
the paper-facing HotpotQA protocol. The current full7405 bracket is closed for
the first passing/failing pair and can be refined with intervening reuse counts.
A question-text overlap audit found that the old full7405 eval includes the
100 training questions, but removing those 100 samples leaves Patch-only at F1
`0.3514`, EM `0.2516`, so the old result is not primarily explained by the
small overlap.

Current consolidation note: the paper-facing HotpotQA reproduction should now
be checked under the GitHub-compatible `hotpotqa_compressed` protocol with
`MAX_LENGTH=512` and full7405 evaluation. The full-context rows above are
diagnostic evidence that the CMRC-selected route is not universal; they should
not replace the compressed-protocol full7405 route table. The current
full7405 boundary is closed: `reuse04_patch28`, `reuse05_patch27`, and
`reuse06_patch26` pass the absolute F1-drop gate, while `reuse07_patch25` and
`reuse08_patch24` fail. The selected full7405 route is `reuse06_patch26`.

## Historical Files

The repository still contains older configs and scripts for Qwen-1.7B,
reverse-Mistral diagnostics, synthetic/runtime smoke tests, and route searches.
They can remain for traceability, but they should not be advertised as the
default SCD-ICML reproduction path unless separately revalidated.
