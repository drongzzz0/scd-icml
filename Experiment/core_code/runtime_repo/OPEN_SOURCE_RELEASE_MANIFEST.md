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
route is a strict no-fallback/no-recompute mixed reuse+patch route. The doctor
supports both raw `--reuse-ckpt/--patch-ckpt` paths and existing
`--bundle-dir` artifacts.

After converting a validated SCD bundle to `ExampleConnector` storage, use
`scd-vllm launch-spec` to emit the official `vllm.LLM(**llm_kwargs)` launch
contract for that trusted bundle/storage pair. Use `scd-vllm launch-smoke` to
consume that JSON through official vLLM and write a one-batch generation
artifact. Use `scd-vllm batch-benchmark` for repeated-prefix connector
throughput checks. These paths are intentionally narrow: they are
trusted/static-prefix artifacts and must not be reported as general
multi-tenant serving, privacy, arbitrary-routing, or dataset-quality claims.

For connector-development experiments, the package includes
`scd_vllm.preloaded_connector.SCDPreloadedConnector`, which can be loaded by
vLLM with `--kv-connector SCDPreloadedConnector` and
`--kv-connector-module-path scd_vllm.preloaded_connector`. It memoizes loaded
SCD KV tensors per worker to avoid repeated safetensors reads for identical
batch prefixes. `scd-vllm launch-spec` automatically sets the HMA-disable
vLLM kwarg required by this connector in vLLM `0.18.0` and can set
`SCD_PRELOADED_STATS_DIR` for connector stats. This is an experimental
connector prototype, not a production multi-tenant cache manager.

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
configs/scd_vllm_mvp.yaml
configs/train_reuse_mistral_self_layerwise_20260404.json
configs/mistral_self_patch_xnorm_nativekv_shared_residual_mlp_alllayers_e1_h2048_r512_b2_20260404.json
configs/mistral_self_patch_xnorm_nativekv_shared_residual_mlp_paper_k6_e1_h2048_r512_b2_20260404.json
configs/mistral_self_patch_xnorm_nativekv_residual_mlp_parallel_e1_h2048_b2_20260404.json
configs/mistral_self_patch_xnorm_nativekv_linear_parallel_20260404.json
configs/route_selection.yaml
configs/route_reuse_patchonly_all32_20260404.json
configs/routes/scd_icml_mistrallite_to_mistral7_reuse19_patch13_cmrc200.json
configs/routes/scd_icml_mistrallite_to_mistral7_patch20_reuse12_cmrc200.json
configs/routes/scd_icml_mistrallite_to_mistral7_patch26_reuse6_cmrc200.json
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

The `reuse19_patch13` route is the paper-aligned maximum-Reuse reference for
the current Mistral evidence package. The `patch20_reuse12` route is the
current higher-Reuse/speed vLLM plugin demo route because it has full CMRC2018/HotpotQA quality
evidence and passes the reconstructed vLLM greedy smoke check from raw trained
Reuse/Patch banks. The `patch26_reuse6` route is the current quality-first
reference route for CMRC2018 and WikiText-2; it also passes the same
raw-bank-to-vLLM greedy smoke check.

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
scripts/scd_vllm/run_00_repo_audit.sh
scripts/scd_vllm/run_01_env_probe.sh
scripts/scd_vllm/run_02_hf_export.sh
scripts/scd_vllm/run_03_vllm_baseline.sh
scripts/scd_vllm/run_04_truekv_oracle.sh
scripts/scd_vllm/run_05_reconstructed_export.sh
scripts/scd_vllm/run_05_scd_injection.sh
scripts/scd_vllm/run_06_report.sh
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
The `scripts/scd_vllm/run_00_*.sh` through `run_06_*.sh` wrappers are the
release-facing vLLM integration phase commands. They drive the repo audit,
environment probe, HF bundle export, plain vLLM baseline, TrueKV oracle,
reconstructed SCD bundle export, SCD injection smoke, and final status report.
The default knobs live in `configs/scd_vllm_mvp.yaml`, while host/model-specific
paths should be supplied through environment variables.

## Required Documentation

```text
README.md
REPRODUCING_SCD_ICML.md
DATA_PREPARATION.md
OPEN_SOURCE_RELEASE_MANIFEST.md
docs/scd_vllm_plugin_mvp.md
reports/scd_vllm_repo_audit.md
reports/scd_vllm_status.md
LICENSE
```

`REPRODUCING_SCD_ICML.md` defines the no-fallback/no-recompute reproduction
boundary. `DATA_PREPARATION.md` explains portable data/model paths.
`reports/scd_vllm_status.md` is the release-facing vLLM evidence boundary and
records the A800 `conda activate cache` environment, the two vLLM-smoke-checked
routes, and the non-production connector limits.

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

The default SCD method boundary is unchanged: non-patch layers are
reconstructed by Reuse, selected transition layers are corrected by Patch,
Patch and Reuse jointly cover all layers, and `fallback_layers` /
recomputation are not allowed in the claim-facing `scd_full` path. Validate any
published route with:

```bash
python scripts/validate_scd_route.py --route <route.json> --require-mixed
```

The current open-release route roles are:

- `reuse19_patch13`: paper-aligned maximum-Reuse reference route. It remains
  useful for historical comparison, but it is not the current vLLM demo route.
- `patch20_reuse12`: current vLLM plugin demo route. It has raw-bank-to-vLLM
  smoke evidence with `greedy_1tok_match=True` and `greedy_ntok_match=True`.
- `patch26_reuse6`: current quality-first reference route for CMRC2018 and
  WikiText-2. It also has raw-bank-to-vLLM smoke evidence with
  `greedy_1tok_match=True` and `greedy_ntok_match=True`.

The current branch-local release evidence is summarized in
`../../docs/scd-vllm-open-release-status-2026-06-17.md` and in these artifact
directories:

- `../../Experiment/results/a800_scd_vllm_raw_bank_patch20_run_20260618_032217/`
  for higher-Reuse raw-bank-to-vLLM plugin evidence.
- `../../Experiment/results/a800_scd_vllm_raw_bank_patch26_run_20260618_044521/`
  for quality-first raw-bank-to-vLLM plugin evidence.
- `../../Experiment/results/a800_mistral_patch20_scd_vllm_preloadedstats_20260618_cache/`
  and
  `../../Experiment/results/a800_mistral_patch26_scd_vllm_preloadedstats_20260618_cache/`
  for the main Mistral repeated-prefix trusted microbenchmark. These compare
  plain vLLM, vLLM `prefix-cache`, and Mistral `SCDPreloadedConnector` batch
  load for the current `patch20_reuse12` and `patch26_reuse6` routes.
  The artifact also records worker-side preloaded-cache stats:
  `cache_misses=32`, `cache_hits=16352`, and `cache_hit_rate=0.998047` for
  each route. SCDPreloaded is faster than plain vLLM on this repeated-prefix
  shape, but it remains slower than native vLLM prefix caching; do not report
  it as a general serving-speed claim over vLLM.
- `../../Experiment/results/a800_qwen17_scd_vllm_preloadedbench_20260618_cache/`
  for additional-model repeated-prefix trusted microbenchmark evidence. It
  compares plain vLLM, vLLM `prefix-cache`, and Qwen
  `SCDPreloadedConnector` batch load. The
  preloaded connector reduces disk-backed ExampleConnector overhead, but the
  measured SCD load throughput is still lower than both vLLM baselines and must
  not be used as a speedup claim.
- `../../Experiment/results/scd_vllm_multidataset_full_strict_20260617/` for
  full CMRC2018 and HotpotQA strict QA evidence.
- `../../Experiment/results/wikitext2_patch20_patch26_full_current_20260618_034812/`
  for full WikiText-2 NLL/PPL evidence.

For QA datasets, use F1/EM as the primary quality metrics. For WikiText-2, use
NLL/PPL as the primary language-modeling metrics. Do not claim that one route
is universally best across all datasets: `patch20_reuse12` is the higher-Reuse
and speed-oriented vLLM demo route, while `patch26_reuse6` is the safer
quality route for CMRC2018 and WikiText-2. Both strict WikiText-2 routes
improve TTFT but worsen PPL relative to Full Consumer, so report that trade-off
explicitly.

## Historical Files

The repository still contains older configs and scripts for Qwen-1.7B,
reverse-Mistral diagnostics, synthetic/runtime smoke tests, and route searches.
They can remain for traceability, but they should not be advertised as the
default SCD-ICML reproduction path unless separately revalidated.
