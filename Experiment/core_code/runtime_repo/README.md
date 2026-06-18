# SCD Runtime Repository

This directory is the canonical runtime/package snapshot for the final
`scd-new` code line.

## What To Keep

- `src/scd_repro/`
  - Main Python package.
- `configs/`
  - Active training/eval/model configs still referenced by the local workflow.
- `scripts/`
  - Remote execution helpers and reusable utility scripts.
- Top-level `run_ser_*.sh`
  - Stable wrappers still useful for reproducible route runs.

Previously archived runtime backups, temporary prototypes, and build metadata
have been removed so this package surface now reflects only the current kept
version.

## Current Canonical Interpretation

For the claim-facing SCD-ICML story, this runtime package should be read as the
implementation base for the paper method: sparse `Patch` at offline-selected
transition layers and `Reuse` on every remaining layer. The default run uses a
fixed mixed route with `fallback_layers=[]` and no recomputation.

Historical grouped/shared-z defaults, all-layer patch anchors, fallback-heavy
routes, selective recompute configs, strict-stop GitHub-style searches, and
scratch patch variants may still exist in the working tree for traceability, but
they are excluded from the default release preview and must not be treated as
the paper-facing method line.

## Quick Start

Install the package from this directory, then run the release-facing Mistral
smoke test or main reproduction command below. The generic console scripts
(`capture-pair`, `train-reuse`, `train-patch`, and `eval-text-strict`) are still
available, but this release does not ship placeholder `models.yaml`,
`data.yaml`, or `train.yaml` files; use the concrete configs listed in
`REPRODUCING_SCD_ICML.md`.

```bash
python -m pip install -e .
python scripts/check_release_readiness.py
python -m unittest discover -s tests

TRAIN_LIMIT=2 EVAL_LIMIT=1 \
DIRECTIONS=lite_to_7b DATASETS=cmrc2018 \
MAX_LENGTH=256 MAX_NEW_TOKENS=8 \
bash scripts/run_ser_scd_icml_mistral_repro.sh
```

The experimental vLLM adapter is exposed through the `scd-vllm` console
script. Start with the static doctor check before launching GPU work:

```bash
scd-vllm doctor \
  --producer-model /path/to/producer \
  --consumer-model /path/to/consumer \
  --reuse-ckpt /path/to/reuse \
  --patch-ckpt /path/to/patch \
  --route configs/routes/scd_icml_mistrallite_to_mistral7_reuse19_patch13_cmrc200.json
```

For the current vLLM demo route, use
`configs/routes/scd_icml_mistrallite_to_mistral7_patch20_reuse12_cmrc200.json`.
It keeps strict reuse+patch-only semantics and is the route with the strongest
reuse/speed trade-off among the currently validated vLLM smoke routes.
For quality-first CMRC2018 and WikiText-2 comparisons, use
`configs/routes/scd_icml_mistrallite_to_mistral7_patch26_reuse6_cmrc200.json`.
It also has raw-bank-to-vLLM greedy-smoke evidence in this branch.

For an already reconstructed SCD bundle, check the bundle artifact instead:

```bash
scd-vllm doctor \
  --bundle-dir runs/scd_vllm/my_bundle \
  --route runs/scd_vllm/my_bundle/route.json
```

Then use `scd-vllm run` to export a reconstructed reuse+patch bundle, convert
it to vLLM ExampleConnector-compatible storage, and run the smoke injection
path. See `docs/scd_vllm_plugin_mvp.md` for the full command.

For programmatic official-vLLM integration, generate a launch spec and validate
it without occupying a GPU:

```bash
scd-vllm launch-spec \
  --bundle-dir runs/scd_vllm/my_plugin_run/reuse_patch_bundle \
  --storage-dir runs/scd_vllm/my_plugin_run/example_connector_storage \
  --model /path/to/consumer \
  --out runs/scd_vllm/launch_spec.json

python examples/scd_vllm_official_launch.py \
  --launch-spec runs/scd_vllm/launch_spec.json \
  --dry-run
```

## Build Reuse + Patch + Route From Scratch

Use this entry point when the checkpoint banks do not already exist. It is the
standard open-source construction path for the release-facing mixed SCD route:

1. capture train-split A/B pair artifacts;
2. train the Reuse bank;
3. write a minimal fixed-route profile for Patch training;
4. train a full native-KV shared residual Patch bank;
5. copy the fixed paper route selected on CMRC2018 (`reuse19/patch13`);
6. validate that the route is mixed, complete, no-fallback, and no-recompute;
7. run strict CMRC2018 dev evaluation for Full Consumer versus SCD.

```bash
DATA_SOURCE_DIR=/disk_n/c101-3-mql/projects/SCD-ICML/kv/snapshots/scd-github-f7d7854/data/cmrc2018 \
CALIB_SAMPLES=200 EVAL_SAMPLES=200 \
ROUTE_PATCH_BUDGET=13 \
bash scripts/run_ser_build_reuse_patch_route_eval.sh
```

The default script trains Patch before route use and does not run the expensive
full `profile_layers` pass for fixed routes. It does not use the weak semantic
task-level selector as the release path. The claim-facing route is the validated
no-fallback/no-recompute route in
`configs/routes/scd_icml_mistrallite_to_mistral7_reuse19_patch13_cmrc200.json`.

Task-level route search is still available as an explicit diagnostic mode, but
it currently requires a compatible semantic Patch bank and must pass F1/EM
gates before being reported:

```bash
ROUTE_MODE=task_level \
PATCH_TRAIN_CONFIG=configs/train_scd_icml_patch_semantic_alllayers_20260616.json \
python -m scd_repro.cli.build_route \
  --profile "${PROFILE_PATH}" \
  --pairs "${PAIRS_DIR}" \
  --patch-ckpt "${PATCH_DIR}" \
  --eval-config "${GENERATED_EVAL_CONFIG}" \
  --out "${ROUTE_PATH}"
```

## SCD-ICML Mistral Reproduction

For the paper-facing Mistral reproduction, use the strict runtime wrapper. The
default `scd_full` path is no-fallback/no-recompute: non-patch layers use
Reuse, and selected transition layers use Patch. The default command now targets
the current positive claim boundary: CMRC2018 dev `n=200`, the selected
Reuse19/Patch13 route, and Full Consumer versus SCD.

```bash
TRAIN_LIMIT=200 EVAL_LIMIT=200 \
DATASETS=cmrc2018 \
MAX_LENGTH=0 MAX_NEW_TOKENS=32 \
bash scripts/run_ser_scd_icml_mistral_repro.sh
```

The default direction is `lite_to_7b`, matching the canonical
MistralLite-to-Mistral-7B slice described in the SCD-ICML paper. To run the
optional reverse diagnostic, set `DIRECTIONS=lite_to_7b,7b_to_lite` explicitly.
Qwen-32B is not part of the default `ser` reproduction path because the current
machine cannot host that pair reliably.
For the final-paper Qwen-32B claim audit on A800-class machines, use the
explicit guarded entry point:

```bash
CONDA_ENV=cache \
MODELS_CONFIG=configs/models_a800_qwen25_32b_base_to_instruct_template.json \
DATA_ROOT=/home/c101-800-mql/maqianli/no4/data \
DATASETS=cmrc2018 \
bash scripts/run_a800_scd_icml_qwen32_repro.sh
```

This runner is intentionally fail-closed. It first runs
`scripts/preflight_qwen32_a800.py`, rejects missing text-CausalLM model paths,
rejects VL checkpoints, and requires two GPUs with sustained zero utilization
and enough free memory before capture/training/evaluation starts. Do not use the
legacy Qwen1.7B diagnostic scripts as evidence for the Qwen-32B paper row.
The wrapper also waits by default for enough GPU memory before capture, train,
and eval stages; override `WAIT_CAPTURE_FREE_MB`, `WAIT_TRAIN_FREE_MB`, and
`WAIT_EVAL_FREE_MB` only if you have inspected GPU availability.
When `CUDA_VISIBLE_DEVICES` is unset, the wrapper waits until two GPUs satisfy
the capture free-memory gate before selecting them, and can reselect a different
pair before capture/eval if another job occupies the initially selected GPU.

The historical `k=6` sparse Patch route is still available as an explicit
diagnostic baseline by setting `ROUTE_MODE=paper_profile` and
`PATCH_TRAIN_CONFIG=configs/mistral_self_patch_xnorm_nativekv_shared_residual_mlp_paper_k6_e1_h2048_r512_b2_20260404.json`.
It is not the default claim-facing run. The current claim-facing route selection
path is teacher-aligned conditional search, not fixed-k layer ranking. Use the
full consumer as teacher, profile route-level KV drift, hidden drift, logit KL,
PPL delta, and cost, then start from a high-quality/all-Patch route and prune
Patch layers into Reuse only when the conditional quality increase remains
within the configured budgets. Fixed top-k routes from the offline profile are
retained as diagnostic candidates and baselines.
`patch_only` remains a diagnostic capability in the codebase, but it is not part
of the default paper-facing table.

For high-coverage no-fallback experiments, use
`scripts/run_ser_scd_icml_parallel_patch_shards.sh` after Reuse pairs/profile
exist. It trains mergeable per-layer Patch shards on multiple GPUs and merges
them with `scripts/merge_patch_ckpts.py`. Treat the merged high-coverage Patch
checkpoint as a candidate pool or upper-bound diagnostic, not as the final route
by itself. Use `scripts/make_paper_scd_candidate_routes.py` and
`scripts/run_ser_scd_icml_eval_existing_route.sh` to evaluate sparse candidates
and `scripts/select_quality_gated_scd_route.py` to select the maximum-Reuse
route that passes the primary quality gate. Use F1/EM retention for QA datasets
and `--metric ppl` with `--max-ppl-relative-increase` for WikiText-2. For QA
datasets with low Full Consumer F1, `--max-metric-drop` can be used as an
absolute-loss gate.
For a faster diagnostic Patch pool, set
`TRAIN_CONFIG=configs/mistral_self_patch_xnorm_nativekv_linear_parallel_20260404.json`;
this saves direct linear Patch weights without residual MLP refinement and should
be treated as a speed-oriented candidate, not as the default final result.

Historical GitHub/new-main Patch-only anchor:

```bash
DATASETS=cmrc2018,hotpotqa_compressed,wikitext2 \
EVAL_LIMIT=0 \
bash scripts/run_ser_scd_icml_patchonly_anchor.sh
```

This anchor uses dataset-specific Reuse/Patch banks from the
`scd-github-f7d7854` snapshot and reproduces the older strong Patch-only trend:
Reuse-only collapses, while Patch-only preserves CMRC/HotpotQA quality and
WikiText-2 PPL with lower TTFT than the full consumer. Keep it separate from the
default mixed Reuse+Patch route; it is the baseline to beat when optimizing for
the maximum number of Reuse layers.
Set `RESUME=1` to skip any completed dataset/stage that already has a
`per_mode_summary.json` under the same `RUN_ROOT`.

Teacher-aligned route selection entry points:

```bash
python scripts/estimate_layer_mapping.py --config configs/route_selection.yaml
python scripts/profile_routes.py --config configs/route_selection.yaml
python scripts/select_patch_reuse_route.py --config configs/route_selection.yaml --selector conditional_pruning
python scripts/report_pairwise_interactions.py --metrics outputs/route_selection/route_metrics.json --candidate-layers 1,8,25 --out-json outputs/route_selection/pairwise_interactions.json
python scripts/evaluate_selected_route.py --config configs/route_selection.yaml --route outputs/route_selection/selected_route.json
```

For GPU-efficient route screening on `ser`, use
`scripts/run_ser_scd_icml_conditional_route_screen.sh`. It launches one worker
per GPU pair from `GPU_PAIRS`; each worker immediately dequeues the next route
when its current route finishes. Set `REUSE_DIR` and `PATCH_DIR` explicitly when
screening dataset-specific banks from a snapshot, and set `REPORT_DATASET` to
the dataset used by `DATASETS`.

Current CMRC200 F1/EM-primary boundary:

- Selected route:
  `configs/routes/scd_icml_mistrallite_to_mistral7_reuse19_patch13_cmrc200.json`.
- Reuse layers:
  `[3,5,6,8,9,11,13,14,17,20,21,22,23,24,25,27,28,29,30]`.
- Patch layers:
  `[0,1,2,4,7,10,12,15,16,18,19,26,31]`.
- Verified CMRC2018 dev `n=200` Full+SCD result:
  Full Consumer F1 `0.6573`, EM `0.365`; SCD F1 `0.6428`, EM `0.350`.
  F1 retention is `0.9779` and EM retention is `0.9589`.
- All 13 one-layer Reuse20 extensions from the selected Reuse19/Patch13 route
  were tested on CMRC200 and did not preserve F1; the best candidate reached
  F1 retention `0.9300`.
- Avg NLL is reported as an auxiliary calibration-risk signal, not as the
  primary route-selection metric.
- Fresh 5-method x 3-dataset aggregate evidence is available at
  `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_selected_reuse19_5method_aggregate_20260610_034426`.
  It passed `scripts/check_release_readiness.py --run-root` as a structural
  package check. Use its CMRC2018 rows as the selected-route quality evidence.
  Do not use its HotpotQA rows for quality claims: later full-context
  no-truncation HotpotQA runs fixed the abnormal baseline and show large
  Reuse19/Patch13 quality loss. WikiText-2 remains an auxiliary NLL/PPL risk
  signal showing that the CMRC-selected route is not a universal
  language-modeling route.
- The older GitHub snapshot has a stronger HotpotQA Patch-only result under a
  question-aware compressed-evidence prompt. Treat it as a separate historical
  compressed-prompt benchmark; the current loader exposes this protocol as
  `--dataset hotpotqa_compressed`. A question-text overlap audit found that its
  7405-example eval includes the 100 training questions, but removing those
  questions still leaves Patch-only at F1 `0.3514`, EM `0.2516`; this does not
  make it evidence for the diagnostic full-context HotpotQA rows.
- For any HotpotQA-trained checkpoint whose training questions overlap with the
  full7405 dev file, use `scripts/filter_strict_eval_by_qids.py` on the saved
  `per_sample_metrics.jsonl` to recompute an overlap-filtered summary before
  making leakage-sensitive claims.
- Current result consolidation uses the GitHub-compatible
  `hotpotqa_compressed` protocol (`MAX_LENGTH=512`) for the paper-facing
  HotpotQA check. Treat the full-context HotpotQA rows above as diagnostics.
  The current full7405 compressed-protocol gate is closed through the boundary:
  `reuse04_patch28`, `reuse05_patch27`, and `reuse06_patch26` pass the absolute
  F1-drop gate, while `reuse07_patch25` and `reuse08_patch24` fail. The selected
  full7405 route is `reuse06_patch26`.

`select_patch_reuse_route.py` requires real route-level metrics. If a trial
route has not been evaluated, it writes `outputs/route_selection/pending_routes`
for the existing strict eval wrapper instead of fabricating teacher metrics.
When strict eval summaries are the available evidence, use F1/EM retention as
the primary QA quality gate, or use `--max-metric-drop` when the intended QA
criterion is an absolute F1/EM loss. Use PPL relative increase as the
WikiText-2 gate. Report `avg_nll_delta` as an auxiliary calibration-risk
signal, and make it a hard gate only for explicit diagnostic experiments. Leave
unavailable KV/hidden drift fields absent.
The strict eval wrappers default to CPU-loaded Patch checkpoints with
`runtime_patch_window_size=8`; this was the fastest safe setting in the current
24GB-GPU MistralLite -> Mistral-7B smoke sweep. Larger windows can be requested
explicitly, but `32` OOMed for the Patch-30 route.

For smoke commands, the no-fallback/no-recompute route semantics, and the
recommended claim boundary, see `REPRODUCING_SCD_ICML.md`.
For portable data/model layout, see `DATA_PREPARATION.md` and
`configs/models_mistrallite_to_mistral7_template.json`.
For the minimal release-facing file set and self-check, see
`OPEN_SOURCE_RELEASE_MANIFEST.md`.
The package is released under the MIT license; see `LICENSE`.

To regenerate profile-based analysis figures without rerunning models:

```bash
python scripts/build_scd_icml_analysis_figures.py \
  --profile <run-root>/lite_to_7b/route/profile_cmrc2018_seed42.json \
  --out-dir <figure-output-dir>
```

This produces layer-profile CSV/Markdown plus SVG figures for Reuse-vs-Patch
error, expected Patch gain, and cumulative gain curves. The same profile-backed
figures are also emitted as PDF/PNG so the paper can include generated figures
directly. It does not generate t-SNE figures; those require saved
representation samples.

To extract and regenerate the stack/t-SNE visualization from pair artifacts and
a Reuse checkpoint:

```bash
python scripts/extract_stack_viz_inputs.py \
  --pairs <run-root>/lite_to_7b/pairs/cmrc2018_seed42 \
  --reuse-ckpt <run-root>/lite_to_7b/ckpt_reuse/cmrc2018_seed42 \
  --profile <run-root>/lite_to_7b/route/profile_cmrc2018_seed42.json \
  --dataset cmrc2018 \
  --save-vectors-npz <analysis-output-dir>/cmrc2018_stack_viz_vectors.npz \
  --out-dir <analysis-output-dir>
```

Install the optional analysis dependency when you need true t-SNE rather than
the built-in PCA fallback:

```bash
python -m pip install -e ".[analysis]"
```

If the first extraction was run on a server without `scikit-learn`, regenerate
the CSV/meta inputs from the sampled vector NPZ in an analysis-capable
environment:

```bash
python scripts/extract_stack_viz_inputs.py \
  --vectors-npz <analysis-output-dir>/cmrc2018_stack_viz_vectors.npz \
  --dataset cmrc2018 \
  --out-dir <analysis-output-dir>
```

Then render the figure from the generated CSV/meta artifacts:

```bash
python scripts/build_stack_viz.py \
  --distance-csv <analysis-dir>/cmrc2018_kv_distance_distribution.csv \
  --distance-meta <analysis-dir>/cmrc2018_kv_distance_distribution_meta.json \
  --tsne-csv <analysis-dir>/cmrc2018_tsne_kv_alignment.csv \
  --tsne-meta <analysis-dir>/cmrc2018_tsne_kv_alignment_meta.json \
  --out-dir <figure-output-dir>
```

The extractor uses existing pair artifacts and a Reuse checkpoint; it does not
rerun model forward passes unless you first regenerate the pair artifacts.

To regenerate the offline calibration/training cost table from raw
`/usr/bin/time -v` logs:

```bash
python scripts/build_offline_cost_table.py \
  --entry capture_pair=<run-root>/logs/lite_to_7b_capture_pair/time.txt \
  --entry profile_layers=<run-root>/logs/lite_to_7b_profile_layers/time.txt \
  --entry train_reuse=<run-root>/logs/lite_to_7b_train_reuse/time.txt \
  --entry train_patch=<run-root>/logs/lite_to_7b_train_patch/time.txt \
  --out-dir <cost-output-dir>
```

This writes CSV/Markdown/JSON plus `offline_cost_wall_time.svg`. If a time log
exists but lacks standard `/usr/bin/time -v` fields, the row is kept with
`status=no_time_fields`.

To assemble generated tables and figures into a paper-facing artifact bundle:

```bash
python scripts/build_paper_artifact_bundle.py \
  --result-tables-dir <result-table-output-dir> \
  --profile-figures-dir <profile-figure-output-dir> \
  --stack-viz-dir <stack-viz-output-dir> \
  --offline-cost-dir <cost-output-dir> \
  --out-dir <paper-artifact-bundle-dir>
```

The bundle command copies existing generated artifacts and writes
`paper_artifact_manifest.json`; it does not compute or invent missing metrics.
Current result-table bundles include TTFT breakdown CSV/Markdown/LaTeX and
per-dataset latency-quality/Pareto SVGs derived from the same completed
summary JSON files.

To validate that a paper source is using generated figure assets and a complete
artifact bundle:

```bash
python scripts/check_paper_artifacts.py \
  --paper-tex <paper.tex> \
  --bundle-dir <paper-artifact-bundle-dir> \
  --forbid-image patch_error1.pdf \
  --forbid-image layer_selection.pdf \
  --forbid-image stack_viz.pdf \
  --require-text "not part of the current verified release evidence" \
  --require-text "do not use them as a claim-facing comparison" \
  --require-text "HotpotQA-compressed reports the full7405 selected route" \
  --require-text "\label{tab:current_cmrc_selected_route}" \
  --require-text "\label{tab:current_reuse_boundary}" \
  --require-text "\label{tab:current_non_cmrc_diagnostics}" \
  --require-text "\label{tab:offline_cost}" \
  --forbid-text "full7405 remains pending" \
  --forbid-text "full7405 pending" \
  --forbid-text "full7405 is not yet closed" \
  --forbid-text "full7405 remains the final open"
```

The text checks are intended for the current safety revision: they verify that
unsupported neighboring-compression rows remain explicitly excluded instead of
silently returning as claim-facing comparison tables, and that the paper source
does not keep stale full7405-pending language after the compressed-protocol
tables have been generated.

To consolidate completed HotpotQA-compressed full7405 route evaluations,
generate the post-full7405 metrics, selector report, and result tables with:

```bash
POST_ROOT=<postprocess-dir> \
EVAL_ROOTS="<reuse04-eval-root>:<reuse05-eval-root>:<reuse06-eval-root>:<reuse07-eval-root>:<reuse08-eval-root>" \
PATCH_ANCHOR_JSONS="<cmrc-patch-anchor-summary>:<hotpot-wiki-patch-anchor-summary>" \
BASE_ROUTE_METRICS_JSONS="<cmrc-route-metrics>:<hotpot-low-route-metrics>:<hotpot-mid-route-metrics>:<wikitext2-route-metrics>" \
PYTHON=<python> \
bash scripts/wait_postprocess_scd_icml_full7405_tables.sh
```

This postprocess wrapper waits for every eval root to contain
`scd_icml_mistral_summary.json`, then writes
`full7405_route_metrics.json`, `hotpotqa_full7405_f1_drop003_selection.json`,
and `result_tables_with_full7405/`. The table directory includes
`bandwidth_sensitivity.csv`, `bandwidth_sensitivity.md`, and
`bandwidth_sensitivity.svg` for rows with measured `payload_mb`; rows without
payload measurements are left out of that derived bandwidth sweep rather than
filled with synthetic payload values.

After `result_tables_with_full7405/` exists, build the final paper-facing bundle
and run the same paper gate with:

```bash
python scripts/finalize_scd_icml_full7405_bundle.py \
  --result-tables-dir <postprocess-dir>/result_tables_with_full7405 \
  --profile-figures-dir <profile-figure-output-dir> \
  --stack-viz-dir <stack-viz-output-dir> \
  --offline-cost-dir <cost-output-dir> \
  --out-dir <final-paper-artifact-bundle-dir> \
  --paper-tex <paper.tex> \
  --paper-base-dir <paper-dir>
```

This wrapper only assembles and validates generated artifacts; it does not
change metric values or edit the paper source. If
`hotpotqa_full7405_f1_drop003_selection.json` is present next to the
`result_tables_with_full7405/` directory, the finalizer also requires the paper
source to mention the selector's chosen `ReuseXX/PatchYY` route. This catches
the common stale-paper case where a boundary refinement selects a higher reuse
count but the paper still says `Reuse04/Patch28`.

For long remote runs, the finalization can also be left waiting for the
postprocess table:

```bash
POST_ROOT=<postprocess-dir> \
PROFILE_FIGURES_DIR=<profile-figure-output-dir> \
STACK_VIZ_DIR=<stack-viz-output-dir> \
OFFLINE_COST_DIR=<cost-output-dir> \
PYTHON=<python> \
bash scripts/wait_finalize_scd_icml_full7405_bundle.sh
```

The wait wrapper exits only after the mixed-route table exists and contains a
`full7405` source row. It skips paper-source checking when no paper path is
provided, which keeps remote experiment hosts usable without a paper checkout.
Run the finalizer locally with `--paper-tex`, or set `PAPER_TEX` and optionally
`PAPER_BASE_DIR` if the paper source is mounted on the remote host; when
`PAPER_TEX` is present, the wait wrapper enables paper checking by default
unless `SKIP_PAPER_CHECK=1` is set explicitly. With paper checking enabled, the
same selector-route text check is applied automatically when the selector
report exists in `POST_ROOT`.

If a long run is interrupted after dataset summaries are written, regenerate the
aggregate table with:

```bash
python -m scd_repro.cli.summarize_mistral_repro \
  --run-root <run-root> \
  --eval-config configs/eval_scd_icml_no_fallback_no_recompute.json \
  --expected-directions lite_to_7b \
  --expected-datasets cmrc2018,hotpotqa,wikitext2
```

For a claim-boundary check that compares only Full Consumer and SCD on a fixed
selected route, use
`configs/eval_scd_icml_full_scd_no_fallback_no_recompute.json` with
`scripts/run_ser_scd_icml_eval_existing_route.sh`.

Before publishing, run:

```bash
python scripts/check_release_readiness.py
```

## Read This Before Uploading

- `configs/` still includes some older experiment branches for reproducibility.
- If you want an even more minimal GitHub repo, trim to `src/`, the subset of
  `configs/` you still use, selected `scripts/`, and this README.
