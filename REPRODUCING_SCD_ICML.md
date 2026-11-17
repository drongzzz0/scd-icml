# Reproducing the SCD-ICML Mistral Results

This document is the release-facing reproduction entry point for the current
runtime code. It targets an approximate strict-runtime reproduction of the
SCD-ICML Mistral result trend, not a byte-for-byte recreation of every
camera-ready table value.

## Scope

The default reproduction is:

- Direction: `MistralLite -> Mistral-7B` (`lite_to_7b`)
- Calibration data: 200 CMRC2018 prefixes
- Evaluation data: CMRC2018, HotpotQA, WikiText-2
- Methods: full consumer, raw KV BF16, 4-bit KV, reuse-only, and full SCD
- SCD route semantics: non-patch layers use Reuse, selected transition layers
  use Patch, and the claim-facing path does not use fallback or recomputation
- Qwen-32B: skipped by default because the current `ser` machine cannot host
  that pair reliably

The reverse Mistral direction is available as an explicit diagnostic with
`DIRECTIONS=lite_to_7b,7b_to_lite`, but it is not the default paper-facing run.

## Environment

The verified run used:

- Host: `ser`
- Runtime directory:
  `/disk_n/c101-3-mql/projects/SCD-ICML/repro_current_from_codex_20260607-154030/runtime_repo`
- Data root:
  `/disk_n/c101-3-mql/projects/SCD-ICML/kv/snapshots/scd-github-f7d7854`
- Models:
  `/disk_n/c101-3-mql/models/MistralLite`
  and `/disk_n/c101-3-mql/models/Mistral-7B-v0.1`
- Python environment:
  `/disk_n/c101-3-mql/venvs/A800`

Install the package from this directory:

```bash
python -m pip install -e .
```

For portable setup on another machine, see `DATA_PREPARATION.md` and copy
`configs/models_mistrallite_to_mistral7_template.json` to a local model config.
The wrapper accepts `DATA_ROOT` and `MODELS_CONFIG_LITE_TO_7B` overrides, so the
script itself does not need to be edited.

## Smoke Test

Use this only to check that model loading, pair capture, training, route
generation, strict evaluation, and summary generation are wired correctly:

```bash
TRAIN_LIMIT=2 EVAL_LIMIT=1 \
DIRECTIONS=lite_to_7b DATASETS=cmrc2018 \
MAX_LENGTH=256 MAX_NEW_TOKENS=8 \
bash scripts/run_ser_scd_icml_mistral_repro.sh
```

If only one 24 GB GPU is idle, use the single-GPU smoke placement:

```bash
MODELS_CONFIG_LITE_TO_7B=configs/models_ser_mistrallite_to_mistral7_mql_single_gpu.json \
CAPTURE_A_DEVICE=cuda:0 CAPTURE_B_DEVICE=cuda:0 \
PRODUCER_DEVICE=cuda:0 CONSUMER_DEVICE=cuda:0 TRAIN_DEVICE=cuda:0 \
TRAIN_LIMIT=2 EVAL_LIMIT=1 \
DIRECTIONS=lite_to_7b DATASETS=cmrc2018 \
MAX_LENGTH=256 MAX_NEW_TOKENS=8 \
bash scripts/run_ser_scd_icml_mistral_repro.sh
```

Do not use the smoke metrics as claim evidence.

## Historical Patch-only Anchor

The older GitHub/new-main code path has a stronger Patch-only anchor than the
current selected mixed route. Use it to verify the historical trend before
searching for a mixed Reuse+Patch route:

```bash
DATASETS=cmrc2018,hotpotqa_compressed,wikitext2 \
EVAL_LIMIT=0 \
bash scripts/run_ser_scd_icml_patchonly_anchor.sh
```

This wrapper evaluates Full Consumer, Reuse-only, and Patch-only using
dataset-specific Reuse/Patch banks from the `scd-github-f7d7854` snapshot. The
HotpotQA row uses the historical compressed-prompt protocol and should not be
mixed with the full-context HotpotQA limitation runs. The anchor is not the
final SCD route; it is the quality target for later mixed-route optimization.
For interrupted anchor runs, set `RESUME=1` with the same `RUN_ROOT` to skip
dataset/stage outputs that already contain `per_mode_summary.json`.

## Main Reproduction Run

Run from this directory on `ser`:

```bash
TRAIN_LIMIT=200 EVAL_LIMIT=200 \
DATASETS=cmrc2018 \
MAX_LENGTH=0 MAX_NEW_TOKENS=32 \
WAIT_TRAIN_FREE_MB=6000 WAIT_EVAL_FREE_MB=18000 WAIT_CAPTURE_FREE_MB=18000 \
bash scripts/run_ser_scd_icml_mistral_repro.sh
```

For a non-`ser` layout:

```bash
DATA_ROOT=/absolute/path/to/data-root \
MODELS_CONFIG_LITE_TO_7B=configs/models_local_mistrallite_to_mistral7.json \
TRAIN_LIMIT=200 EVAL_LIMIT=200 \
DATASETS=cmrc2018 \
MAX_LENGTH=0 MAX_NEW_TOKENS=32 \
bash scripts/run_ser_scd_icml_mistral_repro.sh
```

The wrapper selects the number of GPUs implied by the configured CUDA devices
when `CUDA_VISIBLE_DEVICES` is unset. The default full run uses two logical
devices; the single-GPU smoke command above uses one. It then waits for the
configured free-memory gates before each stage. Defaults are
`WAIT_CAPTURE_FREE_MB=18000`, `WAIT_TRAIN_FREE_MB=6000`, and
`WAIT_EVAL_FREE_MB=18000`.
If `CUDA_VISIBLE_DEVICES` is unset and the capture gate is positive, the wrapper
first waits until the required number of GPUs meet the capture free-memory
threshold, then selects those GPUs. This avoids locking onto a busy card before
it becomes usable. In automatic selection mode, capture/eval waits can also
re-scan and reselect a different placement if another job occupies the initially
selected GPU before model loading starts.
For a fixed placement, set `CUDA_VISIBLE_DEVICES`, for example:

For mixed-route screening, use
`scripts/run_ser_scd_icml_conditional_route_screen.sh`. It is a simple
work-queue runner: each GPU pair dequeues another candidate route as soon as it
finishes. `REUSE_DIR`, `PATCH_DIR`, and `REPORT_DATASET` can be set explicitly
for dataset-specific banks.

```bash
CUDA_VISIBLE_DEVICES=3,1 TRAIN_LIMIT=200 EVAL_LIMIT=200 \
DATASETS=cmrc2018 \
MAX_LENGTH=0 MAX_NEW_TOKENS=32 \
bash scripts/run_ser_scd_icml_mistral_repro.sh
```

The run writes:

- `<run-root>/<direction>/pairs/...`
- `<run-root>/<direction>/ckpt_reuse/...`
- `<run-root>/<direction>/ckpt_patch/...`
- `<run-root>/<direction>/route/...`
- `<run-root>/<direction>/strict_eval/<dataset>/summary.md`
- `<run-root>/scd_icml_mistral_summary.md`
- `<run-root>/scd_icml_mistral_summary.json`

If the run is interrupted after the dataset summaries exist, regenerate and
validate the aggregate table without rerunning the model:

```bash
python -m scd_repro.cli.summarize_mistral_repro \
  --run-root <run-root> \
  --eval-config configs/eval_scd_icml_no_fallback_no_recompute.json \
  --expected-directions lite_to_7b \
  --expected-datasets cmrc2018,hotpotqa,wikitext2
```

## Current Claim Boundary

The earlier local run at:

```text
/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_mistral_repro_20260607_1605
```

is retained only as a diagnostic artifact. Its `scd_full` route was generated
with non-patch layers listed as `fallback_layers`, so it is not valid evidence
for the no-fallback/no-recompute SCD claim.

The claim-facing run must use:

```text
configs/eval_scd_icml_no_fallback_no_recompute.json
```

and a paper SCD route generated from the offline profile by:

```bash
python scripts/make_paper_scd_route.py \
  --profile <run-root>/lite_to_7b/route/profile_cmrc2018_seed42.json \
  --patch-dir <run-root>/lite_to_7b/ckpt_patch/patch_cmrc2018_seed42 \
  --models-config <models-config> \
  --eval-config configs/eval_scd_icml_no_fallback_no_recompute.json \
  --patch-budget 6 \
  --out <run-root>/lite_to_7b/route/route_cmrc2018_seed42.json
```

The generated route must have:

```text
route_semantics=paper_scd_sparse_patch_reuse_all_other_layers
fallback_layers=[]
selected_recompute_layers=[]
reuse_layers=<all non-patch layers>
patch_layers=<offline-selected transition layers>
```

The wrapper can perform the historical paper-profile step automatically with
`ROUTE_MODE=paper_profile` and `ROUTE_PATCH_BUDGET=6`. Treat that route as a
baseline or diagnostic, not as the final selector. The final reproduction claim
should use teacher-aligned conditional route search:

```text
rho* = argmin Cost(rho)
       subject to J_teacher(rho) <= epsilon

J_teacher(rho) =
    lambda_KV  D_KV(rho)
  + lambda_h   D_hidden(rho)
  + lambda_KL  D_KL(rho)
  + lambda_PPL DeltaPPL(rho)
```

Start from a high-quality/all-Patch route, then change Patch to Reuse only when
the conditional marginal loss stays within the configured KV, hidden, KL, PPL,
and total-drift budgets. Fixed sparse top-k candidates from the offline profile
remain useful baselines and pending-eval seeds, but independent layer ranking is
not the claim-facing selector.
`ROUTE_PATCH_LAYERS` remains available for explicit manual ablations, but every
claim-facing route must be fixed before online evaluation and validated as a
mixed route.

The current CMRC200 F1/EM-primary selected route is:

```text
configs/routes/scd_icml_mistrallite_to_mistral7_reuse19_patch13_cmrc200.json
```

It uses Reuse layers `[3,5,6,8,9,11,13,14,17,20,21,22,23,24,25,27,28,29,30]`
and Patch layers `[0,1,2,4,7,10,12,15,16,18,19,26,31]`, with
`fallback_layers=[]` and no recomputation. On CMRC2018 dev `n=200`, the
Full+SCD confirmation run reported Full Consumer F1 `0.6573`, EM `0.365`, and
SCD F1 `0.6428`, EM `0.350`, corresponding to F1 retention `0.9779` and EM
retention `0.9589`. All 13 one-layer Reuse20 extensions from this route were
tested on CMRC200; the best candidate reached only F1 retention `0.9300`. Avg NLL is
reported as an auxiliary calibration-risk signal, not as the primary
route-selection metric.

To rerun only Full Consumer and SCD on this fixed route:

```bash
SOURCE_RUN_ROOT=<source-run-root-with-reuse> \
PATCH_DIR=<patch-checkpoint-with-selected-patch-layers> \
ROUTE_PATH=configs/routes/scd_icml_mistrallite_to_mistral7_reuse19_patch13_cmrc200.json \
RUN_ROOT=<selected-route-eval-run-root> \
DATASETS=cmrc2018 CMRC_SPLIT=dev EVAL_LIMIT=200 \
EVAL_CONFIG_TEMPLATE=configs/eval_scd_icml_full_scd_no_fallback_no_recompute.json \
bash scripts/run_ser_scd_icml_eval_existing_route.sh
```

For a manual diagnostic layer set:

```bash
python scripts/make_paper_scd_route.py \
  --profile <run-root>/lite_to_7b/route/profile_cmrc2018_seed42.json \
  --patch-dir <run-root>/lite_to_7b/ckpt_patch/patch_cmrc2018_seed42 \
  --models-config <models-config> \
  --eval-config configs/eval_scd_icml_no_fallback_no_recompute.json \
  --patch-layers 10-13,15-18 \
  --out <run-root>/lite_to_7b/route/route_cmrc2018_seed42.json
```

Validate any generated route before evaluation:

```bash
python scripts/validate_scd_route.py \
  --route <run-root>/lite_to_7b/route/route_cmrc2018_seed42.json \
  --require-mixed
```

## High-Coverage Parallel Patch Shards

If the strict `k=6` mixed route is too weak for a given model pair, keep the
same no-fallback/no-recompute semantics and train a high-coverage Patch
candidate pool. The helper below trains layers 1-31 as mergeable layer-local
Patch shards, split across four GPUs:

```bash
SOURCE_RUN_ROOT=<run-root-with-pairs-and-profile> \
RUN_ROOT=<new-run-root> \
SHARDS='0:1-8:shard_01_08;1:9-16:shard_09_16;2:17-24:shard_17_24;3:25-31:shard_25_31' \
PATCH_LAYERS=1-31 \
WAIT_FOR_SHARDS=1 MERGE_AFTER=1 \
bash scripts/run_ser_scd_icml_parallel_patch_shards.sh
```

The helper writes:

```text
<new-run-root>/lite_to_7b/ckpt_patch_shards/<shard-name>/
<new-run-root>/lite_to_7b/ckpt_patch/patch_cmrc2018_seed42/
<new-run-root>/lite_to_7b/route/route_cmrc2018_seed42.json
```

The `PATCH_LAYERS=1-31` route is a high-coverage diagnostic. It is useful as an
upper bound and as a source checkpoint for sparse route selection, but it should
not be treated as the final speed-oriented SCD route. Generate fixed sparse
candidates from the merged Patch pool:

```bash
python scripts/make_paper_scd_candidate_routes.py \
  --profile <source-run-root>/lite_to_7b/route/profile_cmrc2018_seed42.json \
  --patch-dir <new-run-root>/lite_to_7b/ckpt_patch/patch_cmrc2018_seed42 \
  --models-config configs/models_ser_mistrallite_to_mistral7_mql.json \
  --eval-config configs/eval_scd_icml_no_fallback_no_recompute.json \
  --budgets 3,5,6,8,12,16,24,31 \
  --out-dir <new-run-root>/lite_to_7b/route/candidates
```

Then run the teacher-aligned route-selection scaffold. It writes schemas and
pending routes when the required teacher metrics are not available yet:

```bash
python scripts/estimate_layer_mapping.py --config configs/route_selection.yaml
python scripts/profile_routes.py --config configs/route_selection.yaml
python scripts/select_patch_reuse_route.py \
  --config configs/route_selection.yaml \
  --selector conditional_pruning \
  --metrics <route-level-metrics.jsonl>
python scripts/report_pairwise_interactions.py \
  --metrics <route-level-metrics.json> \
  --candidate-layers 1,8,25 \
  --out-json outputs/route_selection/pairwise_interactions.json
python scripts/evaluate_selected_route.py \
  --config configs/route_selection.yaml \
  --route outputs/route_selection/selected_route.json
```

For targeted interaction probes on `ser`, set `REUSE_SETS` on the conditional
route-screen wrapper, for example `REUSE_SETS='1;25;1,8;1,25'`. This generates
only those explicit mixed routes instead of all single-layer or prefix
candidates.

The eval wrappers default to `PATCH_CKPT_DEVICE=cpu`,
`RUNTIME_PATCH_WINDOW_SIZE=8`, and non-persistent Patch shards/modules. This is
the current safe Patch-heavy runtime default on 24GB GPUs; `32` was observed to
OOM for the Patch-30 route. Override these environment variables only when
running a dedicated runtime sweep.

The route-level metrics JSONL must contain real values for the metrics it
claims. Prefer KV drift, hidden drift, logit KL, PPL delta, and cost fields
when they have been profiled. If only strict eval summaries are available,
select QA routes by real F1/EM retention and WikiText-2 routes by PPL relative
increase. Report `avg_nll_delta` as auxiliary calibration-risk metadata unless
a diagnostic run explicitly enables the NLL hard gate; do not backfill invented
drift values.

Evaluate a fixed candidate without retraining:

```bash
SOURCE_RUN_ROOT=<source-run-root-with-reuse> \
PATCH_DIR=<new-run-root>/lite_to_7b/ckpt_patch/patch_cmrc2018_seed42 \
ROUTE_PATH=<new-run-root>/lite_to_7b/route/candidates/route_k08.json \
RUN_ROOT=<candidate-eval-run-root> \
DATASETS=cmrc2018,hotpotqa,wikitext2 \
MAX_LENGTH=0 \
bash scripts/run_ser_scd_icml_eval_existing_route.sh
```

The final selected route must still validate as mixed, with `fallback_layers=[]`,
empty recomputation fields, `Patch` on selected transition layers, and `Reuse`
on all remaining layers. Treat HotpotQA and WikiText-2 in this candidate sweep
as diagnostics/limitations unless they are separately selected and validated
under their own quality gates.

Select the maximum-Reuse route after candidate evals finish:

```bash
python scripts/select_quality_gated_scd_route.py \
  --eval-glob '<new-run-root>/evals/cmrc50_k*' \
  --dataset cmrc2018 \
  --metric f1 \
  --retention-threshold 0.95 \
  --out-report <new-run-root>/quality_gate_route_selection.json \
  --out-route <new-run-root>/lite_to_7b/route/selected_quality_gated_route.json
```

For QA datasets with a low Full Consumer F1 baseline, an absolute F1/EM loss
gate may be more interpretable than a retention ratio. Use `--max-metric-drop`
to select the maximum-Reuse route whose absolute metric loss stays within the
chosen bound:

```bash
python scripts/select_quality_gated_scd_route.py \
  --eval-glob '<new-run-root>/evals/hotpotqa_k*' \
  --dataset hotpotqa_compressed \
  --metric f1 \
  --max-metric-drop 0.03 \
  --out-report <new-run-root>/hotpotqa_f1_drop_gate_route_selection.json
```

For WikiText-2, use PPL as a lower-is-better gate. The default PPL relative
increase gate is `0.10` when `--metric ppl` is selected; pass
`--max-ppl-relative-increase` to change it:

```bash
python scripts/select_quality_gated_scd_route.py \
  --eval-glob '<new-run-root>/evals/wikitext2_k*' \
  --dataset wikitext2 \
  --metric ppl \
  --max-ppl-relative-increase 0.10 \
  --out-report <new-run-root>/wikitext2_ppl_gate_route_selection.json
```

The expected aggregate table has 15 rows: 3 datasets times 5 methods:
`full_consumer`, `raw_kv`, `quantized_4bit`, `reuse_only`, and `scd_full`.

Verified selected-route aggregate run:

```text
/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_selected_reuse19_5method_aggregate_20260610_034426
```

This run uses the selected Reuse19/Patch13 route, disables fallback and
recomputation, and passed:

```bash
python scripts/check_release_readiness.py --root . --run-root /disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_selected_reuse19_5method_aggregate_20260610_034426
```

Key rows and current claim boundary:

| dataset | Full Consumer | SCD Reuse19/Patch13 | interpretation |
|---|---:|---:|---|
| CMRC2018 | F1 `0.6573`, EM `0.365` | F1 `0.6428`, EM `0.350` | QA quality retained; F1 retention `0.9779`, EM retention `0.9589`. |
| HotpotQA full-context | F1 `0.4005`, EM `0.265` | F1 `0.1383`, EM `0.105` | Negative cross-dataset result under complete distractor context and `MAX_LENGTH=0`; do not cite as preserved quality. |
| WikiText-2 | NLL `2.0260`, PPL `7.5840` | NLL `3.2742`, PPL `26.4231` | The CMRC-selected route is not a universal language-modeling route; cite as auxiliary risk. |

The HotpotQA row above comes from
`/disk_n/c101-3-mql/projects/SCD-ICML/runs/hotpotqa_fullcontext_full_scd_20260610_115952`,
not from the older 5-method aggregate. The older aggregate HotpotQA rows used a
now-stale prompt/truncation setup and should not be used for quality claims.
Training HotpotQA-specific Reuse/Patch banks on a disjoint split also did not
rescue this fixed CMRC route: the heldout full-context run
`/disk_n/c101-3-mql/projects/SCD-ICML/runs/hotpotqa_full_banks_20260610_192156`
reported Full Consumer F1 `0.3838`, EM `0.235` versus SCD F1 `0.0920`, EM
`0.065`.

## Legacy Diagnostic 200-Sample Result

Do not cite the following table as the final SCD claim. It is useful only for
tracking the pre-fix behavior that mixed Patch with consumer fallback layers.
Its HotpotQA rows are additionally stale after the full-context/no-truncation
HotpotQA correction.

CMRC2018:

| method | F1 | EM | PPL | TTFT ms |
|---|---:|---:|---:|---:|
| Full Consumer | 0.6573 | 0.3650 | 1.7670 | 101.38 |
| Raw KV BF16 | 0.0918 | 0.0000 | 2.0450 | 29.90 |
| Quantized 4-bit | 0.1097 | 0.0000 | 2.0933 | 52.82 |
| Selective Recompute | 0.5223 | 0.2150 | 1.8929 | 99.27 |
| Reuse-only | 0.0089 | 0.0000 | 2689.6618 | 34.37 |
| SCD | 0.7347 | 0.4250 | 1.7174 | 85.23 |

HotpotQA:

| method | F1 | EM | PPL | TTFT ms |
|---|---:|---:|---:|---:|
| Full Consumer | 0.0343 | 0.0100 | 105.3525 | 140.14 |
| Raw KV BF16 | 0.0133 | 0.0000 | 18.5776 | 32.97 |
| Quantized 4-bit | 0.0151 | 0.0000 | 18.6704 | 56.52 |
| Selective Recompute | 0.0324 | 0.0000 | 73.8773 | 139.63 |
| Reuse-only | 0.0079 | 0.0000 | 7342.7770 | 42.40 |
| SCD | 0.0419 | 0.0100 | 93.5093 | 116.40 |

WikiText-2:

| method | avg NLL | PPL | TTFT ms |
|---|---:|---:|---:|
| Full Consumer | 2.0260 | 7.5840 | 64.11 |
| Raw KV BF16 | 2.3637 | 10.6300 | 25.17 |
| Quantized 4-bit | 2.3719 | 10.7175 | 48.41 |
| Selective Recompute | 2.0590 | 7.8385 | 62.21 |
| Reuse-only | 6.6578 | 778.8231 | 26.50 |
| SCD | 2.0328 | 7.6356 | 56.49 |

## Recommended Claim Boundary

Use cautious wording:

- The code implements the SCD-ICML Mistral path as Reuse on non-patch layers and
  Patch on selected transition layers.
- Do not cite any `scd_full` result whose route contains fallback layers or
  recomputation.
- For CMRC2018 dev `n=200`, the current F1/EM-primary selected route is
  Reuse19/Patch13. Cite the selected-route run root and report the exact model
  paths, sample counts, decoding settings, skipped Qwen-32B scope, and the NLL
  risk note.
- For the broader 5-method x 3-dataset aggregate table, cite only a fresh
  no-fallback/no-recompute run that passes
  `check_release_readiness.py --run-root`, and treat that check as structural
  rather than a quality gate.
- For HotpotQA, treat the full-context/no-truncation runs as diagnostic
  evidence only. They show poor Reuse19/Patch13 quality retention under a
  different prompt protocol and should not replace the paper-facing
  `hotpotqa_compressed` full7405 route table.
- The older GitHub snapshot's patch-only HotpotQA result is a separate
  compressed-prompt benchmark, not evidence for the diagnostic full-context
  HotpotQA rows. The current runtime exposes that protocol as
  `--dataset hotpotqa_compressed`; that GitHub-compatible compressed protocol
  is now the paper-facing HotpotQA benchmark. The current full7405 bracket has
  closed for `reuse04_patch28` versus `reuse08_patch24`: `reuse04_patch28`
  passes the F1-drop gate and `reuse08_patch24` fails it. Its old full7405 eval
  includes the 100 training questions used for the old patch checkpoint, but a
  question-text overlap audit found that removing those 100 questions leaves
  Patch-only at F1 `0.3514`, EM `0.2516`; the result is real for the
  compressed-prompt setting.
- When using HotpotQA-trained checkpoints, regenerate leakage-sensitive metrics
  by filtering saved per-sample outputs:

```bash
python scripts/filter_strict_eval_by_qids.py \
  --strict-eval-dir <route-eval-root>/lite_to_7b/strict_eval/hotpotqa_compressed \
  --exclude-qid-file <train-overlap-qids.txt> \
  --out-dir <route-eval-root>/lite_to_7b/strict_eval/hotpotqa_compressed_no_train_overlap \
  --summary-list-out <route-eval-root>/scd_icml_mistral_summary.no_train_overlap.json \
  --dataset hotpotqa_compressed
```

This command does not rerun GPU inference; it recomputes Full/SCD metrics from
`per_sample_metrics.jsonl` after removing qids that overlap with the training
questions.

## Current HotpotQA Protocol Update

The current SCD-ICML consolidation uses the GitHub-compatible compressed
HotpotQA protocol for the paper-facing HotpotQA check:

```text
dataset=hotpotqa_compressed
MAX_LENGTH=512
full_eval_examples=7405
```

The full-context rows above remain useful diagnostics, but they are not the
final HotpotQA reproduction target for the restored GitHub code path. The
current full-dev validation uses selected mixed Reuse+Patch routes from
HotpotQA-trained Reuse/Patch banks. The full7405 boundary is already closed:
`reuse04_patch28`, `reuse05_patch27`, and `reuse06_patch26` pass the absolute
F1-drop gate, while `reuse07_patch25` and `reuse08_patch24` fail. The current
maximum passing route is `reuse06_patch26`. Regenerate the paper-facing
CSV/Markdown/LaTeX tables with `scripts/build_scd_icml_result_tables.py` and use
`scripts/select_quality_gated_scd_route.py --dataset hotpotqa_compressed
--metric f1 --max-metric-drop 0.03` to select the maximum-Reuse passing route.

## Adapted Neighboring Baselines

The historical appendix comparison can be regenerated with the three
CMRC-strict adapted compression entries:

```bash
LIMIT=200 RUN_ROOT=/disk_n/c101-3-mql/projects/SCD-ICML/runs/neighboring_baselines_cmrc200 \
  bash scripts/run_ser_neighboring_baselines_cmrc.sh
```

The wrapper runs SVD-LLM-style and TensorLLM-style in parallel on two GPU pairs,
then runs SliceGPT-style. It builds the consolidated table only when
`SCD_SUMMARY` is set to a same-protocol selected SCD summary. This avoids
mixing adapted baseline rows from one protocol, such as `max_length=256`, with
an SCD row from another protocol, such as `max_length=1024`. When `SCD_SUMMARY`
is provided, the wrapper passes `--require-compatible-protocol` so protocol
mismatches fail instead of producing claim-facing outputs. The individual
commands are:

```bash
python scripts/run_svdllm_style_kv_cmrc.py \
  --project-root . \
  --models-config configs/models_ser_mistrallite_to_mistral7_mql.json \
  --data-root /disk_n/c101-3-mql/datasets \
  --out <run-root>/svdllm

python scripts/run_slicegpt_style_kv_cmrc.py \
  --project-root . \
  --models-config configs/models_ser_mistrallite_to_mistral7_mql.json \
  --data-root /disk_n/c101-3-mql/datasets \
  --pair-dir <pair-artifact-dir> \
  --out <run-root>/slicegpt

python scripts/run_tensorllm_style_kv_cmrc.py \
  --project-root . \
  --models-config configs/models_ser_mistrallite_to_mistral7_mql.json \
  --data-root /disk_n/c101-3-mql/datasets \
  --out <run-root>/tensorllm
```

After all three runs and a selected SCD run with the same `LIMIT`,
`MAX_LENGTH`, `MAX_NEW_TOKENS`, and `BANDWIDTH_GBPS` are complete, consolidate
the appendix table:

```bash
python scripts/build_neighboring_baseline_table.py \
  --svdllm-summary <run-root>/svdllm/summary.json \
  --slicegpt-summary <run-root>/slicegpt/summary.json \
  --tensorllm-summary <run-root>/tensorllm/summary.json \
  --scd-summary <selected-scd-summary.json> \
  --require-compatible-protocol \
  --out-dir <run-root>/neighboring_baselines
```

Do not restore these rows to claim-facing text unless the generated table is
based on completed current-runtime summaries.

## Regenerating Profile-Based Analysis Figures

Layer-profile figures can be regenerated from a completed
`profile_cmrc2018_seed42.json` artifact:

```bash
python scripts/build_scd_icml_analysis_figures.py \
  --profile <run-root>/lite_to_7b/route/profile_cmrc2018_seed42.json \
  --out-dir <figure-output-dir>
```

The script produces:

- `layer_profile.csv/md`;
- `layer_error_profile.svg/pdf/png`;
- `layer_gain_profile.svg/pdf/png`;
- `patch_gain_curve.csv/md/svg/pdf/png`.

These are reproducible replacements for profile-backed layer error and layer
selection visualizations. They are not a replacement for t-SNE alignment
figures, which require saved representation samples rather than only profile
JSON.

## Regenerating Stack/t-SNE Visualizations

The stack/t-SNE figure can be regenerated from saved pair artifacts and a Reuse
checkpoint. First extract the CSV/meta analysis inputs:

```bash
python scripts/extract_stack_viz_inputs.py \
  --pairs <run-root>/lite_to_7b/pairs/cmrc2018_seed42 \
  --reuse-ckpt <run-root>/lite_to_7b/ckpt_reuse/cmrc2018_seed42 \
  --profile <run-root>/lite_to_7b/route/profile_cmrc2018_seed42.json \
  --dataset cmrc2018 \
  --save-vectors-npz <analysis-output-dir>/cmrc2018_stack_viz_vectors.npz \
  --out-dir <analysis-output-dir>
```

For true t-SNE, install the optional analysis dependency:

```bash
python -m pip install -e ".[analysis]"
```

If the extractor first ran on a server without `scikit-learn`, rerun it from
the sampled vectors in an analysis-capable environment:

```bash
python scripts/extract_stack_viz_inputs.py \
  --vectors-npz <analysis-output-dir>/cmrc2018_stack_viz_vectors.npz \
  --dataset cmrc2018 \
  --out-dir <analysis-output-dir>
```

Then render the PDF/PNG/SVG figure:

```bash
python scripts/build_stack_viz.py \
  --distance-csv <analysis-dir>/cmrc2018_kv_distance_distribution.csv \
  --distance-meta <analysis-dir>/cmrc2018_kv_distance_distribution_meta.json \
  --tsne-csv <analysis-dir>/cmrc2018_tsne_kv_alignment.csv \
  --tsne-meta <analysis-dir>/cmrc2018_tsne_kv_alignment_meta.json \
  --out-dir <figure-output-dir>
```

The extractor uses existing pair artifacts and does not rerun model forward
passes. To make this fully current-runtime end to end, regenerate the pair
artifacts first and then run these two commands.

## Regenerating Offline Cost Tables

Offline calibration/training wall-clock and memory tables should be generated
from raw `/usr/bin/time -v` logs:

```bash
python scripts/build_offline_cost_table.py \
  --entry capture_pair=<run-root>/logs/lite_to_7b_capture_pair/time.txt \
  --entry profile_layers=<run-root>/logs/lite_to_7b_profile_layers/time.txt \
  --entry train_reuse=<run-root>/logs/lite_to_7b_train_reuse/time.txt \
  --entry train_patch=<run-root>/logs/lite_to_7b_train_patch/time.txt \
  --out-dir <cost-output-dir>
```

The script emits `offline_cost_table.csv/md`,
`offline_cost_summary.json`, and `offline_cost_wall_time.svg`. A row with
`status=no_time_fields` means the source file existed but did not contain the
standard timing fields, so that stage must not be used as a measured-cost
claim.

## Assembling Paper Artifacts

After regenerating the result tables, profile figures, stack/t-SNE figure, and
offline-cost table, assemble a paper-facing bundle without recomputing metrics:

```bash
python scripts/build_paper_artifact_bundle.py \
  --result-tables-dir <result-table-output-dir> \
  --profile-figures-dir <profile-figure-output-dir> \
  --stack-viz-dir <stack-viz-output-dir> \
  --offline-cost-dir <cost-output-dir> \
  --neighboring-baselines-dir <neighboring-baseline-table-dir> \
  --out-dir <paper-artifact-bundle-dir>
```

The command writes `paper_artifact_manifest.json` plus `tables/`, `figures/`,
and `manifests/` subdirectories. Use `--allow-missing` only for partial bundles
while long evaluations, such as HotpotQA-compressed full7405, are still running.
The result-table input directory can include TTFT breakdown tables and
latency-quality/Pareto SVGs generated by
`scripts/build_scd_icml_result_tables.py`; these are copied into the bundle
without recomputing model metrics. The same generator also writes
`bandwidth_sensitivity.csv`, `bandwidth_sensitivity.md`, and
`bandwidth_sensitivity.svg` by recomputing payload transfer time for rows that
contain measured `payload_mb`; it does not impute payloads for mixed-route rows
whose summaries lack payload bytes.
Profile-backed PDF figures generated by
`scripts/build_scd_icml_analysis_figures.py` are copied as paper-ready figure
replacements.
Before finalizing the paper source, validate that it references generated
figures and a complete artifact bundle:

```bash
python scripts/check_paper_artifacts.py \
  --paper-tex <paper.tex> \
  --bundle-dir <paper-artifact-bundle-dir> \
  --forbid-image patch_error1.pdf \
  --forbid-image layer_selection.pdf \
  --forbid-image stack_viz.pdf \
  --require-bundle-file tables/neighboring_baselines.md \
  --require-bundle-file tables/neighboring_baselines.tex \
  --require-bundle-file tables/neighboring_baselines.csv \
  --require-bundle-file tables/patch_anchor_table.md \
  --require-bundle-file tables/patch_anchor_table.tex \
  --require-bundle-file tables/patch_anchor_table.csv \
  --require-bundle-file manifests/neighboring_baselines_manifest.json \
  --require-text "\label{tab:neighboring_baselines}" \
  --require-text "\label{tab:patchonly_anchors}" \
  --require-text "same-protocol CMRC2018 rerun" \
  --require-text "HotpotQA-compressed reports the full7405 selected route" \
  --require-text "\label{tab:current_cmrc_selected_route}" \
  --require-text "\label{tab:current_reuse_boundary}" \
  --require-text "\label{tab:current_non_cmrc_diagnostics}" \
  --require-text "\label{tab:offline_cost}" \
  --forbid-text "not part of the current verified release evidence" \
  --forbid-text "do not use them as a claim-facing comparison" \
  --forbid-text "full7405 remains pending" \
  --forbid-text "full7405 pending" \
  --forbid-text "full7405 is not yet closed" \
  --forbid-text "full7405 remains the final open"
```

The neighboring-table checks are for the current final revision. They ensure the
same-protocol appendix table is present in both the paper source and the bundle,
and they forbid the older safety text that excluded neighboring baselines from
claim-facing evidence. The full7405 and label checks ensure the paper source
still contains the current claim-facing tables and does not retain stale pending
language.

When HotpotQA-compressed full7405 route evaluations are running on a remote
host, leave the postprocess step waiting for their aggregate summaries:

```bash
POST_ROOT=<postprocess-dir> \
EVAL_ROOTS="<reuse04-eval-root>:<reuse05-eval-root>:<reuse06-eval-root>:<reuse07-eval-root>:<reuse08-eval-root>" \
PATCH_ANCHOR_JSONS="<cmrc-patch-anchor-summary>:<hotpot-wiki-patch-anchor-summary>" \
BASE_ROUTE_METRICS_JSONS="<cmrc-route-metrics>:<hotpot-low-route-metrics>:<hotpot-mid-route-metrics>:<wikitext2-route-metrics>" \
PYTHON=<python> \
bash scripts/wait_postprocess_scd_icml_full7405_tables.sh
```

The postprocess wrapper waits for each eval root's
`scd_icml_mistral_summary.json`, writes `full7405_route_metrics.json`, runs the
HotpotQA F1-drop selector, and produces `result_tables_with_full7405/`. That
table directory should include the payload-based `bandwidth_sensitivity.*`
artifacts in addition to mixed-route, route-curve, TTFT, and latency-quality
tables.

When `result_tables_with_full7405/` exists, finalize the paper-facing bundle
with:

```bash
python scripts/finalize_scd_icml_full7405_bundle.py \
  --result-tables-dir <postprocess-dir>/result_tables_with_full7405 \
  --profile-figures-dir <profile-figure-output-dir> \
  --stack-viz-dir <stack-viz-output-dir> \
  --offline-cost-dir <cost-output-dir> \
  --neighboring-baselines-dir <neighboring-baseline-table-dir> \
  --out-dir <final-paper-artifact-bundle-dir> \
  --paper-tex <paper.tex> \
  --paper-base-dir <paper-dir>
```

This command wraps `build_paper_artifact_bundle.py` and
`check_paper_artifacts.py`; it does not invent metrics or edit the paper
source. If `hotpotqa_full7405_f1_drop003_selection.json` is present next to the
`result_tables_with_full7405/` directory, the finalizer also requires the paper
source to contain the selected `ReuseXX/PatchYY` route token. This prevents a
boundary-refined selector result from passing while the paper still reports an
older HotpotQA route.

On a remote experiment host, leave the finalization waiting for the postprocess
table with:

```bash
POST_ROOT=<postprocess-dir> \
PROFILE_FIGURES_DIR=<profile-figure-output-dir> \
STACK_VIZ_DIR=<stack-viz-output-dir> \
OFFLINE_COST_DIR=<cost-output-dir> \
NEIGHBORING_BASELINES_DIR=<neighboring-baseline-table-dir> \
PYTHON=<python> \
bash scripts/wait_finalize_scd_icml_full7405_bundle.sh
```

The wait wrapper requires the mixed-route table to contain a `full7405` source
row and requires `NEIGHBORING_BASELINES_DIR` before it invokes the finalizer. It
skips local paper-source checking when no paper path is provided, which keeps
remote experiment hosts usable without a paper checkout. Run the finalizer
locally with `--paper-tex`, or set
`PAPER_TEX` and optionally `PAPER_BASE_DIR` when the paper source is available
on the remote host; when `PAPER_TEX` is present, the wait wrapper enables
paper checking by default unless `SKIP_PAPER_CHECK=1` is set explicitly. The
selector-route text check is applied automatically when the selector report
exists in `POST_ROOT`.
