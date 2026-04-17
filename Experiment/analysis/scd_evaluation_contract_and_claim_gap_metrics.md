# Evaluation Contract and Claim-Gap Metrics for Similar-Pair SCD Validation

## Purpose

This document operationalizes Task 3: refine the evaluation plan and the exact claim-gap metrics needed to judge whether:

1. the original SCD baseline reproduces the paper's claimed effect on similar Qwen3.5 fine-tuned pairs; and
2. route A/B/C close or recover the gap if the original baseline fails.

This is an analysis contract, not an implementation document. Its purpose is to ensure that Task 4 and later runs emit enough structured outputs to support a defensible ICML paper.

## Scope

This contract applies to:

- the original SCD baseline
- route A
- route B
- route C
- all supporting comparison methods used to contextualize the result

It builds on:

- `Experiment/analysis/scd_claim_target_spec.md`
- `Experiment/analysis/scd_qwen35_pair_screening_original_baseline_protocol.md`
- `Experiment/analysis/ICML2026_MaQianli.extracted.txt`

## Primary Evaluation Questions

The evaluation must answer four questions in order:

1. Does the original SCD baseline meet the `table1_qwen32b_primary` claim anchor on a screened close Qwen3.5 pair?
2. If not, how large is the component-wise and aggregate claim gap?
3. On the same pair and matched prefixes, does route A, B, or C achieve a full rescue?
4. If no route achieves a full rescue, does any route materially narrow the original claim gap?

## Claim Anchors

Use the following anchor priority:

- Primary family-aligned anchor: `table1_qwen32b_primary`
- Secondary detailed efficiency anchor: `table2_mistrallite_mistral7b_200gbps`

Rule:

- All Qwen3.5 similar-pair headline judgments default to `table1_qwen32b_primary`.
- `table2_mistrallite_mistral7b_200gbps` is required whenever payload, `KL`, `PPL`, or `Bnet = 200 Gbps` efficiency analysis is reported.
- No new claim row may be used for a headline conclusion unless it is first added to the claim specification file with source provenance.

## Required Comparison Methods

For every final close-pair evaluation family, the contract requires the following methods when feasible:

- `oracle_consumer_full_prefill`
- `raw_kv_bf16`
- `quantized_kv_int4`
- `selective_recompute`
- `scd_reuse_only`
- `scd_original`

If the original baseline fails, also require:

- `route_a_s2s2nativekv`
- `route_b_s2directkv`
- `route_c_s2s_suffix_recompute`

If a comparison method is omitted, the run manifest must record the omission reason.

## Analysis Units

The evaluation uses three nested units:

- `prompt-level`: one method on one prompt prefix under one operating point
- `run-level`: one method on one pair under one operating point, aggregated over prompts
- `claim-level`: one claim anchor on one pair, aggregated over the relevant run-level outputs

All claim decisions are made at the `claim-level`, but every claim must trace back to prompt-level rows.

## Operating Point Definition

An operating point is uniquely defined by:

- `pair_id`
- `method`
- `task`
- `split`
- `context_len`
- `decode_len`
- `bandwidth_gbps`
- `patch_budget_k`
- `patch_layers`
- `rank_k`
- `rank_v`
- `rank_h`
- `seed`

Route-level comparisons are valid only if the operating point matches exactly except for `method`.

## Required Output Artifacts

Task 4 and later execution must produce these structured files under `Experiment/analysis/`:

- `run_manifest.jsonl`
- `prompt_level_metrics.csv`
- `run_level_summary.csv`
- `claim_gap_summary.csv`
- `route_rescue_summary.csv`
- `latency_breakdown_summary.csv`
- `pair_screen_summary.csv`
- `evaluation_readme.md`

These are mandatory. Additional plots or notebooks are optional.

## `run_manifest.jsonl` Contract

Each line must contain:

- `run_id`
- `timestamp_utc`
- `git_commit` if available
- `pair_id`
- `producer_model`
- `consumer_model`
- `method`
- `task`
- `split`
- `seed`
- `context_len`
- `decode_len`
- `bandwidth_gbps`
- `patch_budget_k`
- `patch_layers`
- `rank_k`
- `rank_v`
- `rank_h`
- `claim_anchor`
- `status`
- `artifact_paths`
- `notes`

## `prompt_level_metrics.csv` Contract

One row per prompt per method per operating point.

Required columns:

- `run_id`
- `pair_id`
- `pair_bucket`
- `claim_anchor`
- `method`
- `task`
- `split`
- `seed`
- `prompt_id`
- `prefix_id`
- `context_len`
- `decode_len`
- `bandwidth_gbps`
- `patch_budget_k`
- `patch_layers`
- `rank_k`
- `rank_v`
- `rank_h`
- `payload_bytes`
- `producer_prefill_ms`
- `producer_encode_ms`
- `transfer_ms`
- `consumer_rebuild_ms`
- `consumer_patch_ms`
- `consumer_recompute_ms`
- `first_token_decode_ms`
- `ttft_ms`
- `end_to_end_ms`
- `task_metric_name`
- `task_metric_value`
- `token_kl`
- `token_tv`
- `ppl`
- `native_consumer_ttft_ms`
- `native_consumer_task_metric_value`
- `native_consumer_ppl`
- `raw_kv_payload_bytes`

Every missing value must be represented explicitly and justified in `evaluation_readme.md`.

## `run_level_summary.csv` Contract

One row per `run_id`, aggregated across prompts.

Required columns:

- `run_id`
- `pair_id`
- `pair_bucket`
- `claim_anchor`
- `method`
- `task`
- `split`
- `seed`
- `n_prompts`
- `mean_ttft_ms`
- `ttft_ci95_low`
- `ttft_ci95_high`
- `mean_end_to_end_ms`
- `mean_task_metric_value`
- `task_metric_ci95_low`
- `task_metric_ci95_high`
- `mean_token_kl`
- `kl_ci95_low`
- `kl_ci95_high`
- `mean_token_tv`
- `mean_ppl`
- `ppl_ci95_low`
- `ppl_ci95_high`
- `mean_payload_bytes`
- `mean_payload_ratio_vs_raw_kv_local`
- `mean_ttft_ratio_local`
- `mean_quality_retention_local`
- `mean_ppl_ratio_local`

## `claim_gap_summary.csv` Contract

One row per `pair_id x claim_anchor x method x task`.

Required columns:

- `pair_id`
- `pair_bucket`
- `claim_anchor`
- `method`
- `task`
- `ttft_ratio_target`
- `quality_retention_target`
- `kl_ceiling_target`
- `ppl_ratio_target`
- `payload_ratio_target_vs_raw_kv`
- `ttft_ratio_local`
- `quality_retention_local`
- `kl_local`
- `ppl_ratio_local`
- `payload_ratio_vs_raw_kv_local`
- `gap_ttft`
- `gap_quality`
- `gap_kl`
- `gap_ppl`
- `gap_payload`
- `gap_fidelity`
- `claim_gap_total`
- `pass_label`
- `ci_overlap_flag`

Allowed `pass_label` values:

- `strict_pass`
- `soft_pass`
- `fail`

## `route_rescue_summary.csv` Contract

Create this file only after the original baseline has failed on at least one close pair.

One row per `pair_id x claim_anchor x route_method x task`.

Required columns:

- `pair_id`
- `claim_anchor`
- `task`
- `original_method_claim_gap_total`
- `route_method`
- `route_claim_gap_total`
- `gap_reduction_ratio`
- `ttft_delta_vs_original`
- `quality_delta_vs_original`
- `kl_delta_vs_original`
- `ppl_delta_vs_original`
- `rescue_label`

Allowed `rescue_label` values:

- `full_rescue`
- `material_gap_narrowing`
- `no_rescue`

## `latency_breakdown_summary.csv` Contract

One row per `run_id`.

Required columns:

- `run_id`
- `method`
- `producer_prefill_ms`
- `producer_encode_ms`
- `transfer_ms`
- `consumer_rebuild_ms`
- `consumer_patch_ms`
- `consumer_recompute_ms`
- `first_token_decode_ms`
- `ttft_ms`
- `end_to_end_ms`

This file is required because the paper makes latency-breakdown claims, not just TTFT claims.

## Aggregation Rules

Use these aggregation rules consistently:

- Aggregate prompt-level metrics into run-level summaries with arithmetic means.
- Compute `95%` confidence intervals by paired bootstrap over prompts.
- Do not collapse different tasks into one scalar for the main claim decision.
- For claim judgment, evaluate each task separately first, then summarize across tasks narratively.
- For the headline claim on Qwen3.5 similar pairs, the default decisive task is the task most closely matching the chosen claim anchor. Secondary tasks are supporting evidence, not substitutes.

## Statistical Comparison Rules

For matched-method comparisons on the same prompts:

- Use paired bootstrap confidence intervals for all primary reported deltas.
- Use paired Wilcoxon signed-rank tests as the default non-parametric significance check for TTFT and task metrics.
- Apply Holm-Bonferroni correction within each `pair_id x task x claim_anchor` family when comparing route A/B/C against the original baseline.

If the final close-pair evaluation is run with only one seed, mark the result as `single_seed` in `evaluation_readme.md` and treat the associated significance claims as descriptive rather than confirmatory.

## Seed Policy

Use this seed policy:

- Screening and broad ablations: `1` seed is acceptable.
- Claim-bearing close-pair baseline runs: target `3` seeds.
- Full-rescue candidate routes on the final close pair: target `3` seeds.

If compute limits force a reduction below these targets, record the exception explicitly.

## Core Metric Definitions

Use the same metric definitions as the claim specification:

- `ttft_ratio_local = mean_ttft_ms / native_consumer_ttft_ms`
- `quality_retention_local = mean_task_metric_value / native_consumer_task_metric_value`
- `ppl_ratio_local = mean_ppl / native_consumer_ppl`
- `payload_ratio_vs_raw_kv_local = mean_payload_bytes / raw_kv_payload_bytes`

For `KL`, do not normalize by the Oracle because the Oracle baseline is zero by construction. Use the absolute ceiling formulation:

- `kl_local = mean_token_kl`
- compare against `kl_ceiling_target`

## Claim-Gap Formulas

The component gaps are:

- `gap_ttft = max(0, ttft_ratio_local - 1.10 * ttft_ratio_target)`
- `gap_quality = max(0, (quality_retention_target - 0.02) - quality_retention_local)` when active
- `gap_kl = max(0, kl_local - 1.10 * kl_ceiling_target)` when active
- `gap_ppl = max(0, ppl_ratio_local - 1.10 * ppl_ratio_target)` when active
- `gap_payload = max(0, payload_ratio_vs_raw_kv_local - 1.10 * payload_ratio_target_vs_raw_kv)` when active

Define:

- `gap_fidelity = max(active {gap_quality, gap_kl, gap_ppl})`
- `claim_gap_total = 0.45 * norm(gap_ttft) + 0.45 * norm(gap_fidelity) + 0.10 * norm(gap_payload_or_zero)`

Use the same normalization convention as the claim specification file.

## Pass / Rescue Rules

### Baseline pass

The original baseline is:

- `strict_pass` if all active target constraints are met
- `soft_pass` if one threshold is narrowly missed but the `95%` interval overlaps the tolerance band
- `fail` otherwise

### Route rescue

A route is:

- `full_rescue` if it satisfies the same strict-pass criteria on the same pair and task where the original baseline failed
- `material_gap_narrowing` if `route_claim_gap_total <= 0.50 * original_method_claim_gap_total` without violating the other primary axis tolerance band
- `no_rescue` otherwise

## Fair-Comparison Invariants

No comparison is claim-valid unless these match:

- `pair_id`
- tokenizer
- attention mask
- RoPE semantics
- cache layout
- prefix list
- task
- context length
- decode length
- bandwidth
- aggregation rule

If any invariant is broken, the comparison must be labeled `diagnostic_only`.

## Required Tables and Figures for Later Paper Use

The evaluation outputs must support at least these paper-ready assets:

- model-pair screening table
- original-baseline claim-gap table
- route rescue table
- TTFT vs quality Pareto plot
- bandwidth sweep plot
- patch-budget ablation plot
- layerwise similarity heatmap
- latency breakdown table

## Go / No-Go Rule

Task 4 should start only after the implementation plan is capable of emitting every required field in this contract. Missing CSV columns discovered after large runs count as an execution failure, not as a harmless bookkeeping issue.
