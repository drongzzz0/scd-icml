# SCD Claim Target Specification for Similar Qwen3.5 Pairs

## Purpose

This document defines what it means for the original SCD method to "reach the paper's claimed effect" when retested on more similar Qwen3.5 fine-tuned model pairs. It also defines the follow-up judgment for route A, route B, and route C if the original baseline misses that target.

## Grounding

- Source paper: `Semantic Cache Distillation: Efficient State Transfer via Reuse and Selective Patching` (`ICML2026_MaQianli.pdf`).
- Local route memo: `SCD_three_route_validation_checklist.md`.
- Extracted paper text artifact: `Experiment/analysis/ICML2026_MaQianli.extracted.txt`.
- The primary headline targets below were transcribed from the local PDF after local text extraction. Additional rows can be added later if a new table or figure becomes the main comparison anchor.

## Scope of the Claim

The primary claim under test is:

> On a more similar Qwen3.5 fine-tuned model pair, the original SCD method can recover the paper's advertised TTFT-quality trade-off without changing the method itself.

The secondary claim under test is:

> If the original SCD method still misses the target on the same similar pair, route A/B/C may recover that target or materially narrow the remaining gap.

The unit of judgment is a paper-matched operating point `C`:

- task and metric family
- prefix length and decode length
- layer budget / patch budget
- rank or payload budget
- bandwidth assumption if the paper makes one explicit
- producer-consumer pairing regime

Every judgment must be made on:

- one preregistered `close` Qwen3.5 pair from the pair-screening stage
- matched prefixes across all compared methods
- identical tokenizer, attention mask, RoPE convention, cache layout, and decode settings
- the same local consumer-native full-prefill baseline
- the same oracle patch upper bound when oracle comparisons are reported

## Why the Claim Is Normalized

Absolute TTFT numbers from the paper should not be used as the main success target because hardware, kernels, and model pairs differ. The claim therefore has to be evaluated primarily in normalized form relative to the local consumer-native baseline, while still reporting absolute milliseconds for engineering analysis.

## Claim Target Representation

For each headline claim extracted from the paper, record one row in the table below. Any non-empty target field in the chosen row becomes an active constraint for pass/fail evaluation.

| claim_id | source | task | context | headline relation | ttft_ratio_target | quality_retention_target | kl_ceiling_target | ppl_ratio_target | payload_ratio_target_vs_raw_kv | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `table1_qwen32b_primary` | `Table 1, page 8, "Qwen Pair (32B)"` | `F1 + KL` | `Cross-scale main result on a Qwen-family heterogeneous pair` | `SCD recovers Oracle-level quality across scales while achieving ~2.0x speedup over full prefill` | `0.4839` | `0.9669` | `0.31` | `n/a` | `n/a` | `Computed from Table 1 values: TTFT 243.7/503.6, F1 0.847/0.876. Primary family-aligned anchor for Qwen-like validation.` |
| `table2_mistrallite_mistral7b_200gbps` | `Table 2, page 8, "MistralLite→Mistral-7B at Bnet=200 Gbps"` | `CMRC F1 + WikiText KL/PPL` | `Bandwidth-constrained heterogeneous base→fine-tuned transfer` | `SCD matches Oracle quality while achieving 2.65x speedup` | `0.3772` | `0.9685` | `0.1105` | `1.1205` | `0.7591` | `Computed from Table 2 values: TTFT 125.2/331.9, F1 0.7850/0.8105, PPL 4.65/4.15, payload 87.31/115.02 vs Raw KV.` |

Use the following normalized definitions:

- `ttft_ratio = TTFT_method / TTFT_consumer_full_prefill`
- `quality_retention = Q_method / Q_consumer_full_prefill` for higher-is-better task metrics
- `kl_ceiling` is an absolute upper bound for token-level KL when the paper defines KL relative to the Oracle consumer distribution; this is necessary because `KL_oracle = 0` by construction
- `ppl_ratio = PPL_method / PPL_consumer_full_prefill`
- `payload_ratio_vs_raw_kv = Bytes_method / Bytes_raw_kv_transfer`

If the paper reports only absolute TTFT, convert it into `ttft_ratio_target` using the paper's own Oracle or native-consumer baseline. If the paper reports only quality drop rather than retention, convert it into a retention ratio before comparison.

## Local Measured Quantities

For every local run, compute the same normalized quantities against the local consumer-native full-prefill baseline:

- `ttft_ratio_local`
- `quality_retention_local`
- `kl_local` when KL is reported to Oracle
- `ppl_ratio_local`
- `payload_ratio_vs_raw_kv_local`

Always report the raw quantities too:

- `TTFT_ms`
- task metric
- `KL` and `PPL` if available
- payload bytes
- producer encode time
- transfer time
- consumer reconstruction time
- suffix recompute time

## Original SCD Baseline Pass/Fail Criteria

The original SCD baseline counts as a valid reproduction attempt only if it is paper-faithful and evaluated on a preregistered close pair under matched cache semantics.

### Strict Pass

The original baseline is a strict pass if, on at least one close pair and at least one primary paper-matched operating point:

- `ttft_ratio_local <= 1.10 * ttft_ratio_target`
- every active fidelity target in the chosen claim row is met:
  - if `quality_retention_target` is active, `quality_retention_local >= quality_retention_target - 0.02`
  - if `kl_ceiling_target` is active, `kl_local <= 1.10 * kl_ceiling_target`
  - if `ppl_ratio_target` is active, `ppl_ratio_local <= 1.10 * ppl_ratio_target`
- `payload_ratio_vs_raw_kv_local <= 1.10 * payload_ratio_target_vs_raw_kv` if payload is part of the chosen claim row
- no hidden method changes are introduced beyond paper-faithful implementation fixes

### Soft Pass

The original baseline is a soft pass if it misses one of the thresholds above only marginally, but its 95% CI or repeated-run variation overlaps the threshold band and the measured point lies on the same qualitative Pareto region as the paper claim.

### Fail

The original baseline is a fail if no close pair achieves either a strict pass or a soft pass on any primary operating point.

## Claim Gap Definition

The claim gap is computed per operating point and per close pair.

Define the non-negative component gaps as:

- `gap_ttft = max(0, ttft_ratio_local - 1.10 * ttft_ratio_target)`
- `gap_quality = max(0, (quality_retention_target - 0.02) - quality_retention_local)` when a task-quality target is active
- `gap_kl = max(0, kl_local - 1.10 * kl_ceiling_target)` when a KL ceiling is active
- `gap_ppl = max(0, ppl_ratio_local - 1.10 * ppl_ratio_target)` when a PPL ratio is active
- `gap_payload = max(0, payload_ratio_vs_raw_kv_local - 1.10 * payload_ratio_target_vs_raw_kv)` when a payload target is active

Define the active fidelity gap as:

`gap_fidelity = max(active {gap_quality, gap_kl, gap_ppl})`

Define the normalized aggregate claim gap:

`claim_gap_total = 0.45 * norm(gap_ttft) + 0.45 * norm(gap_fidelity) + 0.10 * norm(gap_payload_or_zero)`

where:

- `gap_fidelity` is the largest active task-quality or fidelity miss on the chosen claim row
- `gap_payload_or_zero` is `gap_payload` if a payload target is active, otherwise `0`
- `norm(.)` means normalization by the corresponding target scale so that 0 means "target met" and larger values mean a bigger miss

The aggregate score is only a decision aid. The raw component gaps must always be reported alongside it.

## Rescue Criteria for Route A/B/C

Routes A/B/C are judged only after the original baseline has failed on the same close pair or pairs.

### Full Rescue

A route is a full rescue if it satisfies the same strict-pass criteria above on the same close pair and matched operating point where the original baseline failed.

### Material Gap Narrowing

A route materially narrows the gap if all of the following hold on the same pair and matched prefixes:

- `claim_gap_total_route <= 0.50 * claim_gap_total_original`
- it improves at least one primary axis:
  - lower `ttft_ratio_local`, or
  - higher `quality_retention_local`, or
  - lower `kl_local`, or
  - lower `ppl_ratio_local`
- it does not worsen the other primary axis beyond the same tolerance band used in the pass/fail rule

### No Rescue

A route is not a rescue if it neither reaches a full rescue nor cuts the original claim gap by at least half on the same pair and matched operating point.

## Required Reported Metrics

Every baseline and route comparison must report:

- pair name and similarity bucket
- exact producer and consumer checkpoints
- task, context length, generation length, and prefix identifier
- patch layer set and patch budget
- rank, hidden width, or decoder capacity used
- payload bytes
- producer encode time
- transfer time
- consumer reconstruction time
- suffix recompute time
- end-to-end TTFT
- task metric
- KL and PPL if available
- oracle upper-bound metric for the same layer budget if available

## Comparison Invariants

The following invariants are mandatory for all baseline vs route comparisons:

- the same pair
- the same tokenizer
- the same prefix list
- the same attention mask and RoPE semantics
- the same cache serialization layout
- the same bandwidth assumption when transfer time is simulated
- the same consumer decoding configuration
- the same aggregation rule across prompts and seeds

If any of these differ, the comparison is diagnostic only and cannot be used for the main claim.

## Decision Logic

Use the following decision order:

1. Screen candidate Qwen3.5 pairs and preregister one or more close pairs.
2. Run the original SCD baseline on those close pairs under the paper-matched protocol.
3. If the original baseline strict-passes or soft-passes, the primary conclusion is that pair similarity may be sufficient for the original method; route A/B/C become secondary analyses.
4. If the original baseline fails on all close pairs, run route A/B/C on the same pairs and matched prefixes.
5. If one route full-rescues the claim, the conclusion is that pair similarity alone is insufficient but the modified route can recover the claimed effect.
6. If no route full-rescues but one route materially narrows the gap, the conclusion is that the paper claim is not reproduced, but the route improves the failure mode.
7. If neither the original baseline nor any route reaches full rescue or material gap narrowing, conclude that the claim is not recovered under the similar-pair Qwen3.5 setting.

## Open Items Before Full Runs

Before large-scale execution, extend the claim table only if a new table or figure becomes the main comparison anchor. Any added row must cite its source table, figure, or page and must not use guessed values.
For Qwen3.5 similar-pair validation, `table1_qwen32b_primary` should be the default primary anchor unless a later Qwen-specific table is added.
