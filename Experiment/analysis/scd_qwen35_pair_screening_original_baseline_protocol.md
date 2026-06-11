# Similar-Pair Screening and Original-SCD Baseline Protocol

## Purpose

This protocol operationalizes Task 2: refine the similar-pair screening workflow and the faithful original-SCD baseline setup for more similar Qwen3.5 fine-tuned model pairs.

The immediate goal is not to implement route A/B/C. The goal is to make the original paper method executable, auditable, and separable from the later follow-up routes.

## Scope and Sources

This protocol is grounded only in local project artifacts:

- `ICML2026_MaQianli.pdf`
- `Experiment/analysis/ICML2026_MaQianli.extracted.txt`
- `Experiment/analysis/scd_claim_target_spec.md`
- `SCD_three_route_validation_checklist.md`

No external code reference repo is currently present under `Experiment/code_references/`. Therefore, this protocol treats the local paper and extracted text as the canonical baseline specification and defers repository-level implementation survey to Task 4 if reference repos are later added.

## Paper-Faithful Invariants

The baseline implementation must preserve the following details from the paper before any similar-pair conclusion is drawn:

- Use the same tokenizer on producer and consumer.
- Use the exact same attention masking on both sides.
- Treat reconstructed caches under the native cache semantics of the inference framework.
- For keys, apply `deRoPE` before encoding and `RoPE` after decoding.
- Compress Key, Value, and Hidden-state channels independently.
- Report TTFT and end-to-end latency under controlled effective bandwidth, with at least one fixed anchor at `Bnet = 200 Gbps` and an optional sweep over `100 Gbps` to `1 Tbps`.
- Evaluate on the paper task families: QA (`CMRC2018`, `HotpotQA`), language modeling (`WikiText-2`), and distributional fidelity (`KL`, `TV`).

These invariants are stated in Sections 4, 6.1, Appendix E, and the extracted paper text.

## Original-SCD Runtime Semantics

For this project, the original baseline is defined as the main-text Algorithm 1 / Sections 4.2-4.4 interpretation of SCD:

- `Reuse` layers transmit low-rank semantic codes for `K` and `V`.
- `Patch` layers transmit low-rank semantic codes for `H_in = x_norm`.
- The consumer reconstructs reuse-layer KV into its native cache space.
- The consumer applies the patch aligner `g_l` and regenerates patched-layer KV using its own native `Wk/Wv`.

### Important Ambiguity Lock

Appendix E contains wording that can be read as partial forward recomputation from a patch breakpoint onward. In this project, that interpretation is **not** part of the original baseline. It is treated as a distinct follow-up route family and therefore belongs to later route-level experiments, not to the baseline.

This lock is necessary to prevent contamination of the original-SCD baseline with what is effectively route C style suffix recomputation.

## Candidate Pair Admission Rules

The candidate pool is limited to Qwen3.5-family fine-tuned checkpoints first.

Each candidate producer-consumer pair must satisfy all hard constraints:

- Same Qwen3.5 family.
- Same parameter scale.
- Same layer count.
- Same hidden size.
- Same head configuration.
- Same tokenizer.
- Same RoPE convention.
- Same cache layout under the serving framework.

Each candidate pair should also satisfy as many soft constraints as possible:

- Same base checkpoint lineage.
- Similar fine-tuning recipe.
- Similar data domain.
- Similar instruction style or specialization target.

Pairs with tokenizer mismatch or architecture mismatch are rejected before screening.

## Close / Medium / Far Regime Definition

The protocol requires three regimes when feasible:

- `close`: top-ranked eligible Qwen3.5 similar pair.
- `medium`: second-tier eligible Qwen3.5 pair with weaker similarity.
- `far`: either the current non-Qwen baseline pair already known to be dissimilar, or the lowest-ranked eligible pair if only Qwen3.5 pairs are available.

If only one eligible Qwen3.5 pair is available, it is designated `close`, and the previous project baseline pair should remain the `far` control.

## Screening Dataset and Split

Use a small calibration-only screening subset before any full benchmark run:

- `256-512` prompts for pair screening.
- Context lengths `{512, 2048}`.
- Identical prefix lists for producer and consumer.
- No translator training during the screening stage.

For subsequent baseline training and evaluation, use a strict split on the calibration pool:

- train: `70%`
- val: `15%`
- test: `15%`

The final benchmark test set must remain disjoint from calibration data.

## Trace Collection Contract

For every candidate pair and every prompt in the screening pool, run full prefills on both models and collect the following per layer:

- `H_in` or `x_norm`
- `K`
- `V`
- next-token logits for token-KL computation

Processing rules:

- For `K`, store both native and `deRoPE` content-space versions.
- For `V`, store native values directly.
- Flatten `batch x heads x tokens` into sample axis `N` for linear probes and SVD/ridge fitting, following the paper.
- Save layer metadata and prompt IDs with every tensor shard.

## Screening Metrics

For each layer `l`, compute:

- `cos_H[l]`: average token cosine between producer and consumer `H_in`.
- `cos_K[l]`, `cos_V[l]`: average cosine on content-space `K` and native `V`.
- `r2_H_lin[l]`: linear probe `H_A -> H_B`.
- `r2_KV_lin[l]`: linear probe `H_A -> K_B/V_B`.
- `r2_KV_fromKV[l]`: linear probe `K_A/V_A -> K_B/V_B` when feasible.
- `kl_token`: average next-token KL between producer and consumer.

Aggregate each pair with the composite ranking score proposed in the local checklist:

```text
sim_score = 0.30 * median(r2_H_lin)
          + 0.30 * median(r2_KV_lin)
          + 0.20 * median(cos_H)
          + 0.20 * (1 - normalized_kl_token)
```

Pair labeling rule:

- The highest-ranked eligible pair becomes `close`.
- The next eligible pair becomes `medium`.
- The weakest control becomes `far`.

The `close` label is only valid if the selected pair beats the `far` control on at least three of these four aggregate axes:

- `median(r2_H_lin)`
- `median(r2_KV_lin)`
- `median(cos_H)`
- `kl_token` (lower is better)

## Original-SCD Baseline: Offline Calibration

The baseline calibration phase must implement the paper's two-module design.

### Reuse Module

Per layer `l` in `L_reuse`:

1. Collect paired `deRoPE(K_A^l)`, `deRoPE(K_B^l)`, `V_A^l`, `V_B^l`.
2. Learn shared low-rank latent subspaces independently for `K` and `V` using truncated SVD.
3. Fit producer-to-latent encoders via ridge regression.
4. Fit consumer-side decoders into native consumer feature space.
5. Serialize layer-wise encoder/decoder parameters.

### Patch Module

Per layer `l` in `L_patch`:

1. Collect paired transition states `H_in,A^l = x_norm,A^l` and `H_in,B^l = x_norm,B^l`.
2. Fit a linear producer encoder `W_enc,H^l` into a low-rank code.
3. Fit a nonlinear consumer aligner `g_l` as an MLP with regression loss to estimate `H_in,B^l`.
4. Regenerate KV with native consumer weights:
   - `K_hat_B^l = RoPE(H_hat_in,B^l Wk_B^l)`
   - `V_hat_B^l = H_hat_in,B^l Wv_B^l`
5. Serialize encoder and aligner artifacts with minimal metadata.

### Patch-Layer Selection Policy

Select patch layers using the appendix policy rather than ad hoc intuition:

1. Start from `all-reuse` baseline.
2. Run restore-one sensitivity to estimate `ΔF_l` for each layer.
3. Build a candidate pool from the top-`M` layers ranked by `ΔF_l`.
4. Run interaction-aware greedy selection over that candidate pool.
5. Choose the budget `k*` by maximizing gain-per-cost.

Paper-aligned defaults and ablations:

- Use `k = 6` as the default patch budget because Figure 5 states this is the default budget.
- Also evaluate `k in {0, 1, 3, 5, 6, 8}` because Table 3 reports nearby operating points and these are sufficient to verify monotonicity and diminishing returns.

## Original-SCD Baseline: Online Execution

For every evaluation prefix:

1. Producer runs full prefill.
2. For reuse layers, producer sends `Z_K`, `Z_V`.
3. For patch layers, producer sends `Z_H`.
4. Consumer reconstructs reuse-layer `K/V` into native cache containers.
5. Consumer applies patch aligners and regenerates patched-layer native KV with its own `Wk/Wv`.
6. Consumer initiates decoding from the assembled cache without a full consumer-side prefill.

This online path must not include suffix recomputation during the original baseline runs.

## Task and Metric Contract

The original baseline must be reported on the same evaluation families as the paper:

- `CMRC2018`: F1
- `HotpotQA`: F1
- `WikiText-2`: PPL
- distributional fidelity: token-level `KL` and `TV`

Mandatory latency and payload fields:

- payload bytes
- producer prefill time
- producer encode time
- transfer time
- consumer rebuild / decompression time
- consumer patch / reconstruction time
- first-token decode time
- TTFT
- end-to-end latency

Mandatory claim-judgment fields from Task 1:

- `ttft_ratio_local`
- `quality_retention_local`
- `kl_local`
- `ppl_ratio_local`
- `payload_ratio_vs_raw_kv_local`
- `claim_gap_total`

## Compute Allocation Plan

This protocol is shaped around the available hardware stated by the user:

- `5070 Ti x1`
- `4090D x4`
- `A800-80G x8` with short-burst availability only

### Phase A: Smoke Tests and Tensor-Shape Validation

Use `5070 Ti` for:

- single-batch dry runs
- hook registration validation
- cache-layout checks
- serializer / deserializer debugging

Do not use this GPU for long trace extraction or full calibration.

### Phase B: Stable Pair Screening and Small-Context Trace Extraction

Use `4090D x4` for:

- candidate pair screening on `256-512` prompts
- contexts `512` and `2048`
- linear probes and similarity scoring
- small-scale calibration subset extraction
- default patch-policy discovery from reduced contexts

This is the stable workhorse phase and should produce the `close / medium / far` decision before any expensive burst run.

### Phase C: Short-Burst Main Baseline Runs

Reserve `A800-80G x8` only for:

- larger selected Qwen3.5 close-pair trace extraction if the stable cluster cannot fit the chosen checkpoints cleanly
- long-context or high-throughput baseline validation on the final selected close pair
- final paper-anchor runs at the `200 Gbps` claim target and optional bandwidth sweep

The A800 burst should not be used for exploratory hyperparameter sweeps. All screening, debugging, and protocol errors should be resolved on the `4090D` cluster first.

## Deliverables Required Before Task 4

Task 2 is complete only if the following are fixed in writing:

- pair admission rules
- close / medium / far labeling rule
- screening metrics and ranking score
- original-SCD baseline runtime semantics
- patch-layer selection policy
- paper-aligned task and metric contract
- hardware allocation plan

The following artifacts should be produced during Task 4 implementation using this protocol:

- `Experiment/analysis/pair_similarity_summary.csv`
- `Experiment/analysis/layerwise_similarity_close_medium_far.csv`
- `Experiment/analysis/original_scd_baseline_protocol_runs.csv`
- `Experiment/core_code/collect_paired_traces.py`
- `Experiment/core_code/pair_screen.py`
- `Experiment/core_code/train_scd_reuse.py`
- `Experiment/core_code/train_scd_patch.py`
- `Experiment/core_code/run_original_scd_baseline.py`

## Go / No-Go Rule for Advancing to Task 4

Advance to implementation only if:

- at least one eligible Qwen3.5 candidate pair satisfies all hard constraints
- the baseline semantics are locked to the non-suffix-recompute interpretation above
- the `close` pair is measurably more similar than the `far` control
- the A800 burst is reserved for final selected runs rather than screening

If these conditions are not met, do not start Task 4. Resolve pair availability or baseline ambiguity first.
