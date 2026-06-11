# Claim-Evidence Register and Paper Claim Boundary

Date: 2026-06-03  
Project: SCD-ICML follow-up / cross-scale Patch route  
Model pair: Qwen2.5-1.5B -> Qwen2.5-7B  
Primary evaluation setting: QA-complete 16k, 200 samples where available

## 1. Executive Boundary

The current evidence supports a method-boundary / negative speed result, not a positive 16k acceleration claim.

The most defensible paper-facing statement is:

> Lightweight cross-scale Patch methods can preserve QA quality for a small safe layer subset, but the current route does not accelerate QA-complete 16k inference for Qwen2.5-1.5B -> Qwen2.5-7B because high-impact layer22 remains unpatchable and must stay exact.

The current evidence does not support claiming 16k wall-clock speedup.

## 2. Claim Status Summary

| id | claim | status | evidence strength | paper use |
|---|---|---|---|---|
| C1 | QA-complete 16k F1/EM evaluation is valid after prompt repair | Supported | Strong | Use |
| C2 | Safe mixed route preserves F1/EM on 200 QA-complete 16k samples | Supported | Strong | Use |
| C3 | Safe mixed route accelerates full large at 16k | Refuted | Strong | Do not claim |
| C4 | Patch compute is the main runtime bottleneck | Refuted | Strong | Do not claim |
| C5 | Exact/recompute dominates mixed-route runtime | Supported | Strong | Use |
| C6 | Layer22 is the current hard quality/speed blocker | Supported | Strong | Use |
| C7 | Layer23 can be repaired in isolation | Supported but limited | Medium | Use only as ablation |
| C8 | AX/AY/AZ layer22 Patch methods should scale to 200 samples | Refuted | Strong | Do not run/claim |
| C9 | Current lightweight Patch family is insufficient for layer22 | Supported | Strong | Use |
| C10 | Cross-scale expansion is impossible in general | Not supported | N/A | Do not claim |
| C11 | A larger teacher-student Patch may solve layer22 | Future work | Speculative | Future work only |

## 3. Supported Claims

### C1. QA-complete 16k F1/EM evaluation is valid after prompt repair

Status: Supported.

Evidence:

- Earlier 16k F1/EM was invalid because prompts lacked final Question / Answer instruction.
- QA-complete prompts restored credible full-large baseline behavior.
- Full-large baseline is no longer near zero.

Paper wording:

> We evaluate end-task quality only on QA-complete 16k prompts whose tail preserves the question and answer instruction; earlier suffix-only or truncated-context probes are not used for F1/EM claims.

Do not say:

> Earlier invalid F1/EM results reflect model weakness.

### C2. Safe mixed route preserves QA F1/EM

Status: Supported.

Evidence:

| method | n | F1 | EM | total ms |
|---|---:|---:|---:|---:|
| full large | 200 | 0.4739 | 0.345 | 2110.62 |
| safe mixed | 200 | 0.4883 | 0.360 | 2449.72 |

Interpretation:

The safe route does not degrade F1/EM relative to full large. This is a quality result only, not a speed result.

Paper wording:

> On QA-complete 16k samples, the conservative safe-layer route preserves answer quality relative to the full-large baseline.

### C5. Exact/recompute dominates mixed-route runtime

Status: Supported.

Evidence:

| component | ms | percent |
|---|---:|---:|
| layer-token recompute | 908.06 | 58.27% |
| small prefill | 580.04 | 37.22% |
| Patch compute | 69.98 | 4.49% |
| KV materialization | 0.39 | 0.03% |

Paper wording:

> The bottleneck in the conservative route is not Patch inference but exact layer-token recomputation induced by high exact anchors.

### C6. Layer22 is the hard blocker

Status: Supported.

Evidence:

| method | target | n | full F1 | mixed F1 | EM | F1 retention | decision |
|---|---|---:|---:|---:|---:|---:|---|
| AX native xnorm/KV | 22 | 40 | 0.4008 | 0.2944 | 0.175 | 0.7346 | fail |
| AY sequence hidden bridge | 22 | 40 | 0.4008 | 0.0288 | 0.000 | 0.0717 | fail |
| AZ cross-layer Transformer residual | 22 | 40 | 0.4008 | 0.2039 | 0.100 | 0.5087 | fail |

Threshold: F1 retention >= 0.90 to continue to 200 samples.

Paper wording:

> Layer22 behaves as a hard exact anchor for this model pair: lightweight KV, native-projection, and hidden-bridge variants all fail the 0.90 F1-retention gate.

### C7. Layer23 can be repaired in isolation, but this does not unlock deployment

Status: Supported but limited.

Evidence:

| route | full F1 | mixed F1 | EM | F1 retention |
|---|---:|---:|---:|---:|
| AX_23 | 0.4008 | 0.4532 | 0.325 | 1.1307 |
| AX23_AS24_27 | 0.4008 | 0.3022 | 0.200 | 0.7540 |

Interpretation:

Layer23 alone is not the bottleneck. It becomes useful only if layer22 and boundary interactions are solved.

Paper wording:

> Some upper layers can be repaired in isolation, but isolated repair does not imply deployable speedup because adjacent hard anchors still determine recompute depth.

## 4. Refuted or Prohibited Claims

### C3. Safe mixed route accelerates full large at 16k

Status: Refuted.

Evidence:

| method | total ms | saving vs full large |
|---|---:|---:|
| full large | 2110.62 | 0.00% |
| safe mixed | 2449.72 | -21.99% |

Do not claim 16k speedup for the current route.

Allowed wording:

> The conservative quality-preserving route is not speed-positive under the current runtime implementation.

### C4. Patch compute is the main runtime bottleneck

Status: Refuted.

Evidence: Patch compute is only 4.49% of the measured mixed-route runtime. Exact/recompute and small prefill dominate.

Do not focus optimization claims on Patch kernel speed alone.

### C8. AY/AZ should be promoted to 200 samples

Status: Refuted.

Reason:

Both failed the n=40 gate. Scaling to 200 would be GPU waste under the pre-registered continuation rule.

### C10. Cross-scale expansion is impossible in general

Status: Not supported.

The experiments falsify the current lightweight Patch family in this setting. They do not rule out larger teacher-student Patch models, attention-output supervision, or a runtime that avoids prefix-depth recompute.

## 5. Revised Speed Gate

The previous speed target:

```text
16k wall-clock saving >= 40%
```

is disabled for the current lightweight Patch route.

Revised gates:

| gate | threshold | status |
|---|---:|---|
| QA quality | F1 retention >= 0.90 | retained |
| Speed claim | measured wall-clock saving > 0 | required before any speed claim |
| 40% speed claim | disabled | re-enable only after layer22 is solved |
| AY/AZ continuation | n=40 F1 retention >= 0.90 | failed |

## 6. Recommended Paper Framing

Recommended framing:

1. Present the route as a diagnostic and method-boundary study.
2. Show that QA-complete F1/EM can be preserved by conservative layer selection.
3. Show that wall-clock speedup fails because exact anchors dominate runtime.
4. Identify layer22 as the empirical hard anchor.
5. Position stronger teacher-student Patch training or runtime redesign as future work.

Not recommended:

1. Do not claim 16k acceleration.
2. Do not use NLL-only quality to support end-task F1/EM claims.
3. Do not claim that patching layer23 alone solves deployment.
4. Do not report serial estimates as wall-clock speedup.

## 7. Paper-Ready Paragraph

On QA-complete 16k evaluation, the conservative mixed-KV route preserved answer quality but did not accelerate inference. The full-large Qwen2.5-7B baseline achieved F1 0.4739 and EM 0.345 with 2110.62 ms average runtime, while the quality-safe mixed route achieved F1 0.4883 and EM 0.360 but required 2449.72 ms. Runtime decomposition showed that Patch compute accounted for only 4.49% of the mixed-route runtime, whereas exact layer-token recomputation accounted for 58.27%. The limiting factor is therefore not Patch inference cost but the need to retain high-impact exact layers. Layer22 was the dominant blocker: native xnorm/KV projection, sequence hidden-bridge Patch, and cross-layer Transformer residual Patch reached only 0.7346, 0.0717, and 0.5087 F1 retention respectively, all below the 0.90 continuation threshold. These results bound the current lightweight Patch family: it can preserve quality for a safe layer subset, but it does not support a 16k wall-clock speedup claim for this model pair.

## 8. Evidence Artifacts

Local documents:

```text
PATCH_ROUTE_PROGRESS_SUMMARY_2026-06-03.md
PATCH_ROUTE_DECISION_2026-06-03.md
SPEED_GATE_REVISION_2026-06-03.md
PATCH_NEGATIVE_RESULT_WRITEUP_2026-06-03.md
```

Remote result files:

```text
Experiment/analysis/long_context/AY_layer22_sequence_bridge_summary.md
Experiment/analysis/long_context/AY_layer22_seq_bridge_xnorm_train20_n40_summary.csv
Experiment/analysis/long_context/AZ_layer22_xlayer_transformer_resid_train40_n40_summary.csv
Experiment/core_code/run_task_ay_layer22_sequence_bridge.py
Experiment/core_code/run_task_az_layer22_transformer_residual.py
```
