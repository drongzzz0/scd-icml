# Local Model Survey for Task 4

## Sources Used

- `ICML2026_MaQianli.pdf`
- `Experiment/analysis/ICML2026_MaQianli.extracted.txt`
- `Experiment/analysis/scd_qwen35_pair_screening_original_baseline_protocol.md`
- `Experiment/analysis/scd_evaluation_contract_and_claim_gap_metrics.md`
- `SCD_three_route_validation_checklist.md`

## Why No External Repo Was Cloned Yet

Task 4 currently needs a project-local scaffold, not borrowed training code. The blocking input is not missing GitHub code; it is missing project-local checkpoint paths and real prompt-level traces for the chosen Qwen-style pairs.

## Concept-to-Implementation Mapping

| Concept | Local implementation file | Status |
| --- | --- | --- |
| Similar-pair composite score and close/medium/far buckets | `Experiment/core_code/scd_validation/pair_screening.py` | implemented |
| Restore-one plus greedy patch-layer selection | `Experiment/core_code/scd_validation/claim_eval.py` | implemented as config-driven selector |
| Original SCD prompt/run/claim aggregation | `Experiment/core_code/scd_validation/claim_eval.py` | implemented |
| Pair screening CLI | `Experiment/core_code/run_pair_screening.py` | implemented |
| Original baseline aggregation CLI | `Experiment/core_code/run_original_scd_baseline.py` | implemented |
| Real Hugging Face trace extraction | not yet in workspace | pending |
| Real low-rank reuse / patch training | not yet in workspace | pending |
| Real online cache assembly against chosen checkpoints | not yet in workspace | pending |

## Current Engineering Boundary

The scaffold is deliberately honest:

- it can rank candidate pairs once real screening metrics are available
- it can select patch layers from measured restore-one / greedy gains
- it can emit the full Task 3 schema from real prompt-level measurements
- it does not invent real traces, checkpoints, or claimed results

## Immediate Next Integration Point

Add a project-local trace collector that writes measured prompt rows into `Experiment/core_code/configs/original_scd_baseline_template.json`-compatible format, then replace the smoke config with a real close-pair run config.
