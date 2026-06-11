# MistralLite -> Mistral-7B Multi-Layer Strict-Stop Report (2026-04-07)

## Scope

- Host: `ser`
- Run root: `/disk_n/c101-3-1/projects/scd-repro-codex-mistral-instruct-20260323/runs/mistrallite_to_mistral7_cmrc_blockwise_serrerun_20260406`
- Search root:
  - `blockwise_search_serrerun_20260406_b1_replayk5_f1pct5_multilayer_f1first`
- Final strict output mirror:
  - `/Users/weimingming/dr-claw/scd-new/Experiment/analysis/remote_artifacts/mistrallite_to_mistral7_multilayer_strictstop_step15_20260407`

Search rule for this report:

- `block_size=1`
- `replay_top_k=5`
- replay selector: `quality_threshold_f1_first_ttft_second`
- replay decision gate: relative `F1 drop <= 5%` versus `patch-only`
- `EM` gate disabled
- objective: under the F1 gate, keep as many reuse layers as possible; once replay has no quality-passing candidate, stop immediately
- `fallback=[]`
- `recompute=[]`

## Stop-Rule Fix

Before this report, the multi-layer searcher had a stop-rule bug: when replay candidates existed but none passed the replay quality gate, the search could still continue through a fallback path instead of stopping.

This has now been fixed in:

- `Experiment/core_code/runtime_repo/src/scd_repro/cli/search_blockwise_hybrid.py`

The current behavior is:

- when replay quality-passing set is empty, append `stop_reason = "no_replay_quality_pass"`
- stop the greedy search immediately

So the earlier over-extended multi-layer route is no longer valid for the user's objective.

## Verified Strict Stop Point

Patch-only anchor on this run:

- `patch-only F1 = 0.6285388243746529`
- allowed absolute F1 drop = `0.03142694121873265`
- minimum replay-passing F1 = `0.5971118831559202`

Replay verification on `ser`:

| step | replay candidates | quality-passing count | note |
| --- | ---: | ---: | --- |
| 14 | 5 | 2 | search may continue |
| 15 | 5 | 0 | strict search must stop |

Observed step-15 replay F1 values were:

- `0.5861`
- `0.5848`
- `0.5934`
- `0.5842`
- `0.5865`

All are below the minimum passing F1 `0.5971118831559202`.

Therefore, the correct final route under the strict rule is the route at the start of step 15.

## Final Multi-Layer Route

- Reuse layers:
  - `2, 3, 4, 8, 9, 10, 20, 22, 23, 24, 25, 26, 27, 30`
- Patch layers:
  - `0, 1, 5, 6, 7, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 28, 29, 31`
- Reuse layer count: `14`
- Patch layer count: `18`
- Fallback layers: `[]`
- Recompute layers: `[]`

## Final Strict Comparison

These are the current canonical numbers for this strict-stop multi-layer route.

| method | F1 | EM | TTFT ms | encode | transfer | rebuild | patch_infer | decode_first |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 0.5969 | 0.250 | 103.62 | 0.00 | 0.00 | 0.00 | 0.00 | 22.17 |
| reuse-only | 0.0141 | 0.000 | 33.52 | 0.71 | 8.17 | 2.95 | 0.00 | 21.70 |
| patch-only | 0.6285 | 0.285 | 53.93 | 0.81 | 12.70 | 0.00 | 18.48 | 21.95 |
| hybrid (14 reuse layers) | 0.6072 | 0.250 | 46.18 | 0.76 | 12.22 | 1.47 | 10.19 | 21.54 |

## Delta Vs Patch-Only

- `TTFT -7.75 ms`
- `F1 -0.02132`
- `EM -0.035`

Interpretation:

- the hybrid route is faster than `patch-only`
- the F1 loss stays inside the allowed relative 5% drop budget because:
  - allowed absolute F1 drop = `0.03143`
  - actual absolute F1 drop = `0.02132`
- so this is a valid strict-stop multi-layer route under the user's rule

## Hybrid TTFT Breakdown

Hybrid TTFT is charged strictly as:

- `encode_ms = 0.7649`
- `transfer_ms = 12.2229`
- `rebuild_ms = 1.4666`
- `patch_infer_ms = 10.1908`
- `decode_first_ms = 21.5379`

Additional hybrid patch sub-breakdown:

- `patch_shard_load_ms = 1.3989`
- `patch_shard_gpu_copy_ms = 0.2771`
- `patch_kproj_vproj_ms = 2.9334`
- `patch_rotary_ms = 2.9057`
- `patch_knorm_ms = 0.0`

## Conclusion

Under the corrected strict-stop rule, the answer is yes: on `MistralLite -> Mistral-7B`, the multi-layer hybrid search can keep `14` reuse layers without fallback or recompute and still satisfy the user's replay-stage quality constraint.

The current best strict multi-layer point is:

- `hybrid F1 0.6072 / EM 0.250 / TTFT 46.18 ms`

Relative to `patch-only`, this is a real trade-off rather than a Pareto win:

- faster by `7.75 ms`
- lower by `0.0213 F1`
- still inside the allowed `5%` relative F1-drop budget

## Local Artifact Mirror

- report:
  - `/Users/weimingming/dr-claw/scd-new/Experiment/analysis/mistrallite_to_mistral7_multilayer_strictstop_step15_20260407.md`
- synced strict outputs:
  - `/Users/weimingming/dr-claw/scd-new/Experiment/analysis/remote_artifacts/mistrallite_to_mistral7_multilayer_strictstop_step15_20260407`
- key files:
  - `summary.json`
  - `selected_hybrid_route_step15.json`
  - `strict_baseline_reuse_only/final_comparison.json`
  - `strict_hybrid/final_comparison.json`
