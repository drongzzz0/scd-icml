# Qwen3-1.7B Self-Self True Patch-only Run on `ser`

Date: 2026-04-02

Scope:
- Host: `ser`
- Repo: `/disk_n/c101-3-1/projects/scd-repro-codex-mistral-instruct-20260323`
- Pair: `Qwen3-1.7B -> Qwen3-1.7B`
- Task: CMRC dev, `50` samples
- Route: all 28 layers in `patch_layers`
- Fallback: none
- Recompute: none
- New runtime method: `patch_only`

Implementation change:
- Added a dedicated `patch_only` runtime branch to strict runtime.
- This branch requires full layer coverage by `patch_layers` and skips the old reuse scaffold entirely.
- Added a CLI flag `run_patch_only` so the route can be selected independently of `reuse_patch`.

Run artifact:
- Remote output: `/disk_n/c101-3-1/projects/scd-repro-codex-mistral-instruct-20260323/runs/qwen17_self_cmrc/strict_patch_only_diag50_20260402`
- Local mirror: `/Users/weimingming/dr-claw/scd-new/Experiment/analysis/remote_artifacts/qwen17_self_patchonly_true_patch_only_diag50_20260402/summary.json`

Main result:

| method | F1 | EM | TTFT (ms) | encode (ms) | transfer (ms) | rebuild (ms) | patch_infer (ms) | decode_first (ms) | payload (MB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `full_consumer` | 0.7375 | 0.38 | 39.68 | 0.00 | 0.00 | 0.00 | 0.00 | 1.44 | 0.00 |
| `patch_only` | 0.7700 | 0.46 | 61.04 | 0.00 | 0.00 | 0.00 | 36.56 | 24.48 | 41.32 |

Direct comparison against the previous `reuse_patch` implementation on the same self-self all-28 route:

| route impl | F1 | TTFT (ms) | encode (ms) | transfer (ms) | rebuild (ms) | patch_infer (ms) | decode_first (ms) | payload (MB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| old `reuse_patch` | 0.7700 | 167.74 | 12.18 | 3.47 | 30.64 | 97.44 | 24.01 | 82.63 |
| new `patch_only` | 0.7700 | 61.04 | 0.00 | 0.00 | 0.00 | 36.56 | 24.48 | 41.32 |

Interpretation:
- The true `patch_only` branch preserves the same task quality as the earlier healthy all-28 `reuse_patch` route on `ser`.
- Removing the reuse scaffold cuts TTFT by `106.71 ms`, from `167.74` to `61.04 ms`.
- The saved time comes almost entirely from eliminating reuse-side `encode`, `transfer`, and `rebuild`, plus a secondary reduction in `patch_infer` because the old implementation also paid the overwrite cost after building the reuse scaffold.
- Payload drops from `82.63 MB` to `41.32 MB`, which is exactly consistent with removing the reused latent payload and keeping only the patch-side hidden payload.
- This confirms the earlier suspicion: for an all-layer patch route with no fallback and no recompute, the reuse scaffold was redundant overhead rather than part of the effective algorithm.
