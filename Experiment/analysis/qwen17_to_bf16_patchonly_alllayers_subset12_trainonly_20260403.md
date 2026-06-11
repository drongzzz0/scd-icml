# Qwen3-1.7B -> Qwen3-1.7B-bf16 All-Layer Patch-Only Training Progress (2026-04-03)

Objective:
- Pair: `Qwen3-1.7B -> Qwen3-1.7B-bf16`
- Target serving comparison: `baseline / reuse-only / patch-only`
- Strict setting: no fallback, no recompute
- Patch family: all 28 layers, `xnorm_nativekv_residual_mlp_v1`

Remote artifacts:
- Pair subset dir: `ser:/disk_n/c101-3-1/projects/scd-repro-codex-mistral-instruct-20260323/runs/qwen17_to_bf16_cmrc/pairs_cmrc_seed42_subset12`
- Route: `ser:/disk_n/c101-3-1/projects/scd-repro-codex-mistral-instruct-20260323/runs/qwen17_to_bf16_cmrc/route_reuse_patchonly_all28_20260403.json`
- Eval config: `ser:/disk_n/c101-3-1/projects/scd-repro-codex-mistral-instruct-20260323/runs/qwen17_to_bf16_cmrc/eval_baseline_reuse_patchonly_all28_50_20260403.json`
- Train config: `ser:/disk_n/c101-3-1/projects/scd-repro-codex-mistral-instruct-20260323/configs/qwen17_to_bf16_patch_xnorm_nativekv_residual_mlp_alllayers_subset12_e1_h2048_b2_20260403.json`
- Trained ckpt: `ser:/disk_n/c101-3-1/projects/scd-repro-codex-mistral-instruct-20260323/runs/qwen17_to_bf16_cmrc/ckpt_patch_xnorm_nativekv_residual_mlp_alllayers_subset12_e1_h2048_b2_20260403`

What was run:
- First attempt: full `200`-chunk all-layer training, `e2`, `h2048`, `2` blocks.
- Finding from that attempt: after stats finished, `layer 0` linear base already reached extremely small error (`linear_base_k_rel_mse=0.000017`, `linear_base_v_rel_mse=0.000008`), confirming this is a very close pair.
- Problem: the implementation then entered a per-layer all-chunk error-stat loop that scaled to hour-level runtime. That run was stopped.
- Replacement run: `subset12`, all `28` layers, `1` epoch, `h2048`, `2` blocks.

Key training observations from the completed `subset12` run:
- Stats phase completed quickly: `12/12` chunks in `15.5s`.
- Linear base error on shallow layers was sometimes small, but middle/deep layers showed very large native-KV mismatch under direct linear projection.
- Residual-layer logs consistently showed `base_h_rel_mse` near `0`, often exactly `0.000000`.
- For many layers, the final `layer complete` line was numerically unchanged from init on `h/k/v` relative MSE, despite nonzero one-epoch training loss.

Interpretation:
- On this near pair, the hidden-state channel appears much more aligned than the native-KV channel.
- The all-layer residual MLP did not obviously learn a meaningful non-identity hidden correction in the `subset12/e1` setting.
- That makes the next evaluation especially important: if `patch-only` still works well, it would suggest the useful effect mainly comes from re-projecting near-matched hidden states with the consumer's native `Wk/Wv`, not from a learned hidden-state transport.

Current blocker:
- After the training checkpoint was written, SSH access to `ser` started failing at the transport layer:
  - `Connection closed by 198.18.0.14 port 31005`
- Because of that outage, the following steps were not completed:
  - convert the monolithic ckpt to `sharded_w4`
  - run strict `baseline / reuse-only / patch-only` eval on `50` CMRC samples
  - mirror `summary.json` and `per_mode_summary.json` back to the local project

Next commands to run once `ser` access recovers:
- shard ckpt with `save_patch_ckpt(..., sharded=True, layers_per_shard=4)`
- evaluate with `python -m scd_repro.cli.eval_cmrc_strict`
- collect `summary.json` for the new three-way table
