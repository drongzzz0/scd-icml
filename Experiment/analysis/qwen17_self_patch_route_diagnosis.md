# Qwen3-1.7B Self-Self Patch Route Diagnosis

## Setup

- Producer: `Qwen3-1.7B`
- Consumer: `Qwen3-1.7B`
- Task: CMRC2018 dev
- Samples: `50`
- Reuse checkpoint: `ckpt_reuse_layerwise_seed42`
- Strict runtime: no fallback, no recompute
- Base patch route: overwrite selected patch layers on top of the healthy layerwise `reuse_only` cache

## What Was Broken

The original self-self patch route failed for two distinct reasons:

1. Pair capture silently truncated training prefixes to `256` tokens even when eval ran at `1024`.
2. The `direct_kv_v1` path on Qwen3 used a target/runtime formulation that was not faithful to Qwen3 attention:
   - training directly fit post-cache K/V from `xnorm_a`
   - runtime used the repo's generic `re_rope_k_torch`
   - Qwen3 attention applies `k_proj -> k_norm -> native rotary`, and the generic rope helper did not reproduce the actual cached keys

The key tensor-level check after the second diagnosis was:

- `native k_proj + k_norm + repo re_rope_k_torch` vs real `k_b`: still large MSE
- `native k_proj + k_norm + native apply_rotary_pos_emb` vs real `k_b`: near-zero MSE

So the real runtime bug was the rotary application, not only the direct-KV regression target.

## Fixes Applied On March 31, 2026

### Fix 1: capture length precedence

- explicit CLI `--max-length` now overrides config `capture.max_length`
- new training pairs were recaptured with long prefixes:
  - old broken pair support: effectively `256`
  - new pair support: up to `734` prefix tokens in the 50-sample set

### Fix 2: Qwen-aware direct-KV target

- added `direct_kv_knorm_v1`
- training target changed from repo-style post-cache `de_rope(K)` to:
  - consumer native pre-`k_norm` K
  - consumer native V
- this removes the largest Qwen-specific mismatch from the regression target

### Fix 3: native rotary at runtime

- the `direct_kv_knorm_v1` runtime path now applies consumer-native rotary instead of `re_rope_k_torch`
- after this change, the runtime path matches real cached Qwen3 keys at near machine precision when the pre-rotary key is correct

## Training Signal After The Fix

New checkpoint:

- `runs/qwen17_self_cmrc/ckpt_patch_directkv_knorm_all28_len1024_seed42`

Patch training error changed substantially:

- `layer 0 patch_kv_error`: `178272799.108549` -> `1014534.836084`
- `layer 1 patch_kv_error`: `1.935608` -> `0.000041`
- `layer 2 patch_kv_error`: `1.076786` -> `0.000137`
- most layers `1..21` are now extremely small
- deep layers `22..27` remain nonzero but still far below the old broken run

This means the remaining dominant outlier is now clearly `layer 0`.

## Final Strict Results

### Before the Qwen-aware fix

| Mode | Route | Fit mode | F1 | TTFT (ms) |
|---|---|---|---:|---:|
| `reuse_patch` | patch `0..27` | `direct_kv_v1` | 0.0049 | 62.38 |
| `reuse_patch` | patch `1..27` | `direct_kv_v1` | 0.0508 | 71.02 |

### After the Qwen-aware fix and native rotary fix

| Mode | Route | Fit mode | F1 | EM | TTFT (ms) | Payload (MB) |
|---|---|---|---:|---:|---:|---:|
| `full_consumer` | native | n/a | 0.7375 | 0.38 | 37.81 | 0.00 |
| `reuse_only` | layerwise reuse | n/a | 0.7372 | 0.38 | 58.49 | 41.32 |
| `reuse_patch` | patch `0..27` | `direct_kv_knorm_v1` | 0.0119 | 0.00 | 75.35 | 82.63 |
| `reuse_patch` | patch `1..27` | `direct_kv_knorm_v1` | 0.7425 | 0.44 | 88.10 | 81.16 |

## Interpretation After The Linear Repair

- The patch route is now repaired enough to work on `Qwen3-1.7B -> Qwen3-1.7B`, but only when `layer 0` is excluded.
- `patch 1..27` no longer collapses. It slightly exceeds same-run `full_consumer`/`reuse_only` on this 50-sample CMRC slice.
- `patch 0..27` still fails badly, which isolates the remaining unresolved issue to `layer 0`.
- Therefore the current practical default for Qwen3 self-self patch experiments should be:
  - fit mode: `direct_kv_knorm_v1`
  - runtime rotary: consumer-native
  - route: exclude `layer 0`

## Practical Decision Before The MLP Run

For current Qwen3 self-self experiments:

- treat `reuse_patch` with `patch_layers = 1..27` as the healthy default patch route
- do not use all-28-layer patch as evidence, because `layer 0` is still unstable
- this directly motivated the dedicated `layer 0` audit below

## Layer 0 Control Audit On March 31, 2026

The dedicated `layer 0` audit resolved the remaining ambiguity about whether the failure was caused by runtime semantics or by the learned translator itself.

### Tensor-level controls

Using the repaired self-self capture set:

- `xnorm_a.0` and `xnorm_b.0` are exactly identical on all 50 sampled prefixes
- running consumer-native `k_proj/v_proj` directly on `xnorm_a.0`, then applying consumer `k_norm` and native rotary, reproduces the true cached layer-0 KV almost exactly
- the learned layer-0 `direct_kv_knorm_v1` translator remains catastrophically wrong

Aggregated relative MSE over 50 samples:

| Control | K rel-MSE | V rel-MSE |
|---|---:|---:|
| native projection from `xnorm_a.0` | `5.59e-6` | `4.79e-6` |
| learned layer-0 translator | `1.49` | `2.58e7` |

This means the layer-0 problem is **not**:

- a mismatch between producer and consumer hidden states in self-self mode
- a wrong `k_norm` or rotary runtime path
- a generic cache overwrite bug

It is the trained layer-0 translator itself.

### Strict route-level control

A fresh strict run patched only `layer 0` on top of the healthy layerwise reuse baseline:

| Mode | Route | F1 | EM | TTFT (ms) |
|---|---|---:|---:|---:|
| `reuse_only` | no patch | 0.7372 | 0.38 | 71.19 |
| `reuse_patch` | learned patch on `layer 0` only | 0.0124 | 0.00 | 51.51 |
| `oracle_kv_patchonly` | true consumer KV on `layer 0` only | 0.7372 | 0.38 | 49.13 |

So the same runtime/cache assembly path works perfectly when `layer 0` receives the true KV, but collapses when it receives the learned layer-0 patch output.

### Updated interpretation

- The remaining failure is now localized to the **learned layer-0 direct-KV translator**, not to self-self hidden-state mismatch and not to runtime patch application.
- In self-self mode, `layer 0` is in principle patchable: the oracle and native-projection controls both behave correctly.
- Therefore the current blocker is specifically that `direct_kv_knorm_v1` training produces a broken `Wk/Wv` solution for `layer 0`.

## Nonlinear MLP Follow-up On March 31, 2026

The next self-self follow-up replaced the linear direct-KV patch with the repo's existing nonlinear `xnorm_semantic_v1` route:

- fit mode: `xnorm_semantic_v1`
- translator: `PatchMLP`
- state target: `xnorm`
- apply mode: `overwrite`
- loss weights: `lambda_h = 1.0`, `lambda_k = 0.0`, `lambda_v = 0.0`, `lambda_sem = 0.0`
- MLP hidden width: `512`

This is a genuinely nonlinear patch route, but it is **not** a nonlinear version of `direct_kv_knorm_v1`. It predicts patched hidden state and then lets the consumer runtime derive KV from that state.

### Strict results

| Mode | Route | Fit mode | F1 | EM | TTFT (ms) | Rebuild (ms) | Patch infer (ms) |
|---|---|---|---:|---:|---:|---:|---:|
| `full_consumer` | native | n/a | 0.7375 | 0.38 | 59.10 | 0.00 | 0.00 |
| `reuse_only` | layerwise reuse | n/a | 0.7372 | 0.38 | 103.07 | 31.40 | 0.00 |
| `reuse_patch` | patch `0..27` | `xnorm_semantic_v1` | 0.0000 | 0.00 | 102.16 | 28.57 | 34.11 |
| `reuse_patch` | patch `1..27` | `xnorm_semantic_v1` | 0.0000 | 0.00 | 126.85 | 37.91 | 43.20 |

### What changed relative to the repaired linear route

- The repaired linear route `direct_kv_knorm_v1` was healthy once `layer 0` was removed.
- The nonlinear `PatchMLP` route is worse: both `all28` and `excl0` collapse to `0.0` F1.
- So the current nonlinear route does **not** rescue the self-self patch problem, and excluding `layer 0` no longer helps.

### Training-side diagnosis

The saved checkpoint metrics show that only the first few layers fit the hidden-state target well:

- `layer 0 hidden_mse = 0.000198`
- `layer 1 hidden_mse = 0.0260`
- `layer 2 hidden_mse = 0.0483`

From `layer 3` onward, hidden-state fit deteriorates sharply and generally worsens with depth:

- `layer 3 hidden_mse = 249.2`
- `layer 15 hidden_mse = 102.3`
- `layer 22 hidden_mse = 179.9`
- `layer 27 hidden_mse = 576.4`

Deep-layer auxiliary errors are also large:

- `layer 22 k_mse = 1551.4`, `v_mse = 299.3`
- `layer 27 k_mse = 24999.8`, `v_mse = 1484.1`

Inference from these metrics:

- the current nonlinear formulation is not just failing because of `layer 0`
- the hidden-state patch itself becomes unstable through much of the stack
- simply "making patch nonlinear" is therefore not enough on this setup

## Deeper Direct-KV Residual MLP Follow-up On March 31, 2026

The next follow-up kept the repaired linear `direct_kv_knorm_v1` path as the base for all layers and added a deeper residual MLP that learns `delta K / delta V` directly:

- fit mode: `direct_kv_knorm_residual_mlp_v1`
- base path: solved linear `direct_wk/direct_wv` for all `28` layers
- nonlinear residual layers: only `layer 0`
- residual trunk: width `4096`, `4` residual blocks
- training: `12` epochs, batch size `512`, `lr = 3e-4`

This is a stronger nonlinear design than the earlier `PatchMLP` route because it keeps the healthy direct-KV formulation for layers `1..27` and concentrates the nonlinear capacity on the only broken layer.

### Implementation note

- the first remote run failed during residual dataset preload because the native consumer projection weights were on `cuda:1` while `xnorm_b` was being projected on CPU
- that device-placement bug was fixed locally and on `ser`
- the successful rerun is the `20260331_resid_v2` result below

### Strict results

| Mode | Route | Fit mode | F1 | EM | TTFT (ms) | Rebuild (ms) | Patch infer (ms) |
|---|---|---|---:|---:|---:|---:|---:|
| `full_consumer` | native | n/a | 0.7375 | 0.38 | 56.41 | 0.00 | 0.00 |
| `reuse_only` | layerwise reuse | n/a | 0.7372 | 0.38 | 84.25 | 30.30 | 0.00 |
| `reuse_patch` | patch `0..27` | `direct_kv_knorm_residual_mlp_v1` | 0.0087 | 0.00 | 193.11 | 29.12 | 124.15 |
| `reuse_patch` | patch `1..27` | `direct_kv_knorm_residual_mlp_v1` | 0.7425 | 0.44 | 122.10 | 34.86 | 40.73 |

### Layer-0 control under the residual route

| Mode | Route | F1 | EM | TTFT (ms) | Patch infer (ms) |
|---|---|---:|---:|---:|---:|
| `full_consumer` | native | 0.7375 | 0.38 | 61.09 | 0.00 |
| `reuse_only` | no patch | 0.7372 | 0.38 | 92.09 | 0.00 |
| `reuse_patch` | residual patch on `layer 0` only | 0.0113 | 0.00 | 177.57 | 104.81 |
| `oracle_kv_patchonly` | true consumer KV on `layer 0` only | 0.7372 | 0.38 | 68.06 | 0.00 |

### What changed relative to the repaired linear route

- `patch 0..27` did **not** improve: F1 moved from `0.0119` to `0.0087`, while TTFT worsened from `75.35 ms` to `193.11 ms`
- `patch 1..27` did **not** improve either: F1 stayed at `0.7425`, but TTFT worsened from `88.10 ms` to `122.10 ms`
- `layer 0` remains broken: the residual route reached only `0.0113` F1 on the dedicated layer-0 control, essentially unchanged from the earlier learned linear layer-0 result
- `oracle_kv_patchonly` on `layer 0` is still healthy, so the failure remains in the learned layer-0 translator rather than in runtime cache application

### Training-side diagnosis

The saved residual metrics confirm that the deeper nonlinear model barely changes the layer-0 error:

- `base_k_rel_mse`: `1.213e7 -> 1.199e7`
- `base_v_rel_mse`: `2.330e7 -> 2.184e7`
- total loss remains dominated by the layer-0 regression error rather than by a useful correction term

So even with a direct-KV residual objective and a much larger nonlinear model, the optimizer still fails to find a usable layer-0 mapping.

## Identity-Anchored Hidden Residual Route On March 31, 2026

The next follow-up changed the parameterization only for the broken `layer 0`:

- fit mode: `xnorm_nativekv_residual_mlp_v1`
- layers `1..27`: unchanged healthy linear `direct_kv_knorm_v1`
- layer `0`: predict `xb_hat = xa + delta_h`
- then apply consumer-native `k_proj/v_proj + k_norm + native rotary`

This is the "fix hidden first, then let the consumer derive KV natively" design. The crucial difference from the failed direct-KV routes is that the residual module starts exactly from identity.

### Training-side result

For `layer 0`, the saved metrics show that the identity initialization is already the correct solution in self-self mode:

- `base_h_rel_mse = 0.0`
- `base_k_rel_mse = 1.43e-11`
- `base_v_rel_mse = 1.55e-11`
- best checkpoint stayed at the initialization state

During training, later epochs moved away from the optimum, but checkpoint selection kept the epoch-0 identity solution.

This resolves the earlier ambiguity:

- `layer 0` is not intrinsically unpatchable
- the failure came from the old parameterization, not from runtime cache overwrite

### Strict results

| Mode | Route | Fit mode | F1 | EM | TTFT (ms) | Rebuild (ms) | Patch infer (ms) |
|---|---|---|---:|---:|---:|---:|---:|
| `full_consumer` | native | n/a | 0.7375 | 0.38 | 51.03 | 0.00 | 0.00 |
| `reuse_only` | layerwise reuse | n/a | 0.7372 | 0.38 | 103.67 | 36.68 | 0.00 |
| `reuse_patch` | patch `0..27` | `xnorm_nativekv_residual_mlp_v1` | 0.7700 | 0.46 | 200.01 | 27.91 | 132.23 |
| `reuse_patch` | patch `1..27` | `xnorm_nativekv_residual_mlp_v1` | 0.7425 | 0.44 | 123.42 | 35.10 | 31.08 |

### Layer-0 control under the hidden-residual route

| Mode | Route | F1 | EM | TTFT (ms) | Patch infer (ms) |
|---|---|---:|---:|---:|---:|
| `full_consumer` | native | 0.7375 | 0.38 | 38.45 | 0.00 |
| `reuse_only` | no patch | 0.7372 | 0.38 | 76.22 | 0.00 |
| `reuse_patch` | hidden residual patch on `layer 0` only | 0.7372 | 0.38 | 193.12 | 127.91 |
| `oracle_kv_patchonly` | true consumer KV on `layer 0` only | 0.7372 | 0.38 | 63.45 | 0.00 |

### Interpretation

- This route **does** fix the `layer 0` quality problem.
- Patching all `28` layers is now healthy in self-self mode and no longer collapses.
- The quality gain over the earlier healthy `patch 1..27` route is small but real on this 50-sample slice:
  - `F1 0.7425 -> 0.7700`
  - `EM 0.44 -> 0.46`
- The cost is latency:
  - `TTFT 88.10 ms -> 200.01 ms` relative to the earlier linear `patch 1..27` healthy route
  - almost all of the new cost comes from `patch_infer_ms`, because runtime still executes the residual module even though the selected checkpoint is effectively identity

So the new route solves the **representation** problem at `layer 0`, but it does not yet solve the **efficiency** problem.

## Runtime Identity-Skip Follow-up On March 31, 2026

The next follow-up changed runtime only:

- inspect the loaded `layer 0` hidden-residual module
- if `delta_h_head.weight` and `delta_h_head.bias` are exactly zero, treat the residual as identity
- skip the residual MLP forward
- still apply consumer-native `k_proj/v_proj + k_norm + native rotary`

This does not change the checkpoint. It only removes useless residual inference when the selected solution is already `xb_hat = xa`.

### Strict results after identity skip

| Mode | Route | F1 | EM | TTFT (ms) | Patch infer (ms) |
|---|---|---:|---:|---:|---:|
| `reuse_patch` | patch `0..27` | 0.7700 | 0.46 | 167.74 | 97.44 |
| `reuse_patch` | patch `1..27` | 0.7425 | 0.44 | 126.55 | 46.49 |
| `reuse_patch` | hidden residual `layer 0` only | 0.7372 | 0.38 | 140.55 | 74.26 |

Compared with the pre-skip runtime:

- `all28`: `TTFT 200.01 -> 167.74 ms`, `patch_infer 132.23 -> 97.44 ms`
- `layer0-only`: `TTFT 193.12 -> 140.55 ms`, `patch_infer 127.91 -> 74.26 ms`
- `excl0` is not expected to benefit because it never uses the `layer 0` residual path; its rerun drifted slightly upward

### Interpretation after identity skip

- the fast path works: removing the useless residual MLP produces a clear TTFT reduction on the routes that actually use `layer 0`
- the route is still not fast enough
- after the skip, the main remaining cost is consumer-native hidden-to-KV projection itself:
  - `k_proj/v_proj`
  - `k_norm`
  - native rotary

So the route is now in a cleaner state:

- correctness is healthy
- useless residual inference has been removed
- the next optimization target is the native hidden-to-KV projection path rather than the residual module

## Layer-0 Breakdown Profiling On March 31, 2026

To resolve what "native hidden-to-KV projection path" actually means in runtime cost, the `layer0-only` route was rerun with extra synchronized stage timers inside the `xnorm_nativekv_residual_mlp_v1` branch.

This profiling run is diagnostic only:

- it inserts extra synchronization points
- so the absolute `patch_infer_ms` is for analysis, not for headline TTFT comparison
- the useful signal is the relative composition inside the patched `layer 0` path

### Mean layer-0 patch breakdown

For `reuse_patch` on the strict `layer0-only` route:

| Component | Mean ms |
|---|---:|
| identity check | 73.92 |
| residual forward | 0.00 |
| `k_proj/v_proj` | 0.20 |
| `k_norm` | 0.18 |
| native rotary | 0.68 |

The corresponding profiled run recorded:

- `patch_infer_ms = 75.40`
- `TTFT = 142.63 ms`
- `F1 = 0.7372`

### Interpretation of the profiling result

- the actual native projection stack is small on this route:
  - `k_proj/v_proj + k_norm + rotary` totals only about `1.06 ms`
- but the mean breakdown numbers were later found to be misleading as a steady-state interpretation
- after inspecting `per_sample_metrics.jsonl`, the profiled `layer0-only` run showed:
  - `patch_infer` median only about `1.45 ms`
  - `patch_infer` max about `3698.15 ms`
  - so the large mean was dominated by a single first-sample warm-start event rather than by per-sample native projection cost

The real issue was therefore not "every sample still pays about `74 ms` for identity check", but rather that runtime module materialization and diagnostic identity inspection were being counted inside the first measured sample.

This changed the immediate engineering target:

1. move identity detection to a cached load-time flag
2. move runtime patch-module materialization completely outside the timed sample path

## Runtime Prewarm Follow-up On March 31, 2026

The final runtime optimization added an explicit one-time prewarm step before timed evaluation:

- pre-load runtime patch modules for the selected patch layers
- cache the identity flag when the module is loaded
- ensure the first measured sample no longer includes patch-module materialization

This was tested first on the strict `layer0-only` route because that route isolates the previously problematic `layer 0` path cleanly.

### Strict `layer0-only` results after prewarm

| Mode | Route | F1 | EM | TTFT (ms) | Patch infer (ms) |
|---|---|---:|---:|---:|---:|
| `full_consumer` | native | 0.7375 | 0.38 | 51.29 | 0.00 |
| `reuse_only` | no patch | 0.7372 | 0.38 | 90.78 | 0.00 |
| `reuse_patch` | hidden residual `layer 0` only + prewarm | 0.7372 | 0.38 | 73.67 | 1.42 |
| `oracle_kv_patchonly` | true consumer KV on `layer 0` only | 0.7372 | 0.38 | 63.94 | 0.00 |

Compared with the earlier runtime variants:

- `skipid`: `patch_infer 74.26 ms`, `TTFT 140.55 ms`
- `breakdown`: `patch_infer 75.40 ms`, `TTFT 142.63 ms`
- `loadid only`: `patch_infer 74.33 ms`, `TTFT 143.22 ms`
- `prewarm`: `patch_infer 1.42 ms`, `TTFT 73.67 ms`

### Per-sample timing distribution

For `reuse_patch` on the strict `layer0-only` route:

| Runtime variant | Mean patch infer (ms) | Median patch infer (ms) | Max patch infer (ms) | Mean TTFT (ms) | Median TTFT (ms) |
|---|---:|---:|---:|---:|---:|
| `breakdown` | 75.40 | 1.45 | 3698.15 | 142.63 | 64.75 |
| `loadid` | 74.33 | 1.40 | 3646.50 | 143.22 | 65.03 |
| `prewarm` | 1.42 | 1.39 | 1.95 | 73.67 | 64.31 |

The interpretation is now much cleaner:

- the hidden-residual `layer 0` route was already cheap in steady state
- the apparent `~74 ms` patch cost was almost entirely a first-sample warm-start artifact
- once prewarm moves that cost outside timed evaluation, the measured `layer0-only` patch path becomes close to oracle and clearly faster than `reuse_only`

## All-28-Layer Prewarm Rerun On March 31, 2026

After verifying that `layer0-only` was dominated by first-sample warm-start, the same prewarm treatment was applied to the strict all-28-layer hidden-residual route.

### Strict all-28-layer results after prewarm

| Mode | Route | F1 | EM | TTFT (ms) | Patch infer (ms) |
|---|---|---:|---:|---:|---:|
| `full_consumer` | native | 0.7375 | 0.38 | 50.30 | 0.00 |
| `reuse_only` | layerwise reuse | 0.7372 | 0.38 | 88.33 | 0.00 |
| `reuse_patch` | hidden residual `0..27` + prewarm | 0.7700 | 0.46 | 99.14 | 33.26 |

Compared with the earlier `identity skip` runtime:

- `F1` stayed exactly the same: `0.7700`
- `EM` stayed exactly the same: `0.46`
- `TTFT` dropped from `167.74 ms` to `99.14 ms`
- `patch_infer` dropped from `97.44 ms` to `33.26 ms`

### Per-sample timing distribution

For `reuse_patch` on the strict all-28-layer route:

| Runtime variant | Mean patch infer (ms) | Median patch infer (ms) | Max patch infer (ms) | Mean TTFT (ms) | Median TTFT (ms) |
|---|---:|---:|---:|---:|---:|
| `skipid` | 97.44 | 25.37 | 3596.80 | 167.74 | 90.83 |
| `prewarm` | 33.26 | 32.01 | 40.06 | 99.14 | 96.54 |

This is a different pattern from `layer0-only`:

- prewarm still removes the huge first-sample spike cleanly
- but the all-28-layer steady-state path itself is not cheap
- after warm-start removal, the route still pays about `32-33 ms` patch inference in normal samples

### Updated interpretation

- The all-28-layer hidden-residual route is now much cleaner as a measured operating point:
  - quality is healthy
  - first-sample warm-start is no longer contaminating the mean
- It is still slower than `reuse_only` on the same run:
  - `99.14 ms` vs `88.33 ms`
- But it is no longer dramatically slower as in the pre-prewarm runs, and it still gives the best verified self-self all-28-layer quality.

## Updated Practical Decision

For current `Qwen3-1.7B -> Qwen3-1.7B` strict self-self experiments:

- the strongest correctness result across all `28` patch layers remains `xnorm_nativekv_residual_mlp_v1`
- the cleanest measured `layer 0` runtime result is now `xnorm_nativekv_residual_mlp_v1 + identity skip + prewarm`
- the cleanest measured all-28-layer result is now also the prewarmed hidden-residual route
For overall route choice:

- there are still two healthy patch families:
  - the repaired linear route `direct_kv_knorm_v1` with `patch_layers = 1..27`
  - the identity-anchored hidden residual route `xnorm_nativekv_residual_mlp_v1`
- the repo's current nonlinear `xnorm_semantic_v1` MLP route should not be treated as a rescue path
- the stronger targeted nonlinear route `direct_kv_knorm_residual_mlp_v1` should also not be treated as a rescue path
- if the goal is best self-self quality across all `28` patch layers, the hidden-residual route remains the strongest result
- if the goal is best already-verified TTFT among healthy self-self patch routes, the repaired linear `patch 1..27` route is still the safer default
- if the goal is best verified all-28-layer operating point, use the prewarmed hidden-residual route

## Local Mirrored Artifacts

- `Experiment/analysis/remote_artifacts/qwen17_self_patchonly_directkv_knorm_all28_len1024_50_20260331_v2/summary.json`
- `Experiment/analysis/remote_artifacts/qwen17_self_patchonly_excl0_directkv_knorm_len1024_50_20260331_v2/summary.json`
- `Experiment/analysis/remote_artifacts/qwen17_self_patchonly_layer0_directkv_knorm_vs_oracle_50_20260331/summary.json`
- `Experiment/analysis/remote_artifacts/qwen17_self_patchonly_layer0_directkv_knorm_vs_oracle_50_20260331/layer0_self_patch_tensor_diag_20260331.json`
- `Experiment/analysis/remote_artifacts/qwen17_self_patchonly_semantic_mlp_all28_len1024_50_20260331_mlp_v1/summary.json`
- `Experiment/analysis/remote_artifacts/qwen17_self_patchonly_semantic_mlp_excl0_len1024_50_20260331_mlp_v1/summary.json`
- `Experiment/analysis/remote_artifacts/qwen17_self_patch_semantic_mlp_all28_len1024_seed42/meta.json`
- `Experiment/analysis/remote_artifacts/qwen17_self_patch_semantic_mlp_all28_len1024_seed42/metrics.json`
- `Experiment/analysis/remote_artifacts/qwen17_self_patchonly_directkv_residual_mlp_all28_len1024_50_20260331_resid_v2/summary.json`
- `Experiment/analysis/remote_artifacts/qwen17_self_patchonly_directkv_residual_mlp_excl0_len1024_50_20260331_resid_v2/summary.json`
- `Experiment/analysis/remote_artifacts/qwen17_self_patchonly_directkv_residual_mlp_layer0_len1024_50_20260331_resid_v2/summary.json`
- `Experiment/analysis/remote_artifacts/qwen17_self_patch_directkv_residual_mlp_all28_len1024_seed42/meta.json`
- `Experiment/analysis/remote_artifacts/qwen17_self_patch_directkv_residual_mlp_all28_len1024_seed42/metrics.json`
- `Experiment/analysis/remote_artifacts/qwen17_self_patchonly_xnorm_nativekv_residual_mlp_all28_len1024_50_20260331_xnorm_nativekv_v1/summary.json`
- `Experiment/analysis/remote_artifacts/qwen17_self_patchonly_xnorm_nativekv_residual_mlp_excl0_len1024_50_20260331_xnorm_nativekv_v1/summary.json`
- `Experiment/analysis/remote_artifacts/qwen17_self_patchonly_xnorm_nativekv_residual_mlp_layer0_len1024_50_20260331_xnorm_nativekv_v1/summary.json`
- `Experiment/analysis/remote_artifacts/qwen17_self_patch_xnorm_nativekv_residual_mlp_all28_len1024_seed42/meta.json`
- `Experiment/analysis/remote_artifacts/qwen17_self_patch_xnorm_nativekv_residual_mlp_all28_len1024_seed42/metrics.json`
- `Experiment/analysis/remote_artifacts/qwen17_self_patchonly_xnorm_nativekv_residual_mlp_all28_len1024_50_20260331_xnorm_nativekv_skipid_v1/summary.json`
- `Experiment/analysis/remote_artifacts/qwen17_self_patchonly_xnorm_nativekv_residual_mlp_excl0_len1024_50_20260331_xnorm_nativekv_skipid_v1/summary.json`
- `Experiment/analysis/remote_artifacts/qwen17_self_patchonly_xnorm_nativekv_residual_mlp_layer0_len1024_50_20260331_xnorm_nativekv_skipid_v1/summary.json`
- `Experiment/analysis/remote_artifacts/qwen17_self_patchonly_xnorm_nativekv_residual_mlp_layer0_len1024_50_20260331_xnorm_nativekv_breakdown_v1/summary.json`
- `Experiment/analysis/remote_artifacts/qwen17_self_patchonly_xnorm_nativekv_residual_mlp_layer0_len1024_50_20260331_xnorm_nativekv_prewarm_v1/summary.json`
- `Experiment/analysis/remote_artifacts/qwen17_self_patchonly_xnorm_nativekv_residual_mlp_all28_len1024_50_20260331_xnorm_nativekv_prewarm_v1/summary.json`
