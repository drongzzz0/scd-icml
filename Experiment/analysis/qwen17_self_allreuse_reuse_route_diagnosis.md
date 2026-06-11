# Qwen3-1.7B Self-Self All-Reuse Diagnosis

> Status reset on April 3, 2026:
> this note keeps the full debugging history, but only the fixed
> `layerwise + group_size=1 + batched encode/decode + actual transfer accounting + no fallback`
> reuse route should be used for any current comparison or paper claim.
> All older grouped/shared-z reuse results are archived failure evidence only and are no longer valid baselines.

## Setup

- Producer: `Qwen3-1.7B`
- Consumer: `Qwen3-1.7B`
- Task: CMRC2018 dev
- Samples: `50`
- Route: all 28 layers in `reuse_layers`
- Patch layers: none
- Fallback layers: none
- Recompute layers: none
- Strict runtime methods in the archived reruns below: `full_consumer`, `reuse_only`, `reuse_fallback`

## Fresh strict reruns on March 31, 2026

| Reuse checkpoint | group_size | reuse_only F1 | full_consumer F1 | reuse_only TTFT (ms) | full_consumer TTFT (ms) | payload (MB) | compression |
|---|---:|---:|---:|---:|---:|---:|---:|
| `ckpt_reuse_group4_seed42` | 4 | 0.0019 | 0.7375 | 35.67 | 66.51 | 10.33 | 4.0 |
| `ckpt_reuse_layerwise_seed42` | 1 | 0.7372 | 0.7375 | 65.48 | 43.86 | 41.32 | 1.0 |

Fresh reruns reproduced the previous conclusion. The grouped full-reuse route still collapses on self-self, while the layerwise full-reuse route is nearly lossless.

The rerun summaries are mirrored locally at:

- `Experiment/analysis/remote_artifacts/qwen17_self_allreuse_group4_rerun_50_20260331/summary.json`
- `Experiment/analysis/remote_artifacts/qwen17_self_allreuse_layerwise_rerun_50_20260331/summary.json`

## Batched layerwise fast path on April 1, 2026

To answer why self-self `reuse_only` was still slower than `full_consumer`, the healthy `group_size=1` layerwise route was rerun after replacing the per-layer serial encode/decode loop with one batched path over all 28 single-layer groups.

This rerun still used:

- all 28 layers in `reuse_layers`
- no patch
- no recompute
- no selected fallback layers
- the same `ckpt_reuse_layerwise_seed42` checkpoint
- no payload compression reduction: `payload_mb = 41.32`, `compression_ratio = 1.0`

| Layerwise reuse variant | reuse_only F1 | full_consumer F1 | reuse_only TTFT (ms) | full_consumer TTFT (ms) | encode (ms) | transfer (ms) | rebuild (ms) | decode_first (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| March 31 serial loop | 0.7372 | 0.7375 | 65.48 | 43.86 | 26.86 | 1.73 | 11.89 | 25.00 |
| April 1 batched fast path | 0.7372 | 0.7375 | 40.67 | 45.05 | 0.88 | 1.73 | 12.51 | 25.55 |

The new batched summary is mirrored locally at:

- `Experiment/analysis/remote_artifacts/qwen17_self_allreuse_layerwise_batched_50_20260401/summary.json`

## Aligned first-decode rerun on April 1, 2026

The previous batched rerun still used `align_full_consumer_first_decode = false`, so `full_consumer.decode_first_ms` was not measured on the same cache-based boundary as `reuse_only`. A follow-up aligned rerun used the same route and reuse checkpoint but forced `full_consumer` onto the cache-aligned first-token boundary.

| Layerwise reuse variant | reuse_only TTFT (ms) | reuse_only decode_first (ms) | full_consumer TTFT (ms) | full_consumer prefill (ms) | full_consumer decode_first (ms) |
|---|---:|---:|---:|---:|---:|
| April 1 batched, unaligned | 40.67 | 25.55 | 45.05 | 43.87 | 1.18 |
| April 1 batched, aligned | 34.51 | 25.26 | 87.34 | 60.83 | 26.51 |

The aligned summary is mirrored locally at:

- `Experiment/analysis/remote_artifacts/qwen17_self_allreuse_layerwise_batched_aligned_50_20260401/summary.json`

This aligned rerun confirms the measurement-boundary point directly for the self-self reuse path:

- `reuse_only.decode_first_ms` stays in the `25 ms` range.
- Once aligned, `full_consumer.decode_first_ms` also moves into the same `25 to 26 ms` range instead of the old `1.18 ms`.

## No-double-count timing fix on April 1, 2026

The aligned rerun above also exposed an accounting bug: `transfer_ms` was still a modeled bandwidth term, while `rebuild_ms` still included the real `z.to(consumer_device)` copy inside the reuse decode path. That semantically counted transport twice.

The runtime was then updated so that:

- `transfer_ms` remains the modeled network term
- `staging_copy_ms` records the real in-process `z.to(...)` copy for diagnostics only
- `rebuild_ms` now means pure consumer-side decode/materialization compute only
- `ttft_ms` no longer adds `staging_copy_ms`

With that fix, the same aligned self-self layerwise rerun produced:

| Layerwise reuse variant | reuse_only TTFT (ms) | encode (ms) | transfer (ms) | staging_copy (ms) | rebuild (ms) | reuse_only decode_first (ms) | full_consumer TTFT (ms) | full_consumer decode_first (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| April 1 batched, aligned, old accounting | 34.51 | 0.42 | 1.73 | counted inside rebuild | 7.09 | 25.26 | 87.34 | 26.51 |
| April 1 batched, aligned, no double count | 29.68 | 0.42 | 1.73 | 6.34 | 2.02 | 25.50 | 75.75 | 26.16 |

The corrected summary is mirrored locally at:

- `Experiment/analysis/remote_artifacts/qwen17_self_allreuse_layerwise_batched_aligned_nodouble_50_20260401/summary.json`

Under this corrected accounting:

- `reuse_only` and `reuse_fallback` both become faster by about `5 to 6 ms` because the duplicate transport term is removed from `ttft_ms`.
- The remaining pure rebuild compute on the healthy batched self-self route is now only about `2.02 ms`.
- The dominant reuse-side TTFT terms are now `decode_first_ms` and the modeled `transfer_ms`, not consumer-side decode compute.

## Actual-transfer default on April 1, 2026

The runtime was then changed again so that `transfer_ms` itself now reports the measured cross-device copy time rather than the modeled bandwidth term. The modeled value is still retained separately as `modeled_transfer_ms` for paper-style reference, but the default transfer field now means actual measured transport in the local runtime.

Under the user's default no-fallback policy, the latest aligned self-self all-reuse rerun used only:

- `full_consumer`
- `reuse_only`

and produced:

| Method | F1 | EM | TTFT (ms) | encode (ms) | transfer actual (ms) | transfer modeled (ms) | rebuild (ms) | decode_first (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `full_consumer` | 0.7462 | 0.38 | 85.82 | 0.00 | 0.00 | 0.00 | 0.00 | 29.25 |
| `reuse_only` | 0.7372 | 0.38 | 39.78 | 0.42 | 6.46 | 1.73 | 7.17 | 25.74 |

The latest no-fallback actual-transfer summary is mirrored locally at:

- `Experiment/analysis/remote_artifacts/qwen17_self_allreuse_layerwise_actualtransfer_nofallback_50_20260401/summary.json`

This is now the default interpretation for local runtime comparisons:

- `transfer_ms` = measured local transport time
- `modeled_transfer_ms` = paper-style bandwidth estimate
- `rebuild_ms` = pure consumer-side decode/materialization compute

## Interpretation of the latency bottleneck

- The high `reuse_only` TTFT was not caused by transfer. Transfer stayed at about `1.73 ms` in both runs.
- The main bottleneck was the old serial per-layer encode loop. After batching, `encode_ms` dropped from `26.86` to `0.88 ms`.
- `rebuild_ms` stayed around `12 ms`, and `decode_first_ms` stayed around `25 ms`, so the remaining latency is now concentrated there rather than in encode or transfer.
- With this batched path, the strict self-self layerwise `reuse_only` route is now slightly faster than same-run `full_consumer` (`40.67 ms` vs `45.05 ms`) while preserving essentially identical F1.
- This is still a no-compression result. The gain came from compute parallelization only, not from shrinking the payload.

## Internal reconstruction diagnostics

On the first five captured self-self chunks, internal grouped reconstruction errors were:

| Reuse checkpoint | group_size | k_de_rope | K rel-L2 | V rel-L2 |
|---|---:|---:|---:|---:|
| `ckpt_reuse_seed42` | 4 | true | 0.1676 | 0.6989 |
| `ckpt_reuse_group4_seed42` | 4 | false | 0.1529 | 0.6989 |
| `ckpt_reuse_group4_xkvstyle_seed42` | 4 | false | 0.0731 | 0.0009 |
| `ckpt_reuse_layerwise_seed42` | 1 | true | 0.0072 | 0.0005 |

These diagnostics isolate two grouped failure modes:

1. The older grouped checkpoints have catastrophic V reconstruction.
2. The higher-rank `xkvstyle` grouped checkpoint fixes V but still leaves K error far above the layerwise baseline.

This is consistent with the existing grouped approximation artifact on ser: a same-tensor grouped fake-SVD baseline can reconstruct the packed tensors essentially exactly at the tested ranks, so the bottleneck is not route assembly and not merely the existence of grouped packing. The bottleneck is the current fixed shared-z grouped codec.

## Interpretation

- The strict all-reuse route builder is not the problem here. For `reuse_only`, it already forces:
  - all layers into `reuse_layers`
  - no patch
  - no fallback
  - no recompute
- The failure is inside the grouped reuse representation itself.
- `group_size=1` layerwise reuse is currently the only self-self path that behaves correctly on `Qwen3-1.7B`.

## Practical decision

For current `Qwen3-1.7B` debugging and route validation, the default reuse baseline should be:

- `layerwise`
- `group_size=1`
- batched encode/decode fast path enabled

The current grouped shared-z reuse codec should be treated as an invalid baseline for claim evaluation until it is redesigned rather than lightly retuned.
