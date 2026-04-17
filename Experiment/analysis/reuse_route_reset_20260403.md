# Reuse Route Reset

Superseded for default-route policy by:

- `/Users/weimingming/dr-claw/scd-new/Experiment/analysis/transfer_route_lock_20260405.md`

This note remains the detailed reset record for `reuse-only`. The April 4 lock extends the same treatment to `patch-only` and archives all older patch/reuse families.

Date: `2026-04-03`

## Decision

The `reuse` technical route is reset as of April 3, 2026.

From this point on, only the repaired clean reuse path is valid for experiments, tables, and claim judgment:

- `layerwise`
- `group_size = 1`
- batched encode/decode fast path
- `transfer_ms` reported as actual measured transport time
- `rebuild_ms` kept as pure consumer-side materialization/decode compute
- `no fallback` by default unless the user explicitly requests otherwise

Everything else is deprecated for headline use:

- grouped/shared-z reuse checkpoints
- any reuse result whose main story depends on grouped compression
- older TTFT totals that mixed modeled transfer with actual staging copy
- any route comparison that silently included fallback

## Why The Reset Was Necessary

The old reuse story mixed together two different things:

1. A broken grouped/shared-z codec.
2. A healthy layerwise reuse path with avoidable runtime overhead and timing-accounting issues.

Those two should not have been treated as one route family.

The self-self diagnosis now makes the split clear:

- grouped `group_size=4` reuse collapses even on `Qwen3-1.7B -> Qwen3-1.7B`
- layerwise `group_size=1` reuse is nearly lossless on the same pair
- batching removes the old serial encode bottleneck
- actual-transfer accounting removes the old TTFT double count

So the right interpretation is not "reuse is unstable", but "the old grouped/shared-z reuse codec is invalid, while the repaired layerwise route is the only usable reuse baseline."

## Canonical Reuse Baseline

The canonical reuse baseline is the latest self-self no-fallback actual-transfer run:

- artifact:
  `/Users/weimingming/dr-claw/scd-new/Experiment/analysis/remote_artifacts/qwen17_self_allreuse_layerwise_actualtransfer_nofallback_50_20260401/summary.json`
- paired diagnosis note:
  `/Users/weimingming/dr-claw/scd-new/Experiment/analysis/qwen17_self_allreuse_reuse_route_diagnosis.md`

Headline numbers from that canonical run:

| method | F1 | EM | TTFT ms | encode ms | transfer ms | rebuild ms | decode_first ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `full_consumer` | 0.7462 | 0.38 | 85.82 | 0.00 | 0.00 | 0.00 | 29.25 |
| `reuse_only` | 0.7372 | 0.38 | 39.78 | 0.42 | 6.46 | 7.17 | 25.74 |

Interpretation:

- quality is preserved to within the small self-self measurement noise
- TTFT is materially below aligned baseline
- the repaired route does not rely on grouped compression to show speedup

## What To Use Going Forward

For all future reuse experiments on new model pairs:

- train a new layerwise reuse checkpoint for that pair
- keep `group_size = 1`
- evaluate with no fallback unless explicitly requested
- use actual transfer timing
- report `baseline` versus `reuse-only` directly

For patch experiments:

- compare `patch-only` against the repaired `reuse-only` baseline, not against deprecated grouped reuse runs

## What Is Still Historical But Useful

These archived results are still useful as failure analysis:

- grouped/shared-z reuse collapse on self-self
- timing-boundary misalignment in old `decode_first`
- TTFT double counting before transfer/rebuild cleanup

But they are no longer valid as baselines for claims, tables, or route selection.
