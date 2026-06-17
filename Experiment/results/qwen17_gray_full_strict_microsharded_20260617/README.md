# Qwen3-1.7B Gray Full Strict Micro-Sharded Evaluation

This directory records an additional strict reuse+patch evaluation for
Qwen3-1.7B -> Qwen3-1.7B-gray. It is not release-facing speedup evidence for
the MistralLite -> Mistral-7B vLLM path.

## Setup

- Run root:
  `/disk_n/c101-3-mql/projects/SCD-ICML/runs/extra_model_qwen17_gray_full_strict_20260617_223703`
- Train/calibration split: CMRC2018 train samples.
- Evaluation splits: CMRC2018 dev, HotpotQA dev, WikiText-2 test.
- Strict route: `route_qwen17_gray_patch6_seed42.json`.
- Patch layers: `[0, 22, 24, 25, 26, 27]`.
- Reuse layers:
  `[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23]`.
- No fallback layers and no recompute layers.

## Results

| Dataset | Method | Samples | F1 | EM | NLL | PPL | TTFT ms | Total ms | Total speedup | TTFT speedup |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CMRC2018 | Full Consumer | 3219 | 0.748358 | 0.440509 | 0.386410 | 1.471688 | 61.368 | 787.831 | - | - |
| CMRC2018 | SCD reuse+patch | 3219 | 0.782770 | 0.504815 | 0.406739 | 1.501911 | 103.707 | 829.236 | 0.950069 | 0.591745 |
| HotpotQA | Full Consumer | 7405 | 0.033071 | 0.007292 | 6.166776 | 476.646757 | 71.081 | 791.815 | - | - |
| HotpotQA | SCD reuse+patch | 7405 | 0.038549 | 0.010533 | 6.784828 | 884.328117 | 265.460 | 987.491 | 0.801846 | 0.267766 |
| WikiText-2 | Full Consumer | 763 | 0.245261 | 0.000000 | 3.185116 | 24.170100 | 52.547 | 765.672 | - | - |
| WikiText-2 | SCD reuse+patch | 763 | 0.236718 | 0.000000 | 3.279555 | 26.563944 | 46.799 | 760.346 | 1.007005 | 1.122838 |

## Interpretation

This run is a strictness and risk check for an additional model pair. It shows
that no-fallback reuse+patch can preserve or improve QA F1/EM on CMRC2018 and
HotpotQA, but it does not reproduce the desired speedup on those QA datasets.
WikiText-2 has only a small speed win with a small quality drop. Treat this as
negative or cautionary evidence for Qwen3-1.7B-gray route portability, not as a
paper or release speed claim.

