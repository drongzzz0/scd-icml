# Full-Dataset External Evaluation Partial Results

Date: 2026-06-18

Remote run:
`/disk_n/c101-3-mql/projects/SCD-ICML/runs/extra_datasets_full_cmrcbank_20260618_121640`

Bank:
CMRC2018-trained reuse/patch bank from
`/disk_n/c101-3-mql/projects/SCD-ICML/runs/open_build_native_fixedroute_cmrc200_20260616_133222`.

Route:
`reuse19/patch13`, `fallback=0`, `recompute=0`.

## WikiText-2 Full Test

Primary metric for WikiText-2 is PPL/NLL, not F1.

| Method | Samples | PPL | NLL | F1 | EM | TTFT ms |
|---|---:|---:|---:|---:|---:|---:|
| Full Consumer | 763 | 7.2709 | 1.9839 | 0.2213 | 0.0000 | 60.3179 |
| Reuse-only | 763 | 927.3553 | 6.8323 | 0.0893 | 0.0000 | 24.6725 |
| SCD | 763 | 23.7634 | 3.1681 | 0.1386 | 0.0000 | 30.4605 |

Interpretation:
SCD substantially improves over reuse-only on WikiText-2 PPL/NLL, but remains
well below the full-consumer baseline. This is a quality-risk signal for the
current CMRC-trained route when transferred to WikiText-2.

Status:
HotpotQA full shards and CMRC2018 dev full shards were still running when this
partial summary was written.
