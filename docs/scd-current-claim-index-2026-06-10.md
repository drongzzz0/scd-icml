# SCD-ICML Current Claim Index - 2026-06-10

## Purpose

This index identifies which files are safe to use as the current paper and
open-source claim boundary. It also marks historical artifacts that should not
be cited as current reproducibility evidence.

## Current Source Of Truth

- Runtime release docs:
  - `Experiment/core_code/runtime_repo/README.md`.
  - `Experiment/core_code/runtime_repo/REPRODUCING_SCD_ICML.md`.
  - `Experiment/core_code/runtime_repo/OPEN_SOURCE_RELEASE_MANIFEST.md`.
- Current audits:
  - `docs/scd-main-result-reproducibility-audit-2026-06-10.md`.
  - `docs/scd-paper-claim-to-evidence-audit-2026-06-10.md`.
  - `docs/scd-icml-paper-result-revision-map-2026-06-10.md`.
- Current paper replacement fragment:
  - `Publication/paper/scd-icml-current-results-revision-2026-06-10.tex`.

## Current Positive Claim

The current supported positive claim is limited to the MistralLite ->
Mistral-7B route selected on CMRC2018 dev `n=200`.

Evidence root:

`/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_selected_reuse19_5method_aggregate_20260610_034426`

Evidence file:

`/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_selected_reuse19_5method_aggregate_20260610_034426/scd_icml_mistral_summary.json`

Selected route:

`Experiment/core_code/runtime_repo/configs/routes/scd_icml_mistrallite_to_mistral7_reuse19_patch13_cmrc200.json`

Claim-facing values:

| Method | F1 | EM | Avg NLL | PPL | TTFT ms |
|---|---:|---:|---:|---:|---:|
| Full Consumer | 0.657307 | 0.365 | 0.569284 | 1.767002 | 100.604 |
| SCD Reuse19/Patch13 | 0.642751 | 0.350 | 0.934572 | 2.546122 | 41.598 |

F1 retention is `0.977855`; EM retention is `0.958904`.

## Current Limitations And Diagnostics

HotpotQA full-context is not a preserved-quality result for the selected route:

`/disk_n/c101-3-mql/projects/SCD-ICML/runs/hotpotqa_fullcontext_full_scd_20260610_115952`

Evidence file:

`/disk_n/c101-3-mql/projects/SCD-ICML/runs/hotpotqa_fullcontext_full_scd_20260610_115952/scd_icml_mistral_summary.json`

| Method | F1 | EM | Avg NLL | TTFT ms |
|---|---:|---:|---:|---:|
| Full Consumer | 0.400453 | 0.265 | 0.912496 | 216.153 |
| SCD Reuse19/Patch13 | 0.138272 | 0.105 | 1.423038 | 67.060 |

HotpotQA-specific full-bank composition also failed to recover quality on a
disjoint heldout split:

`/disk_n/c101-3-mql/projects/SCD-ICML/runs/hotpotqa_full_banks_20260610_192156`

Evidence file:

`/disk_n/c101-3-mql/projects/SCD-ICML/runs/hotpotqa_full_banks_20260610_192156/summary.json`

WikiText-2 is an auxiliary NLL/PPL risk signal, not a preserved-quality claim:

| Method | Avg NLL | PPL | TTFT ms |
|---|---:|---:|---:|
| Full Consumer | 2.026045 | 7.584029 | 62.368 |
| SCD Reuse19/Patch13 | 3.274239 | 26.423097 | 31.004 |

## Historical But Useful Compressed HotpotQA Result

The old GitHub snapshot has a real Patch-only result under a different
compressed-prompt HotpotQA setting:

`/disk_n/c101-3-mql/projects/SCD-ICML/kv/snapshots/scd-github-f7d7854/runs/strict_hotpotqa_full7405_baseline_patchonly_20260423`

Evidence files:

`/disk_n/c101-3-mql/projects/SCD-ICML/kv/snapshots/scd-github-f7d7854/runs/strict_hotpotqa_full7405_baseline_patchonly_20260423/reports/hotpotqa/strict_baseline_only_gpu1/summary.json`

`/disk_n/c101-3-mql/projects/SCD-ICML/kv/snapshots/scd-github-f7d7854/runs/strict_hotpotqa_full7405_baseline_patchonly_20260423/reports/hotpotqa/strict_patch_only_gpu0/summary.json`

Full7405 values:

| Method | F1 | EM | Avg NLL | TTFT ms |
|---|---:|---:|---:|---:|
| Full Consumer | 0.197151 | 0.098987 | 2.296656 | 184.539 |
| Patch-only | 0.351659 | 0.252127 | 2.450745 | 114.330 |

The old full7405 eval includes the 100 questions used to train the old Patch
checkpoint. A question-text overlap audit shows that removing those 100
questions leaves the remaining 7305 samples at Patch-only F1 `0.351443`, EM
`0.251608`. This result is therefore not mainly explained by that small
overlap, but it remains a separate compressed-prompt benchmark because the old
loader applies question-aware evidence compression and `max_length=512`.

## Superseded Or Historical Artifacts

Do not use these as current claim boundaries:

- `docs/scd-icml-mistral-reproduction-2026-06-07.md`: old reproduction
  snapshot with stale HotpotQA and WikiText-2 interpretation.
- `docs/scd-icml-paper-release-audit-2026-06-08.md`: superseded `k=6` audit.
- `docs/scd-icml-open-source-release-status-2026-06-08.md`: superseded by the
  selected Reuse19/Patch13 release path.
- `archive/icml-completed-2026-05-24/camera-ready-and-final/final/paper/`:
  historical camera-ready source and PDF. The source still contains old broad
  Mistral/Qwen/WikiText claims and should not be redistributed as the current
  reproducibility statement without revision.

## Release Verification Commands

Local source:

```bash
cd Experiment/core_code/runtime_repo
python -m pip install -e .
python -m unittest discover -s tests
python scripts/check_release_readiness.py --root .
```

Local release preview:

```bash
cd Experiment\core_code\runtime_repo
python scripts\build_release_tree.py --force
cd release_preview\scd_icml_runtime_release
python -m pip install -e .
python -m unittest discover -s tests
python scripts\check_release_readiness.py --root .
```

Remote source and preview:

```bash
cd /disk_n/c101-3-mql/projects/SCD-ICML/repro_current_from_codex_20260607-154030/runtime_repo
python scripts/build_release_tree.py --force
python scripts/check_release_readiness.py --root .
python scripts/check_release_readiness.py --root release_preview/scd_icml_runtime_release
```
