# SCD Paper Claim-to-Evidence Audit - 2026-06-10

## Scope

This note maps paper-facing SCD claims to the current verified Mistral evidence.
Qwen-32B remains out of scope for this pass.

Authoritative current Mistral evidence:

- Runtime repo: `Experiment/core_code/runtime_repo`.
- Selected route:
  `Experiment/core_code/runtime_repo/configs/routes/scd_icml_mistrallite_to_mistral7_reuse19_patch13_cmrc200.json`.
- Verified CMRC run root:
  `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_selected_reuse19_5method_aggregate_20260610_034426`.
- Valid full-context HotpotQA run root:
  `/disk_n/c101-3-mql/projects/SCD-ICML/runs/hotpotqa_fullcontext_full_scd_20260610_115952`.
- Valid HotpotQA full-bank heldout run root:
  `/disk_n/c101-3-mql/projects/SCD-ICML/runs/hotpotqa_full_banks_20260610_192156`.
- Old compressed-prompt HotpotQA snapshot:
  `/disk_n/c101-3-mql/projects/SCD-ICML/kv/snapshots/scd-github-f7d7854/runs/strict_hotpotqa_full7405_baseline_patchonly_20260423`.
  Its Patch-only full7405 result is F1 `0.351659`, EM `0.252127`; after
  removing the 100 training questions by question-text overlap, the remaining
  7305 samples still give Patch-only F1 `0.351443`, EM `0.251608`. Treat it as
  a separate compressed-prompt benchmark, not as full-context HotpotQA evidence.

## Current Safe Claim Boundary

The current safe positive claim is narrow:

> On CMRC2018 dev `n=200`, MistralLite -> Mistral-7B with the selected
> Reuse19/Patch13 route preserves F1/EM relative to Full Consumer while reducing
> TTFT. HotpotQA full-context and WikiText-2 should be reported as limitations
> or diagnostics, not as preserved-quality evidence.

Current CMRC numbers:

| Method | F1 | EM | Avg NLL | PPL | TTFT ms |
|---|---:|---:|---:|---:|---:|
| Full Consumer | 0.657307 | 0.365 | 0.569284 | 1.767002 | 100.604 |
| SCD Reuse19/Patch13 | 0.642751 | 0.350 | 0.934572 | 2.546122 | 41.598 |

Retention:

- F1 retention: `0.977855`.
- EM retention: `0.958904`.

## Archived ICML Camera-Ready TeX

Active archived source:

`archive/icml-completed-2026-05-24/camera-ready-and-final/final/paper/icml2026-camera-ready-source/example_paper.tex`

The archived TeX still contains stronger historical claims than the current
release evidence supports.

| TeX location | Claim | Current status |
|---|---|---|
| lines around `603-625`, Table `main_results_both` | Mistral F1 `0.807 -> 0.769`, Qwen-32B F1 `0.876 -> 0.847`, with about `2x` speedup | Not current release evidence. Qwen-32B is unverified in this scope. Mistral current verified CMRC evidence is `0.657307 -> 0.642751`, not this table. |
| lines around `655-680`, Table `quality_main` | Mistral SCD F1 `0.7850`, PPL `4.65`, TTFT `125.2`, speedup `2.65x` | Not reproduced by the current selected route. Current CMRC is positive but lower absolute F1; current WikiText-2 PPL worsens from Full `7.5840` to SCD `26.4231`. |
| lines around `682-686` | SCD recovers nearly full Oracle performance and the advantage holds for Qwen-32B | Too broad for current evidence. Current supported statement is CMRC-only for Mistral; Qwen-32B is deferred. |
| lines around `711-733`, Patch-budget WikiText-2 table | Patch budget monotonically improves WikiText-2 PPL toward Oracle | Historical/unsupported for the current selected Reuse19/Patch13 path. Current WikiText-2 is an auxiliary risk signal, not preserved quality. |
| lines around `765-799`, WikiText-2 latency breakdown | SCD achieves `2.19x` speedup with quality jump from Patch | Not enough current evidence to use as a release-facing positive quality claim, because current WikiText-2 quality is poor under the selected route. |
| lines around `1091-1109`, neighboring baselines | SCD quality retention `95--97%`, TTFT speedup `1.98--2.65x` | Only the CMRC selected route currently supports roughly this F1/EM retention. Do not generalize to HotpotQA, WikiText-2, or Qwen-32B. |

Conclusion for the archived ICML TeX: do not use it as the current
open-source claim boundary without revision. It is a historical camera-ready
artifact, not the current reproducibility statement.

## Current Publication Drafts

Files under `Publication/paper/` are a separate follow-up direction:

- `Publication/paper/claim-evidence-register.md`.
- `Publication/paper/results-and-limitations-draft.md`.
- `Publication/paper/introduction-and-discussion-draft.md`.

These drafts describe a Qwen2.5-1.5B -> Qwen2.5-7B long-context boundary result
with no positive speedup claim. They are outside the current Mistral release
scope and should not be mixed with the SCD-ICML Mistral claim.

## Release Implication

Before open-source upload or paper-facing redistribution:

1. Use `Experiment/core_code/runtime_repo/README.md` and
   `REPRODUCING_SCD_ICML.md` as the current release-facing claim boundary.
2. Do not advertise the archived TeX's broad Mistral/Qwen tables as current
   reproducibility evidence.
3. If a paper revision is needed, rewrite the result section around the
   verified CMRC Reuse19/Patch13 result and explicitly label HotpotQA/WikiText-2
   as limitations/diagnostics.
4. Keep Qwen-32B marked as deferred until a machine-verified run exists.

Prepared revision artifacts:

- `Publication/paper/scd-icml-current-results-revision-2026-06-10.tex` contains
  a TeX-ready current-results fragment.
- `docs/scd-icml-paper-result-revision-map-2026-06-10.md` maps old archived
  TeX tables/paragraphs to the current safe replacements.
