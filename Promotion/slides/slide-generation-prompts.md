# Slide Generation Prompts

Use `deck-style.md` as the shared style preamble for all generate-from-scratch slides.

## Slide 1

Title: Revisiting SCD on Similar Mistral Pairs

Prompt:
- Create an academic title slide in 16:9 format.
- Use a clean white background with a blue accent footer.
- Main title: `Revisiting SCD on Similar Mistral Pairs`
- Subtitle: `What survives after a strict no-fallback reimplementation?`
- Small supporting line near the bottom:
  - `Use only the validated strict-stop forward artifact`
- Add four small tags:
  - `baseline`
  - `reuse-only`
  - `patch-only`
  - `hybrid`
- Style should feel like an ICML rebuttal or lab update deck.

## Slide 2

Title: Problem Setup

Prompt:
- Create a clean academic content slide.
- Title at top: `Problem Setup`
- Show a two-stage pipeline:
  - Offline: `capture_pair -> train_reuse -> train_patch -> search`
  - Online: `encode -> transfer -> rebuild/patch -> decode_first`
- Add three metric callouts on the right:
  - `F1 / EM`
  - `TTFT / total latency`
  - `payload / latency breakdown`
- Keep the slide minimal, diagram-focused, and easy to read from a distance.

## Slide 3

Title: What SCD Claims

Prompt:
- Create a clean white-background slide with a centered claim table.
- Title: `What SCD Claims`
- Show two benchmark anchor rows:
  - `Qwen-family anchor: TTFT ratio ~0.484, quality retention ~0.967`
  - `MistralLite -> Mistral-7B at 200 Gbps: TTFT ratio ~0.377, quality retention ~0.969`
- Add one sentence beneath the table:
  - `The paper-level target is simultaneous speedup and near-oracle quality, not speed alone.`
- Visual style should be restrained and evidence-first.

## Slide 4

Title: Why Strict Reproduction Matters

Prompt:
- Create a split-panel academic slide.
- Left panel label: `Loose evaluation`
- Right panel label: `Strict evaluation`
- Add three red warning callouts between the panels:
  - `stop-rule bug`
  - `fallback leakage`
  - `misaligned TTFT boundary`
- Bottom takeaway sentence:
  - `If these are not fixed, apparent gains can be methodologically invalid.`

## Slide 5

Title: Fixed Evaluation Contract

Prompt:
- Create a slide with four horizontal method cards.
- Cards:
  - `baseline`
  - `reuse-only`
  - `patch-only`
  - `hybrid`
- Under each card, include a one-line definition:
  - baseline: `consumer full prefill with aligned first decode`
  - reuse-only: `layerwise reuse, actual transfer timing, no fallback`
  - patch-only: `all-layer MLP patch, no reuse scaffold, no fallback, no recompute`
  - hybrid: `patch anchor + greedy reuse search with strict F1 gate`
- Add a small note at the bottom:
  - `All later numbers use this contract unless explicitly marked otherwise.`

## Slide 6

Title: Forward Strict Result

Prompt:
- Create an academic results slide with a chart and a compact table.
- Title: `Forward Strict Result`
- Subtitle: `MistralLite -> Mistral-7B`
- Include these four rows in a small table:
  - `baseline: F1 0.5969, TTFT 103.62 ms`
  - `reuse-only: F1 0.0141, TTFT 33.52 ms`
  - `patch-only: F1 0.6285, TTFT 53.93 ms`
  - `hybrid: F1 0.6072, TTFT 46.18 ms`
- Add one small callout:
  - `hybrid stays within the 5% relative F1-drop budget versus patch-only`
- Highlight the hybrid point in blue and patch-only in dark slate.

## Slide 7

Title: Hybrid Route Anatomy

Prompt:
- Create a route summary slide.
- Title: `Hybrid Route Anatomy`
- Include these key facts:
  - `reuse 14`
  - `patch 18`
  - `fallback 0`
  - `recompute 0`
  - `step 14 replay pass count = 2`
  - `step 15 replay pass count = 0`
- Add a short bottom note:
  - `Strict search stops at the first replay step with zero quality-passing candidate.`

## Slide 8

Title: Why Hybrid Beats Patch-Only on TTFT

Prompt:
- Create a decomposition slide with a compact latency table.
- Title: `Why Hybrid Beats Patch-Only on TTFT`
- Rows:
  - `encode: 0.81 vs 0.76`
  - `transfer: 12.70 vs 12.22`
  - `rebuild: 0.00 vs 1.47`
  - `patch_infer: 18.48 vs 10.19`
  - `decode_first: 21.95 vs 21.54`
  - `total: 53.93 vs 46.18`
- Add a small explanatory sentence:
  - `Hybrid trades a small rebuild cost for a larger reduction in patch inference.`

## Slide 9

Title: What We Exclude From the Final Claim

Prompt:
- Create an academic caution slide.
- Title: `What We Exclude From the Final Claim`
- Subtitle: `Mistral-7B -> MistralLite`
- Left side: a faded old pipeline marked with a red X and the label `old fallback-based reverse route`
- Right side: a neutral box labeled `excluded until same-contract strict rerun is revalidated`
- Bottom note:
  - `Do not mix the old fallback route into the final strict claim.`

## Slide 10

Title: Takeaways

Prompt:
- Create a final summary slide with a checklist layout.
- Title: `Takeaways`
- Checklist items:
  - `validated: forward strict positive result`
  - `fragile: reuse-only quality`
  - `required: strict accounting and no-fallback evaluation`
- Add a closing sentence:
  - `SCD-style gains survive only in a narrower and more controlled form than the original broad claim suggests.`
