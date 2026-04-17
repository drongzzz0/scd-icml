# SCD Reproduction Deck Outline

Deck type: rebuttal / group-meeting update  
Audience: reviewers or lab members familiar with LLM inference and KV-cache transfer  
Duration: about 10 minutes  
Status note: this outline uses only the currently validated strict-stop forward artifact. Reverse-direction claims are intentionally excluded until a same-contract strict rerun is revalidated.

## Slide 1

Title: Revisiting SCD on Similar Mistral Pairs  
Subtitle: What survives after a strict no-fallback reimplementation?

Core message:
- The talk asks a narrow question: can SCD's claimed TTFT-quality trade-off still hold after we remove fallback leakage, align TTFT accounting, and retest on similar Mistral-family pairs?

Suggested visual:
- Title slide with four small method tags:
  - `baseline`
  - `reuse-only`
  - `patch-only`
  - `hybrid`
- Small note that only the forward strict-stop artifact is used.

## Slide 2

Title: Problem Setup

Core message:
- Prefix-state transfer is trained offline and measured online.
- The target metric is the TTFT-quality trade-off, not TTFT alone.

Suggested content:
- Offline chain:
  - `capture_pair -> train_reuse -> train_patch -> search`
- Online chain:
  - `encode -> transfer -> rebuild/patch -> decode_first`
- Metrics:
  - `F1 / EM`
  - `TTFT / total latency`
  - `payload / latency breakdown`

## Slide 3

Title: What SCD Claims

Core message:
- The paper's bar is near-native quality together with large TTFT reduction.

Suggested content:
- Primary normalized anchors from the paper:
  - Qwen-family anchor: TTFT ratio about `0.484`, quality retention about `0.967`
  - MistralLite -> Mistral-7B anchor at 200 Gbps: TTFT ratio about `0.377`, quality retention about `0.969`

## Slide 4

Title: Why Strict Reproduction Matters

Core message:
- Loose routes can look good for the wrong reasons.

Suggested content:
- Stop-rule bug could continue after replay quality failed.
- Fallback or recompute could leak consumer-side information.
- TTFT boundaries could be misaligned across methods.

Suggested visual:
- Split panel:
  - `loose evaluation`
  - `strict evaluation`

## Slide 5

Title: Fixed Evaluation Contract

Core message:
- Every result in the deck is tied to one fixed strict protocol.

Suggested content:
- `baseline`: consumer full prefill with aligned first decode
- `reuse-only`: layerwise reuse, actual transfer timing, no fallback
- `patch-only`: all-layer MLP patch, no reuse scaffold, no fallback, no recompute
- `hybrid`: patch-only anchor plus greedy reuse search
- Search rule:
  - relative `F1 drop <= 5%` versus patch-only
  - `fallback = []`
  - `recompute = []`
  - stop immediately when replay quality pass becomes empty

## Slide 6

Title: Forward Strict Result

Subtitle: MistralLite -> Mistral-7B

Core message:
- Under the corrected strict protocol, a valid no-fallback hybrid exists in the forward direction.

Current canonical strict-stop result:
- `baseline`: `F1 0.5969`, `TTFT 103.62 ms`
- `reuse-only`: `F1 0.0141`, `TTFT 33.52 ms`
- `patch-only`: `F1 0.6285`, `TTFT 53.93 ms`
- `hybrid`: `F1 0.6072`, `TTFT 46.18 ms`

Interpretation:
- `reuse-only` is fast but unusable
- `patch-only` is the quality anchor
- `hybrid` is a real strict trade-off point

## Slide 7

Title: Hybrid Route Anatomy

Core message:
- The route is determined by the strict stop rule, not by hand-picked layers.

Suggested content:
- Reuse layer count: `14`
- Patch layer count: `18`
- Fallback layers: `0`
- Recompute layers: `0`
- Step 14 replay quality-pass count: `2`
- Step 15 replay quality-pass count: `0`

## Slide 8

Title: Why Hybrid Beats Patch-Only on TTFT

Core message:
- The gain comes from lower patch inference, not from hidden accounting changes.

Suggested content:
- TTFT decomposition table:
  - `encode`
  - `transfer`
  - `rebuild`
  - `patch_infer`
  - `decode_first`
- Emphasize:
  - small rebuild penalty
  - larger patch inference savings

## Slide 9

Title: What We Exclude From the Final Claim

Subtitle: Mistral-7B -> MistralLite

Core message:
- The old reverse-direction result is invalid for this deck because it used the older fallback-based path.
- Until a same-contract strict rerun is revalidated, reverse-direction claims must stay out of the rebuttal headline.

Suggested visual:
- Red cross on the old fallback-based reverse path
- Neutral gray box labeled `excluded until revalidated`

## Slide 10

Title: Takeaways

Core message:
- The claim survives only in a narrower and more controlled form than the original broad paper framing suggests.

Suggested content:
- validated: forward strict positive result
- fragile: reuse-only quality
- required: strict accounting and no-fallback evaluation
- rebuttal-safe conclusion:
  - strict SCD-style gains exist in the forward Mistral direction
  - unresolved reverse-direction results are excluded until revalidated

## Appendix

- Exact strict-stop numbers
- Replay F1 threshold and why step 15 was rejected
- Final reuse / patch layer lists
