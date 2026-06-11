# Speaker Notes

Deck target: 10-minute rebuttal / group-meeting update  
Evidence policy: only quote the validated strict-stop forward artifact; exclude reverse-direction claims until the strict rerun is revalidated.

## Slide 1: Revisiting SCD on Similar Mistral Pairs

Goal:
- Open with the narrow validation question, not with engineering detail.

Notes:
- We re-ran SCD under a stricter contract.
- The point is to ask whether the claimed TTFT-quality trade-off still survives once fallback leakage and accounting errors are removed.
- This deck intentionally uses only the forward strict-stop artifact that we currently trust.

## Slide 2: Problem Setup

Goal:
- Give the audience the minimum execution picture they need.

Notes:
- The offline chain learns reuse and patch modules from paired prefixes.
- The online chain is where TTFT is measured.
- Stress that TTFT is only meaningful when the quality comparison is matched and consumer-native.

## Slide 3: What SCD Claims

Goal:
- Put the target bar on screen before showing any local result.

Notes:
- The paper claims simultaneous speedup and near-native quality.
- We normalize our judgment against those anchors instead of celebrating any raw TTFT drop.

## Slide 4: Why Strict Reproduction Matters

Goal:
- Explain why some earlier numbers were discarded.

Notes:
- There were three failure modes: stop-rule continuation, fallback leakage, and mismatched TTFT boundaries.
- Any one of these can create fake wins.
- This slide is the methodological defense for why the rest of the deck is narrower than older internal runs.

## Slide 5: Fixed Evaluation Contract

Goal:
- Define the four compared routes precisely.

Notes:
- Baseline is aligned full prefill.
- Reuse-only is pure no-fallback reuse.
- Patch-only is the quality anchor.
- Hybrid is searched against patch-only under an explicit F1 gate.
- Mention that fallback and recompute are forbidden in the strict route.

## Slide 6: Forward Strict Result

Goal:
- Deliver the headline positive result without overstating it.

Notes:
- Reuse-only is unusable despite low TTFT.
- Patch-only preserves quality.
- Hybrid is faster than patch-only and remains within the allowed F1-drop budget.
- Phrase this as a valid strict trade-off point, not as universal confirmation of the paper's full claim.

## Slide 7: Hybrid Route Anatomy

Goal:
- Show that the selected route is not arbitrary.

Notes:
- The chosen route keeps 14 reuse layers and patches the remaining 18.
- Step 15 is where replay quality fully collapses, so strict search must stop there.
- The route definition is therefore tied to the stop criterion, not to hand tuning.

## Slide 8: Why Hybrid Beats Patch-Only on TTFT

Goal:
- Make the latency gain mechanically believable.

Notes:
- Hybrid introduces some rebuild cost.
- But the savings in patch inference are larger than that rebuild penalty.
- Encode, transfer, and decode-first remain close, so the TTFT gain is not coming from hidden accounting changes.

## Slide 9: What We Exclude From the Final Claim

Goal:
- Protect the rebuttal from mixing incompatible reverse-direction results.

Notes:
- The old reverse result came from the fallback-based route and should not appear in the final claim.
- Until the reverse strict rerun is verified under the same contract, we treat it as unresolved and excluded.

## Slide 10: Takeaways

Goal:
- End with the most rebuttal-safe formulation.

Notes:
- Strict accounting changes the story.
- The forward Mistral direction still has a valid strict hybrid point.
- The overall claim that survives is narrower, directional, and more controlled than the original broad paper framing.

## Appendix

Goal:
- Keep one precise backup slide ready for questions on the stop rule.

Notes:
- If asked why the route stops at 14 reuse layers, point directly to the replay F1 threshold and step-15 failure.
