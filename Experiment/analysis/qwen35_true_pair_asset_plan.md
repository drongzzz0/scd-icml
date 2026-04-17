# Qwen3.5 True-Pair Asset Plan

## Purpose

This note resolves the strict-Qwen3.5 branch of Task 4: identify the cleanest Qwen3.5-family similar pair that can support the original-SCD baseline, record what is already runnable on `ser`, and isolate the remaining blocker.

## Verified Assets on `ser`

Observed local Qwen3.5-family assets:

- post-trained checkpoint: `/disk_n/c101-3-1/models/Qwen3.5-4B`
- prompt-tuning adapter: `/disk_n/c101-3-1/phase01_runs/b2_peft_smoke/adapter`
- prompt-tuning adapter: `/disk_n/c101-3-1/phase01_runs/b2_peft_full/adapter`

Observed runtime facts in `/disk_n/c101-3-1/venvs/A800`:

- `AutoConfig.from_pretrained("/disk_n/c101-3-1/models/Qwen3.5-4B")` resolves to `Qwen3_5Config`
- `AutoModelForCausalLM.from_pretrained("/disk_n/c101-3-1/models/Qwen3.5-4B", local_files_only=True)` loads successfully and resolves to `Qwen3_5ForCausalLM`
- `PeftModel.from_pretrained(base_model, "/disk_n/c101-3-1/phase01_runs/b2_peft_smoke/adapter")` loads successfully and resolves to `PeftModelForCausalLM`

Implication:

- the current `ser` environment can execute text-only causal-LM loading for `Qwen3.5-4B`
- the immediate blocker is no longer model-class support
- the remaining blocker is obtaining a second same-architecture Qwen3.5-family checkpoint for the main similar-pair experiment

## Primary Pair Choice

Primary ICML-aligned pair:

- producer: `Qwen/Qwen3.5-4B-Base`
- consumer: local `/disk_n/c101-3-1/models/Qwen3.5-4B`

Why this is the cleanest pair:

- the local `Qwen3.5-4B` model card identifies `Qwen/Qwen3.5-4B-Base` as its `base_model`
- this gives a same-size, same-family, near-isomorphic pair where the difference is post-training rather than architecture
- this pair is much cleaner for the paper story than mixing `Qwen3` and `Qwen3.5`, or comparing different parameter scales

Inference note:

- the claim that `Qwen/Qwen3.5-4B-Base` shares the same text architecture as the local `Qwen3.5-4B` is an inference from the local model card's `base_model` field and the successful local causal-LM loading path, not from a locally downloaded base-model `config.json`

## Fallback Pair Choice

Fallback near-identity pairs already available on `ser`:

- `Qwen3.5-4B -> qwen35_4b_prompt_smoke`
- `Qwen3.5-4B -> qwen35_4b_prompt_full`

Why these are useful:

- they require no external download
- they are valid same-family, shared-tokenizer, same-architecture pairs
- they can serve as a high-similarity sanity check for whether the original SCD route can ever work on an almost identical Qwen3.5 pair

Why they are not the preferred main claim pair:

- these adapters are prompt-tuning variants, not independently updated full checkpoints
- they are best treated as a fallback diagnostic or upper-bound-style near-pair condition, not the main ICML headline evidence

## Acquisition Blocker

What was observed on `ser`:

- `huggingface.co` DNS resolves
- direct HTTPS access from `ser` to Hugging Face did not complete within a short timeout window

Conservative interpretation:

- outbound Hugging Face access from `ser` is likely blocked, filtered, or too unreliable for direct snapshot download

Practical consequence:

- the main-pair acquisition path should assume download on another machine with stable HF access, then copy to `ser`

## Recommended Acquisition Path

Target location on `ser`:

- `/disk_n/c101-3-1/models/Qwen3.5-4B-Base`

Suggested workflow:

1. Download `Qwen/Qwen3.5-4B-Base` on a machine with reliable Hugging Face access.
2. Copy the snapshot into `/disk_n/c101-3-1/models/Qwen3.5-4B-Base`.
3. Fill the local inventory template at `Experiment/core_code/configs/qwen35_4b_true_pair_inventory_template.json`.
4. Capture pair metrics for `qwen35_4b_base__to__qwen35_4b_post`.
5. Run the original-SCD baseline on that pair before spending compute on route A/B/C.

## Immediate Execution Order

1. Treat `Qwen/Qwen3.5-4B-Base -> Qwen3.5-4B` as the main strict-Qwen3.5 pair.
2. If the base-model acquisition is blocked for more than one iteration, run the prompt-tuning fallback pair as a diagnostic Qwen3.5 sanity check instead of idling.
3. Only after one of those Qwen3.5 pairs is evaluated should route A/B/C consume additional A800 time.

## External References

- Hugging Face model page: `https://huggingface.co/Qwen/Qwen3.5-4B`
- Hugging Face base model page: `https://huggingface.co/Qwen/Qwen3.5-4B-Base`
