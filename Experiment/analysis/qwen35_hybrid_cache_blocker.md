# Qwen3.5 Hybrid Cache Blocker

## Scope

This note records the first real execution blocker encountered while trying to run the original SCD baseline on the strict pair:

- producer: `/disk_n/c101-3-1/models/Qwen3.5-4B-Base`
- consumer: `/disk_n/c101-3-1/models/Qwen3.5-4B`

The objective of the smoke run was narrow: verify whether the existing SCD reproduction repo on `ser` can faithfully capture pair traces for this pair before full CMRC runs.

## What Was Verified

### 1. The strict pair is real and loadable

On `ser`, both checkpoints load through `AutoModelForCausalLM` as:

- class: `Qwen3_5ForCausalLM`
- inner text model: `Qwen3_5TextModel`

Both checkpoints expose 32 decoder layers.

### 2. The architecture is hybrid, not plain KV-only attention

From the model `text_config.layer_types`, the 32 layers follow this repeating pattern:

- layers `0,1,2`: `linear_attention`
- layer `3`: `full_attention`

This repeats across the stack, so the model has:

- 24 `linear_attention` layers
- 8 `full_attention` layers

### 3. The returned cache object is not legacy per-layer KV tuples

Under `transformers 5.3.0`, a forward pass with `use_cache=True` returns:

- cache type: `transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5DynamicCache`

This object has:

- `key_cache`
- `value_cache`
- `conv_states`
- `recurrent_states`

and does not expose:

- `.layers`
- `.to_legacy_cache()`

The upstream class docstring explicitly states that:

- `full_attention` layers use `key_cache` and `value_cache`
- `linear_attention` layers use convolution and recurrent states instead

## Observed Failures

### Failure A: original capture path

The unmodified repo failed in `src/scd_repro/adapters.py` when it tried to iterate directly over `out.past_key_values`:

```text
TypeError: 'Qwen3_5DynamicCache' object is not iterable
```

### Failure B: compatibility wrapper

A temporary wrapper was then used to convert the cache through the repo's own strict-runtime helper. That also failed because the helper only recognizes:

- cache objects with `.layers`
- tuple/list legacy caches

The resulting error was:

```text
TypeError: 不支持的 cache 类型: <class 'transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5DynamicCache'>
```

## Why This Matters For The Research Question

The current remote SCD reproduction repo assumes that every layer can be represented as standard `(K, V)` cache tensors. That assumption holds for the existing Qwen3-1.7B style runs, but it does not hold for Qwen3.5:

- only 8 of 32 layers expose ordinary attention KV
- the other 24 layers carry non-KV state (`conv_states`, `recurrent_states`)

Therefore, a "strict original SCD baseline on Qwen3.5" is currently blocked at the state-semantics level, not merely at the checkpoint-availability level.

## Practical Interpretation

At this point, three different branches must be distinguished:

### Branch 1. Faithful original SCD

Not currently runnable on Qwen3.5 with the existing repo, because the repo is written for KV-only layer state.

### Branch 2. Qwen3.5-specific state-transfer extension

Potentially runnable, but it would no longer be a faithful "original SCD" baseline because it would need to decide how to capture, serialize, transfer, and reconstruct:

- full-attention KV states
- linear-attention convolution states
- linear-attention recurrent states

### Branch 3. Same-family fallback for the original baseline

If the paper story requires a faithful original-SCD baseline before route A/B/C, the cleaner fallback is to move the strict-original baseline to a nearest full-KV Qwen-family pair and keep Qwen3.5 as a separate hybrid-state follow-up branch.

## Immediate Conclusion

The strict Qwen3.5 pair is no longer blocked by missing assets. It is blocked by a method-interface mismatch between:

- the hybrid cache semantics of Qwen3.5
- the KV-only assumptions in the current SCD reproduction pipeline

That distinction matters for the ICML narrative because a failed Qwen3.5 run under the current repo cannot be interpreted as evidence against the SCD claim unless the hybrid-state mismatch is handled explicitly.
