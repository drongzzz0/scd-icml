"""Full-prompt source capture must slice the original forward, not rerun a prefix."""
import dataclasses

import pytest
import torch

from scd_runtime.contracts.types import FeatureRequest
from scd_runtime.errors import IdentityMismatch
from scd_runtime.providers.hf.capture import HFSourceProvider, pairs_from_cache


def tiny_model():
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(12)
    cfg = LlamaConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=128,
    )
    return LlamaForCausalLM(cfg).eval().to(dtype=torch.bfloat16)


def test_full_prompt_source_forward_is_captured_then_sliced(profile, prefix):
    model = tiny_model()
    identity = dataclasses.replace(profile.source, head_dim=8)
    prompt = prefix.token_ids + (17,)
    hidden = {}

    def capture(module, args, kwargs):
        hidden['reference'] = kwargs['hidden_states'].detach().clone()

    hook = model.model.layers[0].self_attn.register_forward_pre_hook(capture, with_kwargs=True)
    try:
        with torch.inference_mode():
            reference = model(torch.tensor([prompt]), use_cache=True)
    finally:
        hook.remove()
    observed = []
    hook = model.register_forward_pre_hook(
        lambda m, args, kwargs: observed.append(tuple(kwargs['input_ids'][0].tolist())),
        with_kwargs=True,
    )
    try:
        actual = HFSourceProvider(model, identity).capture(
            prefix,
            FeatureRequest((0, 1), ((0, 'attention_input'),)),
            full_prompt_ids=prompt,
        )
    finally:
        hook.remove()
    assert observed == [prompt] and actual.prefix == prefix
    assert torch.equal(actual.hidden[(0, 'attention_input')], hidden['reference'][:, :len(prefix.token_ids), :])
    for actual_pair, reference_pair in zip(actual.kv, pairs_from_cache(reference.past_key_values)):
        for a, b in zip(actual_pair, reference_pair):
            assert a.shape[-2] == len(prefix.token_ids) and a.is_contiguous()
            assert torch.equal(a, b[..., :len(prefix.token_ids), :])


@pytest.mark.parametrize('bad', ['changed_prefix', 'missing_final', 'two_final', 'out_of_vocab'])
def test_invalid_full_prompt_is_rejected_before_forward(profile, prefix, bad, monkeypatch):
    model = tiny_model()
    identity = dataclasses.replace(profile.source, head_dim=8)
    prompt = list(prefix.token_ids) + [17]
    if bad == 'changed_prefix':
        prompt[0] = 2
    elif bad == 'missing_final':
        prompt.pop()
    elif bad == 'two_final':
        prompt.append(18)
    else:
        prompt[-1] = 64
    provider = HFSourceProvider(model, identity)
    monkeypatch.setattr(model, 'forward', lambda *a, **k: pytest.fail('invalid original input must not execute'))
    with pytest.raises(IdentityMismatch):
        provider.capture(prefix, FeatureRequest((0, 1)), full_prompt_ids=prompt)


def test_full_capture_releases_hooks_even_when_forward_raises(profile, prefix, monkeypatch):
    model = tiny_model()
    identity = dataclasses.replace(profile.source, head_dim=8)
    before = len(model.model.layers[0].self_attn._forward_pre_hooks)

    def fail(*a, **k):
        raise RuntimeError('test forward failure')

    monkeypatch.setattr(model, 'forward', fail)
    with pytest.raises(RuntimeError, match='test forward failure'):
        HFSourceProvider(model, identity).capture(
            prefix,
            FeatureRequest((0, 1), ((0, 'attention_input'),)),
            full_prompt_ids=prefix.token_ids + (17,),
        )
    assert len(model.model.layers[0].self_attn._forward_pre_hooks) == before
