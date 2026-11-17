"""Explicit v3 mirrors native HF CPU-initialized rotary; v1/v2 remain replayable."""
import dataclasses

import pytest
import torch

from scd_runtime.contracts.identity import profile_fingerprint
from scd_runtime.methods.common import rope_tables
from scd_runtime.methods.icml import backend as B
from scd_runtime.methods.icml.reuse import NATIVE_ROPE_NUMERIC_VERSION


@pytest.mark.parametrize('length', [232, 1273, 457])
def test_native_CPU_tables_match_HF_initialization_at_original_lengths(profile, length):
    from transformers import MistralConfig
    from transformers.models.mistral.modeling_mistral import MistralRotaryEmbedding

    config = MistralConfig(hidden_size=4096, num_attention_heads=32, num_key_value_heads=8, head_dim=128, rope_theta=10000.0)
    with torch.device('cpu'):
        rotary = MistralRotaryEmbedding(config)
    dummy = torch.empty((1, length, 1), dtype=torch.bfloat16)
    native = rotary(dummy, torch.arange(length).unsqueeze(0))
    identity = dataclasses.replace(profile.target, head_dim=128)
    actual = rope_tables(identity, length, 'cpu', torch.bfloat16, native_cpu_init=True)
    assert torch.equal(actual[0], native[0]) and torch.equal(actual[1], native[1])


def test_v3_has_a_new_cache_identity(profile):
    old = dataclasses.replace(profile, method_id='icml_reuse_patch', numeric_version='icml-pinned-v2-batched')
    new = dataclasses.replace(old, numeric_version=NATIVE_ROPE_NUMERIC_VERSION)
    assert profile_fingerprint(old) != profile_fingerprint(new)


def test_explicit_native_mode_is_validated(profile):
    from scd_runtime.errors import UnsupportedProfile

    with pytest.raises(UnsupportedProfile):
        rope_tables(profile.target, 16, 'cpu', torch.bfloat16, native_cpu_init='cpu')


@pytest.mark.parametrize('version,native', [('icml-pinned-v1', False), ('icml-pinned-v2-batched', False), (NATIVE_ROPE_NUMERIC_VERSION, True)])
def test_only_v3_uses_CPU_initialized_native_tables_once(profile, prefix, monkeypatch, version, native):
    from test_icml_identity import setup

    p, state, assets, projections = setup(profile, prefix, version=version)
    calls = []
    original = B.rope_tables

    def traced(*args, **kwargs):
        calls.append(kwargs.get('native_cpu_init', False))
        return original(*args, **kwargs)

    monkeypatch.setattr(B, 'rope_tables', traced)
    output = B.ICMLBackend().convert(state, assets, projections)
    assert calls == ([True] if native else [False, False])
    assert len(output.layers) == 2 and projections.layers == [0, 1]
