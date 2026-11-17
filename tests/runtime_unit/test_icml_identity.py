"""Restore the frozen reference's exact-zero residual fast path without hints."""
import dataclasses
from pathlib import Path

import pytest
import torch

from scd_repro.direct_kv_residual_patch import XNormNativeKVSharedResidualPatch
from scd_runtime.contracts.types import MethodAssets, SourceState
from scd_runtime.errors import AssetMismatch, MissingAsset
from scd_runtime.methods.common import rope_tables, rotate
from scd_runtime.methods.icml import backend as B
from scd_runtime.methods.icml.shared_patch import shared_hidden


def setup(profile, prefix, *, active=(), version='icml-pinned-v2-batched', inference=False):
    torch.manual_seed(87)
    model = XNormNativeKVSharedResidualPatch(
        num_layers=2,
        hidden_dim=4,
        trunk_dim=6,
        bottleneck_dim=3,
        num_blocks=1,
        ffn_multiplier=2.0,
    )
    with torch.no_grad():
        for layer in active:
            model.delta_h_weight[layer].normal_()
            model.delta_h_bias[layer].normal_()
    p = dataclasses.replace(
        profile,
        method_id='icml_reuse_patch',
        numeric_version=version,
        route={
            'reuse_layers': [],
            'patch_layers': [0, 1],
            'fit_mode': 'xnorm_nativekv_shared_residual_mlp_v1',
            'reuse_key_space': 'pre_rope',
        },
    )
    if inference:
        with torch.inference_mode():
            tensors = {'shared.' + k: v.clone() for k, v in model.state_dict().items()}
    else:
        tensors = {'shared.' + k: v.clone() for k, v in model.state_dict().items()}
    assets = MethodAssets(p, Path('.'), tensors, {'shared_num_blocks': 1})
    state = SourceState(
        p.source,
        prefix,
        tuple((torch.ones(1, 1, 16, 4).bfloat16(), torch.ones(1, 1, 16, 4).bfloat16()) for _ in range(2)),
        {(i, 'attention_input'): torch.randn(1, 16, 4).bfloat16() for i in range(2)},
    )

    class Projection:
        identity = p.target

        def __init__(self):
            self.layers = []

        def project(self, layer, hidden):
            self.layers.append(layer)
            return hidden.bfloat16().unsqueeze(1), hidden.bfloat16().unsqueeze(1)

    return p, state, assets, Projection()


def test_all_zero_heads_skip_MLP_but_project_every_patch_layer(profile, prefix, monkeypatch):
    p, state, assets, projection = setup(profile, prefix)
    monkeypatch.setattr(B, 'shared_hidden', lambda *a: pytest.fail('exact-zero heads must not execute the residual MLP'))
    backend = B.ICMLBackend()
    assert backend.shared_identity_layers(assets) == (0, 1)
    result = backend.convert(state, assets, projection)
    assert projection.layers == [0, 1]
    for layer, pair in enumerate(result.layers):
        hidden = state.hidden[layer, 'attention_input']
        expected_k = rotate(hidden.unsqueeze(1), rope_tables(p.target, 16, 'cpu', torch.bfloat16))
        assert torch.equal(pair[0], expected_k) and torch.equal(pair[1], hidden.unsqueeze(1))


@pytest.mark.parametrize('active', [(0,), (1,), (0, 1)])
def test_only_non_identity_layers_use_original_MLP_batch_geometry(profile, prefix, monkeypatch, active):
    p, state, assets, projection = setup(profile, prefix, active=active)
    seen = []

    def observe(x, ids, bank):
        seen.append(ids.tolist())
        return shared_hidden(x, ids, bank)

    monkeypatch.setattr(B, 'shared_hidden', observe)
    result = B.ICMLBackend().convert(state, assets, projection)
    assert seen == [list(active)] and projection.layers == [0, 1]
    inputs = torch.stack([state.hidden[i, 'attention_input'][0].float() for i in active])
    predicted = shared_hidden(inputs, torch.tensor(active), assets)
    for i in range(2):
        hidden = predicted[active.index(i):active.index(i) + 1] if i in active else state.hidden[i, 'attention_input'].float()
        k = rotate(hidden.bfloat16().unsqueeze(1), rope_tables(p.target, 16, 'cpu', torch.bfloat16))
        assert torch.equal(result.layers[i][0], k) and torch.equal(result.layers[i][1], hidden.bfloat16().unsqueeze(1))


def test_identity_cache_avoids_repeat_scans_and_invalidates_head_mutation(profile, prefix, monkeypatch):
    _, _, assets, _ = setup(profile, prefix)
    backend = B.ICMLBackend()
    original = torch.isfinite
    calls = []

    def checked(t):
        calls.append(t)
        return original(t)

    monkeypatch.setattr(torch, 'isfinite', checked)
    assert backend.shared_identity_layers(assets) == (0, 1)
    count = len(calls)
    assert backend.shared_identity_layers(assets) == (0, 1) and len(calls) == count
    assets.tensors['shared.delta_h_weight'][0].fill_(0.001)
    assert backend.shared_identity_layers(assets) == (1,) and len(calls) > count
    assets.tensors['shared.delta_h_bias'][1].fill_(0.001)
    assert backend.shared_identity_layers(assets) == ()


def test_inference_tensors_are_rechecked_when_version_tracking_is_unavailable(profile, prefix):
    _, _, assets, _ = setup(profile, prefix, inference=True)
    backend = B.ICMLBackend()
    assert backend.shared_identity_layers(assets) == (0, 1)
    with torch.inference_mode():
        assets.tensors['shared.delta_h_bias'][0].fill_(1)
    assert backend.shared_identity_layers(assets) == (1,)


@pytest.mark.parametrize('change', ['missing', 'nonfinite', 'head_shape', 'trunk_shape', 'block_shape', 'replace_cached'])
def test_zero_heads_never_bypass_asset_validation(profile, prefix, change):
    _, _, assets, _ = setup(profile, prefix)
    backend = B.ICMLBackend()
    assert backend.shared_identity_layers(assets) == (0, 1)
    if change == 'missing':
        assets.tensors.pop('shared.input_proj.weight')
    elif change == 'nonfinite':
        assets.tensors['shared.blocks.0.up_proj.weight'][0, 0] = float('nan')
    elif change == 'head_shape':
        assets.tensors['shared.delta_h_weight'] = torch.zeros(1, 3, 4)
    elif change == 'trunk_shape':
        assets.tensors['shared.input_proj.bias'] = torch.zeros(7)
    elif change == 'block_shape':
        assets.tensors['shared.blocks.0.down_proj.weight'] = torch.zeros(7, 12)
    else:
        assets.tensors['shared.delta_h_bias'] = torch.ones(2, 4)
    if change == 'replace_cached':
        assert backend.shared_identity_layers(assets) == ()
    else:
        with pytest.raises((AssetMismatch, MissingAsset)):
            backend.shared_identity_layers(assets)


def test_untrusted_metadata_cannot_skip_a_nonzero_residual(profile, prefix):
    _, _, assets, _ = setup(profile, prefix, active=(0,))
    assets.metadata['runtime_layer_hints'] = {'0': {'xnorm_nativekv_shared_residual_identity': True}}
    assert B.ICMLBackend().shared_identity_layers(assets) == (1,)


def test_historical_v1_replay_keeps_its_original_computation_path(profile, prefix, monkeypatch):
    _, state, assets, projection = setup(profile, prefix, version='icml-pinned-v1')
    seen = []

    def observe(x, ids, bank):
        seen.append(ids.tolist())
        return shared_hidden(x, ids, bank)

    monkeypatch.setattr(B, 'shared_hidden', observe)
    B.ICMLBackend().convert(state, assets, projection)
    assert seen == [[0, 1]]
