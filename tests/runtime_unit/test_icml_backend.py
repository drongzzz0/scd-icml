import ast
import dataclasses
from pathlib import Path
from types import SimpleNamespace
import torch
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


def synthetic(profile, prefix):
    from scd_runtime.contracts.types import MethodAssets, SourceState

    profile = dataclasses.replace(
        profile,
        method_id='icml_reuse_patch',
        reader=profile.reader,
        route={'reuse_layers': [0], 'patch_layers': [1], 'fit_mode': 'direct_kv_v1', 'reuse_key_space': 'pre_rope'},
        numeric_version='icml-v1',
    )
    tensors = {f'reuse.0.{name}': torch.eye(4) for name in ('enc_k', 'dec_k', 'enc_v', 'dec_v')}
    tensors['patch.1.wk'] = torch.cat((torch.eye(4), torch.zeros(1, 4)))
    tensors['patch.1.wv'] = torch.cat((2 * torch.eye(4), torch.zeros(1, 4)))
    state = SourceState(
        profile.source,
        prefix,
        tuple((torch.zeros(1, 1, 16, 4, dtype=torch.bfloat16), torch.full((1, 1, 16, 4), 3.0, dtype=torch.bfloat16)) for _ in range(2)),
        {(1, 'attention_input'): torch.ones(1, 16, 4, dtype=torch.bfloat16)},
    )
    return profile, state, MethodAssets(profile, Path('.'), tensors, {})


def test_no_target_oracle_dependency(profile, prefix):
    from scd_runtime.methods.icml.backend import ICMLBackend

    profile, state, assets = synthetic(profile, prefix)
    out = ICMLBackend().convert(state, assets, None)
    assert len(out.layers) == 2
    assert out.layers[0][1].float().unique().tolist() == [3.0]
    assert out.layers[1][1].float().unique().tolist() == [2.0]
    assert out.layers[1][0][0, 0, 0].float().tolist() == [1.0, 1.0, 1.0, 1.0]


def test_route_covers_each_target_layer_once(profile, prefix):
    from scd_runtime.errors import AssetMismatch
    from scd_runtime.methods.icml.backend import ICMLBackend

    profile, state, assets = synthetic(profile, prefix)
    bad = dataclasses.replace(profile, route={**profile.route, 'patch_layers': [0, 1]})
    with pytest.raises(AssetMismatch):
        ICMLBackend().convert(state, dataclasses.replace(assets, profile=bad), None)


def test_unknown_fit_mode_rejected(profile, prefix):
    from scd_runtime.errors import UnsupportedProfile
    from scd_runtime.methods.icml.backend import ICMLBackend

    profile, state, assets = synthetic(profile, prefix)
    bad = dataclasses.replace(profile, route={**profile.route, 'fit_mode': 'guessed'})
    with pytest.raises(UnsupportedProfile):
        ICMLBackend().convert(state, dataclasses.replace(assets, profile=bad), None)


def test_unknown_icml_numeric_version_cannot_fall_back_to_v1(profile, prefix):
    from scd_runtime.errors import UnsupportedProfile
    from scd_runtime.methods.icml.backend import ICMLBackend

    p, state, assets = synthetic(profile, prefix)
    bad = dataclasses.replace(p, numeric_version='icml-pinned-v2-bached')
    with pytest.raises(UnsupportedProfile, match='numeric version'):
        ICMLBackend().convert(state, dataclasses.replace(assets, profile=bad), None)


def test_published_shared_residual_matches_original(profile):
    from scd_repro.direct_kv_residual_patch import XNormNativeKVSharedResidualPatch
    from scd_runtime.contracts.types import MethodAssets
    from scd_runtime.methods.icml.shared_patch import shared_hidden

    torch.manual_seed(42)
    model = XNormNativeKVSharedResidualPatch(
        num_layers=2,
        hidden_dim=4,
        trunk_dim=6,
        bottleneck_dim=3,
        num_blocks=2,
        ffn_multiplier=2.0,
    )
    with torch.no_grad():
        model.delta_h_weight.normal_()
        model.delta_h_bias.normal_()
    x = torch.randn(2, 7, 4)
    ids = torch.tensor([1, 0])
    expected = model.forward_batched(x, ids)[0]
    assets = MethodAssets(profile, Path('.'), {'shared.' + k: v for k, v in model.state_dict().items()}, {'shared_num_blocks': 2})
    assert torch.equal(shared_hidden(x, ids, assets), expected)


def test_reuse_retains_bf16_code_rounding(profile):
    from scd_runtime.contracts.types import MethodAssets
    from scd_runtime.methods.icml.reuse import reconstruct

    tensors = {'reuse.0.enc_v': torch.eye(4) * 1.003, 'reuse.0.dec_v': torch.eye(4) * 100}
    assets = MethodAssets(profile, Path('.'), tensors, {})
    output = reconstruct(torch.ones(1, 1, 2, 4), assets, 0, 'v')
    assert output.unique().tolist() == [100.0]


def test_explicit_shared_token_ids_allow_distinct_tokenizer_files(profile, prefix):
    from scd_runtime.methods.icml.backend import ICMLBackend

    p, state, assets = synthetic(profile, prefix)
    p = dataclasses.replace(p, target=dataclasses.replace(p.target, tokenizer_sha256='8' * 64))
    output = ICMLBackend().convert(state, dataclasses.replace(assets, profile=p), None)
    assert output.prefix.token_ids == tuple(range(16))
    assert output.target.tokenizer_sha256 == '8' * 64


def test_batched_reuse_matches_frozen_original_group_math(profile, prefix):
    from scd_runtime.contracts.types import MethodAssets, SourceState
    from scd_runtime.methods.icml.reuse import reconstruct_layerwise
    from scd_runtime.methods.icml.shared_patch import paired_rope

    source = REPO_ROOT / 'src' / 'scd_repro' / 'strict_runtime.py'
    tree = ast.parse(source.read_text(encoding='utf-8'))
    names = {'_decode_reuse_layers_batched_layerwise_fastpath', '_stack_group_row_vectors', '_apply_optional_group_center'}

    class Timing:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    scope = {
        'require_torch': lambda: torch,
        '_selected_reuse_gids': lambda *a: [0, 1],
        'group_layers_for_gid': lambda w, i: [i],
        '_slice_prefix_tensor': lambda t, n: t[:, :, :n],
        '_synchronize_for_device': lambda *a: None,
        '_tensor_bytes': lambda t: t.numel() * t.element_size(),
        'ReuseDecodeTiming': Timing,
        'time': __import__('time'),
        'de_rope_k_torch': lambda t, s: paired_rope(t, s, True),
        're_rope_k_torch': lambda t, s: paired_rope(t, s),
    }
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    module = ast.Module(
        body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)] + nodes,
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(source), 'exec'), scope)

    torch.manual_seed(71)
    state = SourceState(
        profile.source,
        prefix,
        tuple((torch.randn(1, 1, 16, 4).bfloat16(), torch.randn(1, 1, 16, 4).bfloat16()) for _ in range(2)),
        {},
    )
    weights = {
        i: SimpleNamespace(
            **{name: torch.randn(4, 4) for name in ('enc_k', 'enc_v', 'dec_k_b', 'dec_v_b')},
            **{name: torch.randn(4) for name in ('mean_a_k', 'mean_a_v', 'mean_b_k', 'mean_b_v')},
        )
        for i in range(2)
    }
    bank = SimpleNamespace(num_layers=2, weights=weights, k_de_rope=True, k_re_rope=True)
    tensors = {
        f'reuse.{i}.{new}': getattr(w, old)
        for i, w in weights.items()
        for old, new in [
            ('enc_k', 'enc_k'),
            ('enc_v', 'enc_v'),
            ('dec_k_b', 'dec_k'),
            ('dec_v_b', 'dec_v'),
            ('mean_a_k', 'mean_a_k'),
            ('mean_a_v', 'mean_a_v'),
            ('mean_b_k', 'mean_b_k'),
            ('mean_b_v', 'mean_b_v'),
        ]
    }
    assets = MethodAssets(profile, Path('.'), tensors, {})
    expected, _, _ = scope['_decode_reuse_layers_batched_layerwise_fastpath'](
        SimpleNamespace(input_ids=torch.tensor([prefix.token_ids]), past_key_values=state.kv),
        SimpleNamespace(rope_spec=profile.source),
        SimpleNamespace(rope_spec=profile.target),
        bank,
        bank,
        16,
        torch.bfloat16,
        torch.bfloat16,
        'cpu',
        [0, 1],
    )
    actual = reconstruct_layerwise(state, assets, (0, 1))
    for i in range(2):
        for observed, reference in zip(actual[i], expected[i]):
            assert torch.equal(observed.bfloat16(), reference)
