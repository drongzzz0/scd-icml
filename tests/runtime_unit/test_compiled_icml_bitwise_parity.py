"""Bitwise-parity verification for CompiledICMLPlan, zero-copy packed publish, and paged write_packed_layers_fast."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scd_repro.direct_kv_residual_patch import XNormNativeKVSharedResidualPatch
from scd_runtime.contracts.identity import fingerprint
from scd_runtime.contracts.types import (
    EngineCapability,
    MethodAssets,
    MethodProfile,
    ModelIdentity,
    PrefixContract,
    ReaderIdentity,
    SourceState,
)
from scd_runtime.methods.icml.backend import ICMLBackend
from scd_runtime.methods.icml.reuse import BATCHED_NUMERIC_VERSION, NATIVE_ROPE_NUMERIC_VERSION
from scd_runtime.providers.hf.projections import TargetProjectionProvider
from scd_runtime.transport.resident import ResidentRegistry
from scd_vllm.layout import write_layer, write_packed_layers_fast


def _build_mistral_like_fixture(seq_len: int, numeric_version: str):
    """Build a realistic 8-layer mixed Reuse+Patch fixture with both identity and active SwiGLU heads."""
    torch.manual_seed(20261116 + seq_len)
    L = 8
    H = 4
    D = 16
    hidden_dim = H * D
    reuse_layers = [0, 3, 6]
    patch_layers = [1, 2, 4, 5, 7]
    active_patch = [1, 4, 7]  # layers 2 and 5 stay exact-zero identity residual heads

    source_id = ModelIdentity(
        '1' * 64,
        '2' * 64,
        '3' * 64,
        '4' * 64,
        L,
        H,
        D,
        'bfloat16',
        {'theta': 1000000.0},
        {'layer_types': ['full'] * L, 'sliding_window': None},
    )
    target_id = ModelIdentity(
        '5' * 64,
        '6' * 64,
        '3' * 64,
        '4' * 64,
        L,
        H,
        D,
        'bfloat16',
        {'theta': 10000.0},
        {'layer_types': ['sliding'] * L, 'sliding_window': 4096},
    )
    reader = ReaderIdentity('none', target_id.manifest_sha256)
    profile = MethodProfile(
        'mistral_like_mixed',
        'icml_reuse_patch',
        source_id,
        target_id,
        reader,
        {
            'reuse_layers': reuse_layers,
            'patch_layers': patch_layers,
            'fit_mode': 'xnorm_nativekv_shared_residual_mlp_v1',
            'reuse_key_space': 'pre_rope',
        },
        tuple(range(L)),
        numeric_version,
        '9' * 64,
    )
    prefix = PrefixContract(tuple((i % 50) + 1 for i in range(seq_len)), 0, '3' * 64, '4' * 64)

    tensors = {}
    for layer_idx in reuse_layers:
        for kind in ('k', 'v'):
            tensors[f'reuse.{layer_idx}.enc_{kind}'] = torch.randn(D, D, dtype=torch.float32) * 0.25
            tensors[f'reuse.{layer_idx}.dec_{kind}'] = torch.randn(D, D, dtype=torch.float32) * 0.25
            tensors[f'reuse.{layer_idx}.mean_a_{kind}'] = torch.randn(D, dtype=torch.float32) * 0.1
            tensors[f'reuse.{layer_idx}.mean_b_{kind}'] = torch.randn(D, dtype=torch.float32) * 0.1

    patch_model = XNormNativeKVSharedResidualPatch(
        num_layers=L,
        hidden_dim=hidden_dim,
        trunk_dim=32,
        bottleneck_dim=16,
        num_blocks=2,
        ffn_multiplier=2.0,
    )
    with torch.no_grad():
        patch_model.delta_h_weight.zero_()
        patch_model.delta_h_bias.zero_()
        for layer_idx in active_patch:
            patch_model.delta_h_weight[layer_idx].normal_(std=0.05)
            patch_model.delta_h_bias[layer_idx].normal_(std=0.05)
    for k, v in patch_model.state_dict().items():
        tensors['shared.' + k] = v.clone()

    for layer_idx in patch_layers:
        tensors[f'projection.{layer_idx}.k_weight'] = torch.randn(H * D, hidden_dim, dtype=torch.bfloat16) * 0.1
        tensors[f'projection.{layer_idx}.v_weight'] = torch.randn(H * D, hidden_dim, dtype=torch.bfloat16) * 0.1

    assets = MethodAssets(
        profile,
        Path('.'),
        tensors,
        {'shared_num_blocks': 2, 'projection_model_sha256': target_id.manifest_sha256},
    )
    projections = TargetProjectionProvider.from_assets(assets)

    kv_pairs = tuple(
        (
            torch.randn(1, H, seq_len, D, dtype=torch.bfloat16),
            torch.randn(1, H, seq_len, D, dtype=torch.bfloat16),
        )
        for _ in range(L)
    )
    hidden = {
        (layer_idx, 'attention_input'): torch.randn(1, seq_len, hidden_dim, dtype=torch.bfloat16)
        for layer_idx in patch_layers
    }
    state = SourceState(source_id, prefix, kv_pairs, hidden)
    return profile, prefix, state, assets, projections


@pytest.mark.parametrize('numeric_version', [BATCHED_NUMERIC_VERSION, NATIVE_ROPE_NUMERIC_VERSION])
@pytest.mark.parametrize('seq_len', [1, 15, 16, 17, 48, 129])
def test_compiled_icml_plan_is_bitwise_identical_to_reference_convert(numeric_version, seq_len):
    profile, prefix, state, assets, projections_ref = _build_mistral_like_fixture(seq_len, numeric_version)
    projections_fast = TargetProjectionProvider.from_assets(assets)

    ref_backend = ICMLBackend()
    ref_out = ref_backend.convert(state, assets, projections_ref, return_packed=False)
    assert ref_backend._compiled_plan is None  # Confirm reference actually ran uncompiled

    fast_backend = ICMLBackend()
    plan = fast_backend.compile_plan(assets, projections_fast, device=torch.device('cpu'), max_seq_len=256)
    assert plan is not None
    assert plan.identity_layers & set(plan.patch_layers) == {2, 5}
    assert plan.active_patch_layers == (1, 4, 7)

    fast_unpacked = fast_backend.convert(state, assets, projections_fast, return_packed=False)
    fast_packed = fast_backend.convert(state, assets, projections_fast, return_packed=True)

    assert fast_packed.packed_kv is not None
    assert tuple(fast_packed.packed_kv.shape) == (8, seq_len, 4, 32)
    assert fast_packed.packed_kv.is_contiguous()
    assert not fast_packed.packed_kv.is_inference()

    for layer_idx in range(profile.target.num_layers):
        ref_k, ref_v = ref_out.layers[layer_idx]
        u_k, u_v = fast_unpacked.layers[layer_idx]
        p_k, p_v = fast_packed.layers[layer_idx]
        assert torch.equal(u_k, ref_k), f'unpacked K mismatch at layer {layer_idx}'
        assert torch.equal(u_v, ref_v), f'unpacked V mismatch at layer {layer_idx}'
        assert torch.equal(p_k.contiguous(), ref_k), f'packed view K mismatch at layer {layer_idx}'
        assert torch.equal(p_v.contiguous(), ref_v), f'packed view V mismatch at layer {layer_idx}'


@pytest.mark.parametrize('seq_len', [1, 15, 16, 17, 31, 32, 47])
@pytest.mark.parametrize('sequential_blocks', [True, False])
def test_write_packed_layers_fast_is_bitwise_identical_to_write_layer(seq_len, sequential_blocks):
    profile, prefix, state, assets, projections = _build_mistral_like_fixture(seq_len, NATIVE_ROPE_NUMERIC_VERSION)
    backend = ICMLBackend()
    backend.compile_plan(assets, projections, device=torch.device('cpu'), max_seq_len=128)
    kv = backend.convert(state, assets, projections, return_packed=True)

    block_size = 16
    num_blocks = 8
    L = profile.target.num_layers
    H = profile.target.num_kv_heads
    D = profile.target.head_dim
    cap = SimpleNamespace(block_size=block_size)

    needed_blocks = (seq_len + block_size - 1) // block_size
    if sequential_blocks:
        chosen_blocks = list(range(2, 2 + needed_blocks))
    else:
        chosen_blocks = [7, 1, 5, 3][:needed_blocks]
    slots_list = [chosen_blocks[i // block_size] * block_size + (i % block_size) for i in range(seq_len)]
    slots_tensor = torch.tensor(slots_list, dtype=torch.int64)

    dst_ref = {i: torch.full((num_blocks, H, block_size, 2 * D), -777.0, dtype=torch.bfloat16) for i in range(L)}
    dst_fast = {i: t.clone() for i, t in dst_ref.items()}

    for i in range(L):
        write_layer(dst_ref[i], kv.layers[i], slots_tensor, cap)
    write_packed_layers_fast(dst_fast, kv.packed_kv, slots_list, cap)

    for i in range(L):
        assert torch.equal(dst_fast[i], dst_ref[i]), f'paged KV mismatch at layer {i}'


def test_zero_copy_adopt_packed_and_stream_ordered_load_packed_fast():
    seq_len = 23
    profile, prefix, state, assets, projections = _build_mistral_like_fixture(seq_len, NATIVE_ROPE_NUMERIC_VERSION)
    backend = ICMLBackend()
    backend.compile_plan(assets, projections, device=torch.device('cpu'), max_seq_len=128)
    kv = backend.convert(state, assets, projections, return_packed=True)
    orig_ptr = kv.packed_kv.data_ptr()

    cap = EngineCapability(
        '0.18.0',
        'FLASH_ATTN',
        'bfloat16',
        1,
        1,
        16,
        1,
        profile.target.attention,
        False,
        False,
        False,
        4096,
    )
    reg = ResidentRegistry(profile, 'a' * 32, 16 * 1024**2, device='cpu', capability=cap)
    meta = reg.publish(kv, adopt_packed=True, sync_on_publish=False, check_finite=False)
    assert reg._entry['packed_kind'] == 'lnh2d'
    assert reg._entry['packed'].data_ptr() == orig_ptr  # Zero-copy adoption verified!

    prompt_ids = prefix.token_ids + (42,)
    admission = reg.admit(meta['artifact_id'], prompt_ids, 'req0', 16, import_policy='fullprefix')
    slots_list = [16 + i for i in range(seq_len)]  # block 1 + partial block 2
    L = profile.target.num_layers
    H = profile.target.num_kv_heads
    D = profile.target.head_dim
    dst = {i: torch.zeros((4, H, 16, 2 * D), dtype=torch.bfloat16) for i in range(L)}
    dst_expected = {i: t.clone() for i, t in dst.items()}
    for i in range(L):
        write_layer(dst_expected[i], kv.layers[i], torch.tensor(slots_list, dtype=torch.int64), cap)

    reg.load_packed_fast(admission, 'req0-deadbeef', dst, slots_list, cap, sync_after_write=False)
    for i in range(L):
        assert torch.equal(dst[i], dst_expected[i])
    receipt = reg.finish(admission)
    assert receipt['loaded_tokens'] == seq_len
    assert receipt['import_policy'] == 'fullprefix'
    assert receipt['stream_ordered_nonblocking'] is True
    assert reg.status()['entries'] == 0
