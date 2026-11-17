from types import SimpleNamespace

import pytest
import torch

from scd_runtime.assets.formats import normalize_icml
from scd_runtime.errors import UnsupportedProfile
from scd_runtime.methods.icml.reuse import (
    NATIVE_ROPE_NUMERIC_VERSION, compile_reuse_plan, reconstruct_layerwise,
    reconstruct_layerwise_compiled,
)


def post_rope_profile():
    identity = SimpleNamespace(rope={"theta": 1_000_000, "rope_type": "default"},
                               head_dim=4, num_layers=1)
    return SimpleNamespace(
        source=identity, target=identity, layer_map=(0,), numeric_version=NATIVE_ROPE_NUMERIC_VERSION,
        route={"reuse_layers": [0], "patch_layers": [], "reuse_key_space": "post_rope",
               "fit_mode": "xnorm_nativekv_shared_residual_mlp_v1"},
    )


def test_post_rope_compiled_and_reference_apply_the_learned_matrix_directly():
    profile = post_rope_profile()
    permutation = torch.eye(4).roll(shifts=1, dims=1)
    assets = SimpleNamespace(profile=profile, tensors={
        "reuse.0.enc_k": torch.eye(4), "reuse.0.dec_k": permutation,
        "reuse.0.enc_v": torch.eye(4), "reuse.0.dec_v": torch.eye(4),
    })
    values = torch.arange(12, dtype=torch.bfloat16).reshape(1, 1, 3, 4)
    state = SimpleNamespace(kv=((values, values),))
    plan = compile_reuse_plan(assets, [0], torch.device("cpu"), 16)
    keys, vals = reconstruct_layerwise_compiled(state, plan)
    expected = values.float() @ permutation
    assert torch.equal(keys[0], expected) and torch.equal(vals[0], values.float())
    assert torch.equal(reconstruct_layerwise(state, assets, [0])[0][0], expected)


def test_post_rope_archive_import_requires_matching_rotation():
    profile = post_rope_profile()
    reuse = {
        "meta": {"group_size": 1, "k_de_rope": False, "k_re_rope": False, "group_layers": {"0": [0]}},
        "arrays": {f"{name}.0": torch.eye(4) for name in ("enc_k", "enc_v", "dec_k_b", "dec_v_b")},
    }
    patch = {
        "meta": {"fit_mode": profile.route["fit_mode"], "residual_num_blocks": 2},
        "arrays": {"xnorm_nativekv_shared_residual_patch.delta_h_weight": torch.ones(1)},
    }
    tensors, _ = normalize_icml({"reuse": reuse, "patch": patch}, profile, {})
    assert torch.equal(tensors["reuse.0.dec_k"], reuse["arrays"]["dec_k_b.0"])
    profile.target = SimpleNamespace(rope={"theta": 10_000, "rope_type": "default"})
    with pytest.raises(UnsupportedProfile, match="RoPE"):
        normalize_icml({"reuse": reuse, "patch": patch}, profile, {})
