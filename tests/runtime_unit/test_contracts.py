import dataclasses
import subprocess
import sys
import pytest

from scd_runtime.contracts.capability import VLLM_COMMIT


def test_package_import_does_not_load_models():
    import os
    from pathlib import Path
    import scd_runtime  # noqa: F401

    src_dir = str(Path(__file__).resolve().parents[2] / 'src')
    env = dict(os.environ)
    env['PYTHONPATH'] = src_dir + (os.pathsep + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    result = subprocess.run(
        [sys.executable, '-c', 'import scd_runtime,sys; assert "torch" not in sys.modules; assert "transformers" not in sys.modules'],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize('n,prompt_n,block,want,tail', [(16, 18, 16, 16, 0), (17, 19, 16, 16, 1), (16, 16, 16, 0, 16), (31, 33, 16, 16, 15), (32, 34, 16, 32, 0)])
def test_prefix_boundaries(n, prompt_n, block, want, tail, prefix):
    from scd_runtime.contracts.prefix import plan_import
    from scd_runtime.errors import NoReusablePrefix

    value = dataclasses.replace(prefix, token_ids=tuple(range(n)))
    if want == 0:
        with pytest.raises(NoReusablePrefix):
            plan_import(value, tuple(range(prompt_n)), block)
    else:
        result = plan_import(value, tuple(range(prompt_n)), block)
        assert (result.imported_tokens, result.tail_recomputed_tokens) == (want, tail)
        assert result.suffix_ids == tuple(range(want, prompt_n))


def test_identity_binds_route_and_keeps_suffix_independent(profile, prefix):
    from scd_runtime.contracts.identity import fingerprint
    from scd_runtime.contracts.prefix import plan_import

    other = dataclasses.replace(profile, route={**dict(profile.route), 'reuse_layers': [1], 'patch_layers': [0]})
    assert fingerprint(profile, prefix) != fingerprint(other, prefix)
    assert plan_import(prefix, (*prefix.token_ids, 99), 16).request_digest != plan_import(prefix, (*prefix.token_ids, 98), 16).request_digest


def test_prefix_mismatch_rejected(prefix):
    from scd_runtime.contracts.prefix import plan_import
    from scd_runtime.errors import IdentityMismatch

    with pytest.raises(IdentityMismatch):
        plan_import(prefix, (999, *range(1, 18)), 16)
    with pytest.raises(ValueError):
        dataclasses.replace(prefix, position_start=1)


def test_identity_is_deeply_immutable(profile):
    with pytest.raises(TypeError):
        profile.route['mode'] = 'bad'
    with pytest.raises(TypeError):
        profile.target.attention['layer_types'][0] = 'sliding'


def test_unknown_profile_fields_rejected(profile):
    from scd_runtime.contracts.types import profile_from_dict, to_data

    data = to_data(profile)
    data['silently_ignore'] = True
    with pytest.raises(ValueError):
        profile_from_dict(data)
    assert profile_from_dict(to_data(profile)) == profile


def test_iclr_profile_and_lora_reader_rejected_in_pure_icml_runtime(profile):
    from scd_runtime.contracts.types import ReaderIdentity, profile_from_dict, to_data

    with pytest.raises(ValueError):
        ReaderIdentity('lora', profile.target.manifest_sha256)
    bad = to_data(profile)
    bad['method_id'] = 'iclr_scale_kv'
    with pytest.raises(ValueError):
        profile_from_dict(bad)


def test_engine_capability_validation(profile):
    from scd_runtime.contracts.capability import check_capability
    from scd_runtime.contracts.types import EngineCapability
    from scd_runtime.errors import UnsupportedProfile

    capability = EngineCapability(VLLM_COMMIT, 'FLASH_ATTN', 'bfloat16', 1, 1, 16, 1, profile.target.attention, False, False, False)
    check_capability(profile, capability)
    check_capability(profile, dataclasses.replace(capability, vllm_commit='0.18.0'))
    with pytest.raises(UnsupportedProfile):
        check_capability(profile, dataclasses.replace(capability, vllm_commit='0' * 40), strict_commit_pin=True)
    with pytest.raises(UnsupportedProfile):
        check_capability(profile, dataclasses.replace(capability, hma=True))
    with pytest.raises(UnsupportedProfile):
        check_capability(profile, dataclasses.replace(capability, tp=2))


def test_window_bound_never_disables_attention(profile):
    from scd_runtime.contracts.capability import check_capability
    from scd_runtime.contracts.types import EngineCapability
    from scd_runtime.errors import UnsupportedProfile

    target = dataclasses.replace(profile.target, attention={'layer_types': ['sliding', 'sliding'], 'sliding_window': 32})
    p = dataclasses.replace(profile, target=target)
    cap = EngineCapability(VLLM_COMMIT, 'FLASH_ATTN', 'bfloat16', 1, 1, 16, 1, target.attention, False, False, False, max_model_len=32)
    check_capability(p, cap)
    with pytest.raises(UnsupportedProfile):
        check_capability(p, dataclasses.replace(cap, max_model_len=64))
