"""SWA fullprefix is private, serial and bounded by the real engine window."""
import dataclasses
from types import SimpleNamespace

import pytest
import torch

from scd_runtime.contracts.capability import VLLM_COMMIT
from scd_runtime.contracts.identity import fingerprint
from scd_runtime.contracts.prefix import check_fullprefix_scope
from scd_runtime.contracts.types import EngineCapability, ReaderIdentity, TargetKV
from scd_runtime.errors import IdentityMismatch, UnsupportedProfile
from scd_runtime.resident_runner import GPUResidentConsumer
from scd_runtime.transport.resident import ResidentRegistry


def bounded(profile):
    target = dataclasses.replace(profile.target, attention={'layer_types': ['sliding', 'sliding'], 'sliding_window': 32})
    p = dataclasses.replace(
        profile,
        method_id='icml_reuse_patch',
        target=target,
        reader=ReaderIdentity('none', target.manifest_sha256),
        numeric_version='icml-pinned-v1',
    )
    cap = EngineCapability(VLLM_COMMIT, 'FLASH_ATTN', 'bfloat16', 1, 1, 16, 1, target.attention, False, False, False, 32)
    return p, cap


def test_exact_window_bound_is_supported_only_for_private_resident(profile):
    p, cap = bounded(profile)
    check_fullprefix_scope(p, cap, resident=True)
    with pytest.raises(UnsupportedProfile):
        check_fullprefix_scope(p, cap)
    with pytest.raises(UnsupportedProfile):
        check_fullprefix_scope(p, resident=True)
    check_fullprefix_scope(profile)


@pytest.mark.parametrize(
    ('field', 'value'),
    [
        ('max_model_len', 33),
        ('max_model_len', None),
        ('max_model_len', True),
        ('max_model_len', 0),
        ('kv_groups', 2),
        ('hma', True),
        ('chunked_prefill', True),
        ('prefix_cache', True),
        ('block_size', 8),
        ('backend', 'TRITON_ATTN'),
        ('tp', 2),
    ],
)
def test_unverified_or_unbounded_engine_is_rejected(profile, field, value):
    p, cap = bounded(profile)
    with pytest.raises((UnsupportedProfile, ValueError)):
        check_fullprefix_scope(p, dataclasses.replace(cap, **{field: value}), resident=True)


def test_swa_exception_cannot_enable_mixed_attention(profile):
    p, cap = bounded(profile)
    mixed = dataclasses.replace(p.target, attention={'layer_types': ['full', 'sliding'], 'sliding_window': 32})
    with pytest.raises(UnsupportedProfile):
        check_fullprefix_scope(dataclasses.replace(p, target=mixed), cap, resident=True)


def resident(profile, prefix):
    p, cap = bounded(profile)
    prefix = dataclasses.replace(prefix, token_ids=tuple(range(15)))
    registry = ResidentRegistry(p, 'a' * 32, 65536, device='cpu', synchronize=lambda: None, capability=cap)
    layers = tuple((torch.zeros(1, 1, 15, 4, dtype=torch.bfloat16), torch.ones(1, 1, 15, 4, dtype=torch.bfloat16)) for _ in range(2))
    published = registry.publish(TargetKV(p.target, p.reader, prefix, layers, fingerprint(p, prefix)))
    return p, cap, prefix, registry, published


def test_partial_block_keeps_all_original_tokens_without_tail(profile, prefix):
    p, cap, prefix, registry, published = resident(profile, prefix)
    admission = registry.admit(published['artifact_id'], tuple(range(16)), 'request', 16, import_policy='fullprefix')
    assert admission.plan.imported_tokens == 15 and admission.plan.tail_recomputed_tokens == 0
    assert admission.plan.suffix_ids == (15,)
    with pytest.raises(UnsupportedProfile, match='synchronous only'):
        registry.begin_async_load(admission, 'request', {}, torch.tensor([], dtype=torch.int64), cap)


def test_registry_never_admits_a_prompt_outside_its_registered_extent(profile, prefix):
    p, cap, prefix, registry, published = resident(profile, prefix)
    with pytest.raises(IdentityMismatch, match='generation extent'):
        registry.admit(published['artifact_id'], tuple(range(32)), 'request', 16, import_policy='fullprefix')
    assert registry._entry['phase'] == 'READY'


def test_registry_does_not_keep_a_mutable_capability_attention_alias(profile):
    p, cap = bounded(profile)
    external = {'layer_types': ['sliding', 'sliding'], 'sliding_window': 32}
    registry = ResidentRegistry(p, 'b' * 32, 65536, device='cpu', synchronize=lambda: None, capability=dataclasses.replace(cap, attention=external))
    external['sliding_window'] = 128
    assert registry.capability.attention['sliding_window'] == 32


@pytest.mark.parametrize(
    'features',
    [
        None,
        SimpleNamespace(async_scheduling=True, async_loading=False, routing=False),
        SimpleNamespace(async_scheduling=False, async_loading=True, routing=False),
        SimpleNamespace(async_scheduling=False, async_loading=False, routing=True),
    ],
)
def test_frontend_requires_explicit_synchronous_private_mode(profile, features):
    p, cap = bounded(profile)
    consumer = SimpleNamespace(profile=p, capability=cap, max_batch_size=1, request_features=features)
    with pytest.raises(UnsupportedProfile):
        GPUResidentConsumer._check_fullprefix(consumer)


def test_frontend_accepts_only_the_bounded_serial_mode(profile):
    p, cap = bounded(profile)
    consumer = SimpleNamespace(
        profile=p,
        capability=cap,
        max_batch_size=1,
        request_features=SimpleNamespace(async_scheduling=False, async_loading=False, routing=False),
    )
    GPUResidentConsumer._check_fullprefix(consumer)
