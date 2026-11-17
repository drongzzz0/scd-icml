"""Exact-value and fail-closed contracts for ICML packed private snapshots."""
from dataclasses import replace
import gc
from types import SimpleNamespace
import weakref

import pytest
import torch

from scd_runtime.contracts.types import EngineCapability
from scd_runtime.errors import ArtifactCorrupt, UnsupportedProfile
from scd_runtime.transport.resident import ResidentRegistry
from scd_vllm.layout import prepare_layer_writer, write_layer
from test_resident_transport import destinations, write
from test_store import make_kv


def setup(profile, prefix):
    p = replace(profile, method_id='icml_reuse_patch')
    reg = ResidentRegistry(p, 'a' * 32, 2**20, device='cpu', synchronize=lambda: None)
    kv = make_kv(p, prefix)
    meta = reg.publish(kv)
    a = reg.admit(meta['artifact_id'], tuple(range(18)), 'request', 16)
    return reg, kv, a


def test_packed_owning_copy_one_finite_reduction_each_phase(profile, prefix, monkeypatch):
    seen = []
    original = torch.isfinite

    def finite(t):
        seen.append(tuple(t.shape))
        return original(t)

    monkeypatch.setattr(torch, 'isfinite', finite)
    reg, kv, a = setup(profile, prefix)
    entry = reg._entry
    packed = entry['packed']
    assert packed._base is None and not packed.is_inference() and packed.is_contiguous()
    assert len(seen) == 1 and seen[0] == (4, 1, 1, 16, 4)
    for pair in kv.layers:
        for t in pair:
            t.zero_()
    dst = destinations()
    write(reg, a, dst)
    assert len(seen) == 2 and seen[1] == seen[0]
    assert dst[0][0, 0, 0].tolist() == [1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0]
    assert entry['packed'] is entry['kv'] is None and entry['bytes'] == 0
    reg.finish(a)


def test_normal_versioned_ownership_even_under_outer_inference(profile, prefix):
    with torch.inference_mode():
        reg, _, a = setup(profile, prefix)
        assert not reg._entry['packed'].is_inference()
        assert all(not t.is_inference() for pair in reg._entry['kv'].layers for t in pair)
        write(reg, a)
        reg.finish(a)
    assert reg.status()['entries'] == 0


@pytest.mark.parametrize('change', ['ordinary', 'data_nan', 'replace_layer', 'replace_packed', 'packed_shape'])
def test_private_mutation_rejected_and_storage_released(profile, prefix, change):
    reg, _, a = setup(profile, prefix)
    entry = reg._entry
    if change == 'ordinary':
        entry['kv'].layers[0][0].add_(1)
    elif change == 'data_nan':
        entry['kv'].layers[0][0].data[0, 0, 0, 0] = float('nan')
    elif change == 'replace_layer':
        old = entry['kv']
        pairs = list(old.layers)
        pairs[0] = (pairs[0][0].clone(), pairs[0][1])
        entry['kv'] = replace(old, layers=tuple(pairs))
    elif change == 'replace_packed':
        entry['packed'] = entry['packed'].clone()
    else:
        entry['packed'].resize_(entry['packed'].numel())
    with pytest.raises(ArtifactCorrupt):
        write(reg, a)
    assert entry['packed'] is entry['kv'] is None and entry['bytes'] == 0
    reg.release(a.artifact_id, a.lease_id)


@pytest.mark.parametrize('path', ['success', 'writer_failure', 'cancel', 'clear'])
def test_registry_does_not_retain_freed_packed_storage(profile, prefix, path):
    reg, _, a = setup(profile, prefix)
    ref = weakref.ref(reg._entry['packed'])
    if path == 'success':
        write(reg, a)
    elif path == 'writer_failure':
        def fail(*_a):
            raise RuntimeError('copy failed')

        with pytest.raises(RuntimeError):
            reg.load(a, 'request', destinations(), torch.arange(16), None, fail)
    elif path == 'cancel':
        reg.release(a.artifact_id, a.lease_id)
    else:
        reg.clear()
    gc.collect()
    assert ref() is None


@pytest.mark.parametrize('outcome', ['complete', 'fail'])
def test_async_owner_drain_drops_packed_storage(profile, prefix, outcome):
    reg, _, a = setup(profile, prefix)
    cap = EngineCapability('2b1f75c80f79496b313e28362f0856632dfbf5ea', 'FLASH_ATTN', 'bfloat16', 1, 1, 16, 1, profile.target.attention, False, False, False)
    dst = destinations()
    slots = torch.arange(16, dtype=torch.int64)
    ref = weakref.ref(reg._entry['packed'])
    borrowed = reg.begin_async_load(a, 'request-deadbeef', dst, slots, cap)
    with pytest.raises(UnsupportedProfile):
        reg.clear()
    if outcome == 'complete':
        reg.complete_async_load(a, 'request-deadbeef', {})
    else:
        reg.fail_async_load(a)
    assert reg._entry['packed'] is reg._entry['kv'] is None and reg._entry['bytes'] == 0
    del borrowed
    gc.collect()
    assert ref() is None


def test_nonfinite_publish_cannot_create_entry(profile, prefix):
    reg = ResidentRegistry(profile, 'a' * 32, 2**20, device='cpu', synchronize=lambda: None)
    kv = make_kv(profile, prefix)
    kv.layers[-1][-1][0, 0, -1, -1] = float('inf')
    with pytest.raises(ArtifactCorrupt):
        reg.publish(kv)
    assert reg.status()['entries'] == reg.status()['snapshot_bytes'] == 0


@pytest.mark.parametrize('length', [1, 15, 16, 17, 31])
def test_prepared_slot_writer_exact_and_partial_padding_untouched(length):
    cap = SimpleNamespace(block_size=16)
    slots = torch.randperm(48)[:length].to(dtype=torch.int64)
    dst = {i: torch.full((3, 2, 16, 8), -999.0, dtype=torch.bfloat16) for i in range(2)}
    expected = {i: t.clone() for i, t in dst.items()}
    writer = prepare_layer_writer(dst, slots, cap)
    for i in dst:
        k = torch.arange(2 * length * 8, dtype=torch.float32).reshape(1, 2, length, 8)[..., ::2].to(torch.bfloat16)
        pair = (k, k + 1)
        write_layer(expected[i], pair, slots, cap)
        writer(dst[i], pair, slots, cap)
        assert torch.equal(dst[i], expected[i])
