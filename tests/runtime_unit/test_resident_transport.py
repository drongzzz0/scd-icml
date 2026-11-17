from contextlib import nullcontext
from dataclasses import replace
import threading
import time
from types import SimpleNamespace

import pytest
import torch

from test_store import make_kv


def registry(profile, capacity=2**20, synchronize=lambda: None):
    from scd_runtime.transport.resident import ResidentRegistry

    return ResidentRegistry(profile, 'a' * 32, capacity, device='cpu', synchronize=synchronize)


def destinations():
    return {i: torch.full((1, 1, 16, 8), float('nan'), dtype=torch.bfloat16) for i in range(2)}


def write(reg, admission, dst=None, actual='request-deadbeef'):
    from scd_runtime.contracts.types import EngineCapability
    from scd_vllm.layout import write_layer

    p = reg.profile
    cap = EngineCapability('2b1f75c80f79496b313e28362f0856632dfbf5ea', 'FLASH_ATTN', 'bfloat16', 1, 1, 16, 1, p.target.attention, False, False, False)
    reg.load(admission, actual, dst or destinations(), torch.arange(16), cap, write_layer)


def test_private_snapshot_survives_caller_mutation_and_consumes_once(profile, prefix):
    from scd_runtime.errors import IdentityMismatch, MissingAsset

    reg = registry(profile)
    kv = make_kv(profile, prefix)
    meta = reg.publish(kv)
    for pair in kv.layers:
        for tensor in pair:
            tensor.zero_()
    a = reg.admit(meta['artifact_id'], tuple(range(18)), 'request', 16)
    dst = destinations()
    write(reg, a, dst)
    assert dst[0][0, 0, 0].tolist() == [1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0]
    assert reg.status()['snapshot_bytes'] == 0
    with pytest.raises(IdentityMismatch):
        write(reg, a)
    receipt = reg.finish(a)
    assert receipt['actual_request_id'] == 'request-deadbeef' and receipt['loaded_tokens'] == 16
    assert reg.status()['entries'] == 0
    with pytest.raises(MissingAsset):
        reg.admit(meta['artifact_id'], tuple(range(18)), 'next', 16)


def test_snapshot_version_mutation_is_rejected(profile, prefix):
    from scd_runtime.errors import ArtifactCorrupt

    reg = registry(profile)
    meta = reg.publish(make_kv(profile, prefix))
    a = reg.admit(meta['artifact_id'], tuple(range(18)), 'request', 16)
    reg._entry['kv'].layers[0][0].add_(1)
    with pytest.raises(ArtifactCorrupt):
        write(reg, a)
    assert reg.status()['snapshot_bytes'] == 0
    reg.release(meta['artifact_id'], a.lease_id)
    assert reg.status()['entries'] == 0


def test_wrong_prompt_session_lease_and_actual_request_rejected(profile, prefix):
    from scd_runtime.errors import IdentityMismatch

    reg = registry(profile)
    meta = reg.publish(make_kv(profile, prefix))
    with pytest.raises(IdentityMismatch):
        reg.admit(meta['artifact_id'], [99] + list(range(1, 18)), 'request', 16)
    a = reg.admit(meta['artifact_id'], tuple(range(18)), 'request', 16)
    for bad in (replace(a, session_id='b' * 32), replace(a, lease_id='b' * 64)):
        with pytest.raises(IdentityMismatch):
            write(reg, bad)
    with pytest.raises(IdentityMismatch):
        write(reg, a, actual='different-deadbeef')
    write(reg, a)
    reg.finish(a)


def test_scheduler_binds_actual_tokens_and_rejects_non_token_requests(profile, prefix):
    from scd_runtime.errors import IdentityMismatch, UnsupportedProfile
    from scd_runtime.transport.resident import validate_scheduled_request

    reg = registry(profile)
    meta = reg.publish(make_kv(profile, prefix))
    a = reg.admit(meta['artifact_id'], tuple(range(18)), 'request', 16)
    request = SimpleNamespace(
        request_id='request-deadbeef',
        prompt_token_ids=list(range(18)),
        prompt_embeds=None,
        prompt_is_token_ids=None,
        mm_features=[],
        lora_request=None,
        resumable=False,
    )
    validate_scheduled_request(a, request, profile, 16)
    request.prompt_token_ids[-1] = 99
    with pytest.raises(IdentityMismatch):
        validate_scheduled_request(a, request, profile, 16)
    request.prompt_token_ids = list(range(18))
    request.mm_features = ['image']
    with pytest.raises(UnsupportedProfile):
        validate_scheduled_request(a, request, profile, 16)


def test_client_observed_ttft_and_fused_finalize_cleanup(profile, monkeypatch):
    from scd_runtime.contracts.identity import profile_fingerprint
    from scd_runtime.resident_runner import GPUResidentConsumer
    import scd_runtime.resident_runner as module

    clock = [2.0]
    authorized = {}
    monkeypatch.setattr(module.time, 'perf_counter', lambda: clock[0])

    class Engine:
        active = False
        steps = 0

        def has_unfinished_requests(self):
            return self.active

        def add_request(self, request_id, *args):
            self.active = True
            self.external = request_id
            clock[0] += 1
            return request_id + '-deadbeef'

        def step(self):
            self.steps += 1
            clock[0] += {1: 3, 2: 5, 3: 2}[self.steps]
            if self.steps == 1:
                return []
            self.active = self.steps < 3
            return [
                SimpleNamespace(
                    request_id=self.external,
                    finished=not self.active,
                    outputs=[
                        SimpleNamespace(
                            token_ids=[7] if self.active else [7, 8],
                            finish_reason=None if self.active else 'length',
                        )
                    ],
                )
            ]

    client = object.__new__(GPUResidentConsumer)
    client.engine = SimpleNamespace(llm_engine=Engine())
    client._deadline_factory = lambda seconds: nullcontext()
    client._closed = False
    client._lock = threading.RLock()
    client.session_id = 'a' * 32
    client.profile = profile
    client.enforce_eager = False
    client.has_fixed_reader = False
    client._generation = lambda *args: object()

    def rpc(method, **kwargs):
        if method == 'scd_gpu_admit':
            authorized.update(kwargs)
            clock[0] += 2
            return {'lease_id': 'b' * 64, 'plan': {'request_digest': 'c' * 64, 'imported_tokens': 16, 'tail_recomputed_tokens': 0}}
        if method == 'scd_gpu_finalize':
            clock[0] += 2
            return {
                'receipt': {
                    'session_id': 'a' * 32,
                    'artifact_id': 'd' * 64,
                    'lease_id': 'b' * 64,
                    'request_id': authorized['request_id'],
                    'actual_request_id': authorized['request_id'] + '-deadbeef',
                    'profile_fingerprint': profile_fingerprint(profile),
                    'request_digest': 'c' * 64,
                    'loaded_layers': 2,
                    'loaded_tokens': 16,
                    'device_types': ['cuda'],
                },
                'first_token_completion': None,
                'gpu_timings': {'gpu_convert_ms': 1.25},
                'status': {'entries': 0, 'source_states': 0, 'poisoned': False, 'phases': {}},
            }
        raise AssertionError(method)

    client._rpc = rpc
    prepared = {
        'session_id': 'a' * 32,
        'profile_fingerprint': profile_fingerprint(profile),
        'artifact_id': 'd' * 64,
        **{key: 0.0 for key in ('prepare_rpc_ms', 'source_prefill_ms', 'convert_ms', 'publish_ms', 'peak_allocated_bytes', 'peak_reserved_bytes')},
    }
    result = client.consume_prepared(prepared, tuple(range(18)), {'max_new_tokens': 2}, started=0.0)
    assert result.metrics['pipeline_first_token_ms'] == 13000
    assert result.metrics['pipeline_generation_ms'] == 17000
    assert result.metrics['gpu_convert_ms'] == 1.25


def test_ensure_sampler_jit_safe_disables_flashinfer_without_ninja(monkeypatch):
    import importlib.util
    import os
    import shutil
    from scd_vllm import capability

    monkeypatch.delenv('VLLM_USE_FLASHINFER_SAMPLER', raising=False)
    monkeypatch.setattr(
        importlib.util,
        'find_spec',
        lambda name: object() if name == 'flashinfer' else None,
    )
    monkeypatch.setattr(shutil, 'which', lambda binary: None)
    monkeypatch.setenv('CUDA_HOME', '/nonexistent-cuda-home')
    assert capability.ensure_sampler_jit_safe() is True
    assert os.environ['VLLM_USE_FLASHINFER_SAMPLER'] == '0'
    # Second call is a no-op once set
    assert capability.ensure_sampler_jit_safe() is False


def test_fused_run_batch_uses_two_rpcs_and_tracks_dedup(profile, prefix):
    from scd_runtime.contracts.identity import profile_fingerprint, sha256
    from scd_runtime.resident_runner import GPUResidentConsumer

    rpc_calls = []

    class BatchEngine:
        def __init__(self):
            self.reqs = []
            self.done = False
            self.engine_core = SimpleNamespace(shutdown=lambda timeout=5: None)

        def sleep(self, level=0, mode='keep'):
            return None

        def wake_up(self, tags=None):
            return True

        def has_unfinished_requests(self):
            return bool(self.reqs) and not self.done

        def add_request(self, request_id, payload, params):
            self.reqs.append(request_id)
            return request_id + '-deadbeef'

        def step(self):
            self.done = True
            return [
                SimpleNamespace(
                    request_id=rid,
                    finished=True,
                    outputs=[SimpleNamespace(token_ids=[11, 12], finish_reason='length')],
                )
                for rid in self.reqs
            ]

    client = object.__new__(GPUResidentConsumer)
    client.engine = SimpleNamespace(llm_engine=BatchEngine())
    client._deadline_factory = lambda seconds: nullcontext()
    client._closed = False
    client._lock = threading.RLock()
    client.session_id = 'a' * 32
    client.profile = profile
    client.enforce_eager = False
    client.has_fixed_reader = False
    client.max_batch_size = 4
    client.kv_capacity_tokens = 1024
    client.cache_registration = {'block_size': 16}
    client._use_fused_batch_rpc = True
    client.stream_ordered_prepare = True
    client._generation = lambda *args: object()

    def rpc(method, **kwargs):
        rpc_calls.append(method)
        if method == 'scd_gpu_prepare_and_admit_batch':
            items = []
            for idx, item in enumerate(kwargs['batch_items']):
                art_id = f'{idx:064x}'
                lease_id = f'{idx + 100:064x}'
                items.append({
                    'session_id': 'a' * 32,
                    'artifact_id': art_id,
                    'profile_fingerprint': profile_fingerprint(profile),
                    'prefix_digest': sha256(prefix),
                    'prefix_tokens': 16,
                    'admission': {
                        'session_id': 'a' * 32,
                        'artifact_id': art_id,
                        'lease_id': lease_id,
                        'request_id': item['request_id'],
                        'profile_fingerprint': profile_fingerprint(profile),
                        'prefix_digest': sha256(prefix),
                        'block_size': 16,
                        'plan': {
                            'original_prefix_tokens': 16,
                            'imported_tokens': 16,
                            'tail_recomputed_tokens': 0,
                            'suffix_ids': [16, 17 + idx],
                            'request_digest': f'{idx + 200:064x}',
                        },
                    },
                    'source_prefill_ms': 5.0 if idx == 0 else 0.0,
                    'convert_ms': 1.5 if idx == 0 else 0.1,
                    'publish_ms': 0.05,
                    'memo_hit': False,
                    'batch_dedup_hit': idx > 0,
                    'peak_allocated_bytes': 1024,
                    'peak_reserved_bytes': 2048,
                })
            return {'items': items, 'peak_allocated_bytes': 1024, 'peak_reserved_bytes': 2048}
        if method == 'scd_gpu_finalize_batch':
            receipts = [
                {
                    'session_id': 'a' * 32,
                    'artifact_id': adm['artifact_id'],
                    'lease_id': adm['lease_id'],
                    'request_id': adm['request_id'],
                    'actual_request_id': adm['request_id'] + '-deadbeef',
                    'profile_fingerprint': profile_fingerprint(profile),
                    'request_digest': adm['plan']['request_digest'],
                    'loaded_layers': profile.target.num_layers,
                    'loaded_tokens': 16,
                    'device_types': ['cuda'],
                }
                for adm in kwargs['admissions_data']
            ]
            return {
                'receipts': receipts,
                'status': {
                    'entries': 0,
                    'snapshots': 0,
                    'phases': {},
                    'poisoned': False,
                    'peak_allocated_bytes': 1024,
                    'peak_reserved_bytes': 2048,
                },
            }
        raise AssertionError(f'unexpected RPC: {method}')

    client._rpc = rpc
    requests = [(prefix, tuple(range(16)) + (16, 17 + i)) for i in range(3)]
    out = client.run_batch(requests, {'max_new_tokens': 2})
    assert rpc_calls == ['scd_gpu_prepare_and_admit_batch', 'scd_gpu_finalize_batch']
    assert len(out.generations) == 3
    assert out.metrics['batch_dedup_hits'] == 2
    assert out.metrics['memo_hits'] == 0


def test_multi_entry_registry_requires_publish_all_before_admit_all(profile, prefix):
    from scd_runtime.transport.resident import ResidentRegistry

    reg = ResidentRegistry(profile, 'a' * 32, 2**20, device='cpu', synchronize=lambda: None, max_entries=4)
    receipts = [reg.publish(make_kv(profile, prefix)) for _ in range(4)]
    admissions = [
        reg.admit(r['artifact_id'], tuple(range(16)) + (16, 17 + idx), f'req-{idx}', 16)
        for idx, r in enumerate(receipts)
    ]
    assert reg.status()['admissions'] == 4
    for r, a in zip(receipts, admissions):
        reg.release(r['artifact_id'], a.lease_id)
    assert reg.status()['entries'] == 0
