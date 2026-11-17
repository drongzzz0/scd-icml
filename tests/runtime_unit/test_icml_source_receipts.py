"""Bind original full source prompts to H handles and account for their peak memory."""
import dataclasses
import threading
from types import SimpleNamespace

import pytest
import torch

from scd_runtime.contracts.identity import sha256
from scd_runtime.contracts.types import FeatureRequest
from scd_runtime.errors import IdentityMismatch
from scd_runtime.resident_runner import GPUResidentConsumer
from scd_vllm.resident_worker import GPUResidentWorkerExtension


def state(profile, prefix):
    profile = dataclasses.replace(profile, method_id='icml_reuse_patch')
    registry = SimpleNamespace(
        profile=profile,
        capacity_bytes=3 * 1024**3,
        device='cpu',
        lock=threading.RLock(),
        require_preparable=lambda: None,
    )
    provider = SimpleNamespace(model=SimpleNamespace(config=SimpleNamespace(vocab_size=64, hidden_size=16)))
    backend = SimpleNamespace(required_features=lambda p: FeatureRequest((0, 1), ((0, 'attention_input'),)))
    return {
        'registry': registry,
        'provider': provider,
        'backend': backend,
        'max_model_len': 64,
        'source': None,
    }


def worker(current, prefix):
    obj = GPUResidentWorkerExtension.__new__(GPUResidentWorkerExtension)
    obj._scd_gpu_context = lambda session: current
    obj._scd_gpu_prefix = lambda state, data: prefix
    return obj


def test_full_prompt_receipt_records_actual_original_forward_extent(profile, prefix):
    current = state(profile, prefix)
    obj = worker(current, prefix)
    prompt = prefix.token_ids + (17,)
    actual, receipt = obj._scd_source_prompt(current, prefix, prompt)
    assert actual == prompt and receipt == {
        'source_full_prompt_prefill': True,
        'source_input_tokens': 17,
        'source_prompt_digest': sha256(prompt),
    }
    actual, receipt = obj._scd_source_prompt(current, prefix, None)
    assert actual is None and receipt['source_full_prompt_prefill'] is False and receipt['source_input_tokens'] == 16


@pytest.mark.parametrize('change', ['different_final', 'missing_full_prompt', 'changed_prefix'])
def test_h_handle_cannot_be_reused_for_another_full_prompt(profile, prefix, change):
    current = state(profile, prefix)
    obj = worker(current, prefix)
    prompt = prefix.token_ids + (17,)
    _, receipt = obj._scd_source_prompt(current, prefix, prompt)
    current['source'] = ('owned', SimpleNamespace(prefix=prefix), receipt)
    provided = prefix.token_ids + (18,) if change == 'different_final' else None if change == 'missing_full_prompt' else (1,) + prompt[1:]
    obj._scd_gpu_preflight = lambda *a: pytest.fail('bad H identity must fail before CUDA or conversion')
    with pytest.raises(IdentityMismatch):
        obj.scd_gpu_prepare(session_id='session', prefix_data={}, mode='H', source_handle='owned', full_prompt_ids=provided)
    assert current['source'][0] == 'owned'


def test_capture_peak_bound_grows_without_lowering_the_original_guard(profile, prefix, monkeypatch):
    current = state(profile, prefix)
    obj = worker(current, prefix)
    monkeypatch.setattr(torch.cuda, 'mem_get_info', lambda device: (2**32, 2**33))
    monkeypatch.setattr(torch.cuda, 'memory_reserved', lambda device: 0)
    monkeypatch.setattr(torch.cuda, 'memory_allocated', lambda device: 0)
    original = obj._scd_gpu_preflight(current, prefix, True)
    full = obj._scd_gpu_preflight(current, prefix, True, len(prefix.token_ids) + 1)
    assert full['source_state_bound_bytes'] > 2 * original['source_state_bound_bytes']
    assert full['target_snapshot_bytes'] == original['target_snapshot_bytes']
    assert full['estimated_required_bytes'] > original['estimated_required_bytes']
    assert original['estimated_required_bytes'] == 8 * original['target_snapshot_bytes'] + original['source_state_bound_bytes'] + 512 * 1024**2
    with pytest.raises(IdentityMismatch):
        obj._scd_gpu_preflight(current, prefix, True, len(prefix.token_ids) - 1)


@pytest.mark.parametrize('field,value', [('source_full_prompt_prefill', False), ('source_input_tokens', 16), ('source_prompt_digest', 'wrong')])
def test_frontend_rejects_unconfirmed_full_source_capture(profile, prefix, field, value):
    prompt = prefix.token_ids + (17,)
    receipt = {'source_full_prompt_prefill': True, 'source_input_tokens': 17, 'source_prompt_digest': sha256(prompt)}
    receipt[field] = value
    with pytest.raises(IdentityMismatch):
        GPUResidentConsumer._check_source_receipt(receipt, {'full_prompt_ids': list(prompt)})


def test_frontend_rejects_non_icml_full_source_capture(prefix):
    obj = SimpleNamespace(profile=SimpleNamespace(method_id='other_method'), max_model_len=64, target_vocab_size=64)
    assert GPUResidentConsumer._source_kwargs(obj, prefix, None) == {}
    with pytest.raises(IdentityMismatch):
        GPUResidentConsumer._source_kwargs(obj, prefix, prefix.token_ids + (17,))
