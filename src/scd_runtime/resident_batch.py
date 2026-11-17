"""Per-entry resident batch lifecycle for SCD-ICML on vLLM V1."""
from __future__ import annotations

import time
import uuid

from .batch import BatchResult, observe_batch
from .contracts.identity import profile_fingerprint
from .contracts.types import GenerationResult
from .errors import IdentityMismatch, ReaderMismatch, UnsupportedProfile


def consume_prepared_batch(
    client,
    requests,
    generation,
    *,
    cancel_after_first=(),
    reader_digests=None,
    _preadmitted=None,
):
    items = tuple(requests)
    if reader_digests is not None and any(r is not None for r in reader_digests):
        raise ReaderMismatch('SCD-ICML does not accept LoRA reader digests')
    cancelled_indices = set(cancel_after_first)
    if not items or len(items) > client.max_batch_size:
        raise UnsupportedProfile('resident batch exceeds configured request bound')
    if any(type(i) is not int or not 0 <= i < len(items) for i in cancelled_indices):
        raise ValueError('invalid resident cancellation index')
    handles = [item[0]['artifact_id'] for item in items if item[0] is not None]
    if len(set(handles)) != len(handles):
        raise IdentityMismatch('single-use resident snapshot repeated in batch')
    for prepared, prompt in items:
        if prepared is not None and (
            prepared['session_id'] != client.session_id
            or prepared['profile_fingerprint'] != profile_fingerprint(client.profile)
        ):
            raise IdentityMismatch('foreign resident snapshot in batch')
        client._generation(prompt, generation)

    capacity = getattr(client, 'kv_capacity_tokens', None)
    block = getattr(client, 'cache_registration', {}).get('block_size', 16)
    if (
        capacity is not None
        and sum(((len(prompt) + generation.get('max_new_tokens', 64) + block - 1) // block) * block for _, prompt in items)
        > capacity
    ):
        raise UnsupportedProfile('resident batch would require unsupported preemption')

    started = time.perf_counter()
    with client._lock, client._request_scope(started):
        if client.engine.llm_engine.has_unfinished_requests():
            raise UnsupportedProfile('resident batch requires owned idle frontend')
        admissions = []
        external_ids = []
        finalized_status = None
        result = None
        use_fused_finalize = bool(getattr(client, '_use_fused_batch_rpc', False))
        try:
            t = time.perf_counter()
            if _preadmitted is not None:
                admissions, external_ids = list(_preadmitted[0]), list(_preadmitted[1])
                admission_ms = 0.0
            else:
                for prepared, prompt in items:
                    external_id = uuid.uuid4().hex
                    external_ids.append(external_id)
                    admission = (
                        None
                        if prepared is None
                        else client._rpc(
                            'scd_gpu_admit',
                            session_id=client.session_id,
                            artifact_id=prepared['artifact_id'],
                            prompt_ids=list(prompt),
                            request_id=external_id,
                            max_new_tokens=generation.get('max_new_tokens', 64),
                        )
                    )
                    if admission is not None and admission.get('request_id') != external_id:
                        raise IdentityMismatch('resident admission changed the owned external request ID')
                    admissions.append(admission)
                admission_ms = (time.perf_counter() - t) * 1000

            queued = []
            for (_, prompt), a, rid in zip(items, admissions, external_ids):
                params = client._generation(prompt, generation, a)
                payload = {'prompt_token_ids': list(prompt)}
                queued.append((rid, payload, params))

            generated_at = time.perf_counter()
            observation = observe_batch(
                client.engine.llm_engine,
                queued,
                started=generated_at,
                timeout=client._remaining(),
                deadline_factory=client._deadline_factory,
                cancel_after_first={external_ids[i] for i in cancelled_indices},
            )
            if cancelled_indices or not use_fused_finalize:
                client._rpc('check_health')

            batch_receipts = None
            if use_fused_finalize and any(a is not None for a in admissions):
                finalized_payload = client._rpc(
                    'scd_gpu_finalize_batch',
                    session_id=client.session_id,
                    admissions_data=admissions,
                )
                batch_receipts = finalized_payload['receipts']
                finalized_status = finalized_payload['status']

            generations = []
            for idx, ((prepared, _), admission, rid) in enumerate(zip(items, admissions, external_ids)):
                item = observation['requests'][rid]
                if admission is None:
                    generations.append(
                        GenerationResult(
                            item['token_ids'],
                            None,
                            {
                                'injection_receipt': None,
                                'first_token_ms': (generated_at - started) * 1000 + item['first_token_ms'],
                                'completion_ms': (generated_at - started) * 1000 + item['completion_ms'],
                                'cancelled_after_first_token': item['cancelled'],
                                'imported_tokens': 0,
                                'actual_request_id': item['internal_id'],
                                'native_fallback': False,
                            },
                        )
                    )
                    continue
                receipt = (
                    batch_receipts[idx]
                    if batch_receipts is not None
                    else client._rpc('scd_gpu_finish', session_id=client.session_id, admission_data=admission)
                )
                expected = {
                    'session_id': client.session_id,
                    'artifact_id': prepared['artifact_id'],
                    'lease_id': admission['lease_id'],
                    'request_id': admission['request_id'],
                    'actual_request_id': item['internal_id'],
                    'profile_fingerprint': profile_fingerprint(client.profile),
                    'request_digest': admission['plan']['request_digest'],
                    'loaded_layers': client.profile.target.num_layers,
                    'loaded_tokens': admission['plan']['imported_tokens'],
                }
                if any(receipt.get(key) != value for key, value in expected.items()) or receipt.get('device_types') != ['cuda']:
                    raise IdentityMismatch('resident batch receipt differs from actual request')
                generations.append(
                    GenerationResult(
                        item['token_ids'],
                        None,
                        {
                            'injection_receipt': receipt,
                            'first_token_ms': (generated_at - started) * 1000 + item['first_token_ms'],
                            'completion_ms': (generated_at - started) * 1000 + item['completion_ms'],
                            'cancelled_after_first_token': item['cancelled'],
                            'imported_tokens': receipt['loaded_tokens'],
                            'native_fallback': False,
                            'request_digest': admission['plan']['request_digest'],
                            'transport_kind': 'private_same_worker_gpu',
                            'kv_payload_cpu_bytes': 0,
                            'kv_payload_file_bytes': 0,
                        },
                    )
                )

            complete = sum(not value['cancelled'] for value in observation['requests'].values())
            result = BatchResult(
                tuple(generations),
                {
                    'batch_size': len(items),
                    'completed_requests': complete,
                    'cancelled_requests': len(items) - complete,
                    'generated_tokens': sum(len(item.token_ids) for item in generations),
                    'admission_ms': admission_ms,
                    'generation_ms': observation['wall_ms'],
                    'max_outputs_per_step': observation['max_outputs_per_step'],
                    'steps_with_multiple_outputs': observation['steps_with_multiple_outputs'],
                    'step_count': observation['step_count'],
                    'latency_scope': 'client_batch_call_including_admission',
                    'imported_requests': sum(a is not None for a in admissions),
                    'native_requests': sum(a is None for a in admissions),
                    'scope': (
                        'owned_import_and_native_batch'
                        if any(a is None for a in admissions)
                        else 'prepared_resident_snapshots_to_completed_batch'
                    ),
                    'kv_payload_cpu_bytes': 0,
                    'kv_payload_file_bytes': 0,
                },
            )
        except BaseException as exc:
            if not client._closed:
                client._fail_closed(exc)
            raise
        finally:
            if not client._closed:
                try:
                    if finalized_status is not None:
                        status = finalized_status
                    else:
                        for (prepared, _), admission in zip(items, admissions):
                            if admission is None:
                                continue
                            client._rpc(
                                'scd_gpu_release',
                                session_id=client.session_id,
                                artifact_id=prepared['artifact_id'],
                                lease_id=admission['lease_id'],
                            )
                        status = client.status()
                    phases = status.get('phases', {})
                    if any(handle in phases for handle in handles) or status.get('poisoned'):
                        raise RuntimeError('owned resident batch entries leaked or were poisoned')
                    if result is not None:
                        result.metrics.update(
                            batch_wall_ms=(time.perf_counter() - started) * 1000,
                            worker_peak_allocated_bytes=status.get('peak_allocated_bytes'),
                            worker_peak_reserved_bytes=status.get('peak_reserved_bytes'),
                            remaining_unconsumed_snapshots=status['snapshots'],
                        )
                except BaseException as exc:
                    if not client._closed:
                        client._fail_closed(exc)
                    raise
        return result
