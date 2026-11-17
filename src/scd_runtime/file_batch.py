"""Multi-request file transport with fixed engine identity and per-lease results."""
from __future__ import annotations

import time
import uuid

from .batch import BatchResult, observe_batch
from .contracts.types import GenerationResult, to_data, tokens
from .errors import IdentityMismatch, ReaderMismatch, UnsupportedProfile


def consume_batch(
    consumer,
    requests,
    generation,
    *,
    cancel_after_first=(),
    reader_digests=None,
    import_policy='blocktail',
):
    from vllm import SamplingParams
    from .resident_runner import _signal_deadline

    items = tuple((artifact, tokens(prompt)) for artifact, prompt in requests)
    if reader_digests is not None and any(r is not None for r in reader_digests):
        raise ReaderMismatch('SCD-ICML does not accept LoRA reader digests')

    limit = getattr(consumer, 'max_batch_size', 1)
    if import_policy not in ('blocktail', 'fullprefix'):
        raise ValueError('unknown file import policy')
    if import_policy == 'fullprefix' and (limit != 1 or len(items) != 1):
        raise UnsupportedProfile('file fullprefix requires configured C1 and one request')
    cancel_after_first = set(cancel_after_first)
    if not items or len(items) > limit:
        raise UnsupportedProfile('batch exceeds the configured request bound')
    if any(type(index) is not int or not 0 <= index < len(items) for index in cancel_after_first):
        raise ValueError('invalid cancellation index')
    settings = dict(generation)
    maximum = settings.get('max_new_tokens', 64)
    if (
        settings.keys() - {'max_new_tokens', 'temperature', 'top_p', 'eos_token_id', 'ignore_eos'}
        or type(maximum) is not int
        or maximum < 1
    ):
        raise ValueError('invalid batch generation settings')
    bound = consumer.core.capability.max_model_len
    if bound is not None and any(len(prompt) + maximum > bound for _, prompt in items):
        raise UnsupportedProfile('batch request exceeds qualified context bound')
    capacity = getattr(consumer, 'kv_capacity_tokens', None)
    block = consumer.core.capability.block_size
    if capacity is not None and sum(((len(prompt) + maximum + block - 1) // block) * block for _, prompt in items) > capacity:
        raise UnsupportedProfile('batch would exceed KV capacity and require unsupported preemption')

    deadline_factory = getattr(consumer, '_batch_deadline_factory', _signal_deadline)
    started = time.perf_counter()
    with consumer._consume_lock:
        if getattr(consumer, '_closed', False):
            raise RuntimeError('consumer is closed; create a new engine')
        engine = consumer.engine.llm_engine
        if engine.has_unfinished_requests():
            raise UnsupportedProfile('batch requires its owned engine to be idle')
        queue_ms = (time.perf_counter() - started) * 1000
        admissions = []
        external_ids = []
        engine_started = False
        safe_release = True
        failure = None
        result = None
        try:
            t = time.perf_counter()
            for artifact, prompt in items:
                external_id = uuid.uuid4().hex
                external_ids.append(external_id)
                policy = {} if import_policy == 'blocktail' else {'import_policy': import_policy}
                admissions.append(None if artifact is None else consumer.core.admit(artifact, prompt, external_id, **policy))
            admission_ms = (time.perf_counter() - t) * 1000
            eos = settings.get('eos_token_id')
            stop = None if eos is None else list(eos) if isinstance(eos, (list, tuple)) else [eos]
            queued = []
            for (_, prompt), admission, external_id in zip(items, admissions, external_ids):
                transfer = {} if admission is None else {'scd_admission': to_data(admission)}
                params = SamplingParams(
                    max_tokens=maximum,
                    temperature=settings.get('temperature', 0),
                    top_p=settings.get('top_p', 1),
                    stop_token_ids=stop,
                    ignore_eos=settings.get('ignore_eos', False),
                    extra_args={
                        'skip_writing_prefix_cache': admission is not None,
                        'kv_transfer_params': transfer,
                    },
                )
                payload = {'prompt_token_ids': list(prompt)}
                if admission is not None:
                    payload['cache_salt'] = 'scd-' + admission.lease.lease_id
                queued.append((external_id, payload, params))

            engine_started = True
            generated_at = time.perf_counter()
            observation = observe_batch(
                engine,
                queued,
                started=generated_at,
                cancel_after_first={external_ids[i] for i in cancel_after_first},
                deadline_factory=deadline_factory,
            )
            with deadline_factory(30):
                consumer.engine.collective_rpc('check_health', timeout=30)

            generations = []
            for (artifact, _), admission, external_id in zip(items, admissions, external_ids):
                item = observation['requests'][external_id]
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
                receipt = consumer.core.store.pop_injection(admission.lease)
                expected = {
                    'artifact_id': artifact,
                    'request_id': admission.lease.request_id,
                    'actual_request_id': item['internal_id'],
                    'lease_id': admission.lease.lease_id,
                    'profile_fingerprint': admission.profile_fingerprint,
                    'request_digest': admission.plan.request_digest,
                    'loaded_layers': consumer.profile.target.num_layers,
                    'loaded_tokens': admission.plan.imported_tokens,
                }
                if import_policy == 'fullprefix':
                    expected.update(
                        import_policy='fullprefix',
                        original_prefix_tokens=admission.plan.original_prefix_tokens,
                        tail_recomputed_tokens=0,
                    )
                if any(receipt.get(key) != value for key, value in expected.items()):
                    raise IdentityMismatch('batch injection receipt differs from owned engine request')
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
                            'request_digest': admission.plan.request_digest,
                            'tail_recomputed_tokens': admission.plan.tail_recomputed_tokens,
                            'backend': 'vllm',
                            'native_fallback': False,
                            'vllm_commit': consumer.core.capability.vllm_commit,
                        },
                    )
                )
            wall_ms = (time.perf_counter() - started) * 1000
            completed = sum(not item['cancelled'] for item in observation['requests'].values())
            result = BatchResult(
                tuple(generations),
                {
                    'batch_size': len(items),
                    'completed_requests': completed,
                    'cancelled_requests': len(items) - completed,
                    'generated_tokens': sum(len(item.token_ids) for item in generations),
                    'consumer_queue_ms': queue_ms,
                    'admission_ms': admission_ms,
                    'generation_ms': observation['wall_ms'],
                    'batch_wall_ms': wall_ms,
                    'max_outputs_per_step': observation['max_outputs_per_step'],
                    'steps_with_multiple_outputs': observation['steps_with_multiple_outputs'],
                    'step_count': observation['step_count'],
                    'latency_scope': 'client_batch_call_including_admission',
                    'imported_requests': sum(a is not None for a in admissions),
                    'native_requests': sum(a is None for a in admissions),
                    'scope': (
                        'owned_import_and_native_batch'
                        if any(a is None for a in admissions)
                        else 'prepared_file_artifacts_to_completed_batch'
                    ),
                },
            )
            return result
        except BaseException as exc:
            failure = exc
            if engine_started:
                consumer._closed = True
                safe_release = False
                try:
                    with deadline_factory(10):
                        engine.engine_core.shutdown(timeout=5)
                    safe_release = True
                except BaseException as shutdown_error:
                    if hasattr(exc, 'add_note'):
                        exc.add_note('Owned engine shutdown unconfirmed; leases retained: ' + repr(shutdown_error))
            raise
        finally:
            errors = []
            if safe_release:
                for admission in admissions:
                    if admission is None:
                        continue
                    try:
                        consumer.core.store.discard_injection(admission.lease)
                        consumer.core.release(admission)
                    except BaseException as exc:
                        errors.append(exc)
            if errors:
                if failure is None:
                    raise RuntimeError('batch lease cleanup failed') from errors[0]
                if hasattr(failure, 'add_note'):
                    failure.add_note('Batch cleanup errors: ' + repr(errors))
            if result is not None:
                result.metrics['batch_wall_ms'] = (time.perf_counter() - started) * 1000
