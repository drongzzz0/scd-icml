"""Bounded, single-owner frontend driving several fixed-reader requests."""
from dataclasses import dataclass
import time
from typing import Mapping

from .contracts.types import GenerationResult
from .errors import IdentityMismatch, UnsupportedProfile


@dataclass(frozen=True)
class BatchResult:
    generations: tuple[GenerationResult, ...]
    metrics: Mapping


def observe_batch(engine, requests, *, started, timeout=120, cancel_after_first=(),
                  deadline_factory):
    """Queue the complete group before background scheduling can begin.

    The caller owns failure cleanup: an exception can leave a worker in flight,
    so its engine must stop before leases are reclaimed. A failed pause, add,
    ID validation or wake is deliberately not followed by another wake here.
    """
    requests = tuple(requests)
    if engine.has_unfinished_requests():
        raise UnsupportedProfile('batch requires an exclusively owned idle frontend')
    external_ids = [request[0] for request in requests]
    cancel_after_first = set(cancel_after_first)
    if (not requests or any(not isinstance(value, str) or not value for value in external_ids)
            or len(set(external_ids)) != len(requests)
            or not cancel_after_first.issubset(external_ids)):
        raise ValueError('batch request IDs or cancellation targets are invalid')
    deadline = started + timeout

    def remaining():
        seconds = deadline - time.perf_counter()
        if seconds <= 0:
            raise TimeoutError('batch frontend deadline reached')
        return seconds

    identities = {}
    results = {}
    steps = max_outputs = multiple_steps = 0
    # SyncMPClient sends each ADD immediately to a background engine. Merely
    # delaying frontend step() does not stop that engine from running a partial
    # group. Level zero keeps GPU memory/cache resident and queues new requests.
    with deadline_factory(remaining()):
        engine.sleep(level=0, mode='keep')
    remaining()
    for external_id, prompt, params in requests:
        with deadline_factory(remaining()):
            internal_id = engine.add_request(external_id, prompt, params)
        remaining()
        if (not isinstance(internal_id, str) or not internal_id or internal_id in identities
                or (internal_id in external_ids and internal_id != external_id)):
            raise IdentityMismatch('engine returned an invalid or duplicate internal ID')
        identities[internal_id] = external_id
        identities[external_id] = external_id
        results[external_id] = {'internal_id': internal_id, 'token_ids': (),
            'first_token_ms': None, 'completion_ms': None, 'cancelled': False, 'finished': False}
    # Each operation gets its own remaining deadline: POSIX alarm contexts must
    # not nest. Validate every returned ID before admitting execution of any.
    with deadline_factory(remaining()):
        fully_awake = engine.wake_up(tags=['scheduling'])
    remaining()
    if fully_awake is not True:
        raise RuntimeError('owned batch scheduler did not confirm wake-up')
    while engine.has_unfinished_requests():
        with deadline_factory(remaining()):
            outputs = engine.step()
        remaining()
        steps += 1
        seen = set()
        for output in outputs:
            external_id = identities.get(output.request_id)
            if external_id is None:
                raise IdentityMismatch('engine returned an output from another batch')
            seen.add(external_id)
            item = results[external_id]
            if item['cancelled']:
                continue  # Do not reclassify a late cancelled output as success.
            if output.outputs and output.outputs[0].token_ids:
                if item['first_token_ms'] is None:
                    item['first_token_ms'] = (time.perf_counter() - started) * 1000
                item['token_ids'] = tuple(output.outputs[0].token_ids)
            if output.finished:
                if not output.outputs or output.outputs[0].finish_reason in ('abort', 'error'):
                    raise RuntimeError('one batch request failed during model execution')
                item['finished'] = True
                item['completion_ms'] = (time.perf_counter() - started) * 1000
        max_outputs = max(max_outputs, len(seen))
        multiple_steps += len(seen) > 1
        for external_id in cancel_after_first:
            item = results[external_id]
            if item['first_token_ms'] is None or item['cancelled']:
                continue
            if item['finished']:
                raise RuntimeError('request naturally finished before cancellation probe')
            with deadline_factory(remaining()):
                engine.abort_request([item['internal_id']], internal=True)
            item['cancelled'] = True
            item['completion_ms'] = (time.perf_counter() - started) * 1000
    if any(not item['token_ids'] or not (item['finished'] or item['cancelled'])
           for item in results.values()):
        raise RuntimeError('batch finished without complete owned outputs')
    return {'requests': results, 'step_count': steps, 'max_outputs_per_step': max_outputs,
            'steps_with_multiple_outputs': multiple_steps,
            'wall_ms': (time.perf_counter() - started) * 1000}
