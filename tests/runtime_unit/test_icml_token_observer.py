"""The ICML protocol stop must drain without generating a hidden next token."""
from contextlib import nullcontext
import time
from types import SimpleNamespace

import pytest

from scd_runtime.errors import IdentityMismatch, UnsupportedProfile
from scd_runtime.resident_runner import GPUResidentConsumer


class Engine:
    def __init__(self, steps, drain=True):
        self.steps = list(steps)
        self.active = False
        self.calls = 0
        self.aborts = []
        self.added = 0
        self.drain = drain

    def has_unfinished_requests(self):
        return self.active

    def add_request(self, identifier, prompt, params):
        self.active = True
        self.added += 1
        self.identifier = identifier
        return identifier

    def step(self):
        self.calls += 1
        values, finished, reason = self.steps.pop(0)
        if finished:
            self.active = False
        return [
            SimpleNamespace(
                request_id=self.identifier,
                finished=finished,
                outputs=[SimpleNamespace(token_ids=values, finish_reason=reason)],
            )
        ]

    def abort_request(self, identifiers, internal):
        self.aborts.append((identifiers, internal))
        if self.drain:
            self.active = False


def consumer(engine):
    obj = GPUResidentConsumer.__new__(GPUResidentConsumer)
    obj.engine = SimpleNamespace(llm_engine=engine)
    obj.profile = SimpleNamespace(method_id='icml_reuse_patch')
    obj.max_batch_size = 1
    obj.request_features = SimpleNamespace(async_scheduling=False, async_loading=False, routing=False)
    obj._deadline_factory = lambda seconds: nullcontext()
    obj.syncs = 0
    obj.health = 0
    obj.failures = []

    def sync():
        obj.syncs += 1

    def rpc(name, **kwargs):
        assert name == 'check_health'
        obj.health += 1

    def fail(error):
        obj.failures.append(error)
        engine.active = False

    obj.synchronize = sync
    obj._rpc = rpc
    obj._fail_closed = fail
    return obj


def observe(obj, callback=None, receipt=None, **kwargs):
    args = {} if callback is None else {'token_observer': callback, 'protocol_receipt': receipt}
    return obj._observe((1, 3), SimpleNamespace(), 'owned', time.perf_counter(), **args, **kwargs)


def test_online_text_boundary_aborts_before_another_decode():
    engine = Engine([([13], False, None), ([13, 8], False, None)])
    obj = consumer(engine)
    receipt = {}
    seen = []

    def callback(ids):
        seen.append(ids)
        return 'text_boundary'

    ids, first, request, cancelled = observe(obj, callback, receipt)
    assert ids == (13,) and first >= 0 and request == 'owned' and not cancelled
    assert engine.calls == 1 and engine.aborts == [(['owned'], True)] and not engine.active
    assert obj.syncs == obj.health == 1 and seen == [(13,)] and not obj.failures
    assert receipt['protocol_stop_reason'] == 'text_boundary' and receipt['protocol_request_drained']
    assert receipt['protocol_abort_requested'] and receipt['protocol_abort_confirmed']
    assert receipt['protocol_completion_ms'] >= first and receipt['protocol_stop_cleanup_ms'] >= 0


def test_natural_first_eos_is_valid_without_aborting_an_already_finished_request():
    engine = Engine([([2], True, 'stop')])
    obj = consumer(engine)
    receipt = {}
    ids, first, _, cancelled = observe(obj, lambda ids: 'stop_token', receipt)
    assert ids == (2,) and first >= 0 and not cancelled
    assert engine.calls == 1 and engine.aborts == [] and obj.health == 0
    assert receipt['protocol_stop_reason'] == 'stop_token' and not receipt['protocol_abort_requested']
    assert receipt['protocol_request_drained']


def test_protocol_maximum_is_not_diagnostic_cancellation():
    engine = Engine([([7], False, None), ([7, 8], True, 'length')])
    obj = consumer(engine)
    receipt = {}
    ids, _, _, cancelled = observe(obj, lambda ids: 'max_new_tokens' if len(ids) == 2 else None, receipt)
    assert ids == (7, 8) and not cancelled and not engine.aborts
    assert obj.syncs == 1 and receipt['protocol_observed_token_count'] == 2


@pytest.mark.parametrize('steps', [[([7, 8], False, None)], [([7], False, None), ([8, 9], False, None)]])
def test_multi_token_step_or_changed_history_fails_closed(steps):
    engine = Engine(steps)
    obj = consumer(engine)
    receipt = {}
    with pytest.raises(IdentityMismatch, match='one new cumulative token'):
        observe(obj, lambda ids: None, receipt)
    assert obj.failures and receipt == {}


def test_unrecognized_protocol_reason_is_not_a_success():
    engine = Engine([([7], False, None)])
    obj = consumer(engine)
    receipt = {}
    with pytest.raises(IdentityMismatch, match='unrecognized'):
        observe(obj, lambda ids: 'invented_stop', receipt)
    assert obj.failures and receipt == {}


def test_unconfirmed_abort_never_becomes_a_success_receipt():
    engine = Engine([([7], False, None)], drain=False)
    obj = consumer(engine)
    receipt = {}
    with pytest.raises(RuntimeError, match='did not drain'):
        observe(obj, lambda ids: 'context_span_boundary', receipt)
    assert obj.failures and engine.calls == 1 and receipt == {}


def test_engine_cannot_silently_finish_outside_the_original_protocol():
    engine = Engine([([7], True, 'stop')])
    obj = consumer(engine)
    receipt = {}
    with pytest.raises(IdentityMismatch, match='without an acknowledged'):
        observe(obj, lambda ids: None, receipt)
    assert obj.failures and receipt == {}


@pytest.mark.parametrize('change', ['other_method', 'routing', 'async', 'missing_receipt', 'diagnostic_cancel'])
def test_unverified_policy_is_rejected_before_adding_a_request(change):
    engine = Engine([([7], False, None)])
    obj = consumer(engine)
    receipt = {}
    kwargs = {}
    if change == 'other_method':
        obj.profile.method_id = 'other_method'
    elif change == 'routing':
        obj.request_features.routing = True
    elif change == 'async':
        obj.request_features.async_scheduling = True
    elif change == 'missing_receipt':
        receipt = None
    else:
        kwargs['cancel_after_first'] = True
    with pytest.raises(UnsupportedProfile):
        observe(obj, lambda ids: 'text_boundary', receipt, **kwargs)
    assert engine.added == engine.calls == 0


def completion_consumer(engine, change=None):
    from scd_vllm.output_completion import SOURCES

    obj = consumer(engine)
    obj.output_completion_enabled = True
    obj.session_id = 'session'
    obj.proof_calls = []

    def rpc(name, **kw):
        if name == 'check_health':
            obj.health += 1
            return
        assert name == 'scd_gpu_output_completion' and not engine.active
        obj.proof_calls.append(kw)
        proof = {
            'session_id': 'session',
            'actual_request_id': kw['actual_request_id'],
            'first_token_id': kw['first_token_id'],
            'copy_completed_ns': kw['first_observed_ns'] - 1,
            'kind': 'pinned_request_output_copy_event_v1',
            'actual_copy_event_complete': True,
            'original_get_output_retained': True,
            'source_sha256': dict(SOURCES),
            'evidence_only_no_GPU_work': True,
            'sampled_outputs_seen': engine.calls,
        }
        if change is not None:
            change(proof, kw)
        return proof

    obj._rpc = rpc
    return obj


def test_copy_complete_path_has_no_late_global_barrier_and_preserves_online_stop():
    engine = Engine([([7], False, None), ([7, 8], False, None)])
    obj = completion_consumer(engine)
    receipt = {}

    def callback(ids):
        assert obj.proof_calls == [] and obj.syncs == 0
        return 'text_boundary'

    ids, first, actual, cancelled = observe(obj, callback, receipt)
    assert ids == (7,) and actual == 'owned' and not cancelled and first >= 0
    assert engine.calls == obj.health == 1 and obj.syncs == 0 and len(obj.proof_calls) == 1
    proof = receipt['first_token_completion']
    assert proof['copy_completed_ns'] <= proof['first_observed_ns'] and receipt['protocol_request_drained']


@pytest.mark.parametrize(
    'change',
    [
        lambda p, k: p.clear(),
        lambda p, k: p.update(actual_request_id='foreign'),
        lambda p, k: p.update(first_token_id=9),
        lambda p, k: p.update(copy_completed_ns=k['first_observed_ns'] + 1),
        lambda p, k: p.update(source_sha256={}),
        lambda p, k: p.update(actual_copy_event_complete=False),
        lambda p, k: p.update(evidence_only_no_GPU_work=False),
    ],
)
def test_invalid_copy_evidence_rejects_entire_request_and_its_timing(change):
    engine = Engine([([7], True, 'length')])
    obj = completion_consumer(engine, change)
    receipt = {}
    with pytest.raises(IdentityMismatch, match='completion evidence'):
        observe(obj, lambda ids: 'max_new_tokens', receipt)
    assert obj.failures and receipt == {} and obj.syncs == 0
