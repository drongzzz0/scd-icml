"""Real copy-event evidence is mandatory; a timestamp alone never authorizes TTFT."""
import gc
import threading
from types import SimpleNamespace
import weakref

import pytest

from scd_runtime.errors import IdentityMismatch, UnsupportedProfile
from scd_vllm.output_completion import FirstOutputCompletion, InstalledOutputCompletion, validate_completion
from scd_vllm.resident_worker import GPUResidentWorkerExtension


class Output:
    def __init__(self, request='request', token=7, complete=True):
        self.result = SimpleNamespace(req_ids=[request], sampled_token_ids=[[token]])
        self.calls = 0
        self.copy_event = SimpleNamespace(query=lambda: complete)

    def get_output(self):
        self.calls += 1
        return self.result


def test_original_barrier_retained_result_identity_and_first_request_record():
    store = FirstOutputCompletion('session', Output, clock=lambda: 50)
    output = Output()
    assert store.wrap(output) is output and output.get_output() is output.result and output.calls == 1
    later = Output(token=9)
    store.wrap(later).get_output()
    proof = store.take('request', 7, 60)
    assert proof['copy_completed_ns'] == 50 and proof['sampled_outputs_seen'] == 2
    assert proof['actual_copy_event_complete'] and proof['evidence_only_no_GPU_work'] and store.record is None
    assert validate_completion(proof, 'session', 'request', 7, 60) is proof


@pytest.mark.parametrize('complete', [False, None, 1])
def test_actual_event_must_be_complete_not_merely_an_output_object(complete):
    store = FirstOutputCompletion('s', Output, clock=lambda: 50)
    output = Output(complete=complete)
    with pytest.raises(IdentityMismatch, match='not complete'):
        store.wrap(output).get_output()
    assert store.record is None and output.calls == 1


def test_callback_keeps_pending_tensors_alive_then_breaks_its_completed_cycle():
    enabled = gc.isenabled()
    gc.disable()
    try:
        store = FirstOutputCompletion('s', Output, clock=lambda: 10)
        out = Output()
        reference = weakref.ref(out)
        callback = store.wrap(out).get_output
        del out
        assert reference() is not None
        assert callback().sampled_token_ids == [[7]]
        assert reference() is None
        with pytest.raises(IdentityMismatch, match='more than once'):
            callback()
    finally:
        if enabled:
            gc.enable()


def test_bounded_state_and_close_prevent_late_callbacks_or_reuse():
    store = FirstOutputCompletion('s', Output, clock=lambda: 10)
    for _ in range(64):
        store.wrap(Output()).get_output()
    with pytest.raises(IdentityMismatch, match='overlong'):
        store.wrap(Output()).get_output()
    out = store.wrap(Output())
    store.close()
    with pytest.raises(IdentityMismatch, match='after close'):
        out.get_output()
    with pytest.raises(UnsupportedProfile):
        store.wrap(Output())
    assert store.record is None


@pytest.mark.parametrize('close', [False, True])
def test_request_cleanup_keeps_guard_but_engine_close_restores_runner(close):
    guard = InstalledOutputCompletion.__new__(InstalledOutputCompletion)
    guard.store = FirstOutputCompletion('s', Output)
    guard.runner = SimpleNamespace()
    guard.wrapper = lambda: None
    guard.runner.sample_tokens = guard.wrapper
    guard.had_attribute = False
    guard.attribute = None
    registry = SimpleNamespace(lock=threading.RLock(), clear=lambda: None)
    state = {'registry': registry, 'source': None, 'output_completion': guard}
    worker = SimpleNamespace(_scd_gpu_context=lambda session: state, scd_gpu_status=lambda **kw: {'idle': True})
    result = GPUResidentWorkerExtension.scd_gpu_clear(worker, session_id='s', close_output_completion=close)
    assert result == {'idle': True}
    if close:
        assert state['output_completion'] is None and guard.store.closed and not hasattr(guard.runner, 'sample_tokens')
    else:
        assert state['output_completion'] is guard and guard.runner.sample_tokens is guard.wrapper
        assert not guard.store.closed
