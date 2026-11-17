"""Request-bound evidence of the vLLM V2 runner's original output-copy barrier."""
from __future__ import annotations

import hashlib
import inspect
from pathlib import Path
import time

from scd_runtime.errors import IdentityMismatch, UnsupportedProfile

PINNED_CORE = '2b1f75c80f79496b313e28362f0856632dfbf5ea'
PINNED_SOURCES = {
    'v1/worker/gpu/async_utils.py': 'af54328bad11b63b09ea8fff7756354c59c58a6713499e7e94b566309331896c',
    'v1/worker/gpu/model_runner.py': '91babd0511dc91a1296c783cb987a83a4a8bacfa39f7fb3fc0d8e194f91e1467',
}
CORE = PINNED_CORE
SOURCES = PINNED_SOURCES


def completion_contract(
    session_id: str,
    *,
    core_commit: str = PINNED_CORE,
    source_sha256: dict[str, str] | None = None,
    max_sampled_outputs: int = 64,
) -> dict:
    return {
        'kind': 'pinned_request_output_copy_event_v1',
        'session_id': session_id,
        'core_commit': core_commit,
        'source_sha256': dict(PINNED_SOURCES if source_sha256 is None else source_sha256),
        'original_get_output_retained': True,
        'actual_copy_event_query_required': True,
        'max_live_requests': 1,
        'max_sampled_outputs': max_sampled_outputs,
    }


def _valid_sha_map(value) -> bool:
    return (
        isinstance(value, dict)
        and bool(value)
        and all(
            isinstance(k, str)
            and isinstance(v, str)
            and len(v) == 64
            and all(ch in '0123456789abcdef' for ch in v)
            for k, v in value.items()
        )
    )


def validate_completion(
    receipt,
    session_id: str,
    request_id: str,
    token_id: int,
    observed_ns: int,
    *,
    expected_contract: dict | None = None,
):
    if (
        type(session_id) is not str or not session_id
        or type(request_id) is not str or not request_id
        or type(token_id) is not int or token_id < 0
        or type(observed_ns) is not int or observed_ns <= 0
    ):
        raise IdentityMismatch('actual first-output completion identity is invalid')
    max_outputs = 64 if expected_contract is None else expected_contract.get('max_sampled_outputs', 64)
    expected = {
        'session_id': session_id,
        'actual_request_id': request_id,
        'first_token_id': token_id,
        'kind': 'pinned_request_output_copy_event_v1',
        'actual_copy_event_complete': True,
        'original_get_output_retained': True,
        'evidence_only_no_GPU_work': True,
    }
    if expected_contract is not None:
        expected['source_sha256'] = expected_contract['source_sha256']
    if (
        type(receipt) is not dict
        or any(receipt.get(k) != v for k, v in expected.items())
        or not _valid_sha_map(receipt.get('source_sha256'))
        or receipt.get('actual_copy_event_complete') is not True
        or receipt.get('original_get_output_retained') is not True
        or receipt.get('evidence_only_no_GPU_work') is not True
        or type(receipt.get('copy_completed_ns')) is not int
        or not 0 < receipt['copy_completed_ns'] <= observed_ns
        or type(receipt.get('sampled_outputs_seen')) is not int
        or not 1 <= receipt['sampled_outputs_seen'] <= max_outputs
    ):
        raise IdentityMismatch('actual first-output GPU completion evidence differs')
    return receipt


def _consumed_output():
    raise IdentityMismatch('output completion consumed more than once or after close')


class FirstOutputCompletion:
    """Keep one first-output record and retain only the original copy barrier."""

    def __init__(
        self,
        session_id: str,
        output_type,
        *,
        clock=time.perf_counter_ns,
        source_sha256: dict[str, str] | None = None,
        max_sampled_outputs: int = 64,
    ):
        self.session_id = session_id
        self.output_type = output_type
        self.clock = clock
        self.source_sha256 = dict(PINNED_SOURCES if source_sha256 is None else source_sha256)
        self.max_sampled_outputs = max_sampled_outputs
        self.record = None
        self.closed = False

    def wrap(self, output):
        if self.closed or type(output) is not self.output_type:
            raise UnsupportedProfile('first-output proof requires the V2 async output type')
        original = output.get_output
        if getattr(original, '__self__', None) is not output or 'get_output' in output.__dict__:
            raise UnsupportedProfile('original unmodified async output method required')
        holder = [output, original]
        used = False

        def get_output():
            nonlocal used
            if used:
                _consumed_output()
            used = True
            actual, bound = holder
            try:
                if self.closed:
                    _consumed_output()
                return self._record_output(actual, bound)
            finally:
                actual.get_output = _consumed_output
                holder.clear()

        output.get_output = get_output
        return output

    def _record_output(self, actual, bound):
        result = bound()  # Retains the real upstream copy_event.synchronize().
        if actual.copy_event.query() is not True:
            raise IdentityMismatch('actual output copy event is not complete')
        completed = self.clock()
        ids = result.req_ids
        values = result.sampled_token_ids
        if (
            type(ids) is not list or len(ids) != 1 or type(ids[0]) is not str or not 0 < len(ids[0]) <= 256
            or type(values) is not list or len(values) != 1 or type(values[0]) is not list or len(values[0]) != 1
            or type(values[0][0]) is not int or values[0][0] < 0
        ):
            raise UnsupportedProfile('first-output proof requires one request and one CPU integer token')
        if self.record is None:
            self.record = {
                'actual_request_id': ids[0],
                'first_token_id': values[0][0],
                'copy_completed_ns': completed,
                'sampled_outputs_seen': 0,
            }
        if self.record['actual_request_id'] != ids[0] or self.record['sampled_outputs_seen'] >= self.max_sampled_outputs:
            raise IdentityMismatch('undrained or overlong first-output completion state')
        self.record['sampled_outputs_seen'] += 1
        return result

    def take(self, request_id: str, token_id: int, observed_ns: int):
        if self.closed or self.record is None:
            raise IdentityMismatch('actual first-output evidence missing')
        receipt = {
            **self.record,
            'session_id': self.session_id,
            'kind': 'pinned_request_output_copy_event_v1',
            'actual_copy_event_complete': True,
            'original_get_output_retained': True,
            'source_sha256': dict(self.source_sha256),
            'evidence_only_no_GPU_work': True,
        }
        validate_completion(
            receipt,
            self.session_id,
            request_id,
            token_id,
            observed_ns,
            expected_contract={
                'source_sha256': self.source_sha256,
                'max_sampled_outputs': self.max_sampled_outputs,
            },
        )
        self.record = None
        return receipt

    def close(self):
        self.closed = True
        self.record = None


class InstalledOutputCompletion:
    def __init__(self, runner, session_id: str, *, strict_source_sha: bool = False, max_sampled_outputs: int = 64):
        from vllm import __file__ as origin
        from vllm.v1.worker.gpu.async_utils import AsyncOutput
        from vllm.v1.worker.gpu.model_runner import GPUModelRunner
        from .capability import installed_commit

        commit = installed_commit()
        if type(runner) is not GPUModelRunner:
            raise UnsupportedProfile('first-output proof requires the V2 GPUModelRunner')
        if strict_source_sha and commit != PINNED_CORE:
            raise UnsupportedProfile('strict output completion requires pinned vLLM commit')

        root = Path(origin).resolve().parent
        observed_sources = {}
        for cls, relative in (
            (AsyncOutput, 'v1/worker/gpu/async_utils.py'),
            (GPUModelRunner, 'v1/worker/gpu/model_runner.py'),
        ):
            actual = Path(inspect.getfile(cls)).resolve()
            expected = root / relative
            if actual != expected or actual.stat().st_size > 2 * 1024**2:
                raise IdentityMismatch('actual first-output callchain source path differs')
            digest = hashlib.sha256(actual.read_bytes()).hexdigest()
            if strict_source_sha and digest != PINNED_SOURCES[relative]:
                raise IdentityMismatch('actual first-output callchain source SHA-256 differs')
            observed_sources[relative] = digest

        self.runner = runner
        self.store = FirstOutputCompletion(
            session_id,
            AsyncOutput,
            source_sha256=observed_sources,
            max_sampled_outputs=max_sampled_outputs,
        )
        self.had_attribute = 'sample_tokens' in runner.__dict__
        self.attribute = runner.__dict__.get('sample_tokens')
        original = runner.sample_tokens

        def sample_tokens(*args, **kwargs):
            output = original(*args, **kwargs)
            return None if output is None else self.store.wrap(output)

        self.wrapper = sample_tokens
        runner.sample_tokens = sample_tokens
        self.contract = completion_contract(
            session_id,
            core_commit=commit,
            source_sha256=observed_sources,
            max_sampled_outputs=max_sampled_outputs,
        )

    def take(self, request_id: str, token_id: int, observed_ns: int):
        if self.runner.__dict__.get('sample_tokens') is not self.wrapper:
            raise IdentityMismatch('owned output-copy completion wrapper replaced')
        return self.store.take(request_id, token_id, observed_ns)

    def require_idle(self):
        if (
            self.runner.__dict__.get('sample_tokens') is not self.wrapper
            or self.store.closed
            or self.store.record is not None
        ):
            raise IdentityMismatch('output completion state is not idle or its wrapper changed')

    def close(self):
        self.require_idle()
        if self.had_attribute:
            self.runner.sample_tokens = self.attribute
        else:
            delattr(self.runner, 'sample_tokens')
        self.store.close()
