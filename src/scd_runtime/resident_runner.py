"""Same-worker GPU resident SCD-ICML consumer for vLLM V1."""
from __future__ import annotations

from contextlib import contextmanager
import math
import signal
import threading
import time
import uuid

from .contracts.capability import check_capability
from .contracts.identity import profile_fingerprint, sha256
from .contracts.prefix import check_fullprefix_scope
from .contracts.types import EngineCapability, GenerationResult, to_data, tokens
from .errors import IdentityMismatch, ReaderMismatch, UnsupportedProfile
from .runner import RequestFeatures, RunReceipt, request_prefix_cache_policy_supported


@contextmanager
def _signal_deadline(seconds: float):
    """Bound blocking frontend calls; uses ITIMER_REAL on POSIX main thread, monotonic check otherwise."""
    if not math.isfinite(seconds) or seconds <= 0:
        raise TimeoutError('resident request deadline reached')
    use_posix_timer = (
        hasattr(signal, 'setitimer')
        and hasattr(signal, 'SIGALRM')
        and threading.current_thread() is threading.main_thread()
        and signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)
    )
    if not use_posix_timer:
        deadline = time.perf_counter() + seconds
        yield
        if time.perf_counter() >= deadline:
            raise TimeoutError('resident frontend deadline reached')
        return

    previous = signal.getsignal(signal.SIGALRM)

    def expired(signum, frame):
        raise TimeoutError('resident frontend deadline reached')

    signal.signal(signal.SIGALRM, expired)
    try:
        signal.setitimer(signal.ITIMER_REAL, seconds)
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


class GPUResidentConsumer:
    _deadline_factory = staticmethod(_signal_deadline)

    def __init__(
        self,
        profile,
        assets_root,
        engine_config,
        reader_package=None,
        *,
        enable_diagnostics: bool = False,
        reader_packages=(),
        enable_output_completion: bool = False,
    ):
        if profile.method_id != 'icml_reuse_patch' or profile.reader.kind != 'none':
            raise UnsupportedProfile('GPUResidentConsumer supports SCD-ICML (icml_reuse_patch, reader="none")')
        if reader_package is not None or reader_packages:
            raise ReaderMismatch('unexpected LoRA reader package for SCD-ICML')

        from scd_vllm.capability import ensure_sampler_jit_safe, installed_commit
        from .providers.hf.loading import verify_model

        config = dict(engine_config)
        strict_pin = bool(config.pop('strict_commit_pin', False))
        memo_entries = int(config.pop('memoize_prefix_entries', 0))
        memo_bytes = int(config.pop('memoize_prefix_bytes', 1024 * 1024**2))
        resident_capacity_bytes = config.pop('resident_capacity_bytes', None)
        if resident_capacity_bytes is not None and (type(resident_capacity_bytes) is not int or resident_capacity_bytes <= 0):
            raise UnsupportedProfile('resident_capacity_bytes must be a positive integer')
        source_attn_impl = config.pop('source_attn_implementation', None)
        source_fast_mask = bool(config.pop('source_fast_causal_mask', False))
        self.stream_ordered_prepare = bool(config.pop('stream_ordered_prepare', True))
        self._use_fused_batch_rpc = True

        commit = installed_commit(strict_commit_pin=strict_pin)
        no_store = request_prefix_cache_policy_supported()
        self.request_features = RequestFeatures.from_config(
            profile,
            config,
            (),
            request_policy_supported=no_store,
        )
        prefix_cache = config.get('enable_prefix_caching', False)
        enforce_eager = config.get('enforce_eager', False)
        if type(enforce_eager) is not bool:
            raise UnsupportedProfile('enforce_eager must be boolean')
        if 'kv_cache_memory_bytes' in config and (
            type(config['kv_cache_memory_bytes']) is not int or config['kv_cache_memory_bytes'] <= 0
        ):
            raise UnsupportedProfile('kv_cache_memory_bytes must be a positive integer')

        fixed = {
            'dtype': 'bfloat16',
            'tensor_parallel_size': 1,
            'pipeline_parallel_size': 1,
            'enable_prefix_caching': prefix_cache,
            'enable_chunked_prefill': False,
            'disable_hybrid_kv_cache_manager': True,
            'enforce_eager': enforce_eager,
            'trust_remote_code': False,
            'disable_log_stats': False,
        }
        self.max_batch_size = config.get('max_num_seqs', 1)
        if type(enable_output_completion) is not bool:
            raise ValueError('output completion opt-in must be boolean')
        self.output_completion_enabled = enable_output_completion
        self.output_completion_contract = None
        if enable_output_completion and (
            self.max_batch_size != 1
            or self.request_features.async_scheduling is not False
            or self.request_features.async_loading is not False
            or prefix_cache
        ):
            raise UnsupportedProfile('output completion requires synchronous serial ICML observer mode')
        if type(self.max_batch_size) is not int or not 1 <= self.max_batch_size <= 64:
            raise UnsupportedProfile('resident max_num_seqs must be between 1 and 64')
        config['max_num_seqs'] = self.max_batch_size

        allowed = set(fixed) | {
            'model',
            'model_manifest',
            'block_size',
            'attention_config',
            'compilation_config',
            'kernel_config',
            'max_model_len',
            'gpu_memory_utilization',
            'kv_cache_memory_bytes',
            'max_num_batched_tokens',
            'max_num_seqs',
            'seed',
        }
        if config.keys() - allowed or set(config.get('attention_config', {})) - {'backend'}:
            raise UnsupportedProfile('unbound resident engine override')
        if any(key in config and config[key] != value for key, value in fixed.items()):
            raise UnsupportedProfile('resident engine overrides conflict with ICML serial/batch safety settings')

        root = config.get('model')
        manifest = config.pop('model_manifest', None)
        if not root or not manifest:
            raise UnsupportedProfile('verified target model required')
        verify_model(root, manifest, profile.target)

        block = config.get('block_size')
        bound = config.get('max_model_len')
        if type(block) is not int or type(bound) is not int or bound < 2:
            raise UnsupportedProfile('explicit block_size and max_model_len required')

        cap = EngineCapability(
            commit,
            config.get('attention_config', {}).get('backend'),
            'bfloat16',
            1,
            1,
            block,
            1,
            profile.target.attention,
            False,
            False,
            prefix_cache,
            bound,
            no_store,
        )
        check_capability(profile, cap, strict_commit_pin=strict_pin)
        config.update(fixed)

        self.profile = profile
        self.capability = cap
        self.session_id = uuid.uuid4().hex
        self.max_model_len = bound
        self.enforce_eager = enforce_eager
        self._lock = threading.RLock()
        self._closed = False
        self._shutdown_confirmed = False
        self.engine = None
        self.has_fixed_reader = False

        extra_cfg = {
            'scd_profile': to_data(profile),
            'resident_session': self.session_id,
            'resident_diagnostics': bool(enable_diagnostics),
            'strict_commit_pin': strict_pin,
            'memoize_prefix_entries': memo_entries,
            'memoize_prefix_bytes': memo_bytes,
            'source_fast_causal_mask': source_fast_mask,
            **self.request_features.extra_config(),
        }
        if resident_capacity_bytes is not None:
            extra_cfg['resident_capacity_bytes'] = resident_capacity_bytes
        if source_attn_impl is not None:
            extra_cfg['source_attn_implementation'] = source_attn_impl
        if enable_output_completion:
            extra_cfg['request_output_completion'] = True

        config['worker_extension_cls'] = 'scd_vllm.resident_worker.GPUResidentWorkerExtension'
        config['kv_transfer_config'] = {
            'kv_connector': 'SCDGPUConnectorV1',
            'kv_connector_module_path': 'scd_vllm.resident_connector',
            'kv_role': 'kv_consumer',
            'kv_load_failure_policy': 'fail',
            'kv_connector_extra_config': extra_cfg,
        }
        if self.request_features.async_scheduling is not None:
            config['async_scheduling'] = self.request_features.async_scheduling

        ensure_sampler_jit_safe()
        from vllm import LLM

        self.engine = LLM(**config)
        try:
            self.setup_receipt = self._rpc(
                'scd_gpu_configure',
                session_id=self.session_id,
                profile_data=to_data(profile),
                assets_root=str(assets_root),
            )
            if self.setup_receipt['profile_fingerprint'] != profile_fingerprint(profile):
                raise IdentityMismatch('configured worker differs')
            if enable_output_completion:
                contract = self.setup_receipt.get('output_completion_contract')
                if not isinstance(contract, dict) or contract.get('session_id') != self.session_id:
                    raise IdentityMismatch('configured output-copy completion guard differs')
                self.output_completion_contract = contract
            self.target_vocab_size = self.setup_receipt['target_vocab_size']
            self.cache_registration = self.setup_receipt['registered_cache']
            self.kv_capacity_tokens = self.cache_registration['capacity_tokens']
        except BaseException as exc:
            self._fail_closed(exc)
            raise

    def _fail_closed(self, error):
        self._closed = True
        self._shutdown_confirmed = False
        try:
            with self._deadline_factory(10):
                self.engine.llm_engine.engine_core.shutdown(timeout=5)
            self._shutdown_confirmed = True
        except BaseException as cleanup_error:
            if hasattr(error, 'add_note'):
                error.add_note('Owned engine shutdown also failed: ' + repr(cleanup_error))

    @contextmanager
    def _request_scope(self, started: float):
        previous = getattr(self, '_request_deadline', None)
        self._request_deadline = min(previous, started + 120) if previous is not None else started + 120
        try:
            yield
        finally:
            self._request_deadline = previous

    def _remaining(self) -> float:
        deadline = getattr(self, '_request_deadline', None)
        return 120.0 if deadline is None else min(120.0, deadline - time.perf_counter())

    def _rpc(self, method: str, **kwargs):
        if self._closed:
            raise RuntimeError('resident client is closed; recreate engine')
        try:
            seconds = self._remaining()
            deadline = time.perf_counter() + seconds
            with self._deadline_factory(seconds):
                results = self.engine.collective_rpc(method, timeout=seconds, kwargs=kwargs)
            if time.perf_counter() >= deadline:
                raise TimeoutError('resident RPC exceeded its deadline')
            if not isinstance(results, (tuple, list)) or len(results) != 1:
                raise RuntimeError('one worker response required')
            return results[0]
        except BaseException as exc:
            self._fail_closed(exc)
            raise

    def synchronize(self):
        with self._lock:
            try:
                receipt = self._rpc('scd_gpu_synchronize', session_id=self.session_id)
                if (
                    not isinstance(receipt, dict)
                    or receipt.get('worker_cuda_synchronized') is not True
                    or not str(receipt.get('device', '')).startswith('cuda')
                    or receipt.get('session_id') != self.session_id
                ):
                    raise IdentityMismatch('actual resident synchronization acknowledgement differs')
                return receipt
            except BaseException as exc:
                if not self._closed:
                    self._fail_closed(exc)
                raise

    def status(self):
        return self._rpc('scd_gpu_status', session_id=self.session_id)

    def clear_memo(self):
        with self._lock:
            return self._rpc('scd_gpu_clear_memo', session_id=self.session_id)

    def _source_kwargs(self, prefix, full_prompt_ids):
        if full_prompt_ids is None:
            return {}
        prompt = tokens(full_prompt_ids)
        if (
            self.profile.method_id != 'icml_reuse_patch'
            or len(prompt) != len(prefix.token_ids) + 1
            or prompt[:-1] != prefix.token_ids
            or len(prompt) > self.max_model_len
            or max(prompt) >= self.target_vocab_size
        ):
            raise IdentityMismatch('full source prompt differs from the original ICML request')
        return {'full_prompt_ids': list(prompt)}

    @staticmethod
    def _check_source_receipt(result, kwargs):
        if kwargs:
            prompt = tuple(kwargs['full_prompt_ids'])
            if (
                result.get('source_full_prompt_prefill') is not True
                or result.get('source_input_tokens') != len(prompt)
                or result.get('source_prompt_digest') != sha256(prompt)
            ):
                raise IdentityMismatch('actual full-prompt source capture acknowledgement differs')

    def capture_source(self, prefix, *, full_prompt_ids=None):
        with self._lock:
            try:
                kwargs = self._source_kwargs(prefix, full_prompt_ids)
                started = time.perf_counter()
                result = self._rpc('scd_gpu_capture', session_id=self.session_id, prefix_data=to_data(prefix), **kwargs)
                if result['prefix_digest'] != sha256(prefix):
                    raise IdentityMismatch('H capture prefix differs')
                self._check_source_receipt(result, kwargs)
                return {**result, 'client_capture_ms': (time.perf_counter() - started) * 1000}
            except BaseException as exc:
                if not self._closed:
                    self._fail_closed(exc)
                raise

    def prepare(self, prefix, *, mode='E', source_handle=None, full_prompt_ids=None):
        with self._lock:
            try:
                kwargs = self._source_kwargs(prefix, full_prompt_ids)
                started = time.perf_counter()
                result = self._rpc(
                    'scd_gpu_prepare',
                    session_id=self.session_id,
                    prefix_data=to_data(prefix),
                    mode=mode,
                    source_handle=source_handle,
                    **kwargs,
                )
                if (
                    result['session_id'] != self.session_id
                    or result['profile_fingerprint'] != profile_fingerprint(self.profile)
                    or result['prefix_digest'] != sha256(prefix)
                ):
                    raise IdentityMismatch('prepared snapshot differs')
                self._check_source_receipt(result, kwargs)
                return {**result, 'prepare_rpc_ms': (time.perf_counter() - started) * 1000}
            except BaseException as exc:
                if not self._closed:
                    self._fail_closed(exc)
                raise

    def diagnostic_digest(self, prepared):
        return self._rpc('scd_gpu_diagnostic_digest', session_id=self.session_id, artifact_id=prepared['artifact_id'])

    def cancel_prepared(self, prepared):
        with self._lock:
            return self._rpc('scd_gpu_release', session_id=self.session_id, artifact_id=prepared['artifact_id'])

    def _generation(self, prompt_ids, generation, admission=None):
        from vllm import SamplingParams
        from vllm.sampling_params import RequestOutputKind

        prompt = tokens(prompt_ids)
        maximum = generation.get('max_new_tokens', 64)
        if getattr(self, 'output_completion_enabled', False) and (type(maximum) is not int or not 1 <= maximum <= 64):
            raise UnsupportedProfile('output completion requires max_new_tokens between 1 and 64')
        if generation.keys() - {'max_new_tokens', 'temperature', 'eos_token_id', 'ignore_eos'} or generation.get('temperature', 0) != 0:
            raise ValueError('resident consumer supports explicit greedy generation only')
        if type(maximum) is not int or maximum < 1 or len(prompt) + maximum > self.max_model_len or max(prompt) >= self.target_vocab_size:
            raise IdentityMismatch('request exceeds target token/context bounds')
        eos = generation.get('eos_token_id')
        stop = None if eos is None else (list(eos) if isinstance(eos, (list, tuple)) else [eos])
        extra = None if admission is None else {'kv_transfer_params': {'scd_gpu_admission': admission}}
        return SamplingParams(
            max_tokens=maximum,
            temperature=0.0,
            stop_token_ids=stop,
            ignore_eos=generation.get('ignore_eos', False),
            output_kind=RequestOutputKind.CUMULATIVE,
            extra_args=extra,
        )

    def _observe(
        self,
        prompt_ids,
        params,
        external_id,
        started,
        *,
        cancel_after_first: bool = False,
        token_observer=None,
        protocol_receipt=None,
        deferred_completion: dict | None = None,
    ):
        engine = self.engine.llm_engine
        if engine.has_unfinished_requests():
            raise UnsupportedProfile('resident consumer requires no other live requests')
        if token_observer is not None:
            features = getattr(self, 'request_features', None)
            if (
                not callable(token_observer)
                or cancel_after_first
                or self.profile.method_id != 'icml_reuse_patch'
                or self.max_batch_size != 1
                or getattr(features, 'async_scheduling', None) is not False
                or getattr(features, 'async_loading', None) is not False
                or getattr(features, 'routing', None) is not False
                or type(protocol_receipt) is not dict
                or protocol_receipt
            ):
                raise UnsupportedProfile('token observer requires a synchronous private ICML request and empty receipt')
        elif protocol_receipt is not None:
            raise ValueError('protocol receipt requires a token observer')

        completion_enabled = getattr(self, 'output_completion_enabled', False)
        if completion_enabled and token_observer is None:
            raise UnsupportedProfile('output completion requires an acknowledged original token observer')

        first_observed_ns = None
        first_token_id = None
        deadline = started + 120

        def remaining():
            seconds = deadline - time.perf_counter()
            if seconds <= 0:
                raise TimeoutError('resident request deadline reached')
            return seconds

        first = None
        result = None
        cancelled = False
        seen = ()
        protocol_reason = None
        protocol_abort = False
        protocol_completed = None

        try:
            prompt = {'prompt_token_ids': list(prompt_ids)}
            with self._deadline_factory(remaining()):
                internal_id = engine.add_request(external_id, prompt, params)
            remaining()
            while engine.has_unfinished_requests():
                with self._deadline_factory(remaining()):
                    outputs = engine.step()
                remaining()
                if token_observer is not None and sum(bool(o.outputs and o.outputs[0].token_ids) for o in outputs) > 1:
                    raise IdentityMismatch('token observer received multiple outputs in one engine step')
                for output in outputs:
                    if output.request_id not in (external_id, internal_id):
                        raise IdentityMismatch('unexpected engine output request')
                    if output.outputs and output.outputs[0].token_ids:
                        if first is None:
                            if token_observer is not None and not completion_enabled:
                                self.synchronize()
                            first_observed_ns = time.perf_counter_ns()
                            first_token_id = output.outputs[0].token_ids[0]
                            first = (time.perf_counter() - started) * 1000
                        result = output
                        if token_observer is not None:
                            observed = tuple(output.outputs[0].token_ids)
                            if len(observed) != len(seen) + 1 or observed[:-1] != seen:
                                raise IdentityMismatch('token observer requires exactly one new cumulative token per step')
                            seen = observed
                            with self._deadline_factory(remaining()):
                                protocol_reason = token_observer(seen)
                            remaining()
                            if protocol_reason is not None:
                                if type(protocol_reason) is not str or protocol_reason not in (
                                    'stop_token',
                                    'text_boundary',
                                    'context_span_boundary',
                                    'max_new_tokens',
                                ):
                                    raise IdentityMismatch('unrecognized original generation stop reason')
                                protocol_completed = (time.perf_counter() - started) * 1000
                if protocol_reason is not None:
                    if engine.has_unfinished_requests():
                        with self._deadline_factory(remaining()):
                            engine.abort_request([internal_id], internal=True)
                        self._rpc('check_health')
                        remaining()
                        protocol_abort = True
                        if engine.has_unfinished_requests():
                            raise RuntimeError('protocol stop did not drain its owned request')
                    break
                if first is not None and cancel_after_first:
                    if not engine.has_unfinished_requests():
                        raise RuntimeError('request naturally finished before cancellation probe')
                    with self._deadline_factory(remaining()):
                        engine.abort_request([internal_id], internal=True)
                    self._rpc('check_health')
                    remaining()
                    cancelled = True
                    break

            if result is None or first is None:
                raise RuntimeError('engine produced no tokens')
            part = result.outputs[0]
            if token_observer is not None:
                if protocol_reason is None or engine.has_unfinished_requests():
                    raise IdentityMismatch('engine finished without an acknowledged original protocol stop')
                if part.finish_reason == 'error' or (part.finish_reason == 'abort' and not protocol_abort):
                    raise RuntimeError('protocol generation failed')
                if not protocol_abort and not result.finished:
                    raise RuntimeError('natural protocol completion is unconfirmed')
                completion = {}
                if completion_enabled:
                    if deferred_completion is not None:
                        deferred_completion.update(
                            actual_request_id=internal_id,
                            first_token_id=first_token_id,
                            first_observed_ns=first_observed_ns,
                        )
                    else:
                        from scd_vllm.output_completion import validate_completion

                        proof = self._rpc(
                            'scd_gpu_output_completion',
                            session_id=self.session_id,
                            actual_request_id=internal_id,
                            first_token_id=first_token_id,
                            first_observed_ns=first_observed_ns,
                        )
                        validate_completion(
                            proof,
                            self.session_id,
                            internal_id,
                            first_token_id,
                            first_observed_ns,
                            expected_contract=getattr(self, 'output_completion_contract', None),
                        )
                        completion = {'first_token_completion': {**proof, 'first_observed_ns': first_observed_ns}}
                protocol_receipt.update(
                    **completion,
                    protocol_stop_reason=protocol_reason,
                    protocol_observed_token_count=len(seen),
                    protocol_completion_ms=protocol_completed,
                    protocol_stop_cleanup_ms=(time.perf_counter() - started) * 1000 - protocol_completed,
                    protocol_abort_requested=protocol_abort,
                    protocol_abort_confirmed=protocol_abort,
                    protocol_request_drained=True,
                )
            elif not cancelled and (not result.finished or part.finish_reason in ('abort', 'error')):
                raise RuntimeError('resident generation failed')
            return tuple(part.token_ids), first, internal_id, cancelled
        except BaseException as exc:
            self._fail_closed(exc)
            raise

    def _check_fullprefix(self):
        check_fullprefix_scope(self.profile, getattr(self, 'capability', None), resident=True)
        if self.max_batch_size != 1:
            raise UnsupportedProfile('fullprefix candidate requires a serial resident frontend')
        if self.profile.target.attention.get('sliding_window') is not None:
            features = getattr(self, 'request_features', None)
            if (
                getattr(features, 'async_scheduling', None) is not False
                or getattr(features, 'async_loading', None) is not False
                or getattr(features, 'routing', None) is not False
            ):
                raise UnsupportedProfile('bounded SWA fullprefix requires explicit synchronous private scheduling')

    def _finalize_admitted(
        self,
        prepared,
        admission,
        prompt_ids,
        generation,
        request_id,
        started,
        admission_ms,
        *,
        mode,
        cancel_after_first,
        import_policy,
        token_observer,
    ):
        params = self._generation(prompt_ids, generation, admission)
        protocol_receipt = {}
        deferred_completion = {} if (token_observer is not None and getattr(self, 'output_completion_enabled', False)) else None
        observer_kwargs = (
            {}
            if token_observer is None
            else {
                'token_observer': token_observer,
                'protocol_receipt': protocol_receipt,
                'deferred_completion': deferred_completion,
            }
        )
        ids, ttft, actual_id, cancelled = self._observe(
            prompt_ids,
            params,
            request_id,
            started,
            cancel_after_first=cancel_after_first,
            **observer_kwargs,
        )

        finalized = self._rpc(
            'scd_gpu_finalize',
            session_id=self.session_id,
            admission_data=admission,
            output_completion_args=deferred_completion or None,
        )
        receipt = finalized['receipt']
        status = finalized['status']
        gpu_timings = finalized.get('gpu_timings') or {}
        if deferred_completion:
            from scd_vllm.output_completion import validate_completion

            proof = finalized.get('first_token_completion')
            validate_completion(
                proof,
                self.session_id,
                deferred_completion['actual_request_id'],
                deferred_completion['first_token_id'],
                deferred_completion['first_observed_ns'],
                expected_contract=getattr(self, 'output_completion_contract', None),
            )
            protocol_receipt['first_token_completion'] = {
                **proof,
                'first_observed_ns': deferred_completion['first_observed_ns'],
            }

        expected = {
            'session_id': self.session_id,
            'artifact_id': prepared['artifact_id'],
            'lease_id': admission['lease_id'],
            'request_id': request_id,
            'actual_request_id': actual_id,
            'profile_fingerprint': profile_fingerprint(self.profile),
            'request_digest': admission['plan']['request_digest'],
            'loaded_layers': self.profile.target.num_layers,
            'loaded_tokens': admission['plan']['imported_tokens'],
        }
        if any(receipt.get(k) != v for k, v in expected.items()) or receipt.get('device_types') != ['cuda']:
            raise IdentityMismatch('actual resident CUDA receipt differs')
        if import_policy == 'fullprefix' and (
            admission['plan'].get('import_policy') != 'fullprefix'
            or receipt.get('import_policy') != 'fullprefix'
            or receipt.get('tail_recomputed_tokens') != 0
            or receipt.get('loaded_tokens') != receipt.get('original_prefix_tokens')
        ):
            raise IdentityMismatch('actual fullprefix receipt differs from the selected policy')
        if prepared['artifact_id'] in status.get('phases', {}) or status.get('poisoned'):
            raise RuntimeError('owned resident state leaked or poisoned')

        metadata = {
            'injection_receipt': receipt,
            'fixed_reader_receipt': None,
            'cancelled_after_first_token': cancelled,
            'imported_tokens': receipt['loaded_tokens'],
            'tail_recomputed_tokens': admission['plan']['tail_recomputed_tokens'],
            'native_fallback': False,
            'transport_kind': 'private_same_worker_gpu',
            'kv_payload_cpu_bytes': 0,
            'kv_payload_file_bytes': 0,
        }
        metadata.update(protocol_receipt)
        if import_policy == 'fullprefix':
            metadata['import_policy'] = 'fullprefix'
        if prepared.get('source_full_prompt_prefill'):
            metadata.update({k: prepared[k] for k in ('source_full_prompt_prefill', 'source_input_tokens', 'source_prompt_digest')})

        metrics = {
            k: prepared[k]
            for k in (
                'prepare_rpc_ms',
                'source_prefill_ms',
                'convert_ms',
                'publish_ms',
                'peak_allocated_bytes',
                'peak_reserved_bytes',
            )
        }
        metrics.update(
            **gpu_timings,
            memo_hit=bool(prepared.get('memo_hit', False)),
            enforce_eager=self.enforce_eager,
            mode=mode,
            pipeline_first_token_ms=ttft,
            ttft_scope='client_observed_from_outer_request_start',
            admission_rpc_ms=admission_ms,
            kv_payload_cpu_bytes=0,
            kv_payload_file_bytes=0,
            quality_established=False,
            worker_peak_allocated_bytes=status.get('peak_allocated_bytes'),
            worker_peak_reserved_bytes=status.get('peak_reserved_bytes'),
            pipeline_generation_ms=(time.perf_counter() - started) * 1000,
        )
        return RunReceipt(
            profile_fingerprint(self.profile),
            prepared['artifact_id'],
            receipt['request_digest'],
            mode,
            GenerationResult(ids, None, metadata),
            metrics,
            'cancelled' if cancelled else 'success',
        )

    def consume_prepared(
        self,
        prepared,
        prompt_ids,
        generation,
        *,
        mode='prepared',
        started=None,
        cancel_after_first: bool = False,
        import_policy='blocktail',
        token_observer=None,
    ):
        if import_policy not in ('blocktail', 'fullprefix'):
            raise ValueError('unknown import policy')
        if import_policy == 'fullprefix':
            self._check_fullprefix()
        started = time.perf_counter() if started is None else started
        with self._lock, self._request_scope(started):
            admission = None
            finalized_ok = False
            request_id = uuid.uuid4().hex
            try:
                if prepared['session_id'] != self.session_id or prepared['profile_fingerprint'] != profile_fingerprint(self.profile):
                    raise IdentityMismatch('foreign prepared snapshot')
                if prepared.get('source_full_prompt_prefill') and prepared.get('source_prompt_digest') != sha256(tokens(prompt_ids)):
                    raise IdentityMismatch('target request differs from the captured full source prompt')
                self._generation(prompt_ids, generation)
                t = time.perf_counter()
                kwargs = {
                    'session_id': self.session_id,
                    'artifact_id': prepared['artifact_id'],
                    'prompt_ids': list(prompt_ids),
                    'request_id': request_id,
                    'max_new_tokens': generation.get('max_new_tokens', 64),
                }
                if import_policy == 'fullprefix':
                    kwargs['import_policy'] = 'fullprefix'
                admission = self._rpc('scd_gpu_admit', **kwargs)
                if admission['plan'].get('import_policy', 'blocktail') != import_policy:
                    raise IdentityMismatch('worker admission changed the selected import policy')
                if import_policy == 'fullprefix' and (
                    admission['plan']['imported_tokens'] != prepared['prefix_tokens']
                    or admission['plan']['original_prefix_tokens'] != prepared['prefix_tokens']
                    or admission['plan']['tail_recomputed_tokens'] != 0
                ):
                    raise IdentityMismatch('worker fullprefix admission differs from the prepared prefix')
                admission_ms = (time.perf_counter() - t) * 1000
                outcome = self._finalize_admitted(
                    prepared,
                    admission,
                    prompt_ids,
                    generation,
                    request_id,
                    started,
                    admission_ms,
                    mode=mode,
                    cancel_after_first=cancel_after_first,
                    import_policy=import_policy,
                    token_observer=token_observer,
                )
                finalized_ok = True
                return outcome
            except BaseException as exc:
                if not self._closed:
                    self._fail_closed(exc)
                raise
            finally:
                if not finalized_ok and not self._closed:
                    try:
                        self._rpc(
                            'scd_gpu_release',
                            session_id=self.session_id,
                            artifact_id=prepared['artifact_id'],
                            lease_id=None if admission is None else admission['lease_id'],
                        )
                    except BaseException as exc:
                        if not self._closed:
                            self._fail_closed(exc)
                        raise

    def consume_prepared_batch(self, requests, generation, *, cancel_after_first=(), reader_digests=None, _preadmitted=None):
        from .resident_batch import consume_prepared_batch

        return consume_prepared_batch(
            self,
            requests,
            generation,
            cancel_after_first=cancel_after_first,
            reader_digests=reader_digests,
            _preadmitted=_preadmitted,
        )

    def native_batch(self, prompts, generation, *, reader_digests=None, cancel_after_first=()):
        return self.consume_prepared_batch(
            [(None, prompt) for prompt in prompts],
            generation,
            reader_digests=reader_digests,
            cancel_after_first=cancel_after_first,
        )

    def run_batch(self, requests, generation):
        started = time.perf_counter()
        requests = tuple(requests)
        if not requests or len(requests) > self.max_batch_size:
            raise UnsupportedProfile('resident batch bound exceeded')
        with self._lock, self._request_scope(started):
            created_admissions = []
            batch_done = False
            try:
                for _, prompt in requests:
                    self._generation(prompt, generation)
                if getattr(self, '_use_fused_batch_rpc', False):
                    external_ids = [uuid.uuid4().hex for _ in requests]
                    batch_items = [
                        {
                            'prefix_data': to_data(prefix),
                            'prompt_ids': list(prompt),
                            'request_id': rid,
                            'mode': 'E',
                        }
                        for (prefix, prompt), rid in zip(requests, external_ids)
                    ]
                    t0 = time.perf_counter()
                    fused_batch = self._rpc(
                        'scd_gpu_prepare_and_admit_batch',
                        session_id=self.session_id,
                        batch_items=batch_items,
                        max_new_tokens=generation.get('max_new_tokens', 64),
                        import_policy='blocktail',
                        stream_ordered=getattr(self, 'stream_ordered_prepare', True),
                    )
                    producer_ms = (time.perf_counter() - t0) * 1000
                    prepared_items = fused_batch['items']
                    if len(prepared_items) != len(requests):
                        raise IdentityMismatch('worker batch prepare count differs')
                    prepared = []
                    admissions = []
                    for (prefix, prompt), rid, entry in zip(requests, external_ids, prepared_items):
                        if (
                            entry['session_id'] != self.session_id
                            or entry['profile_fingerprint'] != profile_fingerprint(self.profile)
                            or entry['prefix_digest'] != sha256(prefix)
                            or entry['admission'].get('request_id') != rid
                        ):
                            raise IdentityMismatch('prepared batch snapshot differs')
                        prepared.append(({**entry, 'prepare_rpc_ms': producer_ms / len(requests)}, prompt))
                        admissions.append(entry['admission'])
                    created_admissions = list(zip([p[0]['artifact_id'] for p in prepared], [a['lease_id'] for a in admissions]))
                    consumer_started = time.perf_counter()
                    result = self.consume_prepared_batch(prepared, generation, _preadmitted=(admissions, external_ids))
                    batch_done = True
                    producer_mode = 'fused_batch_E_with_dedup_then_batched_consumer'
                else:
                    if self.status()['entries']:
                        raise UnsupportedProfile('end-to-end batch requires no retained snapshots')
                    prepared = []
                    for prefix, prompt in requests:
                        prepared.append((self.prepare(prefix, mode='E'), prompt))
                    producer_ms = sum(item[0]['prepare_rpc_ms'] for item in prepared)
                    consumer_started = time.perf_counter()
                    result = self.consume_prepared_batch(prepared, generation)
                    batch_done = True
                    producer_mode = 'serial_E_then_batched_consumer'

                from .batch import BatchResult

                return BatchResult(
                    result.generations,
                    {
                        **result.metrics,
                        'pipeline_wall_ms': (time.perf_counter() - started) * 1000,
                        'consumer_start_offset_ms': (consumer_started - started) * 1000,
                        'producer_mode': producer_mode,
                        'producer_ms': producer_ms,
                        'memo_hits': sum(bool(item[0].get('memo_hit')) for item in prepared),
                        'batch_dedup_hits': sum(bool(item[0].get('batch_dedup_hit')) for item in prepared),
                        'pipeline_peak_allocated_bytes': max(
                            [item[0]['peak_allocated_bytes'] for item in prepared]
                            + [result.metrics['worker_peak_allocated_bytes']]
                        ),
                        'pipeline_peak_reserved_bytes': max(
                            [item[0]['peak_reserved_bytes'] for item in prepared]
                            + [result.metrics['worker_peak_reserved_bytes']]
                        ),
                        'peak_scope': 'maximum of all worker Torch-allocator producer/consumer phases',
                    },
                )
            except BaseException as exc:
                if not batch_done and created_admissions and not self._closed:
                    for art_id, lease_id in created_admissions:
                        try:
                            self._rpc('scd_gpu_release', session_id=self.session_id, artifact_id=art_id, lease_id=lease_id)
                        except BaseException:
                            break
                if not self._closed:
                    self._fail_closed(exc)
                raise

    def run(
        self,
        prefix,
        prompt_ids,
        generation,
        *,
        mode='E',
        source_handle=None,
        cancel_after_first: bool = False,
        import_policy='blocktail',
        full_prompt_ids=None,
        token_observer=None,
    ):
        """Fused single-RPC prepare+admit + stream-ordered KV load + single-RPC finalize."""
        if import_policy not in ('blocktail', 'fullprefix'):
            raise ValueError('unknown import policy')
        if import_policy == 'fullprefix':
            self._check_fullprefix()
        started = time.perf_counter()
        with self._lock, self._request_scope(started):
            admission = None
            artifact_id = None
            finalized_ok = False
            request_id = uuid.uuid4().hex
            try:
                self._generation(prompt_ids, generation)
                src_kwargs = self._source_kwargs(prefix, full_prompt_ids)
                t0 = time.perf_counter()
                fused = self._rpc(
                    'scd_gpu_prepare_and_admit',
                    session_id=self.session_id,
                    prefix_data=to_data(prefix),
                    prompt_ids=list(prompt_ids),
                    request_id=request_id,
                    max_new_tokens=generation.get('max_new_tokens', 64),
                    mode=mode,
                    source_handle=source_handle,
                    import_policy=import_policy,
                    stream_ordered=self.stream_ordered_prepare,
                    **src_kwargs,
                )
                rpc_ms = (time.perf_counter() - t0) * 1000
                if (
                    fused['session_id'] != self.session_id
                    or fused['profile_fingerprint'] != profile_fingerprint(self.profile)
                    or fused['prefix_digest'] != sha256(prefix)
                ):
                    raise IdentityMismatch('prepared snapshot differs')
                self._check_source_receipt(fused, src_kwargs)
                artifact_id = fused['artifact_id']
                admission = fused['admission']
                if admission['plan'].get('import_policy', 'blocktail') != import_policy:
                    raise IdentityMismatch('worker admission changed the selected import policy')
                if import_policy == 'fullprefix' and (
                    admission['plan']['imported_tokens'] != fused['prefix_tokens']
                    or admission['plan']['original_prefix_tokens'] != fused['prefix_tokens']
                    or admission['plan']['tail_recomputed_tokens'] != 0
                ):
                    raise IdentityMismatch('worker fullprefix admission differs from the prepared prefix')

                prepared = {**fused, 'prepare_rpc_ms': rpc_ms}
                outcome = self._finalize_admitted(
                    prepared,
                    admission,
                    prompt_ids,
                    generation,
                    request_id,
                    started,
                    0.0,
                    mode=mode,
                    cancel_after_first=cancel_after_first,
                    import_policy=import_policy,
                    token_observer=token_observer,
                )
                finalized_ok = True
                return outcome
            except BaseException as exc:
                if not self._closed:
                    self._fail_closed(exc)
                raise
            finally:
                if not finalized_ok and not self._closed and artifact_id is not None:
                    try:
                        self._rpc(
                            'scd_gpu_release',
                            session_id=self.session_id,
                            artifact_id=artifact_id,
                            lease_id=None if admission is None else admission['lease_id'],
                        )
                    except BaseException as exc:
                        if not self._closed:
                            self._fail_closed(exc)
                        raise

    def native(self, prompt_ids, generation, *, reader_digest=None, token_observer=None):
        """Run native target vLLM generation directly (zero reader-toggle RPCs)."""
        if reader_digest is not None:
            raise UnsupportedProfile('SCD-ICML does not use LoRA reader banks')
        with self._lock:
            protocol_receipt = {}
            deferred_completion = (
                {} if (token_observer is not None and getattr(self, 'output_completion_enabled', False)) else None
            )
            observer_kwargs = (
                {}
                if token_observer is None
                else {
                    'token_observer': token_observer,
                    'protocol_receipt': protocol_receipt,
                    'deferred_completion': deferred_completion,
                }
            )
            params = self._generation(prompt_ids, generation)
            started = time.perf_counter()
            try:
                with self._request_scope(started):
                    ids, ttft, actual_id, _ = self._observe(
                        prompt_ids,
                        params,
                        uuid.uuid4().hex,
                        started,
                        **observer_kwargs,
                    )
                    if deferred_completion:
                        from scd_vllm.output_completion import validate_completion

                        proof = self._rpc(
                            'scd_gpu_output_completion',
                            session_id=self.session_id,
                            actual_request_id=deferred_completion['actual_request_id'],
                            first_token_id=deferred_completion['first_token_id'],
                            first_observed_ns=deferred_completion['first_observed_ns'],
                        )
                        validate_completion(
                            proof,
                            self.session_id,
                            deferred_completion['actual_request_id'],
                            deferred_completion['first_token_id'],
                            deferred_completion['first_observed_ns'],
                            expected_contract=getattr(self, 'output_completion_contract', None),
                        )
                        protocol_receipt['first_token_completion'] = {
                            **proof,
                            'first_observed_ns': deferred_completion['first_observed_ns'],
                        }
                return {
                    'token_ids': list(ids),
                    'first_token_ms': ttft,
                    'full_request_ms': (time.perf_counter() - started) * 1000,
                    'reader': 'none',
                    'enforce_eager': self.enforce_eager,
                    'actual_request_id': actual_id,
                    'source_model_idle_resident': True,
                    **protocol_receipt,
                }
            except BaseException as exc:
                if not self._closed:
                    self._fail_closed(exc)
                raise

    def close(self):
        with self._lock:
            if self._closed:
                return
            clear_kwargs = {'close_output_completion': True} if getattr(self, 'output_completion_enabled', False) else {}
            try:
                self._rpc('scd_gpu_clear', session_id=self.session_id, **clear_kwargs)
            finally:
                if not self._closed:
                    self._closed = True
                    self._shutdown_confirmed = False
                    with self._deadline_factory(10):
                        self.engine.llm_engine.engine_core.shutdown(timeout=5)
                    self._shutdown_confirmed = True
