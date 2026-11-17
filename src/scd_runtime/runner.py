"""Pure-ICML offline preparation and file-backed vLLM consumer."""
from __future__ import annotations

from dataclasses import dataclass
import math
import threading
import time
from typing import Mapping
import uuid

from .benchmarks.metrics import combine_metrics
from .contracts.identity import profile_fingerprint, protocol_fingerprint
from .contracts.types import GenerationResult, to_data, tokens
from .errors import IdentityMismatch, ReaderMismatch, UnsupportedProfile


def request_prefix_cache_policy_supported() -> bool:
    """Inspect the installed vLLM Request API for per-request prefix-cache skip support."""
    try:
        from vllm.v1.request import Request
    except ImportError:
        return False
    return callable(getattr(Request, 'get_skip_writing_prefix_cache', None))


@dataclass
class RequestFeatures:
    """Pure-ICML frontend scheduling/loading configuration."""
    profile: object
    routing: bool
    async_loading: bool
    async_scheduling: bool | None
    namespace: str | None = None
    packages: tuple = ()
    allowed_readers: frozenset = frozenset()
    primary_reader: str | None = None
    math_identity: str | None = None
    generation_counter: int = 0

    @classmethod
    def from_config(cls, profile, config, reader_packages=(), *, request_policy_supported: bool = False):
        routing = config.pop('request_reader_routing', False)
        loading = config.pop('async_loading', False)
        absent = object()
        scheduling = config.pop('async_scheduling', absent)
        if (
            type(routing) is not bool
            or type(loading) is not bool
            or (scheduling is not absent and type(scheduling) is not bool)
        ):
            raise UnsupportedProfile('request routing/loading/scheduling need explicit booleans')
        if routing or reader_packages:
            raise UnsupportedProfile('SCD-ICML uses reader.kind="none" and does not use LoRA reader routing')
        scheduling = None if scheduling is absent else scheduling
        prefix = config.get('enable_prefix_caching', False)
        if type(prefix) is not bool:
            raise UnsupportedProfile('prefix cache setting must be boolean')
        if prefix and request_policy_supported is not True:
            raise UnsupportedProfile('APC needs core request-wide skip_writing_prefix_cache support')
        return cls(profile, False, loading, scheduling)

    def extra_config(self) -> dict:
        return {'request_reader_routing': False, 'async_loading': self.async_loading}

    def select_readers(self, items, reader_digests=None):
        if reader_digests is not None and any(r is not None for r in reader_digests):
            raise ReaderMismatch('SCD-ICML does not accept LoRA reader digests')
        return (None,) * len(items)


@dataclass(frozen=True)
class PreparedArtifact:
    artifact_id: str
    metrics: Mapping


@dataclass(frozen=True)
class RunReceipt:
    profile_fingerprint: str
    artifact_id: str
    request_digest: str
    mode: str
    generation: GenerationResult
    metrics: Mapping
    status: str


def _sync_state(state):
    import torch

    device = state.kv[0][0].device
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def prepare(profile, prefix, assets, store, *, source_state=None, source_provider=None, full_prompt_ids=None):
    if assets.profile != profile:
        raise IdentityMismatch('assets profile differs')
    if profile.method_id != 'icml_reuse_patch' or profile.reader.kind != 'none':
        raise UnsupportedProfile('SCD-ICML prepare requires method_id="icml_reuse_patch" and reader.kind="none"')
    if full_prompt_ids is not None and source_state is not None:
        raise UnsupportedProfile('full-prompt source capture requires a fresh ICML provider path')
    if full_prompt_ids is not None:
        full_prompt_ids = tokens(full_prompt_ids)

    from .methods.icml.backend import ICMLBackend
    from .providers.hf.projections import TargetProjectionProvider

    backend = ICMLBackend()
    load_ms = source_ms = 0.0
    if source_state is None:
        if source_provider is None:
            from .providers.hf.loading import load_source_provider

            started = time.perf_counter()
            source_provider = load_source_provider(assets)
            load_ms = (time.perf_counter() - started) * 1000
        started = time.perf_counter()
        source_state = (
            source_provider.capture(prefix, backend.required_features(profile))
            if full_prompt_ids is None
            else source_provider.capture(prefix, backend.required_features(profile), full_prompt_ids=full_prompt_ids)
        )
        _sync_state(source_state)
        source_ms = (time.perf_counter() - started) * 1000
    elif source_state.prefix != prefix:
        raise IdentityMismatch('existing source prefix differs')

    _sync_state(source_state)
    started = time.perf_counter()
    projections = (
        TargetProjectionProvider.from_assets(assets)
        if profile.route.get('fit_mode') != 'direct_kv_v1'
        else None
    )
    kv = backend.convert(source_state, assets, projections)
    _sync_state(source_state)
    convert_ms = (time.perf_counter() - started) * 1000

    started = time.perf_counter()
    identifier = store.publish(kv)
    publish_ms = (time.perf_counter() - started) * 1000
    byte_count = sum(t.numel() * t.element_size() for pair in kv.layers for t in pair)
    metrics = {
        'source_load_ms': load_ms,
        'source_prefill_ms': source_ms,
        'convert_ms': convert_ms,
        'publish_ms': publish_ms,
        'target_kv_tensor_bytes': byte_count,
        'transport_kind': 'reconstructed_target_kv',
    }
    if full_prompt_ids is not None:
        metrics.update(
            source_full_prompt_prefill=True,
            source_input_tokens=len(full_prompt_ids),
            cache_prefix_tokens=len(prefix.token_ids),
        )
    return PreparedArtifact(identifier, metrics)


class VLLMConsumer:
    """File-store vLLM V1 consumer using SCDConnectorV1."""

    def __init__(self, store, engine_config, profile, reader_package=None, *, reader_packages=()):
        if profile.method_id != 'icml_reuse_patch' or profile.reader.kind != 'none':
            raise UnsupportedProfile('SCD-ICML VLLMConsumer requires icml_reuse_patch with reader="none"')
        if reader_package is not None or reader_packages:
            raise ReaderMismatch('unexpected reader package for SCD-ICML')

        from scd_vllm.capability import ensure_sampler_jit_safe, installed_commit
        from scd_vllm.connector import ConnectorCoordinator
        from .contracts.types import EngineCapability
        from .providers.hf.loading import verify_model

        config = dict(engine_config)
        strict_pin = bool(config.pop('strict_commit_pin', False))
        commit = installed_commit(strict_commit_pin=strict_pin)
        no_store = request_prefix_cache_policy_supported()
        self.request_features = RequestFeatures.from_config(
            profile,
            config,
            (),
            request_policy_supported=no_store,
        )
        self.max_batch_size = config.get('max_num_seqs', 1)
        if type(self.max_batch_size) is not int or self.max_batch_size < 1:
            raise UnsupportedProfile('positive max_num_seqs required')
        config['max_num_seqs'] = self.max_batch_size
        self._closed = False

        prefix_cache = config.get('enable_prefix_caching', False)
        if type(prefix_cache) is not bool:
            raise UnsupportedProfile('prefix cache setting must be boolean')
        enforce_eager = config.get('enforce_eager', False)
        if type(enforce_eager) is not bool:
            raise UnsupportedProfile('enforce_eager must be boolean')

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
            'max_num_seqs',
            'max_num_batched_tokens',
            'seed',
        }
        if config.keys() - allowed:
            raise UnsupportedProfile(f'unbound engine override: {sorted(config.keys() - allowed)}')
        if set(config.get('attention_config', {})) - {'backend'}:
            raise UnsupportedProfile('unbound attention override')

        model_root = config.get('model')
        manifest = config.pop('model_manifest', None)
        if not model_root or not manifest:
            raise UnsupportedProfile('model and verified model_manifest required')
        verify_model(model_root, manifest, profile.target)

        block = config.get('block_size')
        if type(block) is not int or block < 1:
            raise UnsupportedProfile('explicit block_size required')
        backend = config.get('attention_config', {}).get('backend')
        cap = EngineCapability(
            commit,
            backend,
            'bfloat16',
            1,
            1,
            block,
            1,
            profile.target.attention,
            False,
            False,
            prefix_cache,
            config.get('max_model_len'),
            no_store,
        )
        self.core = ConnectorCoordinator(
            store,
            profile,
            cap,
            fullprefix_serial=self.max_batch_size == 1,
            strict_commit_pin=strict_pin,
        )
        for k, v in fixed.items():
            if k in config and config[k] != v:
                raise UnsupportedProfile(f'unsupported override {k}')
            config[k] = v

        config['kv_transfer_config'] = {
            'kv_connector': 'SCDConnectorV1',
            'kv_connector_module_path': 'scd_vllm.connector',
            'kv_role': 'kv_consumer',
            'kv_load_failure_policy': 'fail',
            'kv_connector_extra_config': {
                'scd_profile': to_data(profile),
                'store_root': str(store.root),
                'capacity_bytes': store.capacity_bytes,
                'strict_commit_pin': strict_pin,
                **self.request_features.extra_config(),
            },
        }
        self.lora = None
        self.has_fixed_reader = False
        self._consume_lock = threading.Lock()
        config['worker_extension_cls'] = 'scd_vllm.cache_worker.SCDCacheWorkerExtension'
        if self.request_features.async_scheduling is not None:
            config['async_scheduling'] = self.request_features.async_scheduling

        ensure_sampler_jit_safe()
        from vllm import LLM

        self.engine = LLM(**config)
        self.profile = profile
        try:
            caches = self.engine.collective_rpc('scd_cache_status', timeout=30)
            if (
                len(caches) != 1
                or caches[0]['profile_fingerprint'] != profile_fingerprint(profile)
                or caches[0]['registered_layers'] != profile.target.num_layers
                or caches[0]['device_types'] != ['cuda']
            ):
                raise IdentityMismatch('worker registered cache differs from engine profile')
            self.cache_registration = caches[0]
            self.kv_capacity_tokens = caches[0]['capacity_tokens']
        except BaseException as exc:
            self._closed = True
            try:
                self.engine.llm_engine.engine_core.shutdown(timeout=5)
            except BaseException as cleanup:
                if hasattr(exc, 'add_note'):
                    exc.add_note('Owned engine startup cleanup failed: ' + repr(cleanup))
            raise

    def consume(self, artifact_id, prompt_ids, generation, *, import_policy='blocktail'):
        if getattr(self, '_closed', False):
            raise RuntimeError('consumer is closed; create a new engine')
        if not hasattr(self, '_consume_lock'):
            self._consume_lock = threading.Lock()
        started = time.perf_counter()
        with self._consume_lock:
            if getattr(self, '_closed', False):
                raise RuntimeError('consumer is closed; create a new engine')
            queue_ms = (time.perf_counter() - started) * 1000
            result = self._consume_locked(artifact_id, prompt_ids, generation, import_policy=import_policy)
        return GenerationResult(
            result.token_ids,
            result.first_logits,
            {**result.metadata, 'consumer_queue_ms': queue_ms, 'consume_wall_ms': (time.perf_counter() - started) * 1000},
        )

    def consume_batch(self, requests, generation, *, cancel_after_first=(), reader_digests=None, import_policy='blocktail'):
        from .file_batch import consume_batch

        return consume_batch(
            self,
            requests,
            generation,
            cancel_after_first=cancel_after_first,
            reader_digests=reader_digests,
            import_policy=import_policy,
        )

    def native_batch(self, prompts, generation, *, reader_digests=None, cancel_after_first=()):
        return self.consume_batch(
            [(None, prompt) for prompt in prompts],
            generation,
            reader_digests=reader_digests,
            cancel_after_first=cancel_after_first,
        )

    def native(self, prompt_ids, generation, *, reader_digest=None):
        return self.native_batch([prompt_ids], generation, reader_digests=[reader_digest] if reader_digest is not None else None).generations[0]

    def _consume_locked(self, artifact_id, prompt_ids, generation, *, import_policy='blocktail'):
        from vllm import SamplingParams

        bound = self.core.capability.max_model_len
        if bound is not None and len(prompt_ids) + generation.get('max_new_tokens', 64) > bound:
            raise UnsupportedProfile('prompt plus generation exceeds the qualified context bound')
        policy = {} if import_policy == 'blocktail' else {'import_policy': import_policy}
        started = time.perf_counter()
        admission = self.core.admit(artifact_id, prompt_ids, uuid.uuid4().hex, **policy)
        admission_ms = (time.perf_counter() - started) * 1000
        engine_started = False
        safe_release = True
        try:
            allowed = {'max_new_tokens', 'temperature', 'top_p', 'eos_token_id', 'ignore_eos'}
            if generation.keys() - allowed:
                raise ValueError('unsupported generation settings')
            eos = generation.get('eos_token_id')
            stop_ids = None if eos is None else (list(eos) if isinstance(eos, (list, tuple)) else [eos])
            params = SamplingParams(
                max_tokens=generation.get('max_new_tokens', 64),
                temperature=generation.get('temperature', 0),
                top_p=generation.get('top_p', 1),
                stop_token_ids=stop_ids,
                ignore_eos=generation.get('ignore_eos', False),
                extra_args={
                    'skip_writing_prefix_cache': True,
                    'kv_transfer_params': {'scd_admission': to_data(admission)},
                },
            )
            prompt = {'prompt_token_ids': list(prompt_ids), 'cache_salt': 'scd-' + admission.lease.lease_id}
            engine_started = True
            started = time.perf_counter()
            outputs = self.engine.generate([prompt], params, use_tqdm=False)
            elapsed = (time.perf_counter() - started) * 1000
            output = outputs[0]
            if (
                not output.outputs
                or not output.outputs[0].token_ids
                or getattr(output.outputs[0], 'finish_reason', None) in ('abort', 'error')
            ):
                raise RuntimeError('vLLM request failed or produced no tokens')
            ids = tuple(output.outputs[0].token_ids)
            receipt = self.core.store.pop_injection(admission.lease)
            expected = {
                'artifact_id': artifact_id,
                'request_id': admission.lease.request_id,
                'lease_id': admission.lease.lease_id,
                'profile_fingerprint': admission.profile_fingerprint,
                'request_digest': admission.plan.request_digest,
                'loaded_layers': self.core.profile.target.num_layers,
                'loaded_tokens': admission.plan.imported_tokens,
            }
            if import_policy == 'fullprefix':
                expected.update(
                    import_policy='fullprefix',
                    original_prefix_tokens=admission.plan.original_prefix_tokens,
                    tail_recomputed_tokens=0,
                )
            if any(receipt.get(key) != value for key, value in expected.items()):
                raise IdentityMismatch('worker injection receipt differs from admitted request')
            first = getattr(getattr(output, 'metrics', None), 'first_token_latency', None)
            first_ms = first * 1000 if type(first) in (int, float) and math.isfinite(first) and first >= 0 else None
            return GenerationResult(
                tuple(ids),
                None,
                {
                    'admission_ms': admission_ms,
                    'first_token_ms': first_ms,
                    'generate_wall_ms': elapsed,
                    'reader_before_ms': 0.0,
                    'reader_after_ms': 0.0,
                    'imported_tokens': receipt['loaded_tokens'],
                    'injection_receipt': receipt,
                    'fixed_reader_receipt': None,
                    'tail_recomputed_tokens': admission.plan.tail_recomputed_tokens,
                    'request_digest': admission.plan.request_digest,
                    'backend': 'vllm',
                    'vllm_commit': self.core.capability.vllm_commit,
                    'reader_equivalence': 'not_run',
                    'native_fallback': False,
                },
            )
        except BaseException as exc:
            if engine_started:
                self._closed = True
                safe_release = False
                try:
                    self.engine.llm_engine.engine_core.shutdown(timeout=5)
                    safe_release = True
                except BaseException as cleanup:
                    if hasattr(exc, 'add_note'):
                        exc.add_note('Owned engine shutdown unconfirmed; source lease retained: ' + repr(cleanup))
            raise
        finally:
            if safe_release:
                self.core.store.discard_injection(admission.lease)
                self.core.release(admission)

    def close(self):
        if getattr(self, '_closed', False):
            return
        self._closed = True
        self.engine.llm_engine.engine_core.shutdown(timeout=5)


def run_request(
    profile,
    prefix,
    prompt_ids,
    assets,
    store,
    consumer,
    generation,
    *,
    mode,
    source_state=None,
    source_provider=None,
    sample_id=None,
    import_policy='blocktail',
):
    if mode not in ('E', 'H') or (mode == 'H' and source_state is None) or (mode == 'E' and source_state is not None):
        raise IdentityMismatch('E requires fresh source; H requires existing source')
    prepared = prepare(profile, prefix, assets, store, source_state=source_state, source_provider=source_provider)
    policy = {} if import_policy == 'blocktail' else {'import_policy': import_policy}
    output = consumer.consume(prepared.artifact_id, prompt_ids, generation, **policy)
    metrics = combine_metrics(mode, prepared.metrics, output.metadata)
    if sample_id is not None:
        metrics['sample_id'] = sample_id
    if profile.validation_protocol is not None:
        metrics['validation_protocol_fingerprint'] = protocol_fingerprint(profile.validation_protocol)
    return RunReceipt(
        profile_fingerprint(profile),
        prepared.artifact_id,
        output.metadata.get('request_digest', ''),
        mode,
        output,
        metrics,
        'success',
    )
