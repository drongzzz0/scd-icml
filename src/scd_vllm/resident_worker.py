"""Same-worker GPU source/converter execution for SCD-ICML on vLLM V1."""
from __future__ import annotations

from collections import OrderedDict
import hashlib
import secrets
import time

import torch

from scd_runtime.assets.manifest import load_assets
from scd_runtime.contracts.types import (
    MethodAssets,
    PrefixContract,
    TargetKV,
    profile_from_dict,
    strict_construct,
    to_data,
    tokens,
)
from scd_runtime.contracts.identity import fingerprint, profile_fingerprint, sha256
from scd_runtime.errors import IdentityMismatch, StoreCapacityExceeded, UnsupportedProfile
from scd_runtime.methods.icml.backend import ICMLBackend
from scd_runtime.providers.hf.config_identity import check_config_identity
from scd_runtime.providers.hf.loading import load_source_provider
from scd_runtime.providers.hf.projections import TargetProjectionProvider
from scd_runtime.transport.resident import (
    ResidentRegistry,
    admission_from_data,
    register_registry,
    unregister_registry,
)
from .cache_worker import SCDCacheWorkerExtension


class GPUResidentWorkerExtension(SCDCacheWorkerExtension):
    @torch.inference_mode()
    def scd_gpu_configure(self, *, session_id, profile_data, assets_root):
        if getattr(self, '_scd_gpu_state', None) is not None:
            raise UnsupportedProfile('resident worker already configured')
        cfg = self.vllm_config
        extra = cfg.kv_transfer_config
        p = profile_from_dict(profile_data)
        if (
            p.method_id != 'icml_reuse_patch'
            or p.reader.kind != 'none'
            or extra.get_from_extra_config('resident_session', None) != session_id
            or extra.get_from_extra_config('scd_profile', {}) != to_data(p)
            or cfg.parallel_config.tensor_parallel_size != 1
            or cfg.parallel_config.pipeline_parallel_size != 1
            or cfg.model_config.quantization is not None
            or cfg.lora_config is not None
        ):
            raise UnsupportedProfile('resident worker requires a single-GPU ICML session')

        check_config_identity(cfg.model_config.hf_config, p.target)
        device = next(self.get_model().parameters()).device
        if device.type != 'cuda':
            raise UnsupportedProfile('CUDA worker device required')

        torch.set_float32_matmul_precision('highest')
        torch.backends.cudnn.allow_tf32 = False

        t = time.perf_counter()
        assets = load_assets(assets_root, p)
        backend = ICMLBackend()

        source_attn_impl = extra.get_from_extra_config('source_attn_implementation', None)
        source_fast_mask = bool(extra.get_from_extra_config('source_fast_causal_mask', False))
        provider = load_source_provider(
            assets,
            device=device,
            attn_implementation=source_attn_impl,
            fast_causal_mask=source_fast_mask,
        )
        if provider.model.device != device:
            provider.model.to(device)
        provider.model.requires_grad_(False)
        provider.enable_persistent_hooks(backend.required_features(p).hidden_hooks)

        values = {k: v for k, v in assets.tensors.items() if not k.startswith('lora.')}
        with torch.inference_mode(False), torch.no_grad():
            gpu_assets = MethodAssets(
                p,
                assets.root,
                {k: v.to(device=device).clone() for k, v in values.items()},
                assets.metadata,
            )
        backend.shared_identity_layers(gpu_assets)
        projections = TargetProjectionProvider.from_assets(gpu_assets)
        compiled_plan = backend.compile_plan(
            gpu_assets,
            projections,
            device=device,
            max_seq_len=cfg.model_config.max_model_len,
        )

        # Warm up cuBLAS / RoPE / Patch kernels once during configure so the first request has no JIT/workspace spike.
        if compiled_plan is not None:
            warm_n = min(16, max(2, cfg.model_config.max_model_len - 2))
            dummy_prefix = PrefixContract(
                tuple([1] * warm_n),
                0,
                p.source.tokenizer_sha256,
                p.source.chat_template_sha256,
            )
            with torch.autocast(device_type='cuda', enabled=False):
                dummy_src = provider.capture(dummy_prefix, backend.required_features(p))
                dummy_kv = backend.convert(dummy_src, gpu_assets, projections, return_packed=True)
                del dummy_src, dummy_kv

        registry_options = {'device': device, 'max_entries': cfg.scheduler_config.max_num_seqs}
        if p.target.attention.get('sliding_window') is not None:
            from vllm.distributed.kv_transfer.kv_transfer_state import get_kv_transfer_group
            actual_profile, capability = get_kv_transfer_group()._cache_contract()
            if actual_profile != p or capability.max_model_len != cfg.model_config.max_model_len:
                raise IdentityMismatch('actual SWA cache capability differs from the configured worker')
            registry_options['capability'] = capability

        configured_cap = extra.get_from_extra_config('resident_capacity_bytes', None)
        default_cap = max(3 * 1024**3, min(16 * 1024**3, cfg.scheduler_config.max_num_seqs * 512 * 1024**2))
        capacity_bytes = default_cap if configured_cap is None else int(configured_cap)
        registry = ResidentRegistry(p, session_id, capacity_bytes, **registry_options)
        torch.cuda.synchronize(device)
        baseline_free_bytes, total_memory_bytes = torch.cuda.mem_get_info(device)
        baseline_reserved_bytes = torch.cuda.memory_reserved(device)

        memo_max_entries = int(extra.get_from_extra_config('memoize_prefix_entries', 0))
        memo_max_bytes = int(extra.get_from_extra_config('memoize_prefix_bytes', 1024 * 1024**2))
        strict_pin = bool(extra.get_from_extra_config('strict_commit_pin', False))

        self._scd_gpu_state = {
            'registry': registry,
            'assets': gpu_assets,
            'provider': provider,
            'backend': backend,
            'projections': projections,
            'compiled': compiled_plan is not None,
            'source': None,
            'pending_events': {},
            'max_model_len': cfg.model_config.max_model_len,
            'enforce_eager': bool(cfg.model_config.enforce_eager),
            'diagnostics': bool(extra.get_from_extra_config('resident_diagnostics', False)),
            'total_memory_bytes': total_memory_bytes,
            'baseline_free_bytes': baseline_free_bytes,
            'baseline_reserved_bytes': baseline_reserved_bytes,
            'source_model_bytes': sum(t.numel() * t.element_size() for t in provider.model.parameters()),
            'asset_gpu_bytes': sum(t.numel() * t.element_size() for t in gpu_assets.tensors.values()),
            'memo_cache': OrderedDict(),
            'memo_max_entries': max(0, memo_max_entries),
            'memo_max_bytes': max(0, memo_max_bytes),
            'memo_bytes': 0,
            'memo_hits': 0,
            'memo_misses': 0,
        }
        register_registry(registry)

        completion = extra.get_from_extra_config('request_output_completion', False)
        if type(completion) is not bool:
            raise UnsupportedProfile('output completion opt-in must be boolean')
        if completion:
            scheduler = cfg.scheduler_config
            if (
                scheduler.max_num_seqs != 1
                or scheduler.async_scheduling is not False
                or scheduler.enable_chunked_prefill
                or not scheduler.disable_hybrid_kv_cache_manager
                or cfg.cache_config.enable_prefix_caching
                or cfg.speculative_config is not None
                or not cfg.use_v2_model_runner
                or extra.get_from_extra_config('async_loading', False) is not False
            ):
                raise UnsupportedProfile('output completion requires synchronous serial ICML runner')
            from .output_completion import InstalledOutputCompletion
            self._scd_gpu_state['output_completion'] = InstalledOutputCompletion(
                self.model_runner,
                session_id,
                strict_source_sha=strict_pin,
            )

        contract = {} if not completion else {'output_completion_contract': self._scd_gpu_state['output_completion'].contract}
        return {
            **self.scd_gpu_status(session_id=session_id),
            **contract,
            'producer_setup_ms': (time.perf_counter() - t) * 1000,
            'compiled_fast_path': compiled_plan is not None,
            'enforce_eager': bool(cfg.model_config.enforce_eager),
            'source_attention_backend': getattr(provider.model.config, '_attn_implementation', None),
            'profile_fingerprint': profile_fingerprint(p),
            'target_vocab_size': cfg.model_config.hf_config.vocab_size,
            'registered_cache': self.scd_cache_status(),
        }

    def _scd_gpu_context(self, session_id):
        state = getattr(self, '_scd_gpu_state', None)
        if state is None or state['registry'].session_id != session_id:
            raise IdentityMismatch('unknown worker session')
        return state

    def _scd_gpu_prefix(self, state, prefix_data):
        prefix = strict_construct(PrefixContract, prefix_data)
        p = state['registry'].profile
        if (
            prefix.tokenizer_sha256 != p.source.tokenizer_sha256
            or prefix.chat_template_sha256 != p.source.chat_template_sha256
            or len(prefix.token_ids) >= state['max_model_len']
            or max(prefix.token_ids) >= state['provider'].model.config.vocab_size
        ):
            raise IdentityMismatch('resident prefix exceeds verified source/request contract')
        return prefix

    def _scd_source_prompt(self, state, prefix, full_prompt_ids):
        prompt = None
        if full_prompt_ids is not None:
            prompt = tokens(full_prompt_ids)
            if (
                state['registry'].profile.method_id != 'icml_reuse_patch'
                or len(prompt) != len(prefix.token_ids) + 1
                or prompt[:-1] != prefix.token_ids
                or len(prompt) > state['max_model_len']
                or max(prompt) >= state['provider'].model.config.vocab_size
            ):
                raise IdentityMismatch('full source prompt must be the registered ICML prefix plus its final token')
        return prompt, {
            'source_full_prompt_prefill': prompt is not None,
            'source_input_tokens': len(prefix.token_ids) if prompt is None else len(prompt),
            'source_prompt_digest': None if prompt is None else sha256(prompt),
        }

    def _scd_gpu_preflight(self, state, prefix, need_source, source_input_tokens=None):
        p = state['registry'].profile
        n = len(prefix.token_ids)
        device = state['registry'].device
        source_n = n if source_input_tokens is None else source_input_tokens
        if type(source_n) is not int or source_n not in (n, n + 1):
            raise IdentityMismatch('source capture extent differs')
        target = n * p.target.num_layers * p.target.num_kv_heads * p.target.head_dim * 4
        features = state['backend'].required_features(p)
        per_token = p.source.num_layers * p.source.num_kv_heads * p.source.head_dim * 4
        per_token += len(features.hidden_hooks) * state['provider'].model.config.hidden_size * 2
        source = source_n * per_token
        if source_n != n:
            source += n * per_token
        if source > 8 * 1024**3 or target > state['registry'].capacity_bytes:
            raise StoreCapacityExceeded('resident logical state cap exceeded')

        reserved = torch.cuda.memory_reserved(device)
        allocated = torch.cuda.memory_allocated(device)
        reclaimable = reserved - allocated
        required = 8 * target + (source if need_source else 0) + 512 * 1024**2

        if 'baseline_free_bytes' in state and 'baseline_reserved_bytes' in state:
            estimated_free = state['baseline_free_bytes'] - max(0, reserved - state['baseline_reserved_bytes'])
            if estimated_free + reclaimable < required * 2:
                free, _ = torch.cuda.mem_get_info(device)
            else:
                free = estimated_free
        else:
            free, _ = torch.cuda.mem_get_info(device)

        if free + reclaimable < required:
            raise StoreCapacityExceeded('insufficient conversion/snapshot peak headroom')
        return {
            'free_bytes': free,
            'allocator_reclaimable_bytes': reclaimable,
            'estimated_required_bytes': required,
            'source_state_bound_bytes': source,
            'target_snapshot_bytes': target,
        }

    def _memo_lookup(self, state, prefix: PrefixContract, source_info: dict) -> TargetKV | None:
        if state['memo_max_entries'] <= 0:
            return None
        key = (sha256(prefix), source_info['source_prompt_digest'])
        cache: OrderedDict = state['memo_cache']
        packed_cached = cache.get(key)
        if packed_cached is None:
            state['memo_misses'] += 1
            return None
        cache.move_to_end(key)
        state['memo_hits'] += 1
        p = state['registry'].profile
        D = p.target.head_dim
        with torch.inference_mode(False), torch.no_grad():
            packed_clone = packed_cached.clone()
            layers = tuple(
                (
                    packed_clone[i, :, :, :D].transpose(0, 1).unsqueeze(0),
                    packed_clone[i, :, :, D:].transpose(0, 1).unsqueeze(0),
                )
                for i in range(p.target.num_layers)
            )
        return TargetKV(p.target, p.reader, prefix, layers, fingerprint(p, prefix), packed_kv=packed_clone)

    def _memo_store(self, state, prefix: PrefixContract, source_info: dict, packed_kv: torch.Tensor | None) -> None:
        if state['memo_max_entries'] <= 0 or packed_kv is None:
            return
        nbytes = packed_kv.numel() * packed_kv.element_size()
        if nbytes > state['memo_max_bytes']:
            return
        key = (sha256(prefix), source_info['source_prompt_digest'])
        cache: OrderedDict = state['memo_cache']
        if key in cache:
            old = cache.pop(key)
            state['memo_bytes'] -= old.numel() * old.element_size()
        while cache and (len(cache) >= state['memo_max_entries'] or state['memo_bytes'] + nbytes > state['memo_max_bytes']):
            _, evicted = cache.popitem(last=False)
            state['memo_bytes'] -= evicted.numel() * evicted.element_size()
        with torch.inference_mode(False), torch.no_grad():
            stored = packed_kv.detach().clone()
        cache[key] = stored
        state['memo_bytes'] += nbytes

    def _validate_admit_bounds(self, state, prompt_ids, max_new_tokens, import_policy):
        if (
            type(max_new_tokens) is not int
            or max_new_tokens < 1
            or len(prompt_ids) + max_new_tokens > state['max_model_len']
        ):
            raise IdentityMismatch('resident prompt plus generation exceeds supported window')
        if import_policy == 'fullprefix':
            cfg = self.vllm_config
            if (
                cfg.cache_config.enable_prefix_caching
                or cfg.scheduler_config.enable_chunked_prefill
                or not cfg.scheduler_config.disable_hybrid_kv_cache_manager
                or cfg.scheduler_config.max_num_seqs != 1
            ):
                raise UnsupportedProfile('fullprefix requires serial APC-off single-group resident configuration')
            if state['registry'].profile.target.attention.get('sliding_window') is not None:
                extra = cfg.kv_transfer_config
                if (
                    cfg.scheduler_config.async_scheduling is not False
                    or cfg.speculative_config is not None
                    or extra.get_from_extra_config('async_loading', False) is not False
                ):
                    raise UnsupportedProfile('bounded SWA fullprefix requires synchronous private scheduling')

    @torch.inference_mode()
    def scd_gpu_capture(self, *, session_id, prefix_data, full_prompt_ids=None):
        state = self._scd_gpu_context(session_id)
        registry = state['registry']
        with registry.lock:
            registry.require_preparable()
            if state['source'] is not None:
                raise StoreCapacityExceeded('one source state allowed')
            prefix = self._scd_gpu_prefix(state, prefix_data)
            prompt, source_info = self._scd_source_prompt(state, prefix, full_prompt_ids)
            headroom = self._scd_gpu_preflight(state, prefix, True, source_info['source_input_tokens'])
            torch.cuda.reset_peak_memory_stats(registry.device)
            t = time.perf_counter()
            with torch.autocast(device_type='cuda', enabled=False):
                source = (
                    state['provider'].capture(prefix, state['backend'].required_features(registry.profile))
                    if prompt is None
                    else state['provider'].capture(
                        prefix,
                        state['backend'].required_features(registry.profile),
                        full_prompt_ids=prompt,
                    )
                )
            registry.synchronize()
            handle = secrets.token_hex(32)
            state['source'] = (handle, source, source_info)
            return {
                'source_handle': handle,
                'prefix_digest': sha256(prefix),
                'capture_ms': (time.perf_counter() - t) * 1000,
                'peak_allocated_bytes': torch.cuda.max_memory_allocated(registry.device),
                'headroom': headroom,
                **source_info,
            }

    @torch.inference_mode()
    def scd_gpu_prepare(self, *, session_id, prefix_data, mode, source_handle=None, full_prompt_ids=None):
        state = self._scd_gpu_context(session_id)
        registry = state['registry']
        with registry.lock:
            registry.require_preparable()
            prefix = self._scd_gpu_prefix(state, prefix_data)
            prompt, source_info = self._scd_source_prompt(state, prefix, full_prompt_ids)
            if mode not in ('E', 'H') or (mode == 'E' and (source_handle is not None or state['source'] is not None)):
                raise IdentityMismatch('E needs fresh source, H needs its saved source handle')
            if mode == 'H' and (
                state['source'] is None
                or state['source'][0] != source_handle
                or state['source'][1].prefix != prefix
                or (len(state['source']) == 2 and prompt is not None)
                or (len(state['source']) > 2 and state['source'][2] != source_info)
            ):
                raise IdentityMismatch('H source handle, prefix or full prompt differs')
            headroom = self._scd_gpu_preflight(state, prefix, mode == 'E', source_info['source_input_tokens'])
            torch.cuda.reset_peak_memory_stats(registry.device)
            source_ms = 0.0
            t = time.perf_counter()
            try:
                kv = self._memo_lookup(state, prefix, source_info) if mode == 'E' else None
                memo_hit = kv is not None
                if kv is None:
                    if mode == 'E':
                        with torch.autocast(device_type='cuda', enabled=False):
                            source = (
                                state['provider'].capture(prefix, state['backend'].required_features(registry.profile))
                                if prompt is None
                                else state['provider'].capture(
                                    prefix,
                                    state['backend'].required_features(registry.profile),
                                    full_prompt_ids=prompt,
                                )
                            )
                        registry.synchronize()
                        source_ms = (time.perf_counter() - t) * 1000
                    else:
                        source = state['source'][1]
                    state['source'] = None
                    t = time.perf_counter()
                    with torch.autocast(device_type='cuda', enabled=False):
                        kv = state['backend'].convert(
                            source,
                            state['assets'],
                            state['projections'],
                            return_packed=True,
                        )
                    registry.synchronize()
                    convert_ms = (time.perf_counter() - t) * 1000
                    del source
                    if mode == 'E':
                        self._memo_store(state, prefix, source_info, kv.packed_kv)
                else:
                    convert_ms = 0.0
                t = time.perf_counter()
                receipt = registry.publish(kv, adopt_packed=True, sync_on_publish=True, check_finite=True)
                publish_ms = (time.perf_counter() - t) * 1000
                return {
                    **receipt,
                    'source_prefill_ms': source_ms,
                    'convert_ms': convert_ms,
                    'publish_ms': publish_ms,
                    'memo_hit': memo_hit,
                    'peak_allocated_bytes': torch.cuda.max_memory_allocated(registry.device),
                    'peak_reserved_bytes': torch.cuda.max_memory_reserved(registry.device),
                    'headroom': headroom,
                    **source_info,
                }
            except BaseException:
                try:
                    registry.synchronize()
                except BaseException:
                    registry.poisoned = True
                state['source'] = None
                raise

    @torch.inference_mode()
    def scd_gpu_prepare_and_admit(
        self,
        *,
        session_id,
        prefix_data,
        prompt_ids,
        request_id,
        max_new_tokens,
        mode='E',
        source_handle=None,
        import_policy='blocktail',
        full_prompt_ids=None,
        stream_ordered=True,
    ):
        """Fused single-RPC prepare + publish + admit with optional stream-ordered pipelining."""
        state = self._scd_gpu_context(session_id)
        registry = state['registry']
        with registry.lock:
            registry.require_preparable()
            self._validate_admit_bounds(state, prompt_ids, max_new_tokens, import_policy)
            prefix = self._scd_gpu_prefix(state, prefix_data)
            prompt, source_info = self._scd_source_prompt(state, prefix, full_prompt_ids)
            if mode not in ('E', 'H') or (mode == 'E' and (source_handle is not None or state['source'] is not None)):
                raise IdentityMismatch('E needs fresh source, H needs its saved source handle')
            if mode == 'H' and (
                state['source'] is None
                or state['source'][0] != source_handle
                or state['source'][1].prefix != prefix
                or (len(state['source']) == 2 and prompt is not None)
                or (len(state['source']) > 2 and state['source'][2] != source_info)
            ):
                raise IdentityMismatch('H source handle, prefix or full prompt differs')

            headroom = self._scd_gpu_preflight(state, prefix, mode == 'E', source_info['source_input_tokens'])
            torch.cuda.reset_peak_memory_stats(registry.device)
            stream = torch.cuda.current_stream(registry.device)
            source_ms = 0.0
            convert_ms = 0.0
            ev_pair = {}
            try:
                kv = self._memo_lookup(state, prefix, source_info) if mode == 'E' else None
                memo_hit = kv is not None
                if kv is None:
                    if mode == 'E':
                        t0 = time.perf_counter()
                        ev_s0 = torch.cuda.Event(enable_timing=True)
                        ev_s1 = torch.cuda.Event(enable_timing=True)
                        ev_s0.record(stream)
                        with torch.autocast(device_type='cuda', enabled=False):
                            source = (
                                state['provider'].capture(prefix, state['backend'].required_features(registry.profile))
                                if prompt is None
                                else state['provider'].capture(
                                    prefix,
                                    state['backend'].required_features(registry.profile),
                                    full_prompt_ids=prompt,
                                )
                            )
                        ev_s1.record(stream)
                        if not stream_ordered:
                            registry.synchronize()
                        source_ms = (time.perf_counter() - t0) * 1000
                        ev_pair['source'] = (ev_s0, ev_s1)
                    else:
                        source = state['source'][1]
                    state['source'] = None

                    t1 = time.perf_counter()
                    ev_c0 = torch.cuda.Event(enable_timing=True)
                    ev_c1 = torch.cuda.Event(enable_timing=True)
                    ev_c0.record(stream)
                    with torch.autocast(device_type='cuda', enabled=False):
                        kv = state['backend'].convert(
                            source,
                            state['assets'],
                            state['projections'],
                            return_packed=True,
                        )
                    ev_c1.record(stream)
                    if not stream_ordered:
                        registry.synchronize()
                    convert_ms = (time.perf_counter() - t1) * 1000
                    ev_pair['convert'] = (ev_c0, ev_c1)
                    del source
                    if mode == 'E':
                        self._memo_store(state, prefix, source_info, kv.packed_kv)

                t2 = time.perf_counter()
                receipt = registry.publish(
                    kv,
                    adopt_packed=True,
                    sync_on_publish=not stream_ordered,
                    check_finite=not stream_ordered,
                )
                publish_ms = (time.perf_counter() - t2) * 1000

                block = self.vllm_config.cache_config.block_size
                admission = registry.admit(
                    receipt['artifact_id'],
                    prompt_ids,
                    request_id,
                    block,
                    import_policy=import_policy,
                )
                if ev_pair:
                    state['pending_events'][receipt['artifact_id']] = ev_pair
                return {
                    **receipt,
                    'admission': to_data(admission),
                    'source_prefill_ms': source_ms,
                    'convert_ms': convert_ms,
                    'publish_ms': publish_ms,
                    'memo_hit': memo_hit,
                    'stream_ordered': bool(stream_ordered),
                    'peak_allocated_bytes': torch.cuda.max_memory_allocated(registry.device),
                    'peak_reserved_bytes': torch.cuda.max_memory_reserved(registry.device),
                    'headroom': headroom,
                    **source_info,
                }
            except BaseException:
                try:
                    registry.synchronize()
                except BaseException:
                    registry.poisoned = True
                state['source'] = None
                raise

    @torch.inference_mode()
    def scd_gpu_prepare_and_admit_batch(
        self,
        *,
        session_id,
        batch_items,
        max_new_tokens,
        import_policy='blocktail',
        stream_ordered=True,
    ):
        """Fused single-RPC batch prepare + within-batch prefix deduplication + publish + admit."""
        state = self._scd_gpu_context(session_id)
        registry = state['registry']
        with registry.lock:
            registry.require_preparable()
            if not batch_items or len(registry._entries) + len(batch_items) > registry.max_entries:
                raise StoreCapacityExceeded('resident snapshot entry limit reached')
            if state['source'] is not None:
                raise IdentityMismatch('batch E preparation requires no saved single-request H source state')

            parsed = []
            p = registry.profile
            per_tok_target = p.target.num_layers * p.target.num_kv_heads * p.target.head_dim * 4
            total_new_bytes = 0
            for item in batch_items:
                if item.get('mode', 'E') != 'E':
                    raise UnsupportedProfile('batch fused prepare supports mode="E" only')
                prompt_ids = item['prompt_ids']
                self._validate_admit_bounds(state, prompt_ids, max_new_tokens, import_policy)
                prefix = self._scd_gpu_prefix(state, item['prefix_data'])
                prompt, source_info = self._scd_source_prompt(state, prefix, item.get('full_prompt_ids'))
                total_new_bytes += len(prefix.token_ids) * per_tok_target
                parsed.append((item, prefix, prompt, source_info))

            existing_bytes = sum(e['bytes'] for e in registry._entries.values())
            if existing_bytes + total_new_bytes > registry.capacity_bytes:
                raise StoreCapacityExceeded('resident snapshots exceed aggregate capacity')

            torch.cuda.reset_peak_memory_stats(registry.device)
            block = self.vllm_config.cache_config.block_size
            D = p.target.head_dim
            batch_cache: dict[tuple[str, str | None], torch.Tensor] = {}
            created: list[tuple[str, str | None]] = []
            published_stage = []
            results = []
            try:
                # Pass 1: publish all B snapshots while every entry remains in 'READY' phase
                for item, prefix, prompt, source_info in parsed:
                    dedup_key = (sha256(prefix), source_info['source_prompt_digest'])
                    source_ms = 0.0
                    convert_ms = 0.0
                    memo_hit = False
                    batch_dedup_hit = False
                    headroom = None

                    if dedup_key in batch_cache:
                        batch_dedup_hit = True
                        t1 = time.perf_counter()
                        base_packed = batch_cache[dedup_key]
                        with torch.inference_mode(False), torch.no_grad():
                            packed_clone = base_packed.clone()
                            layers = tuple(
                                (
                                    packed_clone[i, :, :, :D].transpose(0, 1).unsqueeze(0),
                                    packed_clone[i, :, :, D:].transpose(0, 1).unsqueeze(0),
                                )
                                for i in range(p.target.num_layers)
                            )
                        kv = TargetKV(p.target, p.reader, prefix, layers, fingerprint(p, prefix), packed_kv=packed_clone)
                        convert_ms = (time.perf_counter() - t1) * 1000
                    else:
                        kv = self._memo_lookup(state, prefix, source_info)
                        if kv is not None:
                            memo_hit = True
                            batch_cache[dedup_key] = kv.packed_kv
                        else:
                            headroom = self._scd_gpu_preflight(state, prefix, True, source_info['source_input_tokens'])
                            t0 = time.perf_counter()
                            with torch.autocast(device_type='cuda', enabled=False):
                                source = (
                                    state['provider'].capture(prefix, state['backend'].required_features(p))
                                    if prompt is None
                                    else state['provider'].capture(
                                        prefix,
                                        state['backend'].required_features(p),
                                        full_prompt_ids=prompt,
                                    )
                                )
                            if not stream_ordered:
                                registry.synchronize()
                            source_ms = (time.perf_counter() - t0) * 1000

                            t1 = time.perf_counter()
                            with torch.autocast(device_type='cuda', enabled=False):
                                kv = state['backend'].convert(
                                    source,
                                    state['assets'],
                                    state['projections'],
                                    return_packed=True,
                                )
                            if not stream_ordered:
                                registry.synchronize()
                            convert_ms = (time.perf_counter() - t1) * 1000
                            del source
                            self._memo_store(state, prefix, source_info, kv.packed_kv)
                            batch_cache[dedup_key] = kv.packed_kv

                    t2 = time.perf_counter()
                    receipt = registry.publish(
                        kv,
                        adopt_packed=True,
                        sync_on_publish=not stream_ordered,
                        check_finite=not stream_ordered,
                    )
                    publish_ms = (time.perf_counter() - t2) * 1000
                    created.append((receipt['artifact_id'], None))
                    published_stage.append((
                        item,
                        receipt,
                        source_ms,
                        convert_ms,
                        publish_ms,
                        memo_hit,
                        batch_dedup_hit,
                        headroom,
                        source_info,
                    ))

                # Pass 2: admit all B snapshots from 'READY' -> 'ADMITTED'
                for idx, (
                    item,
                    receipt,
                    source_ms,
                    convert_ms,
                    publish_ms,
                    memo_hit,
                    batch_dedup_hit,
                    headroom,
                    source_info,
                ) in enumerate(published_stage):
                    admission = registry.admit(
                        receipt['artifact_id'],
                        item['prompt_ids'],
                        item['request_id'],
                        block,
                        import_policy=import_policy,
                    )
                    created[idx] = (receipt['artifact_id'], admission.lease_id)
                    results.append({
                        **receipt,
                        'admission': to_data(admission),
                        'source_prefill_ms': source_ms,
                        'convert_ms': convert_ms,
                        'publish_ms': publish_ms,
                        'memo_hit': memo_hit,
                        'batch_dedup_hit': batch_dedup_hit,
                        'stream_ordered': bool(stream_ordered),
                        'peak_allocated_bytes': torch.cuda.max_memory_allocated(registry.device),
                        'peak_reserved_bytes': torch.cuda.max_memory_reserved(registry.device),
                        'headroom': headroom,
                        **source_info,
                    })
                return {
                    'items': results,
                    'peak_allocated_bytes': torch.cuda.max_memory_allocated(registry.device),
                    'peak_reserved_bytes': torch.cuda.max_memory_reserved(registry.device),
                }
            except BaseException:
                for art_id, lease_id in created:
                    try:
                        registry.release(art_id, lease_id)
                    except BaseException:
                        registry.poisoned = True
                try:
                    registry.synchronize()
                except BaseException:
                    registry.poisoned = True
                raise

    def scd_gpu_finalize_batch(self, *, session_id, admissions_data):
        """Fused single-RPC finish + release + status for completed batch imports."""
        state = self._scd_gpu_context(session_id)
        registry = state['registry']
        with registry.lock:
            receipts = []
            for adm_data in admissions_data:
                if adm_data is None:
                    receipts.append(None)
                    continue
                admission = admission_from_data(adm_data)
                state['pending_events'].pop(admission.artifact_id, None)
                receipt = registry.finish(admission)
                registry.release(admission.artifact_id, admission.lease_id)
                receipts.append(receipt)
            return {
                'receipts': receipts,
                'status': self.scd_gpu_status(session_id=session_id),
            }

    def scd_gpu_clear_memo(self, *, session_id):
        """Clear worker-side prefix memoization cache without restarting the engine."""
        state = self._scd_gpu_context(session_id)
        with state['registry'].lock:
            state['memo_cache'].clear()
            state['memo_bytes'] = 0
            state['memo_hits'] = 0
            state['memo_misses'] = 0
            return self.scd_gpu_status(session_id=session_id)

    def scd_gpu_admit(self, *, session_id, artifact_id, prompt_ids, request_id, max_new_tokens, import_policy='blocktail'):
        state = self._scd_gpu_context(session_id)
        self._validate_admit_bounds(state, prompt_ids, max_new_tokens, import_policy)
        block = self.vllm_config.cache_config.block_size
        return to_data(state['registry'].admit(artifact_id, prompt_ids, request_id, block, import_policy=import_policy))

    def scd_gpu_finish(self, *, session_id, admission_data):
        state = self._scd_gpu_context(session_id)
        registry = state['registry']
        admission = admission_from_data(admission_data)
        with registry.lock:
            state['pending_events'].pop(admission.artifact_id, None)
            return registry.finish(admission)

    def scd_gpu_finalize(self, *, session_id, admission_data, output_completion_args=None):
        """Fused single-RPC post-generation receipt + optional output_completion + status."""
        state = self._scd_gpu_context(session_id)
        registry = state['registry']
        admission = admission_from_data(admission_data)
        with registry.lock:
            proof = None
            if output_completion_args is not None:
                completion = state.get('output_completion')
                if completion is None:
                    raise UnsupportedProfile('request output completion is not enabled')
                proof = completion.take(
                    output_completion_args['actual_request_id'],
                    output_completion_args['first_token_id'],
                    output_completion_args['first_observed_ns'],
                )
            receipt = registry.finish(admission)
            ev_pair = state['pending_events'].pop(admission.artifact_id, None)
            gpu_timings = {}
            if ev_pair:
                for key, (ev0, ev1) in ev_pair.items():
                    if ev1.query():
                        gpu_timings[f'gpu_{key}_ms'] = float(ev0.elapsed_time(ev1))
            registry.release(admission.artifact_id, admission.lease_id)
            return {
                'receipt': receipt,
                'first_token_completion': proof,
                'gpu_timings': gpu_timings,
                'status': self.scd_gpu_status(session_id=session_id),
            }

    def scd_gpu_release(self, *, session_id, artifact_id, lease_id=None):
        state = self._scd_gpu_context(session_id)
        registry = state['registry']
        with registry.lock:
            state['pending_events'].pop(artifact_id, None)
            registry.release(artifact_id, lease_id)
        return self.scd_gpu_status(session_id=session_id)

    def scd_gpu_clear(self, *, session_id, close_output_completion=False):
        if type(close_output_completion) is not bool:
            raise ValueError('output completion close flag must be boolean')
        state = self._scd_gpu_context(session_id)
        with state['registry'].lock:
            state['registry'].clear()
            state['source'] = None
            if 'pending_events' in state:
                state['pending_events'].clear()
            if 'memo_cache' in state:
                state['memo_cache'].clear()
                state['memo_bytes'] = 0
            completion = state.get('output_completion')
            if completion is not None:
                if close_output_completion:
                    completion.close()
                    state['output_completion'] = None
                else:
                    completion.require_idle()
            elif close_output_completion:
                raise IdentityMismatch('output completion guard missing at close')
            return self.scd_gpu_status(session_id=session_id)

    def scd_gpu_output_completion(self, *, session_id, actual_request_id, first_token_id, first_observed_ns):
        state = self._scd_gpu_context(session_id)
        completion = state.get('output_completion')
        if completion is None:
            raise UnsupportedProfile('request output completion is not enabled')
        with state['registry'].lock:
            return completion.take(actual_request_id, first_token_id, first_observed_ns)

    def scd_gpu_synchronize(self, *, session_id):
        state = self._scd_gpu_context(session_id)
        registry = state['registry']
        with registry.lock:
            if registry.device.type != 'cuda':
                raise UnsupportedProfile('resident synchronization requires its configured CUDA device')
            registry.synchronize()
            return {
                'worker_cuda_synchronized': True,
                'device': str(registry.device),
                'session_id': registry.session_id,
            }

    def scd_gpu_status(self, *, session_id):
        state = self._scd_gpu_context(session_id)
        registry = state['registry']
        with registry.lock:
            return {
                **registry.status(),
                'source_states': int(state['source'] is not None),
                'source_model_bytes': state['source_model_bytes'],
                'asset_gpu_bytes': state['asset_gpu_bytes'],
                'memo_entries': len(state['memo_cache']),
                'memo_bytes': state['memo_bytes'],
                'memo_hits': state['memo_hits'],
                'memo_misses': state['memo_misses'],
                'allocated_bytes': torch.cuda.memory_allocated(registry.device),
                'reserved_bytes': torch.cuda.memory_reserved(registry.device),
                'peak_allocated_bytes': torch.cuda.max_memory_allocated(registry.device),
                'peak_reserved_bytes': torch.cuda.max_memory_reserved(registry.device),
            }

    @torch.inference_mode()
    def scd_gpu_diagnostic_digest(self, *, session_id, artifact_id):
        from safetensors.torch import save

        state = self._scd_gpu_context(session_id)
        registry = state['registry']
        if not state['diagnostics']:
            raise UnsupportedProfile('diagnostic CPU transfer explicitly disabled')
        with registry.lock:
            entry = registry._get(artifact_id)
            if entry['phase'] != 'READY':
                raise IdentityMismatch('diagnostic only on unconsumed snapshot')
            tensors = {
                f'{i}.{kind}': t.detach().cpu().contiguous()
                for i, pair in enumerate(entry['kv'].layers)
                for kind, t in zip(('k', 'v'), pair)
            }
            body = save(tensors)
            return {
                'scope': 'diagnostic_outside_performance_window',
                'sha256': hashlib.sha256(body).hexdigest(),
                'serialized_bytes': len(body),
                'diagnostic_kv_cpu_bytes': entry['bytes'],
            }

    def destroy_worker_extension(self):
        state = getattr(self, '_scd_gpu_state', None)
        if state is not None:
            try:
                state['provider'].close()
                unregister_registry(state['registry'].session_id)
            finally:
                self._scd_gpu_state = None
