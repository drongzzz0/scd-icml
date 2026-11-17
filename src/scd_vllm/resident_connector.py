"""Single-use same-worker GPU connector with stream-ordered fast page scatter."""
from dataclasses import dataclass, field
import torch

from scd_runtime.contracts.types import profile_from_dict
from scd_runtime.errors import SCDRuntimeError, IdentityMismatch, UnsupportedProfile
from scd_runtime.contracts.prefix import get_import_policy, check_fullprefix_scope, allocated_prefix_slots
from scd_runtime.transport.resident import admission_from_data, validate_scheduled_request, get_registry
from .connector import KVConnectorBase_V1, KVConnectorMetadata, _VLLM_AVAILABLE
from .capability import inspect_engine
from .layout import prepare_layer_writer
from .registered_kv import RegisteredKVMixin
from .async_connector import EventConnectorMixin, initialize_request_mode, add_routing_metadata


@dataclass
class ResidentConnectorMetadata(KVConnectorMetadata):
    loads: list = field(default_factory=list)
    cancelled: set = field(default_factory=set)


class SCDGPUConnectorV1(EventConnectorMixin, RegisteredKVMixin, KVConnectorBase_V1):
    def __init__(self, config, role, kv_cache_config):
        if not _VLLM_AVAILABLE:
            raise ImportError('vLLM V1 is required to instantiate SCDGPUConnectorV1')
        super().__init__(config, role, kv_cache_config)
        from scd_runtime.contracts.capability import check_capability
        extra = config.kv_transfer_config
        strict_pin = bool(extra.get_from_extra_config('strict_commit_pin', False))
        self.profile = profile_from_dict(extra.get_from_extra_config('scd_profile', {}))
        self.session = extra.get_from_extra_config('resident_session', '')
        self.capability = inspect_engine(config, kv_cache_config, strict_commit_pin=strict_pin)
        check_capability(self.profile, self.capability, strict_commit_pin=strict_pin)
        initialize_request_mode(self, extra, self.profile)
        if extra.kv_load_failure_policy != 'fail':
            raise UnsupportedProfile('resident connector requires fail-closed loading')
        self._admissions = {}
        self._pending = {}
        self._errors = set()
        self._fullprefix_serial = config.scheduler_config.max_num_seqs == 1
        self._fullprefix_swa_serial = (
            self._fullprefix_serial
            and getattr(config.scheduler_config, 'async_scheduling', None) is False
            and getattr(config, 'speculative_config', False) is None
            and extra.get_from_extra_config('async_loading', False) is False
        )
        self._load_diagnostics = extra.get_from_extra_config('resident_diagnostics', False)
        if type(self._load_diagnostics) is not bool:
            raise UnsupportedProfile('resident diagnostics requires an explicit boolean')
        self._stream_fast_load = bool(extra.get_from_extra_config('stream_fast_load', True))

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        params = request.kv_transfer_params or {}
        if not params:
            return 0, False
        if set(params) != {'scd_gpu_admission'} or params['scd_gpu_admission'] is None:
            raise IdentityMismatch('unexpected resident transfer metadata')
        a = admission_from_data(params['scd_gpu_admission'])
        if a.session_id != self.session:
            raise IdentityMismatch('engine session differs')
        if num_computed_tokens != 0:
            raise UnsupportedProfile('resident connector rejects resumed/preempted imports')
        validate_scheduled_request(a, request, self.profile, self.capability.block_size)
        if get_import_policy(a.plan) == 'fullprefix':
            check_fullprefix_scope(self.profile, self.capability, resident=True)
            if self.profile.target.attention.get('sliding_window') is not None:
                if not self._fullprefix_swa_serial:
                    raise UnsupportedProfile('bounded SWA fullprefix requires synchronous non-speculative serial scheduling')
                maximum = getattr(request, 'max_tokens', None)
                if (type(maximum) is not int or maximum < 1
                        or len(request.prompt_token_ids) + maximum > self.capability.max_model_len):
                    raise IdentityMismatch('actual SWA request exceeds the complete-request bound')
            if not self._fullprefix_serial:
                raise UnsupportedProfile('fullprefix requires one configured resident request')
        self._admissions[request.request_id] = a
        return a.plan.imported_tokens, self._async_loading

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        if num_external_tokens > 0:
            a = self._admissions[request.request_id]
            if num_external_tokens != a.plan.imported_tokens:
                raise IdentityMismatch('partial resident allocation unsupported')
            groups = blocks.get_block_ids()
            if len(groups) != 1:
                raise UnsupportedProfile('resident import requires one actual cache group')
            slots = allocated_prefix_slots(groups[0], self.capability.block_size, a.plan)
            self._pending[request.request_id] = (a, slots)

    def build_connector_meta(self, scheduler_output):
        meta = ResidentConnectorMetadata()
        B = self.capability.block_size
        if self._async_loading:
            meta.loads = [(rid, *value) for rid, value in self._pending.items()]
            self._pending.clear()
        resumed = getattr(scheduler_output.scheduled_cached_reqs, 'resumed_req_ids', ())
        if any(rid in self._admissions for rid in resumed):
            raise UnsupportedProfile('single-use resident import cannot resume after preemption')
        for request in scheduler_output.scheduled_new_reqs:
            rid = request.req_id
            if rid not in self._pending:
                continue
            a, _ = self._pending.pop(rid)
            slots = (
                allocated_prefix_slots(request.block_ids[0], B, a.plan)
                if get_import_policy(a.plan) == 'fullprefix'
                else [block * B + offset for block in request.block_ids[0] for offset in range(B)][:a.plan.imported_tokens]
            )
            if not slots or len(slots) != a.plan.imported_tokens:
                raise IdentityMismatch('resident allocated slots incomplete')
            meta.loads.append((rid, a, slots))
        add_routing_metadata(self, meta, scheduler_output)
        return meta

    def start_load_kv(self, forward_context, **kwargs):
        meta = self._get_connector_metadata()
        if meta is None:
            return
        for rid, admission, indices in meta.loads:
            if self._async_loading:
                self._load_async(rid, admission, indices)
                continue
            B = self.capability.block_size
            blocks = {s // B for s in indices}
            if not blocks:
                raise IdentityMismatch('empty resident load cannot be successful')
            try:
                registry = get_registry(self.session)
                destinations = self._registered_destinations()
                entry = registry._entries.get(admission.artifact_id)
                if (self._stream_fast_load and not self._load_diagnostics
                        and entry is not None and entry.get('packed_kind') == 'lnh2d'):
                    registry.load_packed_fast(
                        admission, rid, destinations, indices, self.capability, sync_after_write=False
                    )
                else:
                    with torch.inference_mode(False), torch.no_grad():
                        slots = torch.tensor(indices, dtype=torch.int64, device='cpu')
                    writer = prepare_layer_writer(destinations, slots, self.capability)
                    if self._load_diagnostics:
                        from scd_runtime.transport.load_diagnostics import LoadDiagnostics
                        registry.load(admission, rid, destinations, slots, self.capability, writer, observer=LoadDiagnostics(admission, rid))
                    else:
                        registry.load(admission, rid, destinations, slots, self.capability, writer)
            except (SCDRuntimeError, ValueError, RuntimeError, KeyError):
                self._errors.update(blocks)

    def get_block_ids_with_load_errors(self):
        result = self._errors
        self._errors = set()
        if self._events is not None:
            result.update(self._events.errors())
        return result

    def wait_for_layer_load(self, layer_name):
        if self._events is not None:
            self._events.wait_layer(layer_name)

    def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs):
        pass

    def wait_for_save(self):
        pass

    def request_finished(self, request, block_ids):
        a = self._admissions.pop(request.request_id, None)
        if a is not None and self._async_loading:
            self._cancelled_requests.add(request.request_id)
        else:
            self._pending.pop(request.request_id, None)
        return False, None

    def _load_async(self, rid, admission, indices):
        registry = get_registry(self.session)
        complete = False
        begun = False

        def release():
            if not complete:
                if begun:
                    registry.fail_async_load(admission)
                registry.release(admission.artifact_id, admission.lease_id)

        if rid in self._events.cancelled:
            self._events.fail_without_writes(rid, indices, release)
            return
        try:
            destinations = self._registered_destinations()
            slots = torch.tensor(indices, dtype=torch.int64, device='cpu')
            kv = registry.begin_async_load(admission, rid, destinations, slots, self.capability)
            begun = True
            if not all(t.is_cuda for pair in kv.layers for t in pair):
                raise UnsupportedProfile('resident asynchronous payload must remain on the worker GPU')

            def received():
                nonlocal complete
                registry.complete_async_load(admission, rid, {
                    'completion': 'copy_stream_event',
                    'device_wide_synchronize': False,
                    'kv_payload_cpu_bytes': 0,
                    'kv_payload_file_bytes': 0,
                })
                complete = True

            self._events.submit(rid, admission.lease_id, kv, destinations, indices, on_complete=received, release=release)
        except (SCDRuntimeError, ValueError, RuntimeError, KeyError):
            if not self._events.pending(rid):
                self._events.fail_without_writes(rid, indices, release)
