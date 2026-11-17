"""Consumer-only V1 file-store adapter.

Supports both direct vLLM plugin usage (via ``vllm.general_plugins``) and
``VLLMConsumer`` orchestration.
"""
from dataclasses import dataclass, field
import os
import time
import torch

from scd_runtime.contracts.types import PrefixContract, ImportPlan, strict_construct, profile_from_dict, to_data
from scd_runtime.contracts.identity import fingerprint, profile_fingerprint, sha256
from scd_runtime.contracts.prefix import (
    plan_import,
    import_plan_from_data,
    get_import_policy,
    check_fullprefix_scope,
    allocated_prefix_slots,
)
from scd_runtime.contracts.capability import check_capability
from scd_runtime.transport.store import ArtifactStore, ArtifactLease
from scd_runtime.errors import SCDRuntimeError, IdentityMismatch, UnsupportedProfile
from .layout import write_layer, write_packed_layers_fast
from .registered_kv import RegisteredKVMixin
from .async_connector import EventConnectorMixin, initialize_request_mode, add_routing_metadata


@dataclass(frozen=True)
class Admission:
    lease: ArtifactLease
    plan: ImportPlan
    profile_fingerprint: str
    prefix_digest: str


def admission_from_data(data):
    if set(data) != {'lease', 'plan', 'profile_fingerprint', 'prefix_digest'}:
        raise IdentityMismatch('invalid admission')
    return Admission(
        ArtifactLease(**data['lease']),
        import_plan_from_data(data['plan']),
        data['profile_fingerprint'],
        data['prefix_digest'],
    )


def _policy_receipt(plan):
    return (
        {
            'import_policy': 'fullprefix',
            'original_prefix_tokens': plan.original_prefix_tokens,
            'tail_recomputed_tokens': plan.tail_recomputed_tokens,
        }
        if get_import_policy(plan) == 'fullprefix'
        else {}
    )


class ConnectorCoordinator:
    def __init__(self, store, profile, capability, *, fullprefix_serial=False, strict_commit_pin=False):
        check_capability(profile, capability, strict_commit_pin=strict_commit_pin)
        if type(fullprefix_serial) is not bool:
            raise UnsupportedProfile('explicit configured C1 flag required')
        self.store = store
        self.profile = profile
        self.capability = capability
        self.errors = {}
        self._fullprefix_serial = fullprefix_serial

    def _check_fullprefix(self):
        check_fullprefix_scope(self.profile, self.capability)
        if self.capability.prefix_cache or not self._fullprefix_serial:
            raise UnsupportedProfile('file fullprefix requires configured serial C1 APC-off engine')

    def admit(self, artifact_id, prompt_ids, request_id, *, import_policy='blocktail'):
        if self.capability.max_model_len is not None and len(prompt_ids) >= self.capability.max_model_len:
            raise UnsupportedProfile('request leaves no room within the supported context bound')
        lease, meta = self.store.pin_with_metadata(artifact_id, request_id)
        try:
            prefix = strict_construct(PrefixContract, meta['prefix'])
            if (meta['target'] != to_data(self.profile.target)
                    or meta['reader'] != to_data(self.profile.reader)
                    or meta['method_fingerprint'] != fingerprint(self.profile, prefix)):
                raise IdentityMismatch('artifact model/reader/method differs from engine')
            if import_policy == 'fullprefix':
                self._check_fullprefix()
            return Admission(
                lease,
                plan_import(prefix, prompt_ids, self.capability.block_size, import_policy=import_policy),
                profile_fingerprint(self.profile),
                sha256(prefix),
            )
        except BaseException:
            self.store.release(lease)
            raise

    def matched(self, admission, num_computed_tokens):
        if admission.profile_fingerprint != profile_fingerprint(self.profile):
            raise IdentityMismatch('admission for different profile')
        if get_import_policy(admission.plan) == 'fullprefix':
            self._check_fullprefix()
        if num_computed_tokens != 0:
            raise UnsupportedProfile('SCD import requires zero local hits and cannot resume')
        return max(0, admission.plan.imported_tokens - num_computed_tokens), False

    def load(self, admission, destinations, slots, *, actual_request_id=None, sync_device=True):
        blocks = set((slots // self.capability.block_size).tolist())
        try:
            self.store.discard_injection(admission.lease)
            if admission.profile_fingerprint != profile_fingerprint(self.profile):
                raise IdentityMismatch('worker profile mismatch')
            load_started = time.perf_counter()
            kv = self.store.load(admission.lease, admission.plan)
            store_load_ms = (time.perf_counter() - load_started) * 1000
            if (sha256(kv.prefix) != admission.prefix_digest
                    or kv.target != self.profile.target
                    or kv.reader != self.profile.reader
                    or kv.method_fingerprint != fingerprint(self.profile, kv.prefix)):
                raise IdentityMismatch('worker artifact identity differs')
            if set(destinations) != set(range(kv.target.num_layers)):
                raise UnsupportedProfile('attention layers incomplete')
            if slots.numel() != admission.plan.imported_tokens:
                raise IdentityMismatch('slot count differs from imported prefix')

            write_started = time.perf_counter()
            first_dst = next(iter(destinations.values()))
            if kv.packed_kv is not None and kv.packed_kv.device == first_dst.device:
                write_packed_layers_fast(destinations, kv.packed_kv, slots.tolist(), self.capability)
            else:
                for layer, pair in enumerate(kv.layers):
                    write_layer(destinations[layer], pair, slots, self.capability)
            kv_write_submit_ms = (time.perf_counter() - write_started) * 1000

            devices = {t.device for t in destinations.values() if t.device.type == 'cuda'}
            sync_started = time.perf_counter()
            if sync_device:
                for device in devices:
                    torch.cuda.synchronize(device)
            kv_sync_ms = (time.perf_counter() - sync_started) * 1000
            self.store.record_injection(admission.lease, {
                'artifact_id': admission.lease.artifact_id,
                'request_id': admission.lease.request_id,
                'actual_request_id': actual_request_id,
                'lease_id': admission.lease.lease_id,
                'profile_fingerprint': admission.profile_fingerprint,
                'request_digest': admission.plan.request_digest,
                'loaded_layers': len(destinations),
                'loaded_tokens': slots.numel(),
                'slot_digest': sha256(slots.tolist()),
                **_policy_receipt(admission.plan),
                'worker_pid': os.getpid(),
                'device_types': sorted({t.device.type for t in destinations.values()}),
                'store_load_ms': store_load_ms,
                'kv_write_submit_ms': kv_write_submit_ms,
                'kv_sync_ms': kv_sync_ms,
            })
            return set()
        except (SCDRuntimeError, ValueError, RuntimeError, KeyError) as exc:
            self.errors[admission.lease.request_id] = f'{type(exc).__name__}: {exc}'
            return blocks

    def release(self, admission):
        self.store.release(admission.lease)
        self.errors.pop(admission.lease.request_id, None)


try:
    from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorBase_V1, KVConnectorMetadata
    _VLLM_AVAILABLE = True
except ModuleNotFoundError:
    KVConnectorBase_V1 = object

    class KVConnectorMetadata:
        pass

    _VLLM_AVAILABLE = False


@dataclass
class SCDConnectorMetadata(KVConnectorMetadata):
    loads: list = field(default_factory=list)
    cancelled: set = field(default_factory=set)


class SCDConnectorV1(EventConnectorMixin, RegisteredKVMixin, KVConnectorBase_V1):
    def __init__(self, config, role, kv_cache_config):
        if not _VLLM_AVAILABLE:
            raise ImportError('vLLM V1 is required to instantiate SCDConnectorV1')
        super().__init__(config, role, kv_cache_config)
        from .capability import inspect_engine
        extra = config.kv_transfer_config
        strict_pin = bool(extra.get_from_extra_config('strict_commit_pin', False))
        profile = profile_from_dict(extra.get_from_extra_config('scd_profile', {}))
        store = ArtifactStore(
            extra.get_from_extra_config('store_root', ''),
            extra.get_from_extra_config('capacity_bytes', 0),
        )
        self.core = ConnectorCoordinator(
            store,
            profile,
            inspect_engine(config, kv_cache_config, strict_commit_pin=strict_pin),
            fullprefix_serial=config.scheduler_config.max_num_seqs == 1,
            strict_commit_pin=strict_pin,
        )
        self._sync_on_load = bool(extra.get_from_extra_config('sync_on_load', False))
        self._admissions = {}
        self._pending = {}
        self._load_errors = set()
        self._lease_requests = {}
        self._receive_done = set()
        self._release_after_receive = {}
        initialize_request_mode(self, extra, profile)
        if extra.kv_load_failure_policy != 'fail':
            raise UnsupportedProfile('strict connector requires kv_load_failure_policy=fail')

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        params = request.kv_transfer_params or {}
        data = params.get('scd_admission')
        if data is None:
            return 0, False
        admission = admission_from_data(data)
        if (getattr(request, 'prompt_embeds', None) is not None
                or getattr(request, 'prompt_is_token_ids', None) is not None
                or getattr(request, 'mm_features', None)
                or getattr(request, 'lora_request', None) is not None
                or getattr(request, 'resumable', False)):
            raise UnsupportedProfile('file imports require non-resumable token-only requests')
        prompt = getattr(request, 'prompt_token_ids', None)
        if prompt is None:
            raise IdentityMismatch('scheduled prompt tokens required')
        p = self.core.profile
        prefix = PrefixContract(
            tuple(prompt[:admission.plan.original_prefix_tokens]),
            0,
            p.source.tokenizer_sha256,
            p.source.chat_template_sha256,
        )
        if sha256(prefix) != admission.prefix_digest or plan_import(
            prefix, prompt, self.core.capability.block_size, import_policy=get_import_policy(admission.plan)
        ) != admission.plan:
            raise IdentityMismatch('scheduled file prompt differs from admission')
        if self.core.capability.prefix_cache and (
            getattr(request, 'skip_writing_prefix_cache', False) is not True
            or getattr(request, 'cache_salt', None) != 'scd-' + admission.lease.lease_id
            or getattr(request, 'skip_reading_prefix_cache', False)
        ):
            raise UnsupportedProfile('APC imports require server salt, no-store, and external reads')
        prior = self._lease_requests.get(admission.lease.lease_id)
        if prior is not None and prior != request.request_id:
            raise IdentityMismatch('file lease already bound to another engine request')
        count, _ = self.core.matched(admission, num_computed_tokens)
        self._lease_requests[admission.lease.lease_id] = request.request_id
        self._admissions[request.request_id] = admission
        return count, self._async_loading and count > 0

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        if num_external_tokens > 0:
            admission = self._admissions[request.request_id]
            groups = blocks.get_block_ids()
            if len(groups) != 1:
                raise UnsupportedProfile('file import requires one actual cache group')
            B = self.core.capability.block_size
            slots = (
                allocated_prefix_slots(groups[0], B, admission.plan)
                if get_import_policy(admission.plan) == 'fullprefix'
                else [block * B + offset for block in groups[0] for offset in range(B)][:num_external_tokens]
            )
            if num_external_tokens != admission.plan.imported_tokens:
                raise IdentityMismatch('partial file allocation unsupported')
            if len(slots) != admission.plan.imported_tokens:
                raise IdentityMismatch('file allocated prefix incomplete')
            self._pending[request.request_id] = (admission, slots)

    def build_connector_meta(self, scheduler_output):
        meta = SCDConnectorMetadata()
        if self._async_loading:
            meta.loads = [(rid, *value) for rid, value in self._pending.items()]
            self._pending.clear()
            add_routing_metadata(self, meta, scheduler_output)
            return meta
        allocations = {r.req_id: r.block_ids[0] for r in scheduler_output.scheduled_new_reqs}
        cached = scheduler_output.scheduled_cached_reqs
        if any(rid in self._admissions for rid in cached.resumed_req_ids):
            raise UnsupportedProfile('file imports cannot resume after preemption')
        for i, rid in enumerate(cached.req_ids):
            if rid in cached.resumed_req_ids and cached.new_block_ids[i] is not None:
                allocations[rid] = cached.new_block_ids[i][0]
        B = self.core.capability.block_size
        for rid, block_ids in allocations.items():
            if rid not in self._pending:
                continue
            admission, _ = self._pending.pop(rid)
            q = admission.plan.imported_tokens
            slots = (
                allocated_prefix_slots(block_ids, B, admission.plan)
                if get_import_policy(admission.plan) == 'fullprefix'
                else [block * B + offset for block in block_ids for offset in range(B)][:q]
            )
            meta.loads.append((rid, admission, slots))
        add_routing_metadata(self, meta, scheduler_output)
        return meta

    def start_load_kv(self, forward_context, **kwargs):
        meta = self._get_connector_metadata()
        if meta is None:
            return
        for rid, admission, slots in meta.loads:
            if self._async_loading:
                self._load_async(rid, admission, slots)
                continue
            indices = torch.tensor(slots, dtype=torch.int64)
            try:
                destinations = self._registered_destinations()
            except UnsupportedProfile as exc:
                self.core.errors[rid] = str(exc)
                self._load_errors.update((indices // self.core.capability.block_size).tolist())
                continue
            self._load_errors.update(
                self.core.load(admission, destinations, indices, actual_request_id=rid, sync_device=self._sync_on_load)
            )

    def get_block_ids_with_load_errors(self):
        result = self._load_errors
        self._load_errors = set()
        if self._events is not None:
            result.update(self._events.errors())
        return result

    def wait_for_layer_load(self, layer_name):
        if self._events is not None:
            self.owner_wait_layer(layer_name)

    def owner_wait_layer(self, layer_name):
        if self._events is not None:
            self._events.wait_layer(layer_name)

    def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs):
        pass

    def wait_for_save(self):
        pass

    def request_finished(self, request, block_ids):
        admission = self._admissions.pop(request.request_id, None)
        if admission is not None and self._async_loading:
            self._cancelled_requests.add(request.request_id)
        else:
            self._pending.pop(request.request_id, None)
            if admission is not None:
                self.core.release(admission)
        if admission is not None:
            if self._async_loading and request.request_id not in self._receive_done:
                self._release_after_receive[request.request_id] = admission.lease.lease_id
            else:
                self._lease_requests.pop(admission.lease.lease_id, None)
                self._receive_done.discard(request.request_id)
        return False, None

    def update_connector_output(self, connector_output):
        super().update_connector_output(connector_output)
        for rid in connector_output.finished_recving or ():
            lease_id = self._release_after_receive.pop(rid, None)
            if lease_id is not None:
                if self._lease_requests.get(lease_id) == rid:
                    self._lease_requests.pop(lease_id)
            elif rid in self._admissions:
                self._receive_done.add(rid)

    def _load_async(self, rid, admission, slots):
        if self._events.lease_owned_by_other(rid, admission.lease.lease_id):
            self._events.fail_without_writes(rid, slots, lambda: None)
            return
        released = False

        def release():
            nonlocal released
            if not released:
                self.core.release(admission)
                released = True

        if rid in self._events.cancelled:
            self._events.fail_without_writes(rid, slots, release)
            return
        try:
            self.core.store.discard_injection(admission.lease)
            if admission.profile_fingerprint != profile_fingerprint(self.core.profile):
                raise IdentityMismatch('worker async profile mismatch')
            kv = self.core.store.load(admission.lease, admission.plan)
            if (sha256(kv.prefix) != admission.prefix_digest
                    or kv.target != self.core.profile.target
                    or kv.reader != self.core.profile.reader
                    or kv.method_fingerprint != fingerprint(self.core.profile, kv.prefix)):
                raise IdentityMismatch('worker async artifact identity differs')
            destinations = self._registered_destinations()
            if set(destinations) != set(range(kv.target.num_layers)):
                raise UnsupportedProfile('async attention layers incomplete')
            receipt = {
                'artifact_id': admission.lease.artifact_id,
                'request_id': admission.lease.request_id,
                'actual_request_id': rid,
                'lease_id': admission.lease.lease_id,
                'profile_fingerprint': admission.profile_fingerprint,
                'request_digest': admission.plan.request_digest,
                'loaded_layers': len(destinations),
                'loaded_tokens': len(slots),
                'slot_digest': sha256(slots),
                'worker_pid': os.getpid(),
                'device_types': ['cuda'],
                'completion': 'copy_stream_event',
                'device_wide_synchronize': False,
                **_policy_receipt(admission.plan),
            }
            self._events.submit(
                rid,
                admission.lease.lease_id,
                kv,
                destinations,
                slots,
                on_complete=lambda: self.core.store.record_injection(admission.lease, receipt),
                release=release,
            )
        except (SCDRuntimeError, ValueError, RuntimeError, KeyError) as exc:
            self.core.errors[rid] = f'{type(exc).__name__}: {exc}'
            if not self._events.pending(rid):
                self._events.fail_without_writes(rid, slots, release)
