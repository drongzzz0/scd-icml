"""Private same-worker GPU snapshots with zero-copy [L, N, H, 2D] fast path."""
from dataclasses import dataclass, replace
import re
import secrets
import threading
import time

import torch

from ..contracts.types import PrefixContract, ImportPlan, strict_construct, to_data, integer, freeze
from ..contracts.identity import fingerprint, profile_fingerprint, sha256
from ..contracts.prefix import plan_import, get_import_policy, import_plan_from_data, check_fullprefix_scope
from ..errors import IdentityMismatch, ArtifactCorrupt, MissingAsset, StoreCapacityExceeded, UnsupportedProfile


@dataclass(frozen=True)
class ResidentAdmission:
    session_id: str
    artifact_id: str
    lease_id: str
    request_id: str
    profile_fingerprint: str
    prefix_digest: str
    block_size: int
    plan: ImportPlan


def admission_from_data(data):
    if not isinstance(data, dict) or set(data) != set(ResidentAdmission.__dataclass_fields__):
        raise IdentityMismatch('invalid resident admission')
    value = dict(data)
    value['plan'] = import_plan_from_data(value['plan'])
    return ResidentAdmission(**value)


def check_actual_request(admission, actual_id):
    if not isinstance(actual_id, str) or (
        actual_id != admission.request_id
        and re.fullmatch(re.escape(admission.request_id) + r'-[0-9a-f]{8}', actual_id) is None
    ):
        raise IdentityMismatch('actual engine request differs from authorized external request')


def validate_scheduled_request(admission, request, profile, block_size):
    check_actual_request(admission, request.request_id)
    if (getattr(request, 'prompt_embeds', None) is not None
            or getattr(request, 'prompt_is_token_ids', None) is not None
            or getattr(request, 'mm_features', None)
            or getattr(request, 'lora_request', None) is not None
            or getattr(request, 'resumable', False)):
        raise UnsupportedProfile('resident connector supports token-only non-resumable requests')
    prompt = getattr(request, 'prompt_token_ids', None)
    if prompt is None:
        raise IdentityMismatch('actual prompt tokens required')
    prefix = PrefixContract(
        tuple(prompt[:admission.plan.original_prefix_tokens]),
        0,
        profile.source.tokenizer_sha256,
        profile.source.chat_template_sha256,
    )
    if (admission.profile_fingerprint != profile_fingerprint(profile)
            or admission.block_size != block_size
            or sha256(prefix) != admission.prefix_digest
            or plan_import(prefix, prompt, block_size, import_policy=get_import_policy(admission.plan)) != admission.plan):
        raise IdentityMismatch('actual scheduled prompt differs from admission')


class ResidentRegistry:
    def __init__(self, profile, session_id, capacity_bytes, *, device='cuda:0', synchronize=None, max_entries=1, capability=None):
        if not isinstance(session_id, str) or re.fullmatch('[0-9a-f]{32}', session_id) is None:
            raise ValueError('invalid engine session')
        integer(capacity_bytes, 'resident capacity')
        integer(max_entries, 'resident entry limit')
        if not 1 <= max_entries <= 64:
            raise ValueError('resident entry limit must be between 1 and 64')
        self.profile = profile
        self.session_id = session_id
        self.capacity_bytes = capacity_bytes
        self.device = torch.device(device)
        if capability is not None:
            from ..contracts.capability import check_capability
            check_capability(profile, capability)
            capability = replace(capability, attention=freeze(capability.attention))
        self.capability = capability
        self.synchronize = synchronize or (lambda: torch.cuda.synchronize(self.device) if self.device.type == 'cuda' else None)
        self.lock = threading.RLock()
        self._entries = {}
        self.max_entries = max_entries
        self.poisoned = False

    @property
    def _entry(self):
        if len(self._entries) > 1:
            raise UnsupportedProfile('select a resident artifact explicitly')
        return next(iter(self._entries.values()), None)

    def require_idle(self):
        if self.poisoned:
            raise UnsupportedProfile('resident session is poisoned; recreate engine')
        if self._entries:
            raise StoreCapacityExceeded('resident engine state must be idle')

    def require_preparable(self):
        if self.poisoned:
            raise UnsupportedProfile('resident session is poisoned; recreate engine')
        if any(entry['phase'] != 'READY' for entry in self._entries.values()):
            raise UnsupportedProfile('prepare snapshots only before requests are admitted')
        if len(self._entries) >= self.max_entries:
            raise StoreCapacityExceeded('resident snapshot entry limit reached')

    def publish(self, kv, *, adopt_packed: bool = False, sync_on_publish: bool = True, check_finite: bool = True):
        """Publish target KV into private worker storage.

        When ``adopt_packed=True`` and ``kv.packed_kv`` is already a private
        contiguous ``[L, n, H, 2*D]`` BF16 buffer on ``self.device``, adopts it
        in O(1) without allocating/copying a second snapshot buffer.
        """
        with self.lock:
            self.require_preparable()
            p = self.profile
            if (kv.target != p.target
                    or kv.reader != p.reader
                    or not kv.rope_applied
                    or kv.method_fingerprint != fingerprint(p, kv.prefix)
                    or kv.prefix.tokenizer_sha256 != p.source.tokenizer_sha256
                    or kv.prefix.chat_template_sha256 != p.source.chat_template_sha256):
                raise IdentityMismatch('resident snapshot model/reader/method/prefix differs')
            n = len(kv.prefix.token_ids)
            shape = (1, p.target.num_kv_heads, n, p.target.head_dim)
            if len(kv.layers) != p.target.num_layers:
                raise ArtifactCorrupt('resident layer count differs')

            # Fast zero-copy path when ICMLBackend already built [L, n, H, 2*D] packed_kv
            if (adopt_packed and kv.packed_kv is not None
                    and isinstance(kv.packed_kv, torch.Tensor)
                    and tuple(kv.packed_kv.shape) == (p.target.num_layers, n, p.target.num_kv_heads, 2 * p.target.head_dim)
                    and kv.packed_kv.dtype == torch.bfloat16
                    and kv.packed_kv.device == self.device
                    and kv.packed_kv.is_contiguous()
                    and kv.packed_kv._base is None
                    and not kv.packed_kv.is_inference()):
                packed = kv.packed_kv
                size = packed.numel() * packed.element_size()
                if size + sum(entry['bytes'] for entry in self._entries.values()) > self.capacity_bytes:
                    raise StoreCapacityExceeded('resident snapshots exceed aggregate capacity')
                if check_finite and not bool(torch.isfinite(packed).all()):
                    raise ArtifactCorrupt('nonfinite resident snapshot')
                D = p.target.head_dim
                layers = tuple(
                    (packed[i, :, :, :D].transpose(0, 1).unsqueeze(0), packed[i, :, :, D:].transpose(0, 1).unsqueeze(0))
                    for i in range(p.target.num_layers)
                )
                snapshot = replace(kv, layers=layers, packed_kv=packed)
                if sync_on_publish:
                    self.synchronize()
                handle = secrets.token_hex(32)
                self._entries[handle] = {
                    'handle': handle,
                    'kv': snapshot,
                    'versions': tuple(t._version for pair in layers for t in pair),
                    'private_layout': self._private_layout(layers),
                    'packed': packed,
                    'packed_kind': 'lnh2d',
                    'packed_layout': self._private_layout(((packed,),)),
                    'packed_version': packed._version,
                    'bytes': size,
                    'phase': 'READY',
                    'admission': None,
                    'receipt': None,
                }
                return {
                    'session_id': self.session_id,
                    'artifact_id': handle,
                    'profile_fingerprint': profile_fingerprint(p),
                    'prefix_digest': sha256(kv.prefix),
                    'prefix_tokens': n,
                    'method_fingerprint': kv.method_fingerprint,
                    'snapshot_bytes': size,
                    'identity_kind': 'ephemeral_private_worker_snapshot',
                    'integrity': 'private_clone_version_geometry_finite_identity',
                    'kv_payload_cpu_bytes': 0,
                    'kv_payload_file_bytes': 0,
                }

            size = 0
            for pair in kv.layers:
                if len(pair) != 2:
                    raise ArtifactCorrupt('resident pair incomplete')
                for tensor in pair:
                    if tuple(tensor.shape) != shape or tensor.dtype != torch.bfloat16 or tensor.device != self.device:
                        raise ArtifactCorrupt('resident geometry/device/dtype differs')
                    size += tensor.numel() * tensor.element_size()
            if size + sum(entry['bytes'] for entry in self._entries.values()) > self.capacity_bytes:
                raise StoreCapacityExceeded('resident snapshots exceed aggregate capacity')

            with torch.inference_mode(False), torch.no_grad():
                packed = torch.empty((2 * p.target.num_layers, *shape), device=self.device, dtype=torch.bfloat16)
                torch.stack(tuple(t.detach() for pair in kv.layers for t in pair), out=packed)
                flat = packed.unbind(0)
                layers = tuple((flat[i], flat[i + 1]) for i in range(0, len(flat), 2))
            if not bool(torch.isfinite(packed).all()):
                raise ArtifactCorrupt('nonfinite resident snapshot')
            if sync_on_publish:
                self.synchronize()
            snapshot = replace(kv, layers=layers, packed_kv=None)
            handle = secrets.token_hex(32)
            self._entries[handle] = {
                'handle': handle,
                'kv': snapshot,
                'versions': tuple(t._version for pair in layers for t in pair),
                'private_layout': self._private_layout(layers),
                'packed': packed,
                'packed_kind': '2l1hnd',
                'packed_layout': self._private_layout(((packed,),)),
                'packed_version': packed._version,
                'bytes': size,
                'phase': 'READY',
                'admission': None,
                'receipt': None,
            }
            return {
                'session_id': self.session_id,
                'artifact_id': handle,
                'profile_fingerprint': profile_fingerprint(p),
                'prefix_digest': sha256(kv.prefix),
                'prefix_tokens': n,
                'method_fingerprint': kv.method_fingerprint,
                'snapshot_bytes': size,
                'identity_kind': 'ephemeral_private_worker_snapshot',
                'integrity': 'private_clone_version_geometry_finite_identity',
                'kv_payload_cpu_bytes': 0,
                'kv_payload_file_bytes': 0,
            }

    def _get(self, handle):
        if self.poisoned:
            raise UnsupportedProfile('resident session is poisoned')
        if handle not in self._entries:
            raise MissingAsset('resident snapshot already released or unknown')
        return self._entries[handle]

    def admit(self, handle, prompt_ids, request_id, block_size, *, import_policy='blocktail'):
        if not isinstance(request_id, str) or not request_id or len(request_id) > 128:
            raise ValueError('invalid request ID')
        with self.lock:
            entry = self._get(handle)
            if entry['phase'] != 'READY':
                raise IdentityMismatch('resident handle is single-use')
            if any(other['admission'] is not None and other['admission'].request_id == request_id
                   for other in self._entries.values()):
                raise IdentityMismatch('resident request ID already admitted')
            if import_policy == 'fullprefix':
                check_fullprefix_scope(self.profile, self.capability, resident=True)
                if (self.profile.target.attention.get('sliding_window') is not None
                        and len(prompt_ids) >= self.capability.max_model_len):
                    raise IdentityMismatch('bounded SWA prompt leaves no verified generation extent')
                if self.max_entries != 1:
                    raise UnsupportedProfile('fullprefix candidate requires a serial resident session')
            prefix = entry['kv'].prefix
            plan = plan_import(prefix, prompt_ids, block_size, import_policy=import_policy)
            admission = ResidentAdmission(
                self.session_id,
                handle,
                secrets.token_hex(32),
                request_id,
                profile_fingerprint(self.profile),
                sha256(prefix),
                block_size,
                plan,
            )
            entry.update(admission=admission, phase='ADMITTED')
            return admission

    def _check(self, admission):
        if admission.session_id != self.session_id or admission.profile_fingerprint != profile_fingerprint(self.profile):
            raise IdentityMismatch('resident session/profile differs')
        entry = self._get(admission.artifact_id)
        if entry['admission'] != admission:
            raise IdentityMismatch('resident lease or import plan differs')
        return entry

    @staticmethod
    def _private_layout(layers):
        return tuple(
            (id(t), t.data_ptr(), tuple(t.shape), tuple(t.stride()), t.dtype, t.device)
            for pair in layers for t in pair
        )

    def _async_snapshot(self, entry, admission):
        kv = entry['kv']
        p = self.profile
        if (kv is None or kv.target != p.target or kv.reader != p.reader or not kv.rope_applied
                or kv.method_fingerprint != fingerprint(p, kv.prefix)
                or sha256(kv.prefix) != admission.prefix_digest
                or len(kv.layers) != p.target.num_layers):
            raise IdentityMismatch('private async snapshot identity differs')
        n = admission.plan.original_prefix_tokens
        shape = (1, p.target.num_kv_heads, n, p.target.head_dim)
        if any(len(pair) != 2 for pair in kv.layers):
            raise ArtifactCorrupt('private async pair count differs')
        tensors = tuple(t for pair in kv.layers for t in pair)
        if any(not isinstance(t, torch.Tensor) or tuple(t.shape) != shape or t.dtype != torch.bfloat16
               or t.device != self.device for t in tensors):
            raise ArtifactCorrupt('private async snapshot geometry differs')
        try:
            versions = tuple(t._version for t in tensors)
        except RuntimeError as exc:
            raise ArtifactCorrupt('private async snapshot lost version tracking') from exc
        if versions != entry['versions'] or self._private_layout(kv.layers) != entry.get('private_layout'):
            raise ArtifactCorrupt('private async snapshot was mutated or replaced')

        packed = entry.get('packed')
        kind = entry.get('packed_kind', '2l1hnd')
        expected_packed_shape = (
            (p.target.num_layers, n, p.target.num_kv_heads, 2 * p.target.head_dim)
            if kind == 'lnh2d'
            else (2 * p.target.num_layers, *shape)
        )
        if (not isinstance(packed, torch.Tensor) or tuple(packed.shape) != expected_packed_shape
                or packed.dtype != torch.bfloat16 or packed.device != self.device
                or not packed.is_contiguous() or packed._base is not None
                or self._private_layout(((packed,),)) != entry.get('packed_layout')):
            raise ArtifactCorrupt('private packed snapshot storage differs')
        try:
            version = packed._version
        except RuntimeError as exc:
            raise ArtifactCorrupt('private packed snapshot lost version tracking') from exc
        if version != entry.get('packed_version'):
            raise ArtifactCorrupt('private packed snapshot was mutated')
        return kv

    def begin_async_load(self, admission, actual_request_id, destinations, slotsCPU, capability):
        with self.lock:
            entry = self._check(admission)
            check_actual_request(admission, actual_request_id)
            if entry['phase'] != 'ADMITTED':
                raise IdentityMismatch('resident lease already consumed or failed')
            if capability.block_size != admission.block_size:
                raise IdentityMismatch('async block size differs from admission')
            if get_import_policy(admission.plan) == 'fullprefix':
                if self.profile.target.attention.get('sliding_window') is not None:
                    raise UnsupportedProfile('bounded SWA fullprefix is synchronous only')
                check_fullprefix_scope(self.profile, capability, resident=True)
            kv = self._async_snapshot(entry, admission)
            p = self.profile
            B = capability.block_size
            if (not isinstance(slotsCPU, torch.Tensor) or slotsCPU.device.type != 'cpu'
                    or slotsCPU.dtype != torch.int64 or slotsCPU.ndim != 1
                    or slotsCPU.numel() != admission.plan.imported_tokens or slotsCPU.numel() == 0
                    or not isinstance(destinations, dict) or set(destinations) != set(range(p.target.num_layers))):
                raise IdentityMismatch('async destinations or CPU slot count differ')
            if any(not isinstance(t, torch.Tensor) or t.ndim != 4 or t.dtype != torch.bfloat16
                   or t.device != self.device or t.shape[0] < 1
                   or tuple(t.shape[1:]) != (p.target.num_kv_heads, B, 2 * p.target.head_dim)
                   for t in destinations.values()):
                raise IdentityMismatch('async destination geometry or device differs')
            if len({t.shape[0] for t in destinations.values()}) != 1:
                raise IdentityMismatch('async destination block counts differ')
            values = tuple(slotsCPU.tolist())
            if len(set(values)) != len(values) or min(values) < 0 or max(values) >= next(iter(destinations.values())).shape[0] * B:
                raise IdentityMismatch('async slots are repeated or outside allocated cache')
            expected = {
                'session_id': self.session_id,
                'artifact_id': admission.artifact_id,
                'lease_id': admission.lease_id,
                'request_id': admission.request_id,
                'actual_request_id': actual_request_id,
                'profile_fingerprint': admission.profile_fingerprint,
                'request_digest': admission.plan.request_digest,
                'loaded_layers': len(destinations),
                'loaded_tokens': len(values),
                'slot_digest': sha256(list(values)),
                'device_types': sorted({t.device.type for t in destinations.values()}),
                'snapshot_version_verified': True,
                'native_fallback': False,
            }
            if get_import_policy(admission.plan) == 'fullprefix':
                expected.update(
                    import_policy='fullprefix',
                    original_prefix_tokens=admission.plan.original_prefix_tokens,
                    tail_recomputed_tokens=0,
                )
            entry.update(
                phase='COPYING_ASYNC',
                async_load={
                    'receipt': expected,
                    'slots': slotsCPU,
                    'slot_layout': self._private_layout(((slotsCPU,),)),
                    'slot_values': values,
                    'destinations': tuple(destinations.values()),
                    'destination_layout': self._private_layout(tuple((t,) for t in destinations.values())),
                },
            )
            return kv

    def complete_async_load(self, admission, actual_request_id, receipt):
        with self.lock:
            entry = self._check(admission)
            check_actual_request(admission, actual_request_id)
            if entry['phase'] != 'COPYING_ASYNC':
                raise IdentityMismatch('resident async load is not in flight')
            state = entry['async_load']
            expected = state['receipt']
            if actual_request_id != expected['actual_request_id'] or not isinstance(receipt, dict):
                raise IdentityMismatch('async completion request or receipt differs')
            try:
                receipt = to_data(freeze(receipt))
            except ValueError as exc:
                raise IdentityMismatch('async receipt must contain finite host JSON values') from exc
            if any(key in receipt and sha256(receipt[key]) != sha256(value) for key, value in expected.items()):
                raise IdentityMismatch('async completion receipt identity differs')
            self._async_snapshot(entry, admission)
            if (self._private_layout(((state['slots'],),)) != state['slot_layout']
                    or tuple(state['slots'].tolist()) != state['slot_values']):
                raise IdentityMismatch('async CPU slots changed after admission')
            if self._private_layout(tuple((t,) for t in state['destinations'])) != state['destination_layout']:
                raise IdentityMismatch('async destinations changed after admission')
            entry.update(phase='COPIED', receipt={**receipt, **expected}, kv=None, packed=None, bytes=0, async_drained=True)
            del entry['async_load']

    def fail_async_load(self, admission):
        with self.lock:
            entry = self._check(admission)
            if entry['phase'] == 'FAILED' and entry.get('async_drained'):
                return
            if entry['phase'] != 'COPYING_ASYNC':
                raise IdentityMismatch('resident async load is not in flight')
            entry.update(phase='FAILED', receipt=None, kv=None, packed=None, bytes=0, async_drained=True)
            del entry['async_load']

    def load_packed_fast(self, admission, actual_request_id, destinations, slots_list, capability, *, sync_after_write: bool = False):
        """Stream-ordered zero-stall load for [L, n, H, 2*D] packed snapshots.

        Enqueues page-level index_copy_ on the worker's active compute stream
        without calling cudaDeviceSynchronize() before model.forward, allowing
        CPU kernel dispatch to overlap KV cache scatter.
        """
        from scd_vllm.layout import write_packed_layers_fast

        with self.lock:
            entry = self._check(admission)
            check_actual_request(admission, actual_request_id)
            if entry['phase'] != 'ADMITTED':
                raise IdentityMismatch('resident lease already consumed or failed')
            entry['phase'] = 'COPYING'
            try:
                self._async_snapshot(entry, admission)
                if get_import_policy(admission.plan) == 'fullprefix':
                    check_fullprefix_scope(self.profile, capability, resident=True)
                packed = entry['packed']
                if entry.get('packed_kind') != 'lnh2d':
                    raise UnsupportedProfile('load_packed_fast requires lnh2d packed layout')
                if len(slots_list) != admission.plan.imported_tokens or not slots_list:
                    raise IdentityMismatch('resident slot count differs')
                if set(destinations) != set(range(self.profile.target.num_layers)):
                    raise IdentityMismatch('resident destinations incomplete')

                write_started = time.perf_counter()
                write_packed_layers_fast(destinations, packed, slots_list, capability)
                if self.device.type == 'cuda':
                    packed.record_stream(torch.cuda.current_stream(self.device))
                write_ms = (time.perf_counter() - write_started) * 1000

                sync_started = time.perf_counter()
                if sync_after_write:
                    self.synchronize()
                sync_ms = (time.perf_counter() - sync_started) * 1000

                entry.update(
                    phase='COPIED',
                    async_drained=not sync_after_write,
                    receipt={
                        'session_id': self.session_id,
                        'artifact_id': admission.artifact_id,
                        'lease_id': admission.lease_id,
                        'request_id': admission.request_id,
                        'actual_request_id': actual_request_id,
                        'profile_fingerprint': admission.profile_fingerprint,
                        'request_digest': admission.plan.request_digest,
                        'loaded_layers': len(destinations),
                        'loaded_tokens': len(slots_list),
                        'slot_digest': sha256(list(slots_list)),
                        'device_types': sorted({t.device.type for t in destinations.values()}),
                        'snapshot_version_verified': True,
                        'kv_payload_cpu_bytes': 0,
                        'kv_payload_file_bytes': 0,
                        'native_fallback': False,
                        'stream_ordered_nonblocking': not sync_after_write,
                        'kv_write_submit_ms': write_ms,
                        'kv_sync_ms': sync_ms,
                    },
                )
                if get_import_policy(admission.plan) == 'fullprefix':
                    entry['receipt'].update(
                        import_policy='fullprefix',
                        original_prefix_tokens=admission.plan.original_prefix_tokens,
                        tail_recomputed_tokens=0,
                    )
            except BaseException:
                entry['phase'] = 'FAILED'
                entry['receipt'] = None
                try:
                    self.synchronize()
                except BaseException:
                    self.poisoned = True
                if not self.poisoned:
                    entry['kv'] = None
                    entry['packed'] = None
                    entry['bytes'] = 0
                raise
            else:
                entry['kv'] = None
                entry['packed'] = None
                entry['bytes'] = 0

    def load(self, admission, actual_request_id, destinations, slots, capability, writer, *, observer=None):
        with self.lock:
            entry = self._check(admission)
            check_actual_request(admission, actual_request_id)
            if entry['phase'] != 'ADMITTED':
                raise IdentityMismatch('resident lease already consumed or failed')
            entry['phase'] = 'COPYING'
            diagnostic_observer = None
            try:
                kv = entry['kv']
                tensors = tuple(t for pair in kv.layers for t in pair)
                if get_import_policy(admission.plan) == 'fullprefix':
                    check_fullprefix_scope(self.profile, capability, resident=True)
                    shape = (1, self.profile.target.num_kv_heads, admission.plan.original_prefix_tokens, self.profile.target.head_dim)
                    if any(tuple(t.shape) != shape for t in tensors):
                        raise ArtifactCorrupt('fullprefix payload must have exactly the original prefix length')
                self._async_snapshot(entry, admission)
                if not bool(torch.isfinite(entry['packed']).all()):
                    raise ArtifactCorrupt('private resident snapshot was mutated')
                if (slots.ndim != 1 or slots.dtype != torch.int64
                        or slots.numel() != admission.plan.imported_tokens or slots.numel() == 0
                        or set(destinations) != set(range(self.profile.target.num_layers))
                        or any(t.device != self.device or t.dtype != torch.bfloat16 for t in destinations.values())):
                    raise IdentityMismatch('resident destinations or slot count differ')
                if observer is not None:
                    from .load_diagnostics import LoadDiagnostics
                    if type(observer) is not LoadDiagnostics:
                        raise UnsupportedProfile('trusted load diagnostic observer required')
                    diagnostic_observer = observer
                    observer.before(admission, actual_request_id, destinations, kv, slots, capability)
                write_started = time.perf_counter()
                for layer, pair in enumerate(kv.layers):
                    writer(destinations[layer], pair, slots, capability)
                write_ms = (time.perf_counter() - write_started) * 1000
                sync_started = time.perf_counter()
                self.synchronize()
                sync_ms = (time.perf_counter() - sync_started) * 1000
                diagnostic = None if observer is None else observer.after(admission, actual_request_id, destinations, slots)
                entry.update(
                    phase='COPIED',
                    receipt={
                        'session_id': self.session_id,
                        'artifact_id': admission.artifact_id,
                        'lease_id': admission.lease_id,
                        'request_id': admission.request_id,
                        'actual_request_id': actual_request_id,
                        'profile_fingerprint': admission.profile_fingerprint,
                        'request_digest': admission.plan.request_digest,
                        'loaded_layers': len(destinations),
                        'loaded_tokens': slots.numel(),
                        'slot_digest': sha256(slots.tolist()),
                        'device_types': sorted({t.device.type for t in destinations.values()}),
                        'snapshot_version_verified': True,
                        'kv_payload_cpu_bytes': 0,
                        'kv_payload_file_bytes': 0,
                        'native_fallback': False,
                        'kv_write_submit_ms': write_ms,
                        'kv_sync_ms': sync_ms,
                    },
                )
                if get_import_policy(admission.plan) == 'fullprefix':
                    entry['receipt'].update(
                        import_policy='fullprefix',
                        original_prefix_tokens=admission.plan.original_prefix_tokens,
                        tail_recomputed_tokens=0,
                    )
                if diagnostic is not None:
                    entry['receipt']['load_diagnostics'] = diagnostic
            except BaseException:
                entry['phase'] = 'FAILED'
                entry['receipt'] = None
                try:
                    self.synchronize()
                except BaseException:
                    self.poisoned = True
                if not self.poisoned:
                    entry['kv'] = None
                    entry['packed'] = None
                    entry['bytes'] = 0
                raise
            else:
                entry['kv'] = None
                entry['packed'] = None
                entry['bytes'] = 0
            finally:
                if diagnostic_observer is not None:
                    diagnostic_observer.close()

    def finish(self, admission):
        with self.lock:
            entry = self._check(admission)
            if entry['phase'] != 'COPIED' or entry['receipt'] is None:
                raise MissingAsset('actual resident injection receipt missing')
            receipt = dict(entry['receipt'])
            del self._entries[admission.artifact_id]
            return receipt

    def release(self, handle, lease_id=None):
        with self.lock:
            if handle not in self._entries:
                return
            entry = self._get(handle)
            if entry['admission'] is not None and entry['admission'].lease_id != lease_id:
                raise IdentityMismatch('release lease differs')
            if entry['phase'] == 'COPYING_ASYNC':
                raise UnsupportedProfile('cannot release an in-flight async snapshot; drain its owner first')
            packed = entry.get('packed')
            if isinstance(packed, torch.Tensor) and packed.device.type == 'cuda':
                packed.record_stream(torch.cuda.current_stream(self.device))
            if entry['phase'] not in ('ADMITTED', 'COPIED') and not entry.get('async_drained'):
                self.synchronize()
            del self._entries[handle]

    def clear(self):
        with self.lock:
            entries = list(self._entries.values())
            if any(entry['phase'] == 'COPYING_ASYNC' for entry in entries):
                raise UnsupportedProfile('cannot clear in-flight async snapshots; drain their owner first')
            if any(entry['phase'] not in ('ADMITTED', 'COPIED') and not entry.get('async_drained') for entry in entries):
                self.synchronize()
            self._entries.clear()

    def status(self):
        with self.lock:
            entries = list(self._entries.values())
            return {
                'session_id': self.session_id,
                'entries': len(entries),
                'snapshot_bytes': sum(e['bytes'] for e in entries),
                'snapshots': sum(e['kv'] is not None for e in entries),
                'admissions': sum(e['phase'] == 'ADMITTED' for e in entries),
                'receipts': sum(e['receipt'] is not None for e in entries),
                'async_loads': sum(e['phase'] == 'COPYING_ASYNC' for e in entries),
                'phase': entries[0]['phase'] if len(entries) == 1 else None,
                'phases': {handle: e['phase'] for handle, e in self._entries.items()},
                'max_entries': self.max_entries,
                'poisoned': self.poisoned,
            }


_REGISTRIES = {}


def register_registry(registry):
    if registry.session_id in _REGISTRIES:
        raise IdentityMismatch('resident engine already configured')
    _REGISTRIES[registry.session_id] = registry


def get_registry(session_id):
    if session_id not in _REGISTRIES:
        raise MissingAsset('resident engine not configured in this worker')
    return _REGISTRIES[session_id]


def unregister_registry(session_id):
    registry = _REGISTRIES.get(session_id)
    if registry is not None:
        registry.clear()
        del _REGISTRIES[session_id]
