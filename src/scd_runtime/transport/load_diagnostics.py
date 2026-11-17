"""Bounded, opt-in inspection of the actual registered load destinations.

This internal observer accepts no addresses, paths or configurable byte budget.
It observes the owning connector's tensors; it never replaces a destination.
"""
import hashlib
import torch

from ..contracts.prefix import get_import_policy
from ..errors import ArtifactCorrupt, IdentityMismatch, UnsupportedProfile

MAX_CPU_READBACK_BYTES = 16 * 1024**2


def _identity(admission, actual_request_id):
    return {'session_id': admission.session_id, 'artifact_id': admission.artifact_id,
            'lease_id': admission.lease_id, 'request_id': admission.request_id,
            'actual_request_id': actual_request_id, 'profile_fingerprint': admission.profile_fingerprint,
            'request_digest': admission.plan.request_digest, 'import_policy': get_import_policy(admission.plan),
            'original_prefix_tokens': admission.plan.original_prefix_tokens,
            'imported_tokens': admission.plan.imported_tokens,
            'tail_recomputed_tokens': admission.plan.tail_recomputed_tokens,
            'block_size': admission.block_size}


def sample_ranges(n):
    if n <= 32:
        return [[0, n]], [[0, n]]
    requested = [[0, 32], [n - 32, n]]
    return requested, ([[0, n]] if n <= 64 else requested)


def planned_readback_bytes(n, layers, heads, head_dim):
    """All layers/ranges preserved; long loads compare before a single CPU read."""
    _, ranges = sample_ranges(n)
    sampled = sum(end - start for start, end in ranges)
    copies = 2 if n <= 32 else 1
    return layers * heads * 2 * head_dim * 2 * copies * (sampled + (-n) % 16)


def _storage(value):
    return str(value.device), value.untyped_storage().data_ptr()


class LoadDiagnostics:
    """Single-use trusted observer, created only by the opt-in connector."""
    def __init__(self, admission, actual_request_id):
        self.binding = _identity(admission, actual_request_id)
        self._source = {}
        self._padding = {}
        self._storage_evidence = {}
        self._closed = False
        self._started = False
        self.cpu_readback_bytes = 0
        self.device_snapshot_bytes = 0
        self.gpu_temporary_snapshot_bytes = 0

    def _check(self, admission, actual_request_id):
        if self._closed or _identity(admission, actual_request_id) != self.binding:
            raise IdentityMismatch('load diagnostic session/lease/request/policy differs')

    def _bits(self, value):
        size = value.numel() * value.element_size()
        if self.cpu_readback_bytes + size > self.planned_cpu_readback_bytes:
            raise UnsupportedProfile('load diagnostic exceeded its preflight CPU readback budget')
        self.cpu_readback_bytes += size
        return value.detach().to(device='cpu', copy=True).contiguous().view(torch.uint8)

    @staticmethod
    def _hash(bits):
        return hashlib.sha256(bits.numpy().tobytes()).hexdigest()

    def _snapshot(self, value, source_tensors, destination):
        bits = value.detach().clone(memory_format=torch.contiguous_format).view(torch.uint8)
        evidence = {'snapshot_kind': 'detached_contiguous_clone_on_worker_device',
                    'independent_from_source_storage': all(_storage(bits) != _storage(t) for t in source_tensors),
                    'independent_from_destination_storage': _storage(bits) != _storage(destination)}
        if not evidence['independent_from_source_storage'] or not evidence['independent_from_destination_storage']:
            raise ArtifactCorrupt('load diagnostic source/destination snapshots share storage')
        self.device_snapshot_bytes += bits.numel()
        if bits.device.type == 'cuda':self.gpu_temporary_snapshot_bytes += bits.numel()
        return bits, evidence

    def before(self, admission, actual_request_id, destinations, kv, slots, capability):
        self._check(admission, actual_request_id)
        if self._started:
            raise IdentityMismatch('load diagnostic observer is single-use')
        self._started = True
        n = admission.plan.imported_tokens
        B = capability.block_size
        L, h, d = kv.target.num_layers, kv.target.num_kv_heads, kv.target.head_dim
        if B != 16 or admission.block_size != B or type(n) is not int or n <= 0:
            raise UnsupportedProfile('load diagnostic requires a nonempty physical block16 import')
        if (slots.device.type != 'cpu' or slots.ndim != 1 or slots.dtype != torch.int64
                or slots.numel() != n or set(destinations) != set(range(L))
                or len(kv.layers) != L):
            raise IdentityMismatch('load diagnostic slot/layer contract differs')
        actual_slots = slots.tolist()
        if any(type(s) is not int or s < 0 for s in actual_slots) or len(set(actual_slots)) != n:
            raise IdentityMismatch('load diagnostic invalid or repeated actual slots')
        prefix_blocks = [actual_slots[i] // B for i in range(0, n, B)]
        if (len(set(prefix_blocks)) != len(prefix_blocks)
                or actual_slots != [prefix_blocks[i // B] * B + i % B for i in range(n)]):
            raise IdentityMismatch('load diagnostic slots do not map ordered private prefix blocks')
        requested, ranges = sample_ranges(n)
        sample_ids = [i for start, end in ranges for i in range(start, end)]
        sampled_slots = [actual_slots[i] for i in sample_ids]
        padding = [] if n % B == 0 else [prefix_blocks[-1] * B + i for i in range(n % B, B)]
        # Tiny: independent source/destination and padding-before/after CPU bytes.
        # Long: independent device snapshots + actual byte comparison, one CPU read.
        planned = planned_readback_bytes(n, L, h, d)
        if planned > MAX_CPU_READBACK_BYTES:
            raise UnsupportedProfile('load diagnostic exceeds fixed 16MiB CPU readback limit')
        self.planned_cpu_readback_bytes = planned
        self.requested_ranges, self.ranges = requested, ranges
        self.sample_ids, self.slots = sample_ids, actual_slots
        self.sampled_slots, self.padding_slots = sampled_slots, padding
        self.prefix_blocks = prefix_blocks
        self.layer_count, self.geometry = L, (h, B, 2 * d)
        self.independent_CPU_hashes = n <= 32
        capacities = set()
        for layer in range(L):
            dst = destinations[layer]
            pair = kv.layers[layer]
            shape = (1, h, admission.plan.original_prefix_tokens, d)
            if (not isinstance(dst, torch.Tensor) or dst.ndim != 4 or dst.dtype != torch.bfloat16
                    or tuple(dst.shape[1:]) != self.geometry or len(pair) != 2
                    or any(tuple(t.shape) != shape or t.dtype != torch.bfloat16 or t.device != dst.device for t in pair)
                    or max(actual_slots + padding) >= dst.shape[0] * B):
                raise IdentityMismatch('load diagnostic registered destination/source geometry differs')
            capacities.add(dst.shape[0])
        if len(capacities) != 1:
            raise IdentityMismatch('load diagnostic registered layer block capacities differ')
        for layer in range(L):
            k, v = kv.layers[layer]
            logical = torch.cat((k[0, :, sample_ids].transpose(0, 1),
                                 v[0, :, sample_ids].transpose(0, 1)), dim=-1)
            if self.independent_CPU_hashes:
                self._source[layer] = self._bits(logical)
                self._storage_evidence[layer] = {'snapshot_kind': 'independent_CPU_copy',
                    'independent_from_source_storage': all(_storage(self._source[layer]) != _storage(t) for t in (k,v)),
                    'independent_from_destination_storage': _storage(self._source[layer]) != _storage(destinations[layer])}
            else:
                self._source[layer], self._storage_evidence[layer] = self._snapshot(logical, (k,v), destinations[layer])
            if padding:
                index = torch.tensor(padding, dtype=torch.int64, device=destinations[layer].device)
                unused = destinations[layer][index // B, :, index % B]
                if self.independent_CPU_hashes:
                    self._padding[layer] = self._bits(unused)
                    self._storage_evidence[layer]['padding_snapshot_independent_from_destination_storage'] = _storage(self._padding[layer]) != _storage(destinations[layer])
                else:
                    self._padding[layer], padding_evidence = self._snapshot(unused, (), destinations[layer])
                    self._storage_evidence[layer]['padding_snapshot_independent_from_destination_storage'] = padding_evidence['independent_from_destination_storage']

    def after(self, admission, actual_request_id, destinations, slots):
        self._check(admission, actual_request_id)
        if not self._started or slots.tolist() != self.slots:
            raise IdentityMismatch('load diagnostic actual slots changed during writer')
        B = self.binding['block_size']
        layers = []
        try:
            for layer in range(self.layer_count):
                index = torch.tensor(self.sampled_slots, dtype=torch.int64, device=destinations[layer].device)
                actual_device = destinations[layer][index // B, :, index % B].contiguous()
                compared = self._bits(actual_device) if self.independent_CPU_hashes else actual_device.view(torch.uint8)
                if not torch.equal(compared, self._source[layer]):
                    raise ArtifactCorrupt('load diagnostic destination differs bitwise from converted source')
                maxabs = float((compared.view(torch.bfloat16).float() - self._source[layer].view(torch.bfloat16).float()).abs().max())
                actual = compared if self.independent_CPU_hashes else self._bits(actual_device)
                source_hash = self._hash(self._source[layer]) if self.independent_CPU_hashes else self._hash(actual)
                comparison_surface = 'gpu' if compared.device.type == 'cuda' else compared.device.type
                record = {'layer': layer, 'source_sha256': source_hash,
                          'source_hash_basis': 'independent_CPU_source_bytes' if self.independent_CPU_hashes else 'derived_after_proven_' + comparison_surface + '_bitwise_equal',
                          'destination_sha256': self._hash(actual), 'sampled_KV_bitwise_equal': True,
                          'comparison_device_type': compared.device.type, 'sampled_KV_max_abs_error': maxabs,
                          'source_cpu_bytes': self._source[layer].numel() if self.independent_CPU_hashes else 0,
                          'destination_cpu_bytes': actual.numel(), 'storage_evidence': self._storage_evidence[layer],
                          'unused_partial_block_offsets_bitwise_unchanged': True,
                          'padding_before_sha256': None, 'padding_after_sha256': None,
                          'padding_hash_basis': 'no_unused_offsets'}
                if self.padding_slots:
                    unused = torch.tensor(self.padding_slots, dtype=torch.int64, device=destinations[layer].device)
                    padding_device = destinations[layer][unused // B, :, unused % B].contiguous()
                    compared_padding = self._bits(padding_device) if self.independent_CPU_hashes else padding_device.view(torch.uint8)
                    if not torch.equal(compared_padding, self._padding[layer]):
                        raise ArtifactCorrupt('load diagnostic writer overwrote unused partial-block offsets')
                    padding = compared_padding if self.independent_CPU_hashes else self._bits(padding_device)
                    record.update(padding_before_sha256=self._hash(self._padding[layer]) if self.independent_CPU_hashes else self._hash(padding),
                                  padding_after_sha256=self._hash(padding),
                                  padding_hash_basis='independent_CPU_before_after_bytes' if self.independent_CPU_hashes else 'derived_after_proven_' + comparison_surface + '_bitwise_equal')
                layers.append(record)
            if self.cpu_readback_bytes != self.planned_cpu_readback_bytes:
                raise ArtifactCorrupt('load diagnostic CPU byte accounting differs')
            return {'schema': 'scd-actual-load-readback-v1', 'scope': 'registered destination readback after existing load sync before source release',
                    **self.binding, 'destination_device_type': destinations[0].device.type,
                    'performance_qualified': False,
                    'sampling': 'all_imported_tokens' if self.binding['imported_tokens'] <= 32 else 'first32_last32_union',
                    'requested_token_ranges_half_open': self.requested_ranges,
                    'sampled_token_ranges_half_open': self.ranges, 'sampled_tokens_per_layer': len(self.sample_ids),
                    'actual_slot_ids': self.slots, 'actual_slot_offsets': [i % B for i in self.slots],
                    'actual_prefix_block_ids': self.prefix_blocks, 'actual_prefix_block_count': len(self.prefix_blocks),
                    'expected_prefix_cdiv_blocks': (self.binding['imported_tokens'] + B - 1) // B,
                    'sampled_actual_slot_ids': self.sampled_slots, 'unused_partial_block_slot_ids': self.padding_slots,
                    'cpu_readback_bytes': self.cpu_readback_bytes, 'maximum_cpu_readback_bytes': MAX_CPU_READBACK_BYTES,
                    'device_temporary_snapshot_bytes': self.device_snapshot_bytes,
                    'gpu_temporary_snapshot_bytes': self.gpu_temporary_snapshot_bytes,
                    'source_hashes_independent_CPU_reads': self.independent_CPU_hashes,
                    'layers': layers, 'all_sampled_layers_bitwise_equal': True,
                    'all_unused_partial_offsets_bitwise_unchanged': True}
        finally:
            self.close()

    def close(self):
        self._source.clear()
        self._padding.clear()
        self._storage_evidence.clear()
        self._closed = True
