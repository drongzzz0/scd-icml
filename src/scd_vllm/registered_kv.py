"""Bind scheduler-block views through the public worker registration interface."""
import re

import torch

from scd_runtime.errors import UnsupportedProfile


class RegisteredKVMixin:
    def _cache_contract(self):
        core = getattr(self, 'core', self)
        return core.profile, core.capability

    def register_kv_caches(self, kv_caches):
        events=getattr(self,'_events',None)
        if events is not None and events.owner is not None and events.owner._jobs:
            raise UnsupportedProfile('cannot replace registered destinations during an in-flight transfer')
        # Failed re-registration must not leave stale destinations usable.
        self._scd_registered_kv = None
        profile, capability = self._cache_contract()
        if not isinstance(kv_caches, dict) or not kv_caches:
            raise UnsupportedProfile('registered KV cache mapping required')
        destinations = {}
        geometries = set()
        for name, tensor in kv_caches.items():
            match = re.search(r'(?:^|\.)layers\.(\d+)\.', name) if isinstance(name, str) else None
            if match is None:
                raise UnsupportedProfile('registered attention layer name is unsupported')
            index = int(match.group(1))
            if index in destinations:
                raise UnsupportedProfile('duplicate registered attention layer')
            if (not isinstance(tensor, torch.Tensor) or tensor.ndim != 4
                    or tensor.dtype != torch.bfloat16 or tensor.device.type not in ('cpu', 'cuda')
                    or tensor.shape[0] < 1 or tuple(tensor.shape[1:]) != (
                        profile.target.num_kv_heads, capability.block_size, 2 * profile.target.head_dim)):
                raise UnsupportedProfile('registered KV layout/dtype/device differs from manager blocks')
            destinations[index] = tensor
            geometries.add((tensor.shape[0], tensor.device))
        if set(destinations) != set(range(profile.target.num_layers)) or len(geometries) != 1:
            raise UnsupportedProfile('registered attention layers incomplete or inconsistent')
        # Keep original references, including non-contiguous views sharing a
        # backing allocation. Cloning would disconnect injection from attention.
        self._scd_registered_kv = destinations

    def unregister_kv_caches(self):
        events=getattr(self,'_events',None)
        if events is not None and events.owner is not None and events.owner._jobs:
            raise UnsupportedProfile('cannot unregister destinations during an in-flight transfer')
        self._scd_registered_kv = None

    def _registered_destinations(self):
        destinations = getattr(self, '_scd_registered_kv', None)
        if destinations is None:
            raise UnsupportedProfile('worker KV caches are not registered')
        return destinations
