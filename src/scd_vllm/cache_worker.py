"""Worker extension exposing registered KV cache status without reader monkeypatches."""
from scd_runtime.contracts.identity import profile_fingerprint
from scd_runtime.errors import UnsupportedProfile


class SCDCacheWorkerExtension:
    def scd_cache_status(self):
        from vllm.distributed.kv_transfer import get_kv_transfer_group
        connector = get_kv_transfer_group()
        destinations = connector._registered_destinations()
        profile, capability = connector._cache_contract()
        first = next(iter(destinations.values()))
        if not destinations:
            raise UnsupportedProfile('no registered KV cache layers')
        return {
            'profile_fingerprint': profile_fingerprint(profile),
            'registered_layers': len(destinations),
            'block_size': capability.block_size,
            'num_blocks': int(first.shape[0]),
            'capacity_tokens': int(first.shape[0] * capability.block_size),
            'device_types': sorted({t.device.type for t in destinations.values()}),
        }
