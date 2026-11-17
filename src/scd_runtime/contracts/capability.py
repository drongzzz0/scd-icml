from .types import integer, to_data
from ..errors import UnsupportedProfile

VLLM_COMMIT = '2b1f75c80f79496b313e28362f0856632dfbf5ea'
SUPPORTED_BACKENDS = frozenset(('FLASH_ATTN', 'TRITON_ATTN', 'FLASHINFER'))


def check_capability(profile, engine, *, strict_commit_pin: bool = False):
    integer(engine.block_size, 'block_size')
    failures = []
    if strict_commit_pin and engine.vllm_commit != VLLM_COMMIT:
        failures.append('unverified vLLM commit')
    elif not isinstance(engine.vllm_commit, str) or not engine.vllm_commit:
        failures.append('missing vLLM version/commit identity')
    if type(engine.prefix_cache) is not bool:
        failures.append('prefix cache boolean')
    if engine.prefix_cache and engine.request_prefix_cache_policy is not True:
        failures.append('APC requires request-wide skip_writing_prefix_cache support')
    if engine.dtype != 'bfloat16' or profile.target.dtype != engine.dtype:
        failures.append('dtype')
    if engine.tp != 1 or engine.pp != 1:
        failures.append('parallelism')
    if engine.kv_groups != 1 or engine.hma:
        failures.append('cache groups/HMA')
    if engine.chunked_prefill:
        failures.append('chunking')
    if engine.backend not in SUPPORTED_BACKENDS:
        failures.append('backend')
    if to_data(engine.attention) != to_data(profile.target.attention):
        failures.append('attention mismatch')
    kinds = set(profile.target.attention.get('layer_types', ()))
    window = profile.target.attention.get('sliding_window')
    if kinds - {'full', 'sliding'} or len(kinds) != 1:
        failures.append('mixed/unknown attention')
    if window is not None:
        if (type(window) is not int or window < 1
                or type(engine.max_model_len) is not int
                or not 0 < engine.max_model_len <= window):
            failures.append('SWA only supported for complete requests within the original window')
    elif kinds != {'full'}:
        failures.append('window definition missing')
    if failures:
        raise UnsupportedProfile('; '.join(failures))
