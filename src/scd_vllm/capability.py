import importlib.metadata
import importlib.util
from importlib import import_module
import json
import os
from pathlib import Path
import shutil
import subprocess
from urllib.parse import urlsplit
from urllib.request import url2pathname

from scd_runtime.contracts.types import EngineCapability
from scd_runtime.contracts.capability import VLLM_COMMIT
from scd_runtime.errors import UnsupportedProfile


def ensure_sampler_jit_safe() -> bool:
    """Auto-disable FlashInfer JIT top-p/top-k sampler when neither flashinfer_cubin nor ninja+nvcc is available.

    Prevents EngineCore warmup_kernels() from crashing with ``FileNotFoundError: 'ninja'``
    when ``enforce_eager=False`` is used in environments without a JIT C++/CUDA build toolchain.
    """
    if 'VLLM_USE_FLASHINFER_SAMPLER' in os.environ:
        return False
    if importlib.util.find_spec('flashinfer') is None:
        return False
    if importlib.util.find_spec('flashinfer_cubin') is not None:
        return False
    has_ninja = shutil.which('ninja') is not None
    cuda_home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH') or '/usr/local/cuda'
    has_nvcc = shutil.which('nvcc') is not None or (Path(cuda_home) / 'bin' / 'nvcc').is_file()
    if not (has_ninja and has_nvcc):
        os.environ['VLLM_USE_FLASHINFER_SAMPLER'] = '0'
        return True
    return False


def installed_commit(*, strict_commit_pin: bool = False, strict: bool | None = None) -> str:
    if strict is not None:
        strict_commit_pin = bool(strict)
    """Resolve installed vLLM commit or release version.

    In open-source mode (strict_commit_pin=False), standard PyPI/conda wheels,
    VCS builds, and editable source checkouts are all accepted. When
    strict_commit_pin=True, enforces the exact reproduction commit VLLM_COMMIT.
    """
    try:
        dist = importlib.metadata.distribution('vllm')
    except importlib.metadata.PackageNotFoundError as exc:
        raise UnsupportedProfile('vLLM installation unavailable') from exc

    raw_direct = dist.read_text('direct_url.json')
    info = {}
    if raw_direct:
        try:
            parsed = json.loads(raw_direct)
            if isinstance(parsed, dict):
                info = parsed
        except ValueError:
            if strict_commit_pin:
                raise UnsupportedProfile('invalid vLLM direct_url.json')

    if 'vcs_info' in info and isinstance(info['vcs_info'], dict):
        commit = info['vcs_info'].get('commit_id', '')
        if strict_commit_pin and commit != VLLM_COMMIT:
            raise UnsupportedProfile('vLLM direct_url must attest the fixed full VCS commit')
        if commit:
            return str(commit)

    directory = info.get('dir_info')
    if isinstance(directory, dict) and directory.get('editable') is True and isinstance(info.get('url'), str):
        try:
            url = urlsplit(info['url'])
            if url.scheme == 'file' and url.netloc in ('', 'localhost') and not url.query and not url.fragment:
                raw = Path(url2pathname(url.path))
                if raw.is_absolute():
                    source = raw.resolve(strict=True)
                    package = import_module('vllm')
                    origin = getattr(package, '__file__', None)
                    if isinstance(origin, str) and Path(origin).resolve(strict=True).is_relative_to(source):
                        git_env = {k: v for k, v in os.environ.items() if not k.startswith('GIT_')}
                        git_env.update(GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM='1', GIT_OPTIONAL_LOCKS='0')
                        result = subprocess.run(
                            ['git', '-C', str(source), 'rev-parse', 'HEAD'],
                            capture_output=True, text=True, timeout=10, check=False, env=git_env,
                        )
                        if result.returncode == 0 and result.stdout.strip():
                            head = result.stdout.strip()
                            if strict_commit_pin and head != VLLM_COMMIT:
                                raise UnsupportedProfile('editable source HEAD differs from pinned commit')
                            return head
        except UnsupportedProfile:
            raise
        except Exception as exc:
            if strict_commit_pin:
                raise UnsupportedProfile('vLLM editable base/source identity is unverified: ' + str(exc)) from exc

    if strict_commit_pin:
        raise UnsupportedProfile('pinned vLLM commit unavailable in non-VCS wheel')
    return f'release:{dist.version}'


def inspect_engine(config, cache_config, *, strict_commit_pin: bool = False) -> EngineCapability:
    groups = cache_config.kv_cache_groups
    if len(groups) != 1 or type(groups[0].kv_cache_spec).__name__ not in ('FullAttentionSpec', 'SlidingWindowSpec'):
        raise UnsupportedProfile('only one dense attention cache group implemented')
    spec = groups[0].kv_cache_spec
    hf = config.model_config.hf_config
    window = getattr(hf, 'sliding_window', None)
    layers = getattr(hf, 'layer_types', None) or [
        ('sliding_attention' if window is not None else 'full_attention')
    ] * hf.num_hidden_layers
    attention = {
        'layer_types': [{'full_attention': 'full', 'sliding_attention': 'sliding'}.get(x, x) for x in layers],
        'sliding_window': window,
    }
    backend = config.attention_config.backend
    backend = getattr(backend, 'name', backend)
    if backend is None:
        backend = 'FLASH_ATTN'
    if not isinstance(backend, str):
        raise UnsupportedProfile('select an explicit verified attention backend')
    scheduler = config.scheduler_config
    from vllm.v1.request import Request
    no_store = callable(getattr(Request, 'get_skip_writing_prefix_cache', None))
    if config.cache_config.enable_prefix_caching and not no_store:
        raise UnsupportedProfile('APC requires request-wide skip_writing_prefix_cache support')
    return EngineCapability(
        installed_commit(strict_commit_pin=strict_commit_pin),
        backend,
        str(spec.dtype).replace('torch.', ''),
        config.parallel_config.tensor_parallel_size,
        config.parallel_config.pipeline_parallel_size,
        spec.block_size,
        len(groups),
        attention,
        not scheduler.disable_hybrid_kv_cache_manager,
        scheduler.enable_chunked_prefill,
        config.cache_config.enable_prefix_caching,
        config.model_config.max_model_len,
        no_store,
    )
