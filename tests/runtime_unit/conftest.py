import os
from pathlib import Path
import sys

_repo_root = Path(__file__).resolve().parents[2]
_src = _repo_root / 'src'
_unit_dir = Path(__file__).resolve().parent
for _p in (str(_src), str(_unit_dir)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')

import pytest


@pytest.fixture
def profile():
    from scd_runtime.contracts.types import MethodProfile, ModelIdentity, ReaderIdentity

    source = ModelIdentity(
        'a' * 64,
        'b' * 64,
        'c' * 64,
        'd' * 64,
        2,
        1,
        4,
        'bfloat16',
        {'theta': 10000.0},
        {'layer_types': ['full', 'full'], 'sliding_window': None},
    )
    target = ModelIdentity(
        'e' * 64,
        'b' * 64,
        'c' * 64,
        'd' * 64,
        2,
        1,
        4,
        'bfloat16',
        {'theta': 10000.0},
        {'layer_types': ['full', 'full'], 'sliding_window': None},
    )
    reader = ReaderIdentity('none', target.manifest_sha256)
    return MethodProfile(
        'synthetic',
        'icml_reuse_patch',
        source,
        target,
        reader,
        {'reuse_layers': [0], 'patch_layers': [1], 'fit_mode': 'direct_kv_v1', 'reuse_key_space': 'pre_rope'},
        (0, 1),
        'icml-v1',
        'f' * 64,
    )


@pytest.fixture
def prefix():
    from scd_runtime.contracts.types import PrefixContract

    return PrefixContract(tuple(range(16)), 0, 'c' * 64, 'd' * 64)
