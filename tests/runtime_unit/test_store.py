import dataclasses
import json
import os
from pathlib import Path
import subprocess
import sys
import torch
import pytest


def make_kv(profile, prefix, fill=1.0):
    from scd_runtime.contracts.identity import fingerprint
    from scd_runtime.contracts.types import TargetKV

    return TargetKV(
        profile.target,
        profile.reader,
        prefix,
        tuple(
            (
                torch.full((1, 1, 16, 4), fill, dtype=torch.bfloat16),
                torch.full((1, 1, 16, 4), 2 * fill, dtype=torch.bfloat16),
            )
            for _ in range(2)
        ),
        fingerprint(profile, prefix),
    )


def test_cancel_one_lease_keeps_other(tmp_path, profile, prefix):
    from scd_runtime.contracts.prefix import plan_import
    from scd_runtime.transport.store import ArtifactStore

    store = ArtifactStore(tmp_path, 2**20)
    identifier = store.publish(make_kv(profile, prefix))
    a = store.pin(identifier, 'a')
    b = store.pin(identifier, 'b')
    src_dir = str(Path(__file__).resolve().parents[2] / 'src')
    env = dict(os.environ)
    env['PYTHONPATH'] = src_dir + (os.pathsep + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    script = (
        'from scd_runtime.transport.store import ArtifactStore,ArtifactLease; '
        'import json,sys; s=ArtifactStore(sys.argv[1],2**20); '
        's.release(ArtifactLease(**json.loads(sys.argv[2])))'
    )
    proc = subprocess.run([sys.executable, '-c', script, str(tmp_path), json.dumps(dataclasses.asdict(a))], capture_output=True, text=True, env=env)
    assert proc.returncode == 0, proc.stderr
    store.release(a)
    result = store.load(b, plan_import(prefix, tuple(range(18)), 16))
    assert result.layers[0][1].unique().float().tolist() == [2.0]
    assert result.prefix == prefix


def test_partial_publish_is_never_ready(tmp_path):
    from scd_runtime.errors import MissingAsset
    from scd_runtime.transport.store import ArtifactStore

    store = ArtifactStore(tmp_path, 2**20)
    pending = tmp_path / 'staging' / 'unfinished'
    pending.mkdir(parents=True)
    (pending / 'weights.safetensors').write_bytes(b'incomplete')
    with pytest.raises(MissingAsset):
        store.pin('a' * 64, 'request')


def test_modified_tensor_is_rejected(tmp_path, profile, prefix):
    from scd_runtime.contracts.prefix import plan_import
    from scd_runtime.errors import ArtifactCorrupt
    from scd_runtime.transport.store import ArtifactStore

    store = ArtifactStore(tmp_path, 2**20)
    identifier = store.publish(make_kv(profile, prefix))
    lease = store.pin(identifier, 'a')
    (tmp_path / 'objects' / identifier / 'kv.safetensors').write_bytes(b'bad')
    with pytest.raises(ArtifactCorrupt):
        store.load(lease, plan_import(prefix, tuple(range(18)), 16))


def test_pinned_objects_not_evicted(tmp_path, profile, prefix):
    from scd_runtime.contracts.prefix import plan_import
    from scd_runtime.errors import StoreCapacityExceeded
    from scd_runtime.transport.store import ArtifactStore

    store = ArtifactStore(tmp_path, 2**20)
    identifier = store.publish(make_kv(profile, prefix))
    lease = store.pin(identifier, 'a')
    huge = dataclasses.replace(prefix, token_ids=tuple(range(65536)))
    kv = make_kv(profile, prefix)
    kv = dataclasses.replace(
        kv,
        prefix=huge,
        layers=tuple(
            (torch.zeros(1, 1, 65536, 4, dtype=torch.bfloat16), torch.zeros(1, 1, 65536, 4, dtype=torch.bfloat16))
            for _ in range(2)
        ),
    )
    with pytest.raises(StoreCapacityExceeded):
        store.publish(kv)
    assert store.load(lease, plan_import(prefix, tuple(range(18)), 16)).prefix == prefix


def test_released_lease_cannot_load(tmp_path, profile, prefix):
    from scd_runtime.contracts.prefix import plan_import
    from scd_runtime.errors import MissingAsset
    from scd_runtime.transport.store import ArtifactStore

    store = ArtifactStore(tmp_path, 2**20)
    identifier = store.publish(make_kv(profile, prefix))
    lease = store.pin(identifier, 'a')
    store.release(lease)
    with pytest.raises(MissingAsset):
        store.load(lease, plan_import(prefix, tuple(range(18)), 16))


@pytest.mark.parametrize('strided', [False, True])
def test_serialized_snapshot_has_independent_storage_and_identical_bytes(tmp_path, profile, prefix, strided):
    from safetensors.torch import save_file
    from scd_runtime.transport.manifest import describe

    kv = make_kv(profile, prefix)
    if strided:
        kv = dataclasses.replace(
            kv,
            layers=tuple(tuple(t.transpose(-1, -2).contiguous().transpose(-1, -2) for t in pair) for pair in kv.layers),
        )
    expected = {f'{i}.{kind}': t.detach().cpu().contiguous().clone() for i, pair in enumerate(kv.layers) for kind, t in zip(('k', 'v'), pair)}
    _, actual = describe(kv)
    save_file(expected, str(tmp_path / 'old.safetensors'))
    save_file(actual, str(tmp_path / 'new.safetensors'))
    assert (tmp_path / 'old.safetensors').read_bytes() == (tmp_path / 'new.safetensors').read_bytes()
    for i, pair in enumerate(kv.layers):
        for kind, t in zip(('k', 'v'), pair):
            copy = actual[f'{i}.{kind}']
            assert copy.is_contiguous() and copy.data_ptr() != t.data_ptr()
            t.zero_()
            assert torch.equal(copy, expected[f'{i}.{kind}'])
