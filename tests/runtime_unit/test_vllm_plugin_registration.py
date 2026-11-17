"""Verify vllm.general_plugins entry-point configuration, idempotent registration, and SCDConnectorV1Impl dispatch."""
from pathlib import Path
import sys
import tomllib
from types import ModuleType, SimpleNamespace

import pytest

import scd_vllm.plugin as plugin_mod
from scd_vllm.vllm_v1_adapter import SCDConnectorV1Impl


def test_pyproject_declares_vllm_general_plugins_entrypoint():
    pyproject = Path(__file__).resolve().parents[2] / 'pyproject.toml'
    data = tomllib.loads(pyproject.read_text(encoding='utf-8'))
    ep = data['project']['entry-points']['vllm.general_plugins']
    assert ep == {'scd_icml': 'scd_vllm.plugin:register'}


def test_plugin_register_is_idempotent_and_registers_all_three_connectors(monkeypatch):
    registered = {}

    class FakeFactory:
        _registry = registered

        @classmethod
        def register_connector(cls, name: str, module_path: str, class_name: str):
            if name in cls._registry:
                raise ValueError(f'duplicate registration: {name}')
            cls._registry[name] = (module_path, class_name)

    parts = ['vllm', 'vllm.distributed', 'vllm.distributed.kv_transfer', 'vllm.distributed.kv_transfer.kv_connector']
    for pkg_name in parts:
        mod = ModuleType(pkg_name)
        mod.__path__ = []  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, pkg_name, mod)
    fake_mod = ModuleType('vllm.distributed.kv_transfer.kv_connector.factory')
    fake_mod.KVConnectorFactory = FakeFactory  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, 'vllm.distributed.kv_transfer.kv_connector.factory', fake_mod)
    monkeypatch.setattr(plugin_mod, '_REGISTERED', False)

    first = plugin_mod.register()
    second = plugin_mod.register()  # Second call must be a clean no-op

    assert first == ['SCDConnectorV1', 'SCDGPUConnectorV1', 'SCDPreloadedConnector']
    assert second == []
    assert registered == {
        'SCDConnectorV1': ('scd_vllm.vllm_v1_adapter', 'SCDConnectorV1Impl'),
        'SCDGPUConnectorV1': ('scd_vllm.resident_connector', 'SCDGPUConnectorV1'),
        'SCDPreloadedConnector': ('scd_vllm.preloaded_connector', 'SCDPreloadedConnector'),
    }


@pytest.mark.parametrize(
    ('extra', 'expected_mode'),
    [
        ({'store_root': '/tmp/store'}, 'file'),
        ({'resident_session': 'a' * 32}, 'resident'),
        ({'shared_storage_path': '/tmp/example'}, 'preloaded'),
        ({'mode': 'gpu'}, 'resident'),
        ({'mode': 'store'}, 'file'),
        ({'mode': 'preloaded'}, 'preloaded'),
    ],
)
def test_scd_connector_v1_impl_mode_dispatch(monkeypatch, extra, expected_mode):
    created = []

    class DummyDelegate:
        def __init__(self, cfg, role, kv_cache_config=None):
            created.append((expected_mode, role, kv_cache_config))

        def get_num_new_matched_tokens(self, req, n):
            return 16, False

    if expected_mode == 'file':
        monkeypatch.setattr('scd_vllm.connector.SCDConnectorV1', DummyDelegate)
    elif expected_mode == 'resident':
        monkeypatch.setattr('scd_vllm.resident_connector.SCDGPUConnectorV1', DummyDelegate)
    else:
        monkeypatch.setattr('scd_vllm.preloaded_connector.SCDPreloadedConnector', DummyDelegate)

    vllm_cfg = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(kv_connector_extra_config=dict(extra))
    )
    adapter = SCDConnectorV1Impl(vllm_cfg, role='worker', kv_cache_config='kv_cfg')
    assert adapter.mode == expected_mode
    assert created == [(expected_mode, 'worker', 'kv_cfg')]
    assert adapter.get_num_new_matched_tokens('r', 0) == (16, False)


def test_scd_connector_v1_impl_rejects_unknown_mode():
    vllm_cfg = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(kv_connector_extra_config={'mode': 'unknown_backend'})
    )
    with pytest.raises(ValueError, match='Unsupported SCDConnectorV1 mode'):
        SCDConnectorV1Impl(vllm_cfg, role='worker')
