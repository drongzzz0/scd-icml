import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture
def source_capture_module(monkeypatch):
    class Connector:
        def __init__(self, config, role, kv_cache_config):
            self._metadata = None

        def bind_connector_metadata(self, metadata):
            self._metadata = metadata

        def has_connector_metadata(self):
            return self._metadata is not None

        def _get_connector_metadata(self):
            return self._metadata

    context = SimpleNamespace(slot_mapping={})
    monkeypatch.setitem(
        sys.modules, "vllm.distributed.kv_transfer.kv_connector.v1.base",
        SimpleNamespace(KVConnectorBase_V1=Connector, KVConnectorMetadata=object),
    )
    monkeypatch.setitem(
        sys.modules, "vllm.forward_context",
        SimpleNamespace(get_forward_context=lambda: context),
    )
    path = Path(__file__).resolve().parents[1] / "scripts" / "scd_raw_apc_source.py"
    spec = importlib.util.spec_from_file_location("scd_raw_apc_source", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module, context


@pytest.mark.parametrize("layers,hidden_size", ((32, 4096), (28, 2048)))
def test_source_snapshot_keeps_raw_block_order_and_reuses_matching_xnorm(source_capture_module, layers, hidden_size):
    module, context = source_capture_module
    config = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=16, enable_prefix_caching=True),
        model_config=SimpleNamespace(
            enforce_eager=True,
            hf_config=SimpleNamespace(num_hidden_layers=layers, num_attention_heads=16,
                                      num_key_value_heads=8, head_dim=128, hidden_size=hidden_size),
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=1, enable_chunked_prefill=False),
    )
    connector = module.APCSourceCaptureConnector(config, object(), object())
    shape = (5, 8, 16, 256)
    caches = {
        f"model.layers.{layer}.self_attn.attn": torch.zeros(shape, dtype=torch.bfloat16)
        for layer in range(layers)
    }
    caches["model.layers.0.self_attn.attn"][4].fill_(23)
    caches["model.layers.0.self_attn.attn"][1].fill_(17)
    caches["model.layers.0.self_attn.attn"][2].fill_(29)
    connector.register_kv_caches(caches)

    class Attention(torch.nn.Module):
        def forward(self, positions, hidden_states):
            return hidden_states

    attn = Attention()
    model_layers = [SimpleNamespace(self_attn=attn)]
    connector.install_hooks(SimpleNamespace(model=SimpleNamespace(layers=model_layers)), [0])

    def schedule(ids, block_ids, cached):
        request = SimpleNamespace(
            req_id="probe", prompt_token_ids=ids, block_ids=(block_ids,),
            num_computed_tokens=cached, sampling_params=SimpleNamespace(max_tokens=1),
        )
        return connector.build_connector_meta(SimpleNamespace(
            scheduled_new_reqs=[request], num_scheduled_tokens={"probe": len(ids) - cached},
        ))

    cold = list(range(18))
    connector.bind_connector_metadata(schedule(cold, (4, 1), 0))
    context.slot_mapping["model.layers.0.self_attn.attn"] = torch.tensor([*range(64, 80), 16, 17])
    attn(positions=None, hidden_states=torch.full((18, hidden_size), 11, dtype=torch.bfloat16))
    connector.wait_for_save()
    packed, hidden = module.latest_source_capture(cold)
    assert packed.shape == (layers, 17, 8, 256) and packed.is_contiguous()
    assert torch.all(packed[0, :16] == 23) and torch.all(packed[0, 16:] == 17)
    assert torch.all(hidden[0, "attention_input"] == 11)

    warm = [*cold[:16], 42, 43]
    connector.bind_connector_metadata(schedule(warm, (4, 2), 16))
    context.slot_mapping["model.layers.0.self_attn.attn"] = torch.tensor([32, 33])
    attn(positions=None, hidden_states=torch.full((2, hidden_size), 19, dtype=torch.bfloat16))
    connector.wait_for_save()
    packed, hidden = module.latest_source_capture(warm)
    assert torch.all(packed[0, :16] == 23) and torch.all(packed[0, 16:] == 29)
    assert torch.all(hidden[0, "attention_input"][:, :16] == 11)
    assert torch.all(hidden[0, "attention_input"][:, 16:] == 19)
    with pytest.raises(RuntimeError, match="sidecar"):
        connector.bind_connector_metadata(schedule([99, *warm[1:]], (4, 2), 16))
