import sys
from types import SimpleNamespace

import pytest

from scripts.cmrc_native_apc import NativeAPC


@pytest.fixture
def native_engine(monkeypatch):
    class Engine:
        def __init__(self, config):
            self.vllm_config = SimpleNamespace(
                cache_config=SimpleNamespace(enable_prefix_caching=config["enable_prefix_caching"]),
            )
            self.config = config
            self.calls = []
            self.replies = []
            self.running = False
            self.engine_core = SimpleNamespace(shutdown=lambda **kwargs: self.calls.append("shutdown"))

        def reset_prefix_cache(self):
            self.calls.append("reset")
            return True

        def has_unfinished_requests(self):
            return self.running

        def add_request(self, request_id, prompt, params):
            self.calls.append(("add", prompt, params))
            self.running = True
            return "internal"

        def step(self):
            ids, cached, finished = self.replies.pop(0)
            self.running = not finished
            return [SimpleNamespace(
                request_id="internal",
                outputs=[SimpleNamespace(token_ids=ids, finish_reason="stop" if finished else None)],
                num_cached_tokens=cached,
                finished=finished,
            )]

        def abort_request(self, ids, *, internal):
            assert ids == ["internal"] and internal
            self.calls.append("abort")
            self.running = False

        def collective_rpc(self, method, *, timeout):
            assert method == "check_health" and timeout > 0
            self.calls.append("health")
            return [None]

    created = []

    def llm(**config):
        engine = Engine(config)
        created.append(engine)
        return SimpleNamespace(llm_engine=engine)

    settings = SimpleNamespace(CUMULATIVE="cumulative")
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(LLM=llm, SamplingParams=dict))
    monkeypatch.setitem(sys.modules, "vllm.sampling_params", SimpleNamespace(RequestOutputKind=settings))
    return created


def test_native_apc_reports_first_output_hit_without_resetting_on_generate(native_engine):
    config = {"model": "fixture", "model_manifest": "fixture.json", "max_num_seqs": 1, "async_loading": False}
    native = NativeAPC(config)
    engine, = native_engine
    assert engine.config["enable_prefix_caching"] is True
    assert engine.config["enforce_eager"] is True
    assert engine.config["enable_chunked_prefill"] is False
    assert "model_manifest" not in engine.config
    assert "async_loading" not in engine.config
    assert "enable_prefix_caching" not in config
    native.reset_prefix_cache()
    settings = {"max_new_tokens": 32, "eos_token_id": [2]}
    prompt = list(range(18))
    for cached in [0, 16]:
        engine.replies = [([13], cached, False), ([13, 2], 0, True)]
        seen = []

        def observe(ids):
            seen.append(ids)
            return "stop_token" if ids[-1] == 2 else None

        assert native.generate(prompt, settings, observe) == ((13, 2), cached)
        assert seen == [(13,), (13, 2)]
    assert engine.calls[1][2] == {
        "max_tokens": 32, "temperature": 0.0, "stop_token_ids": [2],
        "ignore_eos": False, "output_kind": "cumulative",
    }
    assert engine.calls.count("reset") == 1
    assert "abort" not in engine.calls
    native.close()
    native.close()
    assert engine.calls.count("shutdown") == 1


def test_native_apc_aborts_and_checks_drain_at_observer_stop(native_engine):
    native = NativeAPC({"model": "fixture"})
    engine, = native_engine
    engine.replies = [([11], 0, False), ([11, 13], 0, False), ([11, 13, 7], 0, True)]
    ids, hit = native.generate(
        [5, 6], {"max_new_tokens": 32},
        lambda ids: "text_boundary" if ids[-1] == 13 else None,
    )
    assert ids == (11, 13) and hit == 0
    assert engine.calls[-2:] == ["abort", "health"]
    assert not engine.has_unfinished_requests()
    assert len(engine.replies) == 1
    native.close()
