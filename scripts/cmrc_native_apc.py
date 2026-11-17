"""Serial stock vLLM APC arm for the paired CMRC timing evaluation."""

import time
import uuid

from scd_runtime.resident_runner import _signal_deadline


_STOP_REASONS = frozenset(("stop_token", "text_boundary", "context_span_boundary", "max_new_tokens"))


class NativeAPC:
    def __init__(self, engine_config):
        from vllm import LLM

        config = {
            key: value for key, value in engine_config.items()
            if key not in {"model_manifest", "request_reader_routing", "async_loading"}
        }
        config.update(
            dtype="bfloat16",
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            enable_prefix_caching=True,
            enable_chunked_prefill=False,
            disable_hybrid_kv_cache_manager=True,
            enforce_eager=True,
            async_scheduling=False,
            max_num_seqs=1,
            trust_remote_code=False,
            disable_log_stats=False,
        )
        self._llm = LLM(**config)
        self._engine = self._llm.llm_engine
        if self._engine.vllm_config.cache_config.enable_prefix_caching is not True:
            self.close()
            raise RuntimeError("vLLM did not enable prefix caching")

    @property
    def llm_engine(self):
        return self._idle_engine()

    def reset_prefix_cache(self):
        engine = self._idle_engine()
        with _signal_deadline(120):
            if engine.reset_prefix_cache() is not True:
                raise RuntimeError("vLLM failed to reset the native prefix cache")

    def _idle_engine(self):
        if self._engine is None or self._engine.has_unfinished_requests():
            raise RuntimeError("native APC requires an open, idle serial engine")
        return self._engine

    def close(self):
        engine, self._engine = self._engine, None
        if engine is not None:
            try:
                with _signal_deadline(10):
                    engine.engine_core.shutdown(timeout=5)
            finally:
                self._llm = None

    def generate(self, prompt_ids, settings, observer):
        from vllm import SamplingParams
        from vllm.sampling_params import RequestOutputKind

        engine = self._idle_engine()
        deadline = time.perf_counter() + 120

        def remaining():
            return deadline - time.perf_counter()

        eos = settings.get("eos_token_id")
        stops = None if eos is None else (list(eos) if isinstance(eos, (list, tuple)) else [eos])
        params = SamplingParams(
            max_tokens=settings.get("max_new_tokens", 64),
            temperature=settings.get("temperature", 0.0),
            stop_token_ids=stops,
            ignore_eos=settings.get("ignore_eos", False),
            output_kind=RequestOutputKind.CUMULATIVE,
        )
        prompt = {"prompt_token_ids": list(prompt_ids)}
        external_id = uuid.uuid4().hex
        seen = ()
        cached_tokens = None
        try:
            with _signal_deadline(remaining()):
                internal_id = engine.add_request(external_id, prompt, params)
            if not isinstance(internal_id, str) or not internal_id:
                raise RuntimeError("vLLM did not return an internal request ID")
            while engine.has_unfinished_requests():
                with _signal_deadline(remaining()):
                    outputs = engine.step()
                if any(output.request_id not in (external_id, internal_id)
                       or getattr(output, "error", None) is not None for output in outputs):
                    raise RuntimeError("unexpected or failed native APC output")
                active = [output for output in outputs if output.outputs and output.outputs[0].token_ids]
                if not active:
                    continue
                if len(active) != 1 or len(active[0].outputs) != 1:
                    raise RuntimeError("native APC expects one token-bearing output per engine step")
                output = active[0]
                part = output.outputs[0]
                observed = tuple(part.token_ids)
                if len(observed) != len(seen) + 1 or observed[:-1] != seen:
                    raise RuntimeError("native APC expects one new cumulative token per step")
                seen = observed
                with _signal_deadline(remaining()):
                    reason = observer(seen)
                if cached_tokens is None:
                    cached_tokens = getattr(output, "num_cached_tokens", None)
                    if type(cached_tokens) is not int or not 0 <= cached_tokens <= len(prompt["prompt_token_ids"]):
                        raise RuntimeError("vLLM did not report valid native APC cache-hit token counts")
                if part.finish_reason in ("abort", "error"):
                    raise RuntimeError("native APC generation failed")
                if reason is not None:
                    if reason not in _STOP_REASONS:
                        raise RuntimeError("unknown CMRC protocol stop reason")
                    if engine.has_unfinished_requests():
                        with _signal_deadline(remaining()):
                            engine.abort_request([internal_id], internal=True)
                        # Order the abort before the next request can read the cache.
                        with _signal_deadline(remaining()):
                            engine.collective_rpc("check_health", timeout=remaining())
                    elif not output.finished:
                        raise RuntimeError("native APC natural completion is unconfirmed")
                    if engine.has_unfinished_requests():
                        raise RuntimeError("native APC request did not drain")
                    return seen, cached_tokens
            raise RuntimeError("native APC ended without an acknowledged CMRC stop")
        except BaseException as exc:
            try:
                self.close()
            except BaseException as cleanup_error:
                if hasattr(exc, "add_note"):
                    exc.add_note("Native APC engine shutdown also failed: " + repr(cleanup_error))
            raise
