"""Capture one owned request from the pinned vLLM V1 APC producer."""

from contextlib import nullcontext
from dataclasses import dataclass
import re

import torch
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorBase_V1, KVConnectorMetadata
from vllm.forward_context import get_forward_context


APC_SOURCE_CONNECTOR_NAME = "APCSourceCaptureConnector"
_worker_connector = None


@dataclass(frozen=True)
class _SourceCaptureMetadata(KVConnectorMetadata):
    prompt_ids: tuple[int, ...] = ()
    block_ids: tuple[int, ...] = ()
    cached_tokens: int = 0


def latest_source_capture(prompt_ids=None):
    """Get the most recent capture in its owning worker process."""
    if _worker_connector is None or _worker_connector._latest is None:
        raise RuntimeError("source worker has not completed a capture")
    captured_ids, packed, hidden, ready = _worker_connector._latest
    if prompt_ids is not None and tuple(prompt_ids) != captured_ids:
        raise RuntimeError("source capture does not belong to the requested prompt")
    if ready is not None:
        torch.cuda.current_stream(packed.device).wait_event(ready)
    return packed, hidden


class APCSourceCaptureConnector(KVConnectorBase_V1):
    def __init__(self, config, role, kv_cache_config):
        super().__init__(config, role, kv_cache_config)
        if (config.cache_config.enable_prefix_caching is not True
                or not config.model_config.enforce_eager
                or config.scheduler_config.max_num_seqs != 1
                or config.scheduler_config.enable_chunked_prefill):
            raise ValueError("source capture requires an eager serial APC producer with unchunked prefill")
        self._block_size = config.cache_config.block_size
        model = config.model_config.hf_config
        self._num_layers = model.num_hidden_layers
        self._hidden_size = model.hidden_size
        self._num_heads = getattr(model, "num_key_value_heads", model.num_attention_heads)
        self._head_size = getattr(model, "head_dim", None) or model.hidden_size // model.num_attention_heads
        self._cache_names = {}
        self._caches = {}
        self._sidecar = None
        self._hooks = []
        self._block_prefixes = {}
        self._latest = None
        self._slots = ()
        self._captured_layers = set()
        self._slot_map = None
        self._stream = None

    def register_kv_caches(self, kv_caches):
        global _worker_connector
        caches = {}
        names = {}
        geometries = set()
        expected = (self._num_heads, self._block_size, 2 * self._head_size)
        for name, cache in kv_caches.items():
            match = re.search(r"(?:^|\.)layers\.(\d+)\.", name)
            if not match or cache.ndim != 4 or tuple(cache.shape[1:]) != expected or cache.dtype != torch.bfloat16:
                raise ValueError("source cache must be a per-layer [blocks, heads, block_size, 2*head_size] BF16 view")
            layer = int(match.group(1))
            if layer in caches:
                raise ValueError("duplicate registered source cache layer")
            caches[layer], names[layer] = cache, name
            geometries.add((cache.shape[0], cache.device))
        if set(caches) != set(range(self._num_layers)) or len(geometries) != 1:
            raise ValueError("registered source caches must cover all configured layers with uniform block capacity")
        if self._hooks:
            raise RuntimeError("cannot replace source caches after installing xnorm hooks")
        self._caches = caches
        self._cache_names = names
        _worker_connector = self

    def install_hooks(self, model, patch_layers):
        layers = tuple(patch_layers)
        if self._hooks or not layers or len(set(layers)) != len(layers) or set(layers) - self._caches.keys():
            raise ValueError("source capture requires one set of distinct registered Patch layers")
        cache = self._caches[0]
        capacity = cache.shape[0] * self._block_size
        self._sidecar = torch.empty((len(layers), capacity, self._hidden_size), dtype=cache.dtype, device=cache.device)
        self._patch_layers = layers
        for index, layer in enumerate(layers):
            attention = model.model.layers[layer].self_attn

            def hook(_attention, _args, kwargs, layer=layer, index=index):
                if self._slots:
                    self._write_hidden(layer, index, kwargs["hidden_states"])

            self._hooks.append(attention.register_forward_pre_hook(hook, with_kwargs=True))
        return {"patch_layers": layers, "cache_shape": tuple(cache.shape), "device": str(cache.device)}

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        # Use vLLM's local APC; this connector supplies no externally matched KV.
        return 0, False

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        if num_external_tokens:
            raise ValueError("source capture accepts only locally computed KV")

    def build_connector_meta(self, scheduler_output):
        new_requests = scheduler_output.scheduled_new_reqs
        if not new_requests:
            return _SourceCaptureMetadata()
        if len(new_requests) != 1:
            raise RuntimeError("source capture can schedule only one new request")
        request = new_requests[0]
        prompt = tuple(request.prompt_token_ids or ())
        cached = request.num_computed_tokens
        slots_done = scheduler_output.num_scheduled_tokens[request.req_id]
        if (len(prompt) < 2 or len(request.block_ids) != 1 or request.sampling_params.max_tokens != 1
                or cached < 0 or cached % self._block_size or slots_done <= 0 or cached + slots_done != len(prompt)):
            raise RuntimeError("source capture requires full-block APC plus one unchunked single-token decode request")
        blocks = tuple(request.block_ids[0][:(len(prompt) + self._block_size - 1) // self._block_size])
        if len(blocks) * self._block_size < len(prompt) or len(set(blocks)) != len(blocks):
            raise RuntimeError("source request has an incomplete or repeated block table")
        return _SourceCaptureMetadata(prompt, blocks, cached)

    def bind_connector_metadata(self, metadata):
        super().bind_connector_metadata(metadata)
        self._slots = ()
        self._captured_layers.clear()
        self._slot_map = None
        self._stream = None
        if not metadata.prompt_ids:
            return
        self._latest = None
        if self._sidecar is None:
            raise RuntimeError("initialize the source xnorm sidecar before generation")
        block_size = self._block_size
        num_blocks = self._caches[0].shape[0]
        if any(not 0 <= block < num_blocks for block in metadata.block_ids):
            raise RuntimeError("source block IDs exceed registered cache capacity")
        for index, block in enumerate(metadata.block_ids[:metadata.cached_tokens // block_size]):
            prefix = metadata.prompt_ids[:(index + 1) * block_size]
            if self._block_prefixes.get(block) != prefix:
                raise RuntimeError("APC hit is missing the matching source xnorm sidecar")
        self._slots = tuple(
            block * block_size + offset
            for block in metadata.block_ids
            for offset in range(block_size)
        )[:len(metadata.prompt_ids)]

    def _write_hidden(self, layer, index, hidden):
        metadata = self._get_connector_metadata()
        count = len(self._slots) - metadata.cached_tokens
        layer_slots = get_forward_context().slot_mapping[self._cache_names[layer]][:count]
        if (layer in self._captured_layers or hidden.ndim != 2 or hidden.shape[0] < count
                or hidden.shape[1] != self._hidden_size or hidden.dtype != self._sidecar.dtype
                or layer_slots.shape != (count,) or layer_slots.dtype != torch.int64
                or layer_slots.device != self._sidecar.device or hidden.device != self._sidecar.device):
            raise RuntimeError("source attention hook received unexpected xnorm or slot mapping")
        if self._slot_map is None:
            self._slot_map = layer_slots
            if hidden.is_cuda:
                self._stream = torch.cuda.current_stream(hidden.device)
        elif layer_slots.data_ptr() != self._slot_map.data_ptr() and not torch.equal(layer_slots, self._slot_map):
            raise RuntimeError("single-group source attention layers disagree about new slots")
        self._sidecar[index].index_copy_(0, layer_slots, hidden[:count].detach())
        self._captured_layers.add(layer)

    def start_load_kv(self, forward_context, **kwargs):
        pass

    def wait_for_layer_load(self, layer_name):
        pass

    def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs):
        pass

    def wait_for_save(self):
        if not self._slots:
            return
        metadata = self._get_connector_metadata()
        if self._captured_layers != set(self._patch_layers):
            raise RuntimeError("source forward did not write every Patch layer's xnorm")
        stream_scope = nullcontext() if self._stream is None else torch.cuda.stream(self._stream)
        with stream_scope, torch.no_grad():
            slots = torch.tensor(self._slots, device=self._sidecar.device, dtype=torch.int64)
            if not torch.equal(slots[metadata.cached_tokens:], self._slot_map):
                raise RuntimeError("source slot mapping differs from its request block table")
            # The last prompt token receives its own cache entry on the producer,
            # while the target recomputes that token from the first N-1 entries.
            prefix_slots = slots[:-1]
            blocks, offsets = prefix_slots // self._block_size, prefix_slots % self._block_size
            packed = torch.empty((self._num_layers, prefix_slots.numel(), self._num_heads, 2 * self._head_size),
                                 dtype=self._sidecar.dtype, device=self._sidecar.device)
            for layer, cache in self._caches.items():
                packed[layer].copy_(cache[blocks, :, offsets, :])
            activations = self._sidecar.index_select(1, prefix_slots)
            hidden = {(layer, "attention_input"): activations[i].unsqueeze(0)
                      for i, layer in enumerate(self._patch_layers)}
            ready = None
            if self._stream is not None:
                ready = torch.cuda.Event()
                ready.record(self._stream)

        for index, block in enumerate(metadata.block_ids):
            end = (index + 1) * self._block_size
            if end <= len(metadata.prompt_ids):
                self._block_prefixes[block] = metadata.prompt_ids[:end]
            else:
                self._block_prefixes.pop(block, None)
        self._latest = (metadata.prompt_ids, packed, hidden, ready)
        self._slots = ()

    def shutdown(self):
        global _worker_connector
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        self._latest = self._sidecar = None
        self._caches.clear()
        self._block_prefixes.clear()
        if _worker_connector is self:
            _worker_connector = None


class APCSourceWorker:
    def scd_apc_source_initialize(self, *, patch_layers):
        from vllm.distributed.kv_transfer import get_kv_transfer_group

        connector = get_kv_transfer_group()
        if not isinstance(connector, APCSourceCaptureConnector):
            raise RuntimeError("source worker is missing the APC capture connector")
        return connector.install_hooks(self.model_runner.model, patch_layers)
