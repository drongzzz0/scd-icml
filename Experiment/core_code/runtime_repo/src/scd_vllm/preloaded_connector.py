from __future__ import annotations

import atexit
import importlib
import json
import os
import time
import weakref
from pathlib import Path
from typing import Any

_IMPORT_ERROR: Exception | None = None
_CONNECTOR_INSTANCES: weakref.WeakSet[Any] = weakref.WeakSet()

try:
    safetensors_torch = importlib.import_module("safetensors.torch")
    base_mod = importlib.import_module("vllm.distributed.kv_transfer.kv_connector.v1.base")
    example_mod = importlib.import_module("vllm.distributed.kv_transfer.kv_connector.v1.example_connector")
    mla_mod = importlib.import_module("vllm.model_executor.layers.attention.mla_attention")
    attn_mod = importlib.import_module("vllm.v1.attention.backend")
    triton_attn_mod = importlib.import_module("vllm.v1.attention.backends.triton_attn")
    KVConnectorMetadata = base_mod.KVConnectorMetadata
    ExampleConnector = example_mod.ExampleConnector
    ExampleConnectorMetadata = example_mod.ExampleConnectorMetadata
    MLACommonMetadata = mla_mod.MLACommonMetadata
    AttentionMetadata = attn_mod.AttentionMetadata
    TritonAttentionMetadata = triton_attn_mod.TritonAttentionMetadata
except Exception as exc:  # pragma: no cover - exercised only without vLLM.
    _IMPORT_ERROR = exc
    safetensors_torch = None  # type: ignore[assignment]
    KVConnectorMetadata = object  # type: ignore[assignment,misc]
    ExampleConnectorMetadata = object  # type: ignore[assignment,misc]

    class ExampleConnector:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("vLLM is required for SCDPreloadedConnector") from _IMPORT_ERROR

    class MLACommonMetadata:  # type: ignore[no-redef]
        pass

    class TritonAttentionMetadata:  # type: ignore[no-redef]
        pass

    class AttentionMetadata:  # type: ignore[no-redef]
        pass

class SCDPreloadedConnector(ExampleConnector):
    """External vLLM connector that caches loaded SCD KV tensors per worker.

    vLLM's bundled ``ExampleConnector`` is intentionally disk-backed and reads a
    safetensors file for every request/layer load. SCD serving commonly reuses
    the same validated prefix across a batch, so this connector keeps the same
    scheduler metadata semantics but memoizes each loaded KV tensor by filename.
    It is a release-facing prototype for the dedicated SCD connector path; it
    does not claim multi-tenant isolation or arbitrary prompt routing.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if _IMPORT_ERROR is not None:  # pragma: no cover - requires missing vLLM.
            raise RuntimeError("vLLM is required for SCDPreloadedConnector") from _IMPORT_ERROR
        super().__init__(*args, **kwargs)
        self._scd_cache_device = str(
            self._kv_transfer_config.get_from_extra_config("cache_device", "cuda")
        ).lower()
        self._scd_tensor_cache: dict[str, Any] = {}
        self._scd_cache_hits = 0
        self._scd_cache_misses = 0
        self._scd_stats_dir = os.environ.get("SCD_PRELOADED_STATS_DIR", "")
        _CONNECTOR_INSTANCES.add(self)
        if self._scd_stats_dir:
            atexit.register(self._write_scd_stats, True)

    def start_load_kv(self, forward_context: Any, **kwargs: Any) -> None:
        """Load cached KV tensors into vLLM's paged KV buffer."""

        def inject_kv_into_layer(
            dst_kv_cache_layer: Any,
            src_kv_cache: Any,
            slot_mapping: Any,
            attn_metadata: AttentionMetadata,
        ) -> None:
            dst_kv_cache_layer_shape = dst_kv_cache_layer.shape
            if isinstance(attn_metadata, MLACommonMetadata):
                num_pages = dst_kv_cache_layer_shape[0]
                page_size = dst_kv_cache_layer_shape[1]
                dst_kv_cache_layer = dst_kv_cache_layer.reshape(num_pages * page_size, -1)
                dst_kv_cache_layer[slot_mapping, ...] = src_kv_cache
            elif isinstance(attn_metadata, TritonAttentionMetadata):
                block_idxs = slot_mapping // self._block_size
                offsets = slot_mapping % self._block_size
                dst_kv_cache_layer[block_idxs, :, offsets] = src_kv_cache
            else:
                num_pages = dst_kv_cache_layer_shape[1]
                page_size = dst_kv_cache_layer_shape[2]
                dst_kv_cache_layer = dst_kv_cache_layer.reshape(2, num_pages * page_size, -1)
                dst_kv_cache_layer[:, slot_mapping, ...] = src_kv_cache

        metadata: KVConnectorMetadata = self._get_connector_metadata()
        assert isinstance(metadata, ExampleConnectorMetadata)

        attn_metadata = forward_context.attn_metadata
        if attn_metadata is None:
            return

        for request in metadata.requests:
            if request.is_store:
                continue
            for layer_name in forward_context.no_compile_layers:
                layer = forward_context.no_compile_layers[layer_name]
                kv_cache_attr = getattr(layer, "kv_cache", None)
                if kv_cache_attr is None:
                    continue

                kv_cache_layer = kv_cache_attr[forward_context.virtual_engine]
                filename = self._generate_filename_debug(
                    layer_name, request.token_ids, request.mm_hashes
                )
                kv_cache = self._load_scd_kv(filename, kv_cache_layer.device)
                if isinstance(attn_metadata, dict):
                    inject_kv_into_layer(
                        kv_cache_layer,
                        kv_cache,
                        request.slot_mapping,
                        attn_metadata[layer_name],
                    )

    def _load_scd_kv(self, filename: str, target_device: Any) -> Any:
        cached = self._scd_tensor_cache.get(filename)
        if cached is None:
            tensor = safetensors_torch.load_file(filename)["kv_cache"]  # type: ignore[union-attr]
            if self._scd_cache_device in {"cuda", "gpu"}:
                tensor = tensor.to(device=target_device, non_blocking=True)
            self._scd_tensor_cache[filename] = tensor
            self._scd_cache_misses += 1
            cached = tensor
        else:
            self._scd_cache_hits += 1
        self._maybe_write_scd_stats()

        if getattr(cached, "device", None) != target_device:
            return cached.to(device=target_device, non_blocking=True)
        return cached

    def _maybe_write_scd_stats(self) -> None:
        if not self._scd_stats_dir:
            return
        total = self._scd_cache_hits + self._scd_cache_misses
        if total <= 1 or total % 1024 == 0:
            self._write_scd_stats(False)

    def _write_scd_stats(self, final: bool = False) -> None:
        if not self._scd_stats_dir:
            return
        stats_dir = Path(self._scd_stats_dir)
        stats_dir.mkdir(parents=True, exist_ok=True)
        pid = os.getpid()
        payload = {
            "connector": "SCDPreloadedConnector",
            "pid": pid,
            "cache_device": self._scd_cache_device,
            "cache_hits": int(self._scd_cache_hits),
            "cache_misses": int(self._scd_cache_misses),
            "cached_files": int(len(self._scd_tensor_cache)),
            "cached_bytes": int(sum(_tensor_nbytes(tensor) for tensor in self._scd_tensor_cache.values())),
            "final": bool(final),
            "timestamp_unix": time.time(),
        }
        target = stats_dir / f"scd_preloaded_connector_{pid}.json"
        tmp = target.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        tmp.replace(target)


def _tensor_nbytes(tensor: Any) -> int:
    try:
        return int(tensor.numel()) * int(tensor.element_size())
    except Exception:
        return 0


def flush_scd_preloaded_connector_stats(final: bool = False) -> int:
    """Flush stats for live connector instances in this process."""

    flushed = 0
    for instance in list(_CONNECTOR_INSTANCES):
        instance._write_scd_stats(final)
        flushed += 1
    return flushed
