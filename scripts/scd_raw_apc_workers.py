"""Worker RPCs for the paired A800 experiment with one vLLM source and target."""

import hashlib
import time

import torch

from scd_raw_apc_source import APCSourceWorker, latest_source_capture
from scd_raw_apc_transport import export_payload, receive_payload
from scd_runtime.assets.manifest import load_assets
from scd_runtime.contracts.identity import fingerprint, profile_fingerprint
from scd_runtime.contracts.types import (
    MethodAssets, PrefixContract, SourceState, TargetKV, profile_from_dict, strict_construct,
)
from scd_runtime.methods.icml.backend import ICMLBackend
from scd_runtime.providers.hf.config_identity import check_config_identity
from scd_runtime.providers.hf.projections import TargetProjectionProvider
from scd_runtime.transport.resident import ResidentRegistry, register_registry, unregister_registry
from scd_vllm.resident_worker import GPUResidentWorkerExtension


def _digest(tensor):
    return hashlib.sha256(tensor.detach().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


def _kv_views(packed, head_dim):
    return tuple(
        (layer[..., :head_dim].transpose(0, 1).unsqueeze(0),
         layer[..., head_dim:].transpose(0, 1).unsqueeze(0))
        for layer in packed
    )


class SourceWorker(APCSourceWorker):
    @torch.inference_mode()
    def cmrc_conversion_setup(self, *, profile_data, assets_root):
        profile = profile_from_dict(profile_data)
        cfg = self.vllm_config
        check_config_identity(cfg.model_config.hf_config, profile.source)
        device = next(self.get_model().parameters()).device
        if device.type != "cuda" or not cfg.cache_config.enable_prefix_caching:
            raise RuntimeError("source needs CUDA and native vLLM APC")
        self._cmrc_converted = None
        self._cmrc_exported = None
        self._cmrc_conversion = None
        torch.set_float32_matmul_precision("highest")
        torch.backends.cudnn.allow_tf32 = False
        if assets_root is None:
            return {"profile_fingerprint": profile_fingerprint(profile), "source_apc": True, "conversion_loaded": False}
        assets = load_assets(assets_root, profile)
        with torch.inference_mode(False), torch.no_grad():
            gpu_assets = MethodAssets(
                profile, assets.root, {key: val.to(device) for key, val in assets.tensors.items()},
                assets.metadata,
            )
        backend = ICMLBackend()
        backend.shared_identity_layers(gpu_assets)
        projections = TargetProjectionProvider.from_assets(gpu_assets)
        if backend.compile_plan(gpu_assets, projections, device=device, max_seq_len=4096) is None:
            raise RuntimeError("the configured SCD Patch20/Reuse12 compiled path is unavailable")
        self._cmrc_conversion = profile, gpu_assets, backend, projections
        return {"profile_fingerprint": profile_fingerprint(profile), "source_apc": True, "conversion_loaded": True}

    @torch.inference_mode()
    def cmrc_convert(self, *, prefix_data, source_prompt_ids):
        if self._cmrc_conversion is None:
            raise RuntimeError("SCD conversion requires assets bound to the source and target")
        if self._cmrc_exported is not None or self._cmrc_converted is not None:
            raise RuntimeError("previous source payload is still owned")
        profile, assets, backend, projections = self._cmrc_conversion
        prefix = strict_construct(PrefixContract, prefix_data)
        packed, hidden = latest_source_capture(source_prompt_ids)
        state = SourceState(profile.source, prefix, _kv_views(packed, profile.source.head_dim), hidden)
        self._cmrc_converted = backend.convert(state, assets, projections, return_packed=True).packed_kv
        if self._cmrc_converted is None:
            raise RuntimeError("SCD did not produce a packed payload")
        torch.cuda.current_stream(packed.device).synchronize()

    @torch.inference_mode()
    def cmrc_export(self, *, arm, source_prompt_ids, verify_bits):
        if self._cmrc_exported is not None:
            raise RuntimeError("previous IPC export is still leased")
        raw, _ = latest_source_capture(source_prompt_ids)
        if arm == "raw":
            payload = raw
        elif arm == "scd" and self._cmrc_converted is not None:
            payload = self._cmrc_converted
        else:
            raise RuntimeError("missing conversion or invalid arm")
        self._cmrc_exported = payload
        return {"ipc": export_payload(payload), "digest": _digest(payload) if verify_bits else None}

    def cmrc_release_export(self):
        self._cmrc_exported = None
        self._cmrc_converted = None


class TimedRegistry(ResidentRegistry):
    def load_packed_fast(self, admission, *args, **kwargs):
        began = time.perf_counter_ns()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        stream = torch.cuda.current_stream(self.device)
        start.record(stream)
        super().load_packed_fast(admission, *args, **kwargs)
        end.record(stream)
        end.synchronize()
        finished = time.perf_counter_ns()
        self._entries[admission.artifact_id]["receipt"].update(
            kv_load_start_ns=began,
            kv_load_end_ns=finished,
            kv_load_gpu_ms=float(start.elapsed_time(end)),
        )


class TargetWorker(GPUResidentWorkerExtension):
    @torch.inference_mode()
    def scd_gpu_configure(self, *, session_id, profile_data, assets_root):
        from vllm.distributed.kv_transfer import get_kv_transfer_group

        cfg = self.vllm_config
        connector = get_kv_transfer_group()
        profile, capability = connector._cache_contract()
        device = next(self.get_model().parameters()).device
        if (profile != profile_from_dict(profile_data) or connector.session != session_id
                or device.type != "cuda" or cfg.cache_config.enable_prefix_caching):
            raise RuntimeError("invalid paired target engine configuration")
        check_config_identity(cfg.model_config.hf_config, profile.target)
        registry = TimedRegistry(profile, session_id, 512 * 1024**2, device=device, capability=capability)
        self._scd_gpu_state = {
            "registry": registry, "max_model_len": cfg.model_config.max_model_len,
            "pending_events": {}, "source": None, "source_model_bytes": 0, "asset_gpu_bytes": 0,
            "memo_cache": {}, "memo_bytes": 0, "memo_hits": 0, "memo_misses": 0,
        }
        self._cmrc_incoming = None
        register_registry(registry)
        return {
            "profile_fingerprint": profile_fingerprint(profile),
            "target_vocab_size": cfg.model_config.hf_config.vocab_size,
            "registered_cache": self.scd_cache_status(),
        }

    def destroy_worker_extension(self):
        state = getattr(self, "_scd_gpu_state", None)
        if state is not None:
            registry = state["registry"]
            try:
                registry.clear()
            finally:
                unregister_registry(registry.session_id)
                self._scd_gpu_state = None
                self._cmrc_incoming = None

    def cmrc_receive(self, *, descriptor, expected_digest):
        if self._cmrc_incoming is not None:
            raise RuntimeError("target has an unconsumed transfer")
        with torch.inference_mode(False), torch.no_grad():
            packed, copy_ms = receive_payload(descriptor)
        if expected_digest is not None and _digest(packed) != expected_digest:
            raise RuntimeError("source IPC export and target BF16 payload differ")
        self._cmrc_incoming = packed
        return {"transfer_gpu_ms": copy_ms, "transfer_bytes": packed.numel() * packed.element_size()}

    def cmrc_register(self, *, session_id, prefix_data):
        registry = self._scd_gpu_context(session_id)["registry"]
        profile = registry.profile
        prefix = strict_construct(PrefixContract, prefix_data)
        packed = self._cmrc_incoming
        expected = (profile.target.num_layers, len(prefix.token_ids),
                    profile.target.num_kv_heads, 2 * profile.target.head_dim)
        if packed is None or tuple(packed.shape) != expected or packed.is_inference():
            raise RuntimeError("received payload does not match the target KV geometry")
        layers = _kv_views(packed, profile.target.head_dim)
        snapshot = TargetKV(profile.target, profile.reader, prefix, layers,
                            fingerprint(profile, prefix), packed_kv=packed)
        self._cmrc_incoming = None
        receipt = registry.publish(snapshot, adopt_packed=True, sync_on_publish=False, check_finite=False)
        return {
            **receipt,
            "prepare_rpc_ms": 0.0,
            "source_prefill_ms": 0.0,
            "convert_ms": 0.0,
            "publish_ms": 0.0,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(registry.device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(registry.device),
        }
