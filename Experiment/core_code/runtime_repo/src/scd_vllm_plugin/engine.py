from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scd_vllm.injection import convert_bundle_to_example_storage, run_injection_smoke
from scd_vllm.reconstructed_export import export_reconstructed_bundle


@dataclass(frozen=True)
class SCDVLLMConfig:
    """Minimal configuration for the external SCD-vLLM adapter.

    This facade intentionally keeps vLLM out of the core SCD runtime. It
    prepares a reconstructed SCD bundle, converts it to vLLM connector storage,
    and can run the current smoke path. Production serving should build on top
    of this boundary instead of patching vLLM internals directly.
    """

    producer_model: str
    consumer_model: str
    reuse_ckpt: str
    patch_ckpt: str
    route: str
    tokenizer: str = ""
    dtype: str = "bfloat16"
    transfer_dtype: str = ""
    producer_device: str = "cuda:0"
    consumer_device: str = "cuda:1"
    patch_ckpt_device: str = "cpu"
    max_prefix_tokens: int = 256
    bandwidth_gbps: float = 200.0
    quant_group_size: int = 128
    runtime_patch_window_size: int = 8
    runtime_patch_persist_shards: bool = False
    runtime_patch_persist_modules: bool = False


class SCDVLLMEngine:
    """Plug-and-play adapter around the existing SCD-vLLM MVP.

    The class is a thin API layer. It does not fork vLLM and does not claim a
    production connector. The supported MVP path is:
    reconstructed SCD bundle -> ExampleConnector-compatible storage -> smoke.
    """

    def __init__(self, config: SCDVLLMConfig) -> None:
        self.config = config

    def export_bundle(
        self,
        *,
        out_dir: str | Path,
        prompt: str = "",
        prefix_file: str = "",
    ) -> dict[str, Any]:
        prompt_text = _resolve_prompt(prompt=prompt, prefix_file=prefix_file)
        return export_reconstructed_bundle(
            producer_model=self.config.producer_model,
            consumer_model=self.config.consumer_model,
            tokenizer_id=self.config.tokenizer or self.config.consumer_model,
            reuse_ckpt=self.config.reuse_ckpt,
            patch_ckpt=self.config.patch_ckpt,
            route_path=self.config.route,
            prompt=prompt_text,
            dtype=self.config.dtype,
            transfer_dtype=self.config.transfer_dtype or self.config.dtype,
            producer_device=self.config.producer_device,
            consumer_device=self.config.consumer_device,
            patch_ckpt_device=self.config.patch_ckpt_device,
            max_prefix_tokens=self.config.max_prefix_tokens,
            bandwidth_gbps=self.config.bandwidth_gbps,
            quant_group_size=self.config.quant_group_size,
            runtime_patch_window_size=self.config.runtime_patch_window_size,
            runtime_patch_persist_shards=self.config.runtime_patch_persist_shards,
            runtime_patch_persist_modules=self.config.runtime_patch_persist_modules,
            out_dir=out_dir,
        )

    def convert_bundle(
        self,
        *,
        bundle_dir: str | Path,
        storage_dir: str | Path,
        block_size: int = 16,
        overwrite: bool = False,
        layer_name_template: str = "model.layers.{layer}.self_attn.attn",
    ) -> dict[str, Any]:
        return convert_bundle_to_example_storage(
            bundle_dir=bundle_dir,
            storage_dir=storage_dir,
            block_size=block_size,
            overwrite=overwrite,
            layer_name_template=layer_name_template,
        )

    def smoke_test(
        self,
        *,
        bundle_dir: str | Path,
        storage_dir: str | Path,
        out_dir: str | Path,
        model: str = "",
        tokenizer: str = "",
        prompt: str = "",
        prefix_file: str = "",
        block_size: int = 16,
        overwrite: bool = False,
        max_model_len: int = 512,
        max_new_tokens: int = 16,
        gpu_memory_utilization: float = 0.85,
        seed: int = 0,
        temperature: float = 0.0,
        top_p: float = 1.0,
        enforce_eager: bool = False,
        enable_prefix_caching: bool = False,
        enable_chunked_prefill: bool = False,
        disable_hybrid_kv_cache_manager: bool = False,
    ) -> dict[str, Any]:
        args = argparse.Namespace(
            bundle_dir=str(bundle_dir),
            storage_dir=str(storage_dir),
            block_size=int(block_size),
            layer_name_template="model.layers.{layer}.self_attn.attn",
            overwrite=bool(overwrite),
            out=str(out_dir),
            model=model or self.config.consumer_model,
            tokenizer=tokenizer or self.config.tokenizer or model or self.config.consumer_model,
            prompt=prompt,
            prefix_file=prefix_file,
            dtype=self.config.dtype,
            max_model_len=int(max_model_len),
            max_new_tokens=int(max_new_tokens),
            gpu_memory_utilization=float(gpu_memory_utilization),
            seed=int(seed),
            temperature=float(temperature),
            top_p=float(top_p),
            enforce_eager=bool(enforce_eager),
            enable_prefix_caching=bool(enable_prefix_caching),
            enable_chunked_prefill=bool(enable_chunked_prefill),
            disable_hybrid_kv_cache_manager=bool(disable_hybrid_kv_cache_manager),
        )
        return run_injection_smoke(args)

    def run_smoke_pipeline(
        self,
        *,
        out_dir: str | Path,
        prompt: str = "",
        prefix_file: str = "",
        bundle_name: str = "reuse_patch_bundle",
        block_size: int = 16,
        overwrite: bool = False,
        max_model_len: int = 512,
        max_new_tokens: int = 16,
        gpu_memory_utilization: float = 0.85,
        enforce_eager: bool = False,
        enable_prefix_caching: bool = False,
        enable_chunked_prefill: bool = False,
        disable_hybrid_kv_cache_manager: bool = False,
    ) -> dict[str, Any]:
        root = Path(out_dir)
        bundle_dir = root / bundle_name
        storage_dir = root / "example_connector_storage"
        smoke_dir = root / "injection_smoke"
        export_metrics = self.export_bundle(out_dir=bundle_dir, prompt=prompt, prefix_file=prefix_file)
        smoke_metrics = self.smoke_test(
            bundle_dir=bundle_dir,
            storage_dir=storage_dir,
            out_dir=smoke_dir,
            prompt=prompt,
            prefix_file=prefix_file,
            block_size=block_size,
            overwrite=overwrite,
            max_model_len=max_model_len,
            max_new_tokens=max_new_tokens,
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=enforce_eager,
            enable_prefix_caching=enable_prefix_caching,
            enable_chunked_prefill=enable_chunked_prefill,
            disable_hybrid_kv_cache_manager=disable_hybrid_kv_cache_manager,
        )
        return {
            "mode": "scd_vllm_plugin_smoke_pipeline",
            "out_dir": str(root),
            "bundle_dir": str(bundle_dir),
            "storage_dir": str(storage_dir),
            "smoke_dir": str(smoke_dir),
            "export": export_metrics,
            "smoke": smoke_metrics,
            "pass": bool(smoke_metrics.get("scd_injection_pass")),
        }


def _resolve_prompt(*, prompt: str, prefix_file: str) -> str:
    if prompt and prefix_file:
        raise ValueError("provide either prompt or prefix_file, not both")
    if prefix_file:
        return Path(prefix_file).read_text(encoding="utf-8").strip()
    return prompt
