from __future__ import annotations

import argparse
import importlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from scd_vllm.batch_benchmark import _collect_preloaded_connector_stats
from scd_vllm.bundle import array_sha256, read_manifest, text_sha256
from scd_vllm.hf_export import load_prompt
from scd_vllm.injection import convert_bundle_to_example_storage, run_injection_smoke
from scd_vllm.reconstructed_export import export_reconstructed_bundle
from scd_vllm.truekv_oracle import _cuda_memory_info, _cuda_synchronize, _load_optional, summarize_storage


DEFAULT_PRELOADED_CONNECTOR = "SCDPreloadedConnector"
DEFAULT_PRELOADED_CONNECTOR_MODULE = "scd_vllm.preloaded_connector"


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

    def build_launch_spec(
        self,
        *,
        bundle_dir: str | Path,
        storage_dir: str | Path,
        model: str = "",
        tokenizer: str = "",
        dtype: str = "",
        max_model_len: int = 512,
        gpu_memory_utilization: float = 0.85,
        seed: int = 0,
        enforce_eager: bool = False,
        enable_prefix_caching: bool = False,
        enable_chunked_prefill: bool = False,
        kv_connector: str = DEFAULT_PRELOADED_CONNECTOR,
        kv_connector_module_path: str = DEFAULT_PRELOADED_CONNECTOR_MODULE,
        connector_cache_device: str = "cuda",
        stats_dir: str | Path = "",
        require_storage: bool = True,
    ) -> dict[str, Any]:
        return build_preloaded_launch_spec(
            model=model or self.config.consumer_model,
            tokenizer=tokenizer or self.config.tokenizer or model or self.config.consumer_model,
            bundle_dir=bundle_dir,
            storage_dir=storage_dir,
            dtype=dtype or self.config.dtype,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            seed=seed,
            enforce_eager=enforce_eager,
            enable_prefix_caching=enable_prefix_caching,
            enable_chunked_prefill=enable_chunked_prefill,
            kv_connector=kv_connector,
            kv_connector_module_path=kv_connector_module_path,
            connector_cache_device=connector_cache_device,
            stats_dir=stats_dir,
            require_storage=require_storage,
        )

    def launch_smoke(
        self,
        *,
        launch_spec: str | Path,
        prompt: str = "",
        prefix_file: str = "",
        out: str | Path = "",
        batch_size: int = 1,
        warmup_batches: int = 0,
        max_new_tokens: int = 16,
        temperature: float = 0.0,
        top_p: float = 1.0,
        seed: int = 0,
        use_tqdm: bool = False,
    ) -> dict[str, Any]:
        return run_launch_smoke(
            launch_spec=launch_spec,
            prompt=prompt,
            prefix_file=prefix_file,
            out=out,
            batch_size=batch_size,
            warmup_batches=warmup_batches,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            use_tqdm=use_tqdm,
        )

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


def build_preloaded_launch_spec(
    *,
    model: str,
    bundle_dir: str | Path,
    storage_dir: str | Path,
    tokenizer: str = "",
    dtype: str = "bfloat16",
    max_model_len: int = 512,
    gpu_memory_utilization: float = 0.85,
    seed: int = 0,
    enforce_eager: bool = False,
    enable_prefix_caching: bool = False,
    enable_chunked_prefill: bool = False,
    kv_connector: str = DEFAULT_PRELOADED_CONNECTOR,
    kv_connector_module_path: str = DEFAULT_PRELOADED_CONNECTOR_MODULE,
    connector_cache_device: str = "cuda",
    stats_dir: str | Path = "",
    require_storage: bool = True,
) -> dict[str, Any]:
    """Build the official-vLLM launch contract for a trusted SCD bundle.

    The returned payload is deliberately JSON-serializable so users can inspect,
    version, and pass the ``llm_kwargs`` into ``vllm.LLM`` without patching vLLM.
    It is a single-tenant trusted-bundle launch contract, not a multi-tenant
    cache-sharing policy.
    """

    bundle = Path(bundle_dir)
    storage = Path(storage_dir)
    manifest = read_manifest(bundle / "manifest.json")
    storage_summary = summarize_storage(storage)
    storage_ready = bool(
        storage_summary.get("exists")
        and int(storage_summary.get("safetensor_files", 0) or 0) > 0
    )
    if require_storage and not storage_ready:
        raise ValueError(f"storage_dir must contain safetensors files for vLLM launch: {storage}")

    connector_extra = {
        "shared_storage_path": str(storage),
        "cache_device": connector_cache_device,
    }
    kv_transfer_config: dict[str, Any] = {
        "kv_connector": kv_connector,
        "kv_role": "kv_both",
        "kv_load_failure_policy": "fail",
        "kv_connector_extra_config": connector_extra,
    }
    if kv_connector_module_path:
        kv_transfer_config["kv_connector_module_path"] = kv_connector_module_path

    disable_hybrid = _connector_requires_hma_disabled(kv_connector, kv_connector_module_path)
    llm_kwargs = {
        "model": model,
        "tokenizer": tokenizer or model,
        "dtype": dtype,
        "max_model_len": int(max_model_len),
        "gpu_memory_utilization": float(gpu_memory_utilization),
        "seed": int(seed),
        "enforce_eager": bool(enforce_eager),
        "enable_prefix_caching": bool(enable_prefix_caching),
        "enable_chunked_prefill": bool(enable_chunked_prefill),
        "disable_hybrid_kv_cache_manager": disable_hybrid,
        "kv_transfer_config": kv_transfer_config,
    }
    environment = {}
    if stats_dir:
        environment["SCD_PRELOADED_STATS_DIR"] = str(stats_dir)

    return {
        "mode": "scd_vllm_launch_spec",
        "launch_spec_pass": True,
        "claim_boundary": (
            "official vLLM LLM kwargs for a single-tenant trusted SCD bundle; "
            "not a multi-tenant cache-sharing or dynamic-routing claim"
        ),
        "model": model,
        "tokenizer": tokenizer or model,
        "bundle_dir": str(bundle),
        "storage_dir": str(storage),
        "storage_ready": storage_ready,
        "bundle": {
            "schema_version": manifest.schema_version,
            "bundle_type": manifest.bundle_type,
            "producer_model": manifest.producer_model,
            "consumer_model": manifest.consumer_model,
            "tokenizer_id": manifest.tokenizer_id,
            "token_hash": manifest.token_hash,
            "route_id": manifest.route_id,
            "route_hash": manifest.route_hash,
            "num_tokens": manifest.num_tokens,
            "dtype": manifest.dtype,
            "kv_spec": manifest.kv_spec,
        },
        "storage": storage_summary,
        "environment": environment,
        "llm_kwargs": llm_kwargs,
        "usage": [
            "Set each key in environment before constructing the vLLM engine.",
            "Use: from vllm import LLM; llm = LLM(**payload['llm_kwargs']).",
            "Run only for trusted bundles whose token_hash and route_id match the intended request prefix.",
        ],
        "safety": {
            "multi_tenant_safe": False,
            "dynamic_request_routing": False,
            "trusted_bundle_required": True,
        },
    }


def validate_launch_spec_safety(payload: dict[str, Any], *, source: str | Path = "<launch-spec>") -> None:
    """Reject launch specs that do not declare the trusted single-tenant contract."""

    safety = payload.get("safety")
    if not isinstance(safety, dict):
        raise ValueError(f"launch spec missing safety contract: {source}")
    expected = {
        "multi_tenant_safe": False,
        "dynamic_request_routing": False,
        "trusted_bundle_required": True,
    }
    for key, value in expected.items():
        if safety.get(key) is not value:
            raise ValueError(f"launch spec safety.{key} must be {value!r}: {source}")
    boundary = str(payload.get("claim_boundary", ""))
    for snippet in ("single-tenant", "trusted SCD bundle", "not a multi-tenant"):
        if snippet not in boundary:
            raise ValueError(f"launch spec claim_boundary missing {snippet!r}: {source}")


def _connector_requires_hma_disabled(kv_connector: str, kv_connector_module_path: str) -> bool:
    return kv_connector == DEFAULT_PRELOADED_CONNECTOR or kv_connector_module_path.endswith("preloaded_connector")


def run_launch_smoke(
    *,
    launch_spec: str | Path,
    prompt: str = "",
    prefix_file: str = "",
    out: str | Path = "",
    batch_size: int = 1,
    warmup_batches: int = 0,
    max_new_tokens: int = 16,
    temperature: float = 0.0,
    top_p: float = 1.0,
    seed: int = 0,
    use_tqdm: bool = False,
) -> dict[str, Any]:
    """Instantiate official ``vllm.LLM`` from a launch spec and generate text."""

    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if warmup_batches < 0:
        raise ValueError(f"warmup_batches must be non-negative, got {warmup_batches}")

    spec_path = Path(launch_spec)
    payload = json.loads(spec_path.read_text(encoding="utf-8-sig"))
    validate_launch_spec_safety(payload, source=spec_path)
    llm_kwargs = dict(payload.get("llm_kwargs") or {})
    if not llm_kwargs:
        raise ValueError(f"launch spec does not contain llm_kwargs: {spec_path}")

    environment = {str(key): str(value) for key, value in (payload.get("environment") or {}).items()}
    for key, value in environment.items():
        os.environ[key] = value

    bundle_dir = Path(payload.get("bundle_dir") or "")
    prompt_text = _resolve_launch_prompt(bundle_dir=bundle_dir, prompt=prompt, prefix_file=prefix_file)
    prompts = [prompt_text] * int(batch_size)

    vllm = importlib.import_module("vllm")
    torch = _load_optional("torch")
    llm_cls = getattr(vllm, "LLM")
    sampling_params_cls = getattr(vllm, "SamplingParams")

    memory_before = _cuda_memory_info(torch)
    start_load = time.perf_counter()
    llm = llm_cls(**llm_kwargs)
    load_seconds = time.perf_counter() - start_load
    memory_after_load = _cuda_memory_info(torch)

    sampling = sampling_params_cls(
        max_tokens=int(max_new_tokens),
        temperature=float(temperature),
        top_p=float(top_p),
        seed=int(seed),
    )
    for _ in range(int(warmup_batches)):
        llm.generate(prompts, sampling, use_tqdm=bool(use_tqdm))
    _cuda_synchronize(torch)

    _cuda_synchronize(torch)
    start_generation = time.perf_counter()
    outputs = llm.generate(prompts, sampling, use_tqdm=bool(use_tqdm))
    _cuda_synchronize(torch)
    generation_seconds = time.perf_counter() - start_generation

    prompt_token_ids: list[int] | None = None
    total_output_tokens = 0
    first_output_token_ids: list[int] = []
    first_output_text = ""
    for request_idx, request_output in enumerate(outputs):
        completion = request_output.outputs[0] if getattr(request_output, "outputs", None) else None
        token_ids = [int(token) for token in (getattr(completion, "token_ids", []) or [])]
        total_output_tokens += len(token_ids)
        if request_idx == 0:
            first_output_token_ids = token_ids
            first_output_text = str(getattr(completion, "text", "") or "")
            prompt_token_ids = [int(token) for token in (getattr(request_output, "prompt_token_ids", []) or [])]

    prompt_token_hash = array_sha256(np.asarray([prompt_token_ids], dtype=np.int64)) if prompt_token_ids else ""
    expected_token_hash = str((payload.get("bundle") or {}).get("token_hash") or "")
    bundle_token_match = bool(prompt_token_hash and expected_token_hash and prompt_token_hash == expected_token_hash)
    flushed_connectors = _flush_preloaded_connector_stats(final=True)
    stats_dir = environment.get("SCD_PRELOADED_STATS_DIR", "")
    preloaded_stats = _collect_preloaded_connector_stats(Path(stats_dir)) if stats_dir else None
    storage_dir = str(payload.get("storage_dir") or "")
    generation_ok = bool(len(outputs) == int(batch_size) and total_output_tokens > 0 and generation_seconds > 0.0)

    metrics = {
        "mode": "scd_vllm_launch_smoke",
        "launch_smoke_pass": bool(generation_ok and bundle_token_match),
        "claim_boundary": (
            "official vLLM LLM constructed from a single trusted launch-spec JSON; "
            "static prefix smoke, not multi-tenant serving or dynamic routing"
        ),
        "launch_spec": str(spec_path),
        "vllm_version": getattr(vllm, "__version__", None),
        "model": llm_kwargs.get("model"),
        "tokenizer": llm_kwargs.get("tokenizer"),
        "bundle_dir": str(bundle_dir),
        "storage_dir": storage_dir,
        "storage": summarize_storage(storage_dir) if storage_dir else {"exists": False},
        "environment": environment,
        "kv_connector": (llm_kwargs.get("kv_transfer_config") or {}).get("kv_connector"),
        "kv_connector_module_path": (llm_kwargs.get("kv_transfer_config") or {}).get("kv_connector_module_path"),
        "disable_hybrid_kv_cache_manager": bool(llm_kwargs.get("disable_hybrid_kv_cache_manager")),
        "prompt_text_hash": text_sha256(prompt_text),
        "prompt_token_count": len(prompt_token_ids or []),
        "prompt_token_ids": prompt_token_ids or [],
        "prompt_token_hash": prompt_token_hash,
        "expected_token_hash": expected_token_hash,
        "bundle_token_match": bundle_token_match,
        "batch_size": int(batch_size),
        "warmup_batches": int(warmup_batches),
        "max_new_tokens": int(max_new_tokens),
        "load_seconds": load_seconds,
        "generation_seconds": generation_seconds,
        "request_count": len(outputs),
        "output_tokens": int(total_output_tokens),
        "requests_per_second": _safe_div(len(outputs), generation_seconds),
        "tokens_per_second": _safe_div(total_output_tokens, generation_seconds),
        "generation_ok": generation_ok,
        "first_output_token_ids": first_output_token_ids,
        "first_output_text": first_output_text,
        "preloaded_connector_stats_dir": stats_dir,
        "preloaded_connector_stats": preloaded_stats,
        "preloaded_connector_flushed_instances": flushed_connectors,
        "memory_before": memory_before,
        "memory_after_load": memory_after_load,
    }
    if out:
        out_path = Path(out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return metrics


def _resolve_launch_prompt(*, bundle_dir: Path, prompt: str, prefix_file: str) -> str:
    if prompt or prefix_file:
        return load_prompt(prompt, prefix_file)
    prefix_path = bundle_dir / "prefix.txt"
    if not prefix_path.is_file():
        raise ValueError("no prompt/prefix-file supplied and bundle prefix.txt is missing")
    return load_prompt(prefix_path.read_text(encoding="utf-8"), "")


def _flush_preloaded_connector_stats(*, final: bool = False) -> int:
    try:
        mod = importlib.import_module("scd_vllm.preloaded_connector")
        flush = getattr(mod, "flush_scd_preloaded_connector_stats", None)
        if flush is None:
            return 0
        return int(flush(final=final) or 0)
    except Exception:
        return 0


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if float(denominator) else 0.0
