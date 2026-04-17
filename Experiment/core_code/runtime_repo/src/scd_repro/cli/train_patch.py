from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from ..direct_kv_residual_patch import PATCH_FIT_MODE_DIRECT_KV_KNORM_RESIDUAL_MLP_V1, PATCH_FIT_MODE_XNORM_NATIVEKV_RESIDUAL_MLP_V1, PATCH_FIT_MODE_XNORM_NATIVEKV_SHARED_RESIDUAL_MLP_V1, train_direct_kv_knorm_residual_patch, train_xnorm_nativekv_residual_patch, train_xnorm_nativekv_shared_residual_patch
from ..io_utils import load_bundle, load_json, save_json
from ..patch import (
    PATCH_FIT_MODE_DIRECT_KV_V1,
    PATCH_FIT_MODE_DIRECT_KV_KNORM_V1,
    PATCH_FIT_MODE_XNORM_SEMANTIC_V1,
    SemanticProbe,
    XNormSemanticPatchLayer,
    aux_k_loss,
    aux_v_loss,
    create_direct_kv_error_stats_torch,
    create_direct_kv_stats_torch,
    freeze_module,
    hidden_loss,
    pack_module_state,
    pack_xnorm_semantic_patch_layer,
    semantic_kl_loss,
    solve_direct_kv_from_stats_torch,
    topk_match_at_k,
)
from ..pipeline import load_pairs_meta, save_patch_ckpt
from ..rope import de_rope_k_torch, resolve_rope_specs
from ..seed import set_global_seed
from ..torch_utils import clear_torch_cache, configure_torch_acceleration, resolve_device
from .common import add_accel_args, load_cfg, resolve_accel_options, resolve_logger


@dataclass
class ConsumerNativeComponents:
    k_proj_weight: torch.Tensor
    k_proj_bias: torch.Tensor | None
    v_proj_weight: torch.Tensor
    v_proj_bias: torch.Tensor | None


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="训练补丁对齐器")
    p.add_argument("--pairs", required=True)
    p.add_argument("--profile", required=True)
    p.add_argument("--route", default="")
    p.add_argument("--train-config", required=True)
    p.add_argument("--models-config", default="")
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log", default="")
    p.add_argument("--train-device", default="")
    add_accel_args(p)
    return p


def _iter_chunk_paths(pairs_dir: str):
    root = Path(pairs_dir)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    for row in manifest.get("chunks", []):
        yield root / str(row["path"])


def _select_layers(profile: dict, patch_budget: int, route: dict | None = None) -> list[int]:
    if route is not None:
        route_layers = [int(layer) for layer in route.get("patch_layers", [])]
        if route_layers:
            return route_layers[:patch_budget]
    if "recommended_patch_layers" in profile:
        return list(profile["recommended_patch_layers"][:patch_budget])
    scored = sorted(profile["layers"], key=lambda item: float(item["expected_gain"]), reverse=True)
    return [int(row["layer"]) for row in scored[:patch_budget]]


def _resolve_decoder_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise RuntimeError("无法从 consumer 模型中定位 decoder layers")


def _resolve_attn_module(layer_module):
    if hasattr(layer_module, "self_attn"):
        return layer_module.self_attn
    if hasattr(layer_module, "attn"):
        return layer_module.attn
    raise RuntimeError("无法从 decoder layer 中定位 attention 模块")


def _tensor_or_none(value: Any, *, device: torch.device) -> torch.Tensor | None:
    if value is None:
        return None
    return value.detach().to(device=device, dtype=torch.float32).contiguous()


def _coerce_consumer_native_components(value: ConsumerNativeComponents | dict[str, Any]) -> ConsumerNativeComponents:
    if isinstance(value, ConsumerNativeComponents):
        return value
    return ConsumerNativeComponents(
        k_proj_weight=value["k_proj_weight"],
        k_proj_bias=value.get("k_proj_bias"),
        v_proj_weight=value["v_proj_weight"],
        v_proj_bias=value.get("v_proj_bias"),
    )


def _project_consumer_native_kv_from_xnorm_b(
    arrays: dict[str, torch.Tensor],
    layer: int,
    hidden_dim: int,
    device: torch.device,
    native_components: ConsumerNativeComponents | dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    components = _coerce_consumer_native_components(native_components)
    xb = arrays[f"xnorm_b.{layer}"].reshape(-1, hidden_dim).to(device=device, dtype=torch.float32)
    k_weight = components.k_proj_weight.to(device=device, dtype=torch.float32)
    v_weight = components.v_proj_weight.to(device=device, dtype=torch.float32)
    k_bias = None if components.k_proj_bias is None else components.k_proj_bias.to(device=device, dtype=torch.float32)
    v_bias = None if components.v_proj_bias is None else components.v_proj_bias.to(device=device, dtype=torch.float32)
    kb_raw = F.linear(xb, k_weight, k_bias)
    vb = F.linear(xb, v_weight, v_bias)
    return kb_raw, vb


def _load_consumer_native_components(
    models_config: str,
    selected_layers: list[int],
    hidden_dim: int,
    kv_dim: int,
    device: torch.device,
    dtype: str,
) -> dict[int, ConsumerNativeComponents]:
    if not models_config:
        raise ValueError("fit_mode=xnorm_semantic_v1 时必须提供 --models-config")
    try:
        from transformers import AutoModelForCausalLM
    except Exception as exc:  # pragma: no cover - 依赖缺失时直接报错
        raise RuntimeError("缺少运行依赖: transformers") from exc

    models_cfg = load_cfg(models_config)
    model_b_cfg = models_cfg.get("model_b", {}) if isinstance(models_cfg, dict) else {}
    model_b_path = str(model_b_cfg.get("path") or "").strip()
    if not model_b_path:
        raise ValueError("models-config 中缺少 model_b.path")

    model = AutoModelForCausalLM.from_pretrained(
        model_b_path,
        torch_dtype=torch.float32 if str(dtype).lower() == "float32" else torch.float32,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.eval()
    layers = _resolve_decoder_layers(model)
    components: dict[int, ConsumerNativeComponents] = {}
    try:
        for layer in selected_layers:
            attn = _resolve_attn_module(layers[layer])
            k_proj = attn.k_proj
            v_proj = attn.v_proj
            if int(k_proj.weight.shape[1]) != int(hidden_dim):
                raise ValueError(f"layer={layer} 的 k_proj 输入维不匹配 hidden_dim={hidden_dim}")
            if int(k_proj.weight.shape[0]) != int(kv_dim):
                raise ValueError(f"layer={layer} 的 k_proj 输出维不匹配 kv_dim={kv_dim}")
            if int(v_proj.weight.shape[0]) != int(kv_dim):
                raise ValueError(f"layer={layer} 的 v_proj 输出维不匹配 kv_dim={kv_dim}")
            components[layer] = ConsumerNativeComponents(
                k_proj_weight=k_proj.weight.detach().to(device=device, dtype=torch.float32).contiguous(),
                k_proj_bias=_tensor_or_none(k_proj.bias, device=device),
                v_proj_weight=v_proj.weight.detach().to(device=device, dtype=torch.float32).contiguous(),
                v_proj_bias=_tensor_or_none(v_proj.bias, device=device),
            )
    finally:
        del model
        clear_torch_cache()
    return components


def _accumulate_direct_kv_layer_stats(
    arrays: dict[str, torch.Tensor],
    layer: int,
    hidden_dim: int,
    kv_dim: int,
    device: torch.device,
    stats: dict[str, torch.Tensor],
    rope_b,
    native_components: ConsumerNativeComponents | None = None,
) -> None:
    from ..patch import accumulate_direct_kv_stats_torch

    xa = arrays[f"xnorm_a.{layer}"].reshape(-1, hidden_dim).to(device=device, dtype=torch.float32)
    if native_components is None:
        kb_raw = de_rope_k_torch(arrays[f"k_b.{layer}"], rope_b).permute(0, 2, 1, 3).reshape(-1, kv_dim).to(device=device, dtype=torch.float32)
        vb = arrays[f"v_b.{layer}"].permute(0, 2, 1, 3).reshape(-1, kv_dim).to(device=device, dtype=torch.float32)
    else:
        kb_raw, vb = _project_consumer_native_kv_from_xnorm_b(arrays, layer, hidden_dim, device, native_components)
    accumulate_direct_kv_stats_torch(xa, kb_raw, vb, stats)


def _accumulate_direct_kv_error_layer_stats(
    arrays: dict[str, torch.Tensor],
    layer: int,
    hidden_dim: int,
    kv_dim: int,
    device: torch.device,
    stats: dict[str, torch.Tensor],
    rope_b,
    wk: torch.Tensor,
    wv: torch.Tensor,
    native_components: ConsumerNativeComponents | None = None,
) -> None:
    from ..patch import accumulate_direct_kv_error_stats_torch

    xa = arrays[f"xnorm_a.{layer}"].reshape(-1, hidden_dim).to(device=device, dtype=torch.float32)
    if native_components is None:
        kb_raw = de_rope_k_torch(arrays[f"k_b.{layer}"], rope_b).permute(0, 2, 1, 3).reshape(-1, kv_dim).to(device=device, dtype=torch.float32)
        vb = arrays[f"v_b.{layer}"].permute(0, 2, 1, 3).reshape(-1, kv_dim).to(device=device, dtype=torch.float32)
    else:
        kb_raw, vb = _project_consumer_native_kv_from_xnorm_b(arrays, layer, hidden_dim, device, native_components)
    accumulate_direct_kv_error_stats_torch(xa, kb_raw, vb, wk, wv, stats)


def _accumulate_direct_kv_layers_with_retry(
    arrays: dict[str, torch.Tensor],
    layers: list[int],
    hidden_dim: int,
    kv_dim: int,
    device: torch.device,
    stats_by_layer: dict[int, dict[str, torch.Tensor]],
    rope_b,
    logger,
    depth: int = 0,
    native_components_by_layer: dict[int, ConsumerNativeComponents] | None = None,
) -> None:
    try:
        for layer in layers:
            native_components = None if native_components_by_layer is None else native_components_by_layer[int(layer)]
            _accumulate_direct_kv_layer_stats(arrays, layer, hidden_dim, kv_dim, device, stats_by_layer[layer], rope_b, native_components)
    except RuntimeError as exc:
        text = str(exc).lower()
        if "out of memory" not in text and "cuda oom" not in text:
            raise
        clear_torch_cache()
        if len(layers) <= 1:
            raise
        mid = len(layers) // 2
        left = layers[:mid]
        right = layers[mid:]
        logger.warning("patch KV 统计触发 OOM，按层块二分重试 depth=%d size=%d -> %d + %d", depth, len(layers), len(left), len(right))
        _accumulate_direct_kv_layers_with_retry(arrays, left, hidden_dim, kv_dim, device, stats_by_layer, rope_b, logger, depth + 1, native_components_by_layer)
        _accumulate_direct_kv_layers_with_retry(arrays, right, hidden_dim, kv_dim, device, stats_by_layer, rope_b, logger, depth + 1, native_components_by_layer)


def _accumulate_direct_kv_error_layers_with_retry(
    arrays: dict[str, torch.Tensor],
    layers: list[int],
    hidden_dim: int,
    kv_dim: int,
    device: torch.device,
    stats_by_layer: dict[int, dict[str, torch.Tensor]],
    rope_b,
    wk_by_layer: dict[int, torch.Tensor],
    wv_by_layer: dict[int, torch.Tensor],
    logger,
    depth: int = 0,
    native_components_by_layer: dict[int, ConsumerNativeComponents] | None = None,
) -> None:
    try:
        for layer in layers:
            native_components = None if native_components_by_layer is None else native_components_by_layer[int(layer)]
            _accumulate_direct_kv_error_layer_stats(
                arrays,
                layer,
                hidden_dim,
                kv_dim,
                device,
                stats_by_layer[layer],
                rope_b,
                wk_by_layer[layer],
                wv_by_layer[layer],
                native_components,
            )
    except RuntimeError as exc:
        text = str(exc).lower()
        if "out of memory" not in text and "cuda oom" not in text:
            raise
        clear_torch_cache()
        if len(layers) <= 1:
            raise
        mid = len(layers) // 2
        left = layers[:mid]
        right = layers[mid:]
        logger.warning("patch 误差统计触发 OOM，按层块二分重试 depth=%d size=%d -> %d + %d", depth, len(layers), len(left), len(right))
        _accumulate_direct_kv_error_layers_with_retry(arrays, left, hidden_dim, kv_dim, device, stats_by_layer, rope_b, wk_by_layer, wv_by_layer, logger, depth + 1, native_components_by_layer)
        _accumulate_direct_kv_error_layers_with_retry(arrays, right, hidden_dim, kv_dim, device, stats_by_layer, rope_b, wk_by_layer, wv_by_layer, logger, depth + 1, native_components_by_layer)


def _train_direct_kv_patch(
    *,
    args,
    logger,
    cfg: dict[str, Any],
    meta: dict[str, Any],
    profile: dict[str, Any],
    route: dict[str, Any] | None,
    rope_b,
    device: torch.device,
    accel: dict[str, Any],
    fit_mode: str,
) -> None:
    from ..patch import finalize_direct_kv_error_stats_torch

    patch_cfg = cfg.get("patch", {})
    patch_budget = int(patch_cfg["patch_budget"])
    ridge_lambda = float(patch_cfg["ridge_lambda"])
    selected = _select_layers(profile, patch_budget, route=route)
    hidden_dim = int(meta["hidden_dim"])
    kv_dim = int(meta["num_key_value_heads"]) * int(meta["head_dim"])
    layer_block_size = int(accel["layer_block_size"])
    use_consumer_native_targets = fit_mode == PATCH_FIT_MODE_DIRECT_KV_KNORM_V1

    logger.info(
        "patch 训练开始: fit_mode=%s device=%s accel=%s layer_block_size=%d selected=%s",
        fit_mode,
        device,
        accel["accel_mode"],
        layer_block_size,
        selected,
    )

    native_components_by_layer: dict[int, ConsumerNativeComponents] | None = None
    if use_consumer_native_targets:
        native_components_by_layer = _load_consumer_native_components(
            args.models_config,
            selected,
            hidden_dim,
            kv_dim,
            device,
            "float32",
        )
        logger.info("direct-kv knorm-aware 训练目标: consumer pre-k_norm K + native V")

    direct_stats_by_layer: dict[int, dict[str, torch.Tensor]] = {layer: create_direct_kv_stats_torch(hidden_dim, kv_dim, device) for layer in selected}
    chunk_paths = list(_iter_chunk_paths(args.pairs))

    start = time.perf_counter()
    for idx, chunk_path in enumerate(chunk_paths, start=1):
        bundle = load_bundle(chunk_path, device=str(device), dtype="float32")
        arrays = bundle.arrays
        for start_idx in range(0, len(selected), layer_block_size):
            block_layers = selected[start_idx : start_idx + layer_block_size]
            _accumulate_direct_kv_layers_with_retry(
                arrays,
                block_layers,
                hidden_dim,
                kv_dim,
                device,
                direct_stats_by_layer,
                rope_b,
                logger,
                native_components_by_layer=native_components_by_layer,
            )
        del arrays, bundle
        clear_torch_cache()
        if idx == 1 or idx % 5 == 0 or idx == len(chunk_paths):
            elapsed = time.perf_counter() - start
            eta_s = (elapsed / idx) * (len(chunk_paths) - idx) if idx > 0 else 0.0
            logger.info("patch KV 统计进度: chunk %d/%d, elapsed=%.1fs, eta=%.1fs", idx, len(chunk_paths), elapsed, eta_s)

    out_arrays: dict[str, object] = {}
    wk_by_layer: dict[int, torch.Tensor] = {}
    wv_by_layer: dict[int, torch.Tensor] = {}
    for layer in selected:
        wk, wv, _patch_k_formula, _patch_v_formula, _patch_kv_formula = solve_direct_kv_from_stats_torch(
            direct_stats_by_layer[layer],
            ridge_lambda=ridge_lambda,
        )
        wk_by_layer[layer] = wk.to(device=device, dtype=torch.float32)
        wv_by_layer[layer] = wv.to(device=device, dtype=torch.float32)

    del direct_stats_by_layer
    clear_torch_cache()

    error_stats_by_layer: dict[int, dict[str, torch.Tensor]] = {layer: create_direct_kv_error_stats_torch(device) for layer in selected}
    start = time.perf_counter()
    for idx, chunk_path in enumerate(chunk_paths, start=1):
        bundle = load_bundle(chunk_path, device=str(device), dtype="float32")
        arrays = bundle.arrays
        for start_idx in range(0, len(selected), layer_block_size):
            block_layers = selected[start_idx : start_idx + layer_block_size]
            _accumulate_direct_kv_error_layers_with_retry(
                arrays,
                block_layers,
                hidden_dim,
                kv_dim,
                device,
                error_stats_by_layer,
                rope_b,
                wk_by_layer,
                wv_by_layer,
                logger,
                native_components_by_layer=native_components_by_layer,
            )
        del arrays, bundle
        clear_torch_cache()
        if idx == 1 or idx % 5 == 0 or idx == len(chunk_paths):
            elapsed = time.perf_counter() - start
            eta_s = (elapsed / idx) * (len(chunk_paths) - idx) if idx > 0 else 0.0
            logger.info("patch 误差统计进度: chunk %d/%d, elapsed=%.1fs, eta=%.1fs", idx, len(chunk_paths), elapsed, eta_s)

    for layer in selected:
        wk = wk_by_layer[layer]
        wv = wv_by_layer[layer]
        patch_k_error, patch_v_error, patch_kv_error = finalize_direct_kv_error_stats_torch(error_stats_by_layer[layer], kv_dim=kv_dim)
        out_arrays[f"direct_wk.{layer}"] = wk.detach().to(device="cpu", dtype=torch.float32).contiguous()
        out_arrays[f"direct_wv.{layer}"] = wv.detach().to(device="cpu", dtype=torch.float32).contiguous()
        logger.info(
            "已完成补丁层训练: layer=%d patch_k_error=%.6f patch_v_error=%.6f patch_kv_error=%.6f",
            layer,
            patch_k_error,
            patch_v_error,
            patch_kv_error,
        )

    save_patch_ckpt(
        args.out,
        out_arrays,
        {
            "selected_layers": selected,
            "hidden_dim": hidden_dim,
            "code_dim": hidden_dim,
            "num_key_value_heads": int(meta["num_key_value_heads"]),
            "head_dim": int(meta["head_dim"]),
            "k_space": "consumer_prenorm_k_native_v" if use_consumer_native_targets else "raw_derope",
            "rope": {"model_b": rope_b.to_dict()},
            "fit_mode": fit_mode,
            "route_layers": [] if route is None else [int(layer) for layer in route.get("patch_layers", [])],
        },
    )
    logger.info("补丁检查点已保存到 %s", args.out)


def _build_semantic_probe(patch_cfg: dict[str, Any], hidden_dim: int, device: torch.device) -> SemanticProbe:
    semantic_source = str(patch_cfg.get("semantic_source") or "probe")
    if semantic_source == "lm_head":
        raise NotImplementedError("semantic_source=lm_head 在 Step A 中保留为实验分支，当前默认仅实现 probe")
    if semantic_source != "probe":
        raise ValueError(f"未知 semantic_source: {semantic_source}")
    probe = SemanticProbe(
        hidden_dim=hidden_dim,
        probe_dim=int(patch_cfg.get("semantic_probe_dim") or hidden_dim),
        num_classes=int(patch_cfg.get("semantic_probe_num_classes") or hidden_dim),
    ).to(device=device, dtype=torch.float32)
    if bool(patch_cfg.get("semantic_probe_frozen", True)) and not bool(patch_cfg.get("semantic_probe_pretrain", False)):
        freeze_module(probe)
    return probe


def _apply_projection(hidden_states: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
    if hidden_states.ndim != 2:
        raise ValueError(f"辅助投影输入必须是 [N,H]，实际为 {tuple(hidden_states.shape)}")
    return torch.nn.functional.linear(hidden_states, weight, bias)


def _prepare_semantic_layer_tensors(
    arrays: dict[str, torch.Tensor],
    layer: int,
    hidden_dim: int,
    kv_dim: int,
    device: torch.device,
    storage_device: torch.device | None,
    rope_b,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    target_device = storage_device or device
    xa = arrays[f"xnorm_a.{layer}"].reshape(-1, hidden_dim).to(device=target_device, dtype=torch.float32)
    xb = arrays[f"xnorm_b.{layer}"].reshape(-1, hidden_dim).to(device=target_device, dtype=torch.float32)
    resid_b = arrays.get(f"resid_b.{layer}")
    if resid_b is not None:
        resid_b = resid_b.reshape(-1, hidden_dim).to(device=target_device, dtype=torch.float32)
    k_raw_b = (
        de_rope_k_torch(arrays[f"k_b.{layer}"], rope_b)
        .permute(0, 2, 1, 3)
        .reshape(-1, kv_dim)
        .to(device=target_device, dtype=torch.float32)
    )
    v_b = arrays[f"v_b.{layer}"].permute(0, 2, 1, 3).reshape(-1, kv_dim).to(device=target_device, dtype=torch.float32)
    if xa.shape != xb.shape:
        raise ValueError(f"xnorm_a/xnorm_b shape mismatch: {tuple(xa.shape)} vs {tuple(xb.shape)}")
    if resid_b is not None and tuple(resid_b.shape) != tuple(xb.shape):
        raise ValueError(f"resid_b shape mismatch: {tuple(resid_b.shape)} vs {tuple(xb.shape)}")
    if xa.shape[0] != k_raw_b.shape[0] or xa.shape[0] != v_b.shape[0]:
        raise ValueError("xnorm and KV row counts must match")
    return xa, xb, k_raw_b, v_b, resid_b


def _state_target_space(patch_cfg: dict[str, Any]) -> str:
    return str(patch_cfg.get("state_target_space") or "residual_pre_ln")


def _preload_semantic_layer_dataset(
    *,
    chunk_paths: list[Path],
    layer: int,
    hidden_dim: int,
    kv_dim: int,
    device: torch.device,
    storage_device: torch.device | None,
    rope_b,
    logger=None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    xa_parts: list[torch.Tensor] = []
    xb_parts: list[torch.Tensor] = []
    k_parts: list[torch.Tensor] = []
    v_parts: list[torch.Tensor] = []
    total_rows = 0

    for idx, chunk_path in enumerate(chunk_paths, start=1):
        load_device = storage_device or device
        bundle = load_bundle(chunk_path, device=str(load_device), dtype="float32")
        arrays = bundle.arrays
        xa, xb, k_raw_b, v_b, resid_b = _prepare_semantic_layer_tensors(arrays, layer, hidden_dim, kv_dim, device, storage_device, rope_b)
        target_b = resid_b if resid_b is not None else xb
        xa_parts.append(xa.contiguous())
        xb_parts.append(target_b.contiguous())
        k_parts.append(k_raw_b.contiguous())
        v_parts.append(v_b.contiguous())
        total_rows += int(xa.shape[0])
        del arrays, bundle
        clear_torch_cache()
        if logger is not None and (idx == 1 or idx % 5 == 0 or idx == len(chunk_paths)):
            logger.info("semantic patch preload: layer=%d chunk=%d/%d rows=%d", layer, idx, len(chunk_paths), total_rows)

    if not xa_parts:
        raise ValueError("semantic patch training set is empty")

    return (
        torch.cat(xa_parts, dim=0),
        torch.cat(xb_parts, dim=0),
        torch.cat(k_parts, dim=0),
        torch.cat(v_parts, dim=0),
    )


def _preload_semantic_datasets(
    *,
    chunk_paths: list[Path],
    layers: list[int],
    hidden_dim: int,
    kv_dim: int,
    device: torch.device,
    storage_device: torch.device | None,
    rope_b,
    logger=None,
) -> tuple[dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]], torch.Tensor]:
    parts: dict[int, dict[str, list[torch.Tensor] | int]] = {
        int(layer): {"xa": [], "xb": [], "k": [], "v": [], "rows": 0}
        for layer in layers
    }
    next_token_parts: list[torch.Tensor] = []

    for idx, chunk_path in enumerate(chunk_paths, start=1):
        load_device = storage_device or device
        bundle = load_bundle(chunk_path, device=str(load_device), dtype="float32")
        arrays = bundle.arrays
        if "next_token_ids" not in arrays:
            raise ValueError("semantic probe pretrain requires pairs to include next_token_ids; recapture pairs with the updated format")
        next_token_parts.append(arrays["next_token_ids"].reshape(-1).to(device=load_device, dtype=torch.long).contiguous())
        for layer in layers:
            xa, xb, k_raw_b, v_b, resid_b = _prepare_semantic_layer_tensors(
                arrays,
                int(layer),
                hidden_dim,
                kv_dim,
                device,
                storage_device,
                rope_b,
            )
            target_b = resid_b if resid_b is not None else xb
            parts[int(layer)]["xa"].append(xa.contiguous())  # type: ignore[index]
            parts[int(layer)]["xb"].append(target_b.contiguous())  # type: ignore[index]
            parts[int(layer)]["k"].append(k_raw_b.contiguous())  # type: ignore[index]
            parts[int(layer)]["v"].append(v_b.contiguous())  # type: ignore[index]
            parts[int(layer)]["rows"] = int(parts[int(layer)]["rows"]) + int(xa.shape[0])  # type: ignore[index]
        del arrays, bundle
        clear_torch_cache()
        if logger is not None and (idx == 1 or idx % 5 == 0 or idx == len(chunk_paths)):
            logger.info("semantic patch preload(all-layers): chunk=%d/%d", idx, len(chunk_paths))

    out: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    for layer in layers:
        row = parts[int(layer)]
        xa_parts = row["xa"]  # type: ignore[assignment]
        if not xa_parts:
            raise ValueError(f"semantic patch training set is empty for layer={layer}")
        out[int(layer)] = (
            torch.cat(row["xa"], dim=0),  # type: ignore[arg-type]
            torch.cat(row["xb"], dim=0),  # type: ignore[arg-type]
            torch.cat(row["k"], dim=0),  # type: ignore[arg-type]
            torch.cat(row["v"], dim=0),  # type: ignore[arg-type]
        )
    return out, torch.cat(next_token_parts, dim=0)


def _iter_row_batches(num_rows: int, batch_size: int):
    for start in range(0, num_rows, batch_size):
        stop = min(start + batch_size, num_rows)
        yield start, stop


def _pretrain_semantic_probe(
    *,
    semantic_probe: SemanticProbe,
    datasets: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    next_token_ids: torch.Tensor,
    patch_cfg: dict[str, Any],
    meta: dict[str, Any],
    device: torch.device,
    logger,
) -> tuple[dict[str, float], torch.nn.Module]:
    if next_token_ids.ndim != 1:
        raise ValueError(f"next_token_ids must be flattened [N], got {tuple(next_token_ids.shape)}")
    if not datasets:
        raise ValueError("probe pretrain datasets are empty")

    probe_epochs = max(1, int(patch_cfg.get("semantic_probe_pretrain_epochs") or 1))
    probe_batch_size = max(1, int(patch_cfg.get("semantic_probe_pretrain_batch_size") or patch_cfg.get("patch_batch_size") or 64))
    probe_lr = float(patch_cfg.get("semantic_probe_pretrain_lr") or patch_cfg.get("lr") or 5e-4)
    probe_weight_decay = float(patch_cfg.get("semantic_probe_pretrain_weight_decay") or 0.0)
    vocab_size = int(meta["vocab_size"])
    classifier = torch.nn.Linear(int(semantic_probe.fc2.out_features), vocab_size, bias=True).to(device=device, dtype=torch.float32)
    optimizer = torch.optim.AdamW(list(semantic_probe.parameters()) + list(classifier.parameters()), lr=probe_lr, weight_decay=probe_weight_decay)

    xb_all = torch.cat([row[1] for row in datasets.values()], dim=0)
    labels_all = next_token_ids.repeat(len(datasets))
    valid_mask = labels_all.ge(0)
    xb_all = xb_all[valid_mask]
    labels_all = labels_all[valid_mask]
    if int(labels_all.numel()) <= 0:
        raise ValueError("probe pretrain found no valid next-token labels")

    metrics = {"probe_pretrain_loss": 0.0, "probe_pretrain_acc": 0.0}
    for epoch in range(1, probe_epochs + 1):
        semantic_probe.train()
        classifier.train()
        running_loss = 0.0
        running_correct = 0.0
        running_rows = 0
        for start, stop in _iter_row_batches(int(labels_all.shape[0]), probe_batch_size):
            batch_xb = xb_all[start:stop].to(device=device, dtype=torch.float32, non_blocking=True)
            batch_labels = labels_all[start:stop].to(device=device, dtype=torch.long, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            semantic_logits = semantic_probe(batch_xb)
            vocab_logits = classifier(semantic_logits)
            loss = F.cross_entropy(vocab_logits, batch_labels)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * int(batch_labels.shape[0])
            running_correct += float((torch.argmax(vocab_logits.detach(), dim=-1) == batch_labels).to(dtype=torch.float32).sum().item())
            running_rows += int(batch_labels.shape[0])
        denom = max(1, running_rows)
        metrics = {
            "probe_pretrain_loss": running_loss / denom,
            "probe_pretrain_acc": running_correct / denom,
        }
        logger.info(
            "semantic probe pretrain: epoch=%d/%d loss=%.6f acc=%.4f rows=%d",
            epoch,
            probe_epochs,
            metrics["probe_pretrain_loss"],
            metrics["probe_pretrain_acc"],
            running_rows,
        )
    return metrics, classifier


def _collect_optimizer_params(layer_module: XNormSemanticPatchLayer, semantic_probe: SemanticProbe | None) -> list[torch.nn.Parameter]:
    params = list(layer_module.parameters())
    if semantic_probe is not None:
        params.extend([param for param in semantic_probe.parameters() if param.requires_grad])
    return params


def _train_semantic_patch_layer(
    *,
    layer: int,
    xa: torch.Tensor,
    xb: torch.Tensor,
    k_raw_b: torch.Tensor,
    v_b: torch.Tensor,
    hidden_dim: int,
    kv_dim: int,
    device: torch.device,
    patch_cfg: dict[str, Any],
    native_components: ConsumerNativeComponents,
    semantic_probe: SemanticProbe,
    semantic_vocab_head: torch.nn.Module | None,
    logger,
) -> tuple[dict[str, Any], dict[str, float]]:
    code_dim = int(patch_cfg.get("code_dim") or hidden_dim)
    mlp_hidden_dim = int(patch_cfg.get("patch_mlp_hidden_dim") or hidden_dim)
    patch_num_epochs = int(patch_cfg.get("patch_num_epochs") or 1)
    patch_batch_size = max(1, int(patch_cfg.get("patch_batch_size") or 32))
    lr = float(patch_cfg.get("lr") or 5e-4)
    weight_decay = float(patch_cfg.get("weight_decay") or 0.0)
    tau = float(patch_cfg.get("tau") or 1.0)
    semantic_source = str(patch_cfg.get("semantic_source") or "probe")
    semantic_loss_direction = str(patch_cfg.get("semantic_loss_direction") or "teacher_to_student")
    enc_h_is_identity = bool(patch_cfg.get("enc_h_identity_if_possible", True)) and code_dim == hidden_dim

    lambda_h = float(patch_cfg.get("lambda_h", 1.0))
    lambda_k = float(patch_cfg.get("lambda_k", 0.5))
    lambda_v = float(patch_cfg.get("lambda_v", 0.5))
    lambda_sem = float(patch_cfg.get("lambda_sem", 1.0))

    layer_module = XNormSemanticPatchLayer(
        hidden_dim=hidden_dim,
        code_dim=code_dim,
        mlp_hidden_dim=mlp_hidden_dim,
        enc_h_is_identity=enc_h_is_identity,
    ).to(device=device, dtype=torch.float32)

    optimizer = torch.optim.AdamW(_collect_optimizer_params(layer_module, semantic_probe), lr=lr, weight_decay=weight_decay)
    final_metrics: dict[str, float] = {}
    num_rows = int(xa.shape[0])

    for epoch in range(1, patch_num_epochs + 1):
        layer_module.train()
        if any(param.requires_grad for param in semantic_probe.parameters()):
            semantic_probe.train()
        else:
            semantic_probe.eval()

        running = {
            "rows": 0.0,
            "hidden_mse": 0.0,
            "k_mse": 0.0,
            "v_mse": 0.0,
            "sem_kl": 0.0,
            "total_loss": 0.0,
            "topk_match": 0.0,
        }
        for start, stop in _iter_row_batches(num_rows, patch_batch_size):
            batch_xa = xa[start:stop].to(device=device, dtype=torch.float32, non_blocking=True)
            batch_xb = xb[start:stop].to(device=device, dtype=torch.float32, non_blocking=True)
            batch_k = k_raw_b[start:stop].to(device=device, dtype=torch.float32, non_blocking=True)
            batch_v = v_b[start:stop].to(device=device, dtype=torch.float32, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            xnorm_b_hat = layer_module(batch_xa)
            k_raw_hat = _apply_projection(xnorm_b_hat, native_components.k_proj_weight, native_components.k_proj_bias)
            v_hat = _apply_projection(xnorm_b_hat, native_components.v_proj_weight, native_components.v_proj_bias)

            if semantic_source != "probe":
                raise NotImplementedError("Step A semantic path currently only supports semantic_source=probe")
            with torch.no_grad():
                teacher_semantic = semantic_probe(batch_xb).detach()
            student_semantic = semantic_probe(xnorm_b_hat)
            teacher_logits = teacher_semantic
            student_logits = student_semantic
            if semantic_vocab_head is not None and str(patch_cfg.get("semantic_target_space") or "probe_logits") == "next_token_logits":
                with torch.no_grad():
                    teacher_logits = semantic_vocab_head(teacher_semantic).detach()
                student_logits = semantic_vocab_head(student_semantic)

            loss_hidden = hidden_loss(xnorm_b_hat, batch_xb)
            loss_k = aux_k_loss(k_raw_hat, batch_k)
            loss_v = aux_v_loss(v_hat, batch_v)
            loss_sem = semantic_kl_loss(student_logits, teacher_logits, tau=tau, direction=semantic_loss_direction)
            total_loss = (lambda_h * loss_hidden) + (lambda_k * loss_k) + (lambda_v * loss_v) + (lambda_sem * loss_sem)
            total_loss.backward()
            optimizer.step()

            batch_rows = int(batch_xa.shape[0])
            running["rows"] += float(batch_rows)
            running["hidden_mse"] += float(loss_hidden.item()) * batch_rows
            running["k_mse"] += float(loss_k.item()) * batch_rows
            running["v_mse"] += float(loss_v.item()) * batch_rows
            running["sem_kl"] += float(loss_sem.item()) * batch_rows
            running["total_loss"] += float(total_loss.item()) * batch_rows
            running["topk_match"] += topk_match_at_k(student_logits.detach(), teacher_logits.detach(), k=min(3, int(teacher_logits.shape[-1]))) * batch_rows

        denom = max(1.0, running["rows"])
        final_metrics = {
            "hidden_mse": running["hidden_mse"] / denom,
            "k_mse": running["k_mse"] / denom,
            "v_mse": running["v_mse"] / denom,
            "sem_kl": running["sem_kl"] / denom,
            "total_loss": running["total_loss"] / denom,
            "topk_match_at_3": running["topk_match"] / denom,
        }
        logger.info(
            "semantic patch epoch: layer=%d epoch=%d/%d hidden_loss=%.6f k_loss=%.6f v_loss=%.6f semantic_loss=%.6f total_loss=%.6f topk_match_at_3=%.4f",
            layer,
            epoch,
            patch_num_epochs,
            final_metrics["hidden_mse"],
            final_metrics["k_mse"],
            final_metrics["v_mse"],
            final_metrics["sem_kl"],
            final_metrics["total_loss"],
            final_metrics["topk_match_at_3"],
        )

    return pack_xnorm_semantic_patch_layer(layer, layer_module), final_metrics


def _train_semantic_patch(
    *,
    args,
    logger,
    cfg: dict[str, Any],
    meta: dict[str, Any],
    profile: dict[str, Any],
    route: dict[str, Any] | None,
    rope_b,
    device: torch.device,
    accel: dict[str, Any],
) -> None:
    patch_cfg = cfg.get("patch", {})
    patch_budget = int(patch_cfg["patch_budget"])
    selected = _select_layers(profile, patch_budget, route=route)
    hidden_dim = int(meta["hidden_dim"])
    kv_dim = int(meta["num_key_value_heads"]) * int(meta["head_dim"])
    semantic_source = str(patch_cfg.get("semantic_source") or "probe")
    semantic_probe_shared = bool(patch_cfg.get("semantic_probe_shared", True))
    semantic_probe_frozen = bool(patch_cfg.get("semantic_probe_frozen", True))
    semantic_probe_pretrain = bool(patch_cfg.get("semantic_probe_pretrain", False))

    if semantic_probe_pretrain and not semantic_probe_shared:
        raise NotImplementedError("semantic_probe_pretrain currently requires semantic_probe_shared=true")

    logger.info(
        "patch training start: fit_mode=%s device=%s accel=%s selected=%s semantic_source=%s semantic_probe_shared=%s semantic_probe_frozen=%s",
        PATCH_FIT_MODE_XNORM_SEMANTIC_V1,
        device,
        accel["accel_mode"],
        selected,
        semantic_source,
        semantic_probe_shared,
        semantic_probe_frozen,
    )

    native_components = _load_consumer_native_components(
        args.models_config,
        selected_layers=selected,
        hidden_dim=hidden_dim,
        kv_dim=kv_dim,
        device=device,
        dtype="float32",
    )

    chunk_paths = list(_iter_chunk_paths(args.pairs))
    storage_device = torch.device("cpu") if device.type == "cuda" else device
    preloaded, next_token_ids = _preload_semantic_datasets(
        chunk_paths=chunk_paths,
        layers=selected,
        hidden_dim=hidden_dim,
        kv_dim=kv_dim,
        device=device,
        storage_device=storage_device,
        rope_b=rope_b,
        logger=logger,
    )
    shared_probe = _build_semantic_probe(patch_cfg, hidden_dim=hidden_dim, device=device)
    probe_pretrain_metrics: dict[str, float] = {}
    semantic_vocab_head: torch.nn.Module | None = None
    if semantic_probe_pretrain:
        probe_pretrain_metrics, semantic_vocab_head = _pretrain_semantic_probe(
            semantic_probe=shared_probe,
            datasets=preloaded,
            next_token_ids=next_token_ids,
            patch_cfg=patch_cfg,
            meta=meta,
            device=device,
            logger=logger,
        )
    if semantic_probe_frozen:
        freeze_module(shared_probe)
    out_arrays: dict[str, Any] = {}
    layer_metrics: dict[str, dict[str, float]] = {}

    for layer in selected:
        probe = shared_probe if semantic_probe_shared else _build_semantic_probe(patch_cfg, hidden_dim=hidden_dim, device=device)
        xa, xb, k_raw_b, v_b = preloaded[int(layer)]
        layer_arrays, metrics = _train_semantic_patch_layer(
            layer=layer,
            xa=xa,
            xb=xb,
            k_raw_b=k_raw_b,
            v_b=v_b,
            hidden_dim=hidden_dim,
            kv_dim=kv_dim,
            device=device,
            patch_cfg=patch_cfg,
            native_components=_coerce_consumer_native_components(native_components[layer]),
            semantic_probe=probe,
            semantic_vocab_head=semantic_vocab_head,
            logger=logger,
        )
        del xa, xb, k_raw_b, v_b
        clear_torch_cache()
        out_arrays.update(layer_arrays)
        if semantic_source == "probe" and not semantic_probe_shared:
            out_arrays.update(pack_module_state(f"semantic_probe.{layer}", probe))
        layer_row = dict(metrics)
        layer_row.update(probe_pretrain_metrics)
        layer_metrics[str(layer)] = layer_row
        logger.info(
            "completed semantic patch layer: layer=%d hidden_mse=%.6f k_mse=%.6f v_mse=%.6f sem_kl=%.6f total_loss=%.6f",
            layer,
            metrics["hidden_mse"],
            metrics["k_mse"],
            metrics["v_mse"],
            metrics["sem_kl"],
            metrics["total_loss"],
        )

    if semantic_source == "probe" and semantic_probe_shared:
        out_arrays.update(pack_module_state("semantic_probe", shared_probe))
    if semantic_vocab_head is not None:
        out_arrays.update(pack_module_state("semantic_vocab_head", semantic_vocab_head))

    code_dim = int(patch_cfg.get("code_dim") or hidden_dim)
    enc_h_is_identity = bool(patch_cfg.get("enc_h_identity_if_possible", True)) and code_dim == hidden_dim
    meta_payload = {
        "fit_mode": PATCH_FIT_MODE_XNORM_SEMANTIC_V1,
        "selected_layers": selected,
        "hidden_dim": hidden_dim,
        "code_dim": code_dim,
        "mlp_hidden_dim": int(patch_cfg.get("patch_mlp_hidden_dim") or hidden_dim),
        "num_key_value_heads": int(meta["num_key_value_heads"]),
        "head_dim": int(meta["head_dim"]),
        "k_space": "raw_derope",
        "semantic_source": semantic_source,
        "semantic_probe_shared": semantic_probe_shared,
        "semantic_probe_frozen": semantic_probe_frozen,
        "semantic_probe_pretrained": semantic_probe_pretrain,
        "semantic_probe_arch": "mlp_gelu_v1",
        "semantic_probe_version": "v1",
        "semantic_probe_dim": int(patch_cfg.get("semantic_probe_dim") or hidden_dim),
        "semantic_probe_num_classes": int(patch_cfg.get("semantic_probe_num_classes") or hidden_dim),
        "state_target_space": _state_target_space(patch_cfg),
        "semantic_target_space": str(patch_cfg.get("semantic_target_space") or ("next_token_logits" if semantic_vocab_head is not None else "probe_logits")),
        "vocab_size": int(meta["vocab_size"]),
        "patch_apply_mode": str(patch_cfg.get("patch_apply_mode") or "residual_blend"),
        "patch_blend_alpha_k": float(patch_cfg.get("patch_blend_alpha_k", 0.25)),
        "patch_blend_alpha_v": float(patch_cfg.get("patch_blend_alpha_v", 0.5)),
        "loss_weights": {
            "lambda_h": float(patch_cfg.get("lambda_h", 1.0)),
            "lambda_k": float(patch_cfg.get("lambda_k", 0.5)),
            "lambda_v": float(patch_cfg.get("lambda_v", 0.5)),
            "lambda_sem": float(patch_cfg.get("lambda_sem", 1.0)),
        },
        "tau": float(patch_cfg.get("tau", 1.0)),
        "route_layers": [] if route is None else [int(layer) for layer in route.get("patch_layers", [])],
        "enc_h_is_identity": enc_h_is_identity,
        "rope": {"model_b": rope_b.to_dict()},
    }
    save_patch_ckpt(args.out, out_arrays, meta_payload)
    save_json(Path(args.out) / "metrics.json", {"fit_mode": PATCH_FIT_MODE_XNORM_SEMANTIC_V1, "layers": layer_metrics})
    logger.info("semantic patch checkpoint saved to %s", args.out)


def main() -> None:
    args = build_parser().parse_args()
    logger = resolve_logger(args.log)
    set_global_seed(args.seed)

    meta = load_pairs_meta(args.pairs)
    profile = load_json(args.profile)
    route = load_json(args.route) if args.route else None
    cfg = load_cfg(args.train_config)
    _rope_a, rope_b = resolve_rope_specs(meta, args.models_config or None)

    accel = resolve_accel_options(cfg, args)
    configure_torch_acceleration(str(accel["accel_mode"]))

    compute_cfg = cfg.get("compute", {})
    runtime_cfg = cfg.get("runtime", {})
    device = resolve_device(args.train_device or compute_cfg.get("train_device") or "cuda:0", cpu_fallback=bool(runtime_cfg.get("cpu_fallback", True)))

    patch_cfg = cfg.get("patch", {})
    fit_mode = str(patch_cfg.get("fit_mode") or PATCH_FIT_MODE_DIRECT_KV_V1)
    if fit_mode in {PATCH_FIT_MODE_DIRECT_KV_V1, PATCH_FIT_MODE_DIRECT_KV_KNORM_V1}:
        _train_direct_kv_patch(
            args=args,
            logger=logger,
            cfg=cfg,
            meta=meta,
            profile=profile,
            route=route,
            rope_b=rope_b,
            device=device,
            accel=accel,
            fit_mode=fit_mode,
        )
        return
    if fit_mode == PATCH_FIT_MODE_XNORM_SEMANTIC_V1:
        _train_semantic_patch(
            args=args,
            logger=logger,
            cfg=cfg,
            meta=meta,
            profile=profile,
            route=route,
            rope_b=rope_b,
            device=device,
            accel=accel,
        )
        return
    if fit_mode == PATCH_FIT_MODE_DIRECT_KV_KNORM_RESIDUAL_MLP_V1:
        train_direct_kv_knorm_residual_patch(
            args=args,
            logger=logger,
            cfg=cfg,
            meta=meta,
            profile=profile,
            route=route,
            rope_b=rope_b,
            device=device,
            accel=accel,
        )
        return
    if fit_mode == PATCH_FIT_MODE_XNORM_NATIVEKV_RESIDUAL_MLP_V1:
        train_xnorm_nativekv_residual_patch(
            args=args,
            logger=logger,
            cfg=cfg,
            meta=meta,
            profile=profile,
            route=route,
            rope_b=rope_b,
            device=device,
            accel=accel,
        )
        return
    if fit_mode == PATCH_FIT_MODE_XNORM_NATIVEKV_SHARED_RESIDUAL_MLP_V1:
        train_xnorm_nativekv_shared_residual_patch(
            args=args,
            logger=logger,
            cfg=cfg,
            meta=meta,
            profile=profile,
            route=route,
            rope_b=rope_b,
            device=device,
            accel=accel,
        )
        return
    raise ValueError(f"未知 patch fit_mode: {fit_mode}")


if __name__ == "__main__":
    main()
