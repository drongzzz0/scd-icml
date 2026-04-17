from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from ..io_utils import load_bundle, save_json
from ..patch import (
    PATCH_FIT_MODE_DIRECT_KV_V1,
    accumulate_patch_align_stats_torch,
    accumulate_direct_kv_error_stats_torch,
    accumulate_direct_kv_stats_torch,
    create_patch_align_stats_torch,
    create_direct_kv_error_stats_torch,
    create_direct_kv_stats_torch,
    finalize_direct_kv_error_stats_torch,
    solve_patch_align_from_stats_torch,
    solve_direct_kv_from_stats_torch,
)
from ..pipeline import load_pairs_meta, load_reuse_weights
from ..reuse import decode_group, encode_group, group_layers_for_gid, move_reuse_weights, pack_group, unpack_group
from ..rope import de_rope_k_torch, re_rope_k_torch, resolve_rope_specs
from ..routing import compute_layer_profiles, expected_patch_gain_curve
from ..torch_utils import clear_torch_cache, configure_torch_acceleration, resolve_device
from .common import add_accel_args, resolve_accel_options, resolve_logger


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="分析层敏感度与补丁收益")
    p.add_argument("--pairs", required=True)
    p.add_argument("--reuse-ckpt", required=True)
    p.add_argument("--models-config", default="")
    p.add_argument("--out", required=True)
    p.add_argument("--log", default="")
    p.add_argument("--train-device", default="")
    add_accel_args(p)
    return p


def _iter_chunk_paths(pairs_dir: str):
    root = Path(pairs_dir)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    for row in manifest.get("chunks", []):
        yield root / str(row["path"])


def _mse_from_sse(sse: float, count: float) -> float:
    if count <= 0:
        return 0.0
    return float(sse / count)


def _align_tensor(x: torch.Tensor, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if x.device != device or x.dtype != dtype:
        return x.to(device=device, dtype=dtype)
    return x


def _accumulate_patch_align_block(arrays: dict[str, torch.Tensor], layers: list[int], stats_by_layer: dict[int, dict[str, torch.Tensor]], hidden_dim: int, device: torch.device) -> None:
    for layer in layers:
        xa = arrays[f"xnorm_a.{layer}"].reshape(-1, hidden_dim)
        xb = arrays[f"xnorm_b.{layer}"].reshape(-1, hidden_dim)
        xa = _align_tensor(xa, device=device, dtype=torch.float32)
        xb = _align_tensor(xb, device=device, dtype=torch.float32)
        accumulate_patch_align_stats_torch(xa, xb, stats_by_layer[layer])


def _accumulate_patch_align_block_with_retry(
    arrays: dict[str, torch.Tensor],
    layers: list[int],
    stats_by_layer: dict[int, dict[str, torch.Tensor]],
    hidden_dim: int,
    device: torch.device,
    logger,
    depth: int = 0,
) -> None:
    try:
        _accumulate_patch_align_block(arrays, layers, stats_by_layer, hidden_dim, device)
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
        logger.warning("profile patch 统计触发 OOM，按层块二分重试 depth=%d size=%d -> %d + %d", depth, len(layers), len(left), len(right))
        _accumulate_patch_align_block_with_retry(arrays, left, stats_by_layer, hidden_dim, device, logger, depth + 1)
        _accumulate_patch_align_block_with_retry(arrays, right, stats_by_layer, hidden_dim, device, logger, depth + 1)


def _accumulate_patch_kv_block(
    arrays: dict[str, torch.Tensor],
    layers: list[int],
    stats_by_layer: dict[int, dict[str, torch.Tensor]],
    hidden_dim: int,
    kv_dim: int,
    device: torch.device,
    rope_b,
) -> None:
    for layer in layers:
        xa = arrays[f"xnorm_a.{layer}"].reshape(-1, hidden_dim)
        xa = _align_tensor(xa, device=device, dtype=torch.float32)
        kb_raw = de_rope_k_torch(arrays[f"k_b.{layer}"], rope_b).permute(0, 2, 1, 3).reshape(-1, kv_dim)
        vb = arrays[f"v_b.{layer}"].permute(0, 2, 1, 3).reshape(-1, kv_dim)
        kb_raw = _align_tensor(kb_raw, device=device, dtype=torch.float32)
        vb = _align_tensor(vb, device=device, dtype=torch.float32)
        accumulate_direct_kv_stats_torch(xa, kb_raw, vb, stats_by_layer[layer])


def _accumulate_patch_kv_block_with_retry(
    arrays: dict[str, torch.Tensor],
    layers: list[int],
    stats_by_layer: dict[int, dict[str, torch.Tensor]],
    hidden_dim: int,
    kv_dim: int,
    device: torch.device,
    rope_b,
    logger,
    depth: int = 0,
) -> None:
    try:
        _accumulate_patch_kv_block(arrays, layers, stats_by_layer, hidden_dim, kv_dim, device, rope_b)
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
        logger.warning("profile patch KV 统计触发 OOM，按层块二分重试 depth=%d size=%d -> %d + %d", depth, len(layers), len(left), len(right))
        _accumulate_patch_kv_block_with_retry(arrays, left, stats_by_layer, hidden_dim, kv_dim, device, rope_b, logger, depth + 1)
        _accumulate_patch_kv_block_with_retry(arrays, right, stats_by_layer, hidden_dim, kv_dim, device, rope_b, logger, depth + 1)


def _accumulate_patch_error_block(
    arrays: dict[str, torch.Tensor],
    layers: list[int],
    stats_by_layer: dict[int, dict[str, torch.Tensor]],
    hidden_dim: int,
    kv_dim: int,
    device: torch.device,
    rope_b,
    wk_by_layer: dict[int, torch.Tensor],
    wv_by_layer: dict[int, torch.Tensor],
) -> None:
    for layer in layers:
        xa = arrays[f"xnorm_a.{layer}"].reshape(-1, hidden_dim)
        xa = _align_tensor(xa, device=device, dtype=torch.float32)
        kb_raw = de_rope_k_torch(arrays[f"k_b.{layer}"], rope_b).permute(0, 2, 1, 3).reshape(-1, kv_dim)
        vb = arrays[f"v_b.{layer}"].permute(0, 2, 1, 3).reshape(-1, kv_dim)
        kb_raw = _align_tensor(kb_raw, device=device, dtype=torch.float32)
        vb = _align_tensor(vb, device=device, dtype=torch.float32)
        accumulate_direct_kv_error_stats_torch(xa, kb_raw, vb, wk_by_layer[layer], wv_by_layer[layer], stats_by_layer[layer])


def _accumulate_patch_error_block_with_retry(
    arrays: dict[str, torch.Tensor],
    layers: list[int],
    stats_by_layer: dict[int, dict[str, torch.Tensor]],
    hidden_dim: int,
    kv_dim: int,
    device: torch.device,
    rope_b,
    wk_by_layer: dict[int, torch.Tensor],
    wv_by_layer: dict[int, torch.Tensor],
    logger,
    depth: int = 0,
) -> None:
    try:
        _accumulate_patch_error_block(arrays, layers, stats_by_layer, hidden_dim, kv_dim, device, rope_b, wk_by_layer, wv_by_layer)
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
        logger.warning("profile patch 误差统计触发 OOM，按层块二分重试 depth=%d size=%d -> %d + %d", depth, len(layers), len(left), len(right))
        _accumulate_patch_error_block_with_retry(arrays, left, stats_by_layer, hidden_dim, kv_dim, device, rope_b, wk_by_layer, wv_by_layer, logger, depth + 1)
        _accumulate_patch_error_block_with_retry(arrays, right, stats_by_layer, hidden_dim, kv_dim, device, rope_b, wk_by_layer, wv_by_layer, logger, depth + 1)


def _accumulate_reuse_groups_with_retry(
    arrays: dict[str, torch.Tensor],
    gids: list[int],
    reuse,
    rope_a,
    rope_b,
    device: torch.device,
    hd: int,
    bsz: int,
    kvh: int,
    t_steps: int,
    reuse_k_sse: torch.Tensor,
    reuse_v_sse: torch.Tensor,
    reuse_elem: torch.Tensor,
    logger,
    depth: int = 0,
) -> None:
    try:
        for gid in gids:
            layers = group_layers_for_gid(reuse, gid)
            a_k = pack_group([de_rope_k_torch(arrays[f"k_a.{layer}"], rope_a).to(dtype=torch.float32) for layer in layers])
            a_v = pack_group([arrays[f"v_a.{layer}"].to(dtype=torch.float32) for layer in layers])
            z_k, z_v = encode_group(a_k, a_v, reuse.weights[gid])
            k_hat_raw, v_hat = decode_group(z_k, z_v, reuse.weights[gid])
            k_layers_raw = unpack_group(k_hat_raw, len(layers), hd, bsz, kvh, t_steps)
            v_layers = unpack_group(v_hat, len(layers), hd, bsz, kvh, t_steps)
            for layer, rk_raw, rv in zip(layers, k_layers_raw, v_layers):
                rk = re_rope_k_torch(rk_raw, rope_b)
                rk = _align_tensor(rk, device=device, dtype=torch.float32)
                rv = _align_tensor(rv, device=device, dtype=torch.float32)
                kb = _align_tensor(arrays[f"k_b.{layer}"], device=device, dtype=rk.dtype)
                vb = _align_tensor(arrays[f"v_b.{layer}"], device=device, dtype=rv.dtype)
                reuse_k_sse[layer] += torch.sum((rk - kb) ** 2, dtype=torch.float64)
                reuse_v_sse[layer] += torch.sum((rv - vb) ** 2, dtype=torch.float64)
                reuse_elem[layer] += float(kb.numel())
    except RuntimeError as exc:
        text = str(exc).lower()
        if "out of memory" not in text and "cuda oom" not in text:
            raise
        clear_torch_cache()
        if len(gids) <= 1:
            raise
        mid = len(gids) // 2
        left = gids[:mid]
        right = gids[mid:]
        logger.warning("profile reuse 统计触发 OOM，按组块二分重试 depth=%d size=%d -> %d + %d", depth, len(gids), len(left), len(right))
        _accumulate_reuse_groups_with_retry(arrays, left, reuse, rope_a, rope_b, device, hd, bsz, kvh, t_steps, reuse_k_sse, reuse_v_sse, reuse_elem, logger, depth + 1)
        _accumulate_reuse_groups_with_retry(arrays, right, reuse, rope_a, rope_b, device, hd, bsz, kvh, t_steps, reuse_k_sse, reuse_v_sse, reuse_elem, logger, depth + 1)


def main() -> None:
    args = build_parser().parse_args()
    logger = resolve_logger(args.log)

    accel = resolve_accel_options({}, args)
    configure_torch_acceleration(str(accel["accel_mode"]))

    meta = load_pairs_meta(args.pairs)
    reuse, reuse_meta = load_reuse_weights(args.reuse_ckpt, device="cpu")
    rope_meta = reuse_meta if isinstance(reuse_meta.get("rope"), dict) else meta
    rope_a, rope_b = resolve_rope_specs(rope_meta, args.models_config or None)
    device = resolve_device(args.train_device or "cuda:0", cpu_fallback=True)
    reuse = move_reuse_weights(reuse, str(device), dtype="float32", cpu_fallback=True)

    num_layers = int(meta["num_layers"])
    hidden_dim = int(meta["hidden_dim"])
    hd = int(meta["head_dim"])
    layer_block_size = int(accel["layer_block_size"])
    group_block_size = int(accel["group_block_size"])

    logger.info(
        "profile 开始: device=%s accel=%s layer_block_size=%d group_block_size=%d num_layers=%d",
        device,
        accel["accel_mode"],
        layer_block_size,
        group_block_size,
        num_layers,
    )

    reuse_k_sse = torch.zeros(num_layers, device=device, dtype=torch.float64)
    reuse_v_sse = torch.zeros(num_layers, device=device, dtype=torch.float64)
    reuse_elem = torch.zeros(num_layers, device=device, dtype=torch.float64)

    align_stats_by_layer: dict[int, dict[str, torch.Tensor]] = {}
    for layer in range(num_layers):
        align_stats_by_layer[layer] = create_patch_align_stats_torch(hidden_dim, device)

    chunk_paths = list(_iter_chunk_paths(args.pairs))
    start_all = time.perf_counter()

    for chunk_idx, chunk_path in enumerate(chunk_paths, start=1):
        bundle = load_bundle(chunk_path, device=str(device), dtype="float32")
        arrays = bundle.arrays
        chunk_meta = bundle.meta
        bsz = int(chunk_meta["batch_size"])
        kvh = int(chunk_meta["num_key_value_heads"])
        t_steps = int(chunk_meta["prefix_tokens"])

        gids = list(range(reuse.num_groups))
        for start in range(0, len(gids), group_block_size):
            block = gids[start : start + group_block_size]
            _accumulate_reuse_groups_with_retry(
                arrays,
                block,
                reuse,
                rope_a,
                rope_b,
                device,
                hd,
                bsz,
                kvh,
                t_steps,
                reuse_k_sse,
                reuse_v_sse,
                reuse_elem,
                logger,
            )

        layers = list(range(num_layers))
        for start in range(0, len(layers), layer_block_size):
            block_layers = layers[start : start + layer_block_size]
            _accumulate_patch_align_block_with_retry(arrays, block_layers, align_stats_by_layer, hidden_dim, device, logger)

        del arrays, bundle
        clear_torch_cache()

        if chunk_idx == 1 or chunk_idx % 5 == 0 or chunk_idx == len(chunk_paths):
            elapsed = time.perf_counter() - start_all
            eta_s = (elapsed / chunk_idx) * (len(chunk_paths) - chunk_idx) if chunk_idx > 0 else 0.0
            logger.info("profile 统计进度: chunk %d/%d, elapsed=%.1fs, eta=%.1fs", chunk_idx, len(chunk_paths), elapsed, eta_s)

    patch_xnorm_errors: list[float] = []
    for layer in range(num_layers):
        _align, patch_xnorm_error = solve_patch_align_from_stats_torch(align_stats_by_layer[layer], ridge_lambda=1e-4)
        patch_xnorm_errors.append(patch_xnorm_error)

    del align_stats_by_layer
    clear_torch_cache()

    kv_dim = int(meta["num_key_value_heads"]) * int(meta["head_dim"])
    kv_stats_by_layer: dict[int, dict[str, torch.Tensor]] = {}
    for layer in range(num_layers):
        kv_stats_by_layer[layer] = create_direct_kv_stats_torch(hidden_dim, kv_dim, device)

    start_patch_kv = time.perf_counter()
    for chunk_idx, chunk_path in enumerate(chunk_paths, start=1):
        bundle = load_bundle(chunk_path, device=str(device), dtype="float32")
        arrays = bundle.arrays

        layers = list(range(num_layers))
        for start in range(0, len(layers), layer_block_size):
            block_layers = layers[start : start + layer_block_size]
            _accumulate_patch_kv_block_with_retry(arrays, block_layers, kv_stats_by_layer, hidden_dim, kv_dim, device, rope_b, logger)

        del arrays, bundle
        clear_torch_cache()

        if chunk_idx == 1 or chunk_idx % 5 == 0 or chunk_idx == len(chunk_paths):
            elapsed = time.perf_counter() - start_patch_kv
            eta_s = (elapsed / chunk_idx) * (len(chunk_paths) - chunk_idx) if chunk_idx > 0 else 0.0
            logger.info("profile patch KV 统计进度: chunk %d/%d, elapsed=%.1fs, eta=%.1fs", chunk_idx, len(chunk_paths), elapsed, eta_s)

    wk_by_layer: dict[int, torch.Tensor] = {}
    wv_by_layer: dict[int, torch.Tensor] = {}
    for layer in range(num_layers):
        wk, wv, _patch_k_formula, _patch_v_formula, _patch_kv_formula = solve_direct_kv_from_stats_torch(kv_stats_by_layer[layer], ridge_lambda=1e-4)
        wk_by_layer[layer] = wk.to(device=device, dtype=torch.float32)
        wv_by_layer[layer] = wv.to(device=device, dtype=torch.float32)

    del kv_stats_by_layer
    clear_torch_cache()

    error_stats_by_layer: dict[int, dict[str, torch.Tensor]] = {}
    for layer in range(num_layers):
        error_stats_by_layer[layer] = create_direct_kv_error_stats_torch(device)

    start_patch_error = time.perf_counter()
    for chunk_idx, chunk_path in enumerate(chunk_paths, start=1):
        bundle = load_bundle(chunk_path, device=str(device), dtype="float32")
        arrays = bundle.arrays

        layers = list(range(num_layers))
        for start in range(0, len(layers), layer_block_size):
            block_layers = layers[start : start + layer_block_size]
            _accumulate_patch_error_block_with_retry(
                arrays,
                block_layers,
                error_stats_by_layer,
                hidden_dim,
                kv_dim,
                device,
                rope_b,
                wk_by_layer,
                wv_by_layer,
                logger,
            )

        del arrays, bundle
        clear_torch_cache()

        if chunk_idx == 1 or chunk_idx % 5 == 0 or chunk_idx == len(chunk_paths):
            elapsed = time.perf_counter() - start_patch_error
            eta_s = (elapsed / chunk_idx) * (len(chunk_paths) - chunk_idx) if chunk_idx > 0 else 0.0
            logger.info("profile patch 误差统计进度: chunk %d/%d, elapsed=%.1fs, eta=%.1fs", chunk_idx, len(chunk_paths), elapsed, eta_s)

    reuse_errors: list[float] = []
    patch_errors: list[float] = []
    per_layer: list[dict[str, float | int]] = []

    for layer in range(num_layers):
        patch_k_error, patch_v_error, pe = finalize_direct_kv_error_stats_torch(
            error_stats_by_layer[layer],
            kv_dim=kv_dim,
        )
        patch_xnorm_error = patch_xnorm_errors[layer]

        re = 0.5 * (
            _mse_from_sse(float(reuse_k_sse[layer].item()), float(reuse_elem[layer].item()))
            + _mse_from_sse(float(reuse_v_sse[layer].item()), float(reuse_elem[layer].item()))
        )
        reuse_errors.append(re)
        patch_errors.append(pe)
        per_layer.append(
            {
                "layer": layer,
                "reuse_error": re,
                "patch_error": pe,
                "patch_xnorm_error": patch_xnorm_error,
                "patch_k_error": patch_k_error,
                "patch_v_error": patch_v_error,
                "patch_kv_error": pe,
                "expected_gain": float(max(0.0, re - pe)),
            }
        )
        logger.info(
            "层分析已完成: %d/%d，reuse_error=%.6f，patch_xnorm_error=%.6f，patch_kv_error=%.6f",
            layer + 1,
            num_layers,
            re,
            patch_xnorm_error,
            pe,
        )

    profiles = compute_layer_profiles(reuse_errors, patch_errors)
    gain_curve = expected_patch_gain_curve(profiles, max_k=min(16, num_layers))
    payload = {
        "num_layers": num_layers,
        "patch_metric": "patch_kv_error",
        "patch_fit_mode": PATCH_FIT_MODE_DIRECT_KV_V1,
        "layers": per_layer,
        "patch_gain_curve": gain_curve,
        "recommended_patch_layers": [
            row["layer"]
            for row in sorted(per_layer, key=lambda item: float(item["expected_gain"]), reverse=True)[: min(8, num_layers)]
        ],
    }
    save_json(args.out, payload)
    logger.info("层分析结果已保存到 %s", args.out)


if __name__ == "__main__":
    main()
