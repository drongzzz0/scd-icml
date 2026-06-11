from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

from ..pipeline import load_pairs_meta, save_reuse_weights
from ..reuse import (
    REUSE_MODE_XKV_GROUP,
    SHARED_Z_ACTIVATION_IDENTITY,
    SVD_BACKEND_MOMENTS_EIGH,
    TRANSLATOR_PATH_SHARED_Z,
    ReuseGroupWeights,
    ReuseWeights,
    build_group_specs,
    group_layers_from_size,
    pack_group,
    resolve_xkv_group_config,
    train_reuse_group_from_moments_torch,
)
from ..rope import de_rope_k_torch, resolve_rope_specs
from ..seed import set_global_seed
from ..torch_utils import clear_torch_cache, configure_torch_acceleration, resolve_device, tensor_to_cpu
from ..tracking import command_from_argv, write_manifest
from ..validate import validate_path
from .common import add_accel_args, load_cfg, resolve_accel_options, resolve_logger


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train grouped shared-z reuse weights for xKV reuse")
    p.add_argument("--pairs", required=True)
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


def _create_branch_stats(feature_dim: int, *, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "aa": torch.zeros((feature_dim, feature_dim), device=device, dtype=torch.float32),
        "ab": torch.zeros((feature_dim, feature_dim), device=device, dtype=torch.float32),
        "bb": torch.zeros((feature_dim, feature_dim), device=device, dtype=torch.float32),
        "sum_a": torch.zeros((feature_dim,), device=device, dtype=torch.float32),
        "sum_b": torch.zeros((feature_dim,), device=device, dtype=torch.float32),
        "count": torch.zeros((), device=device, dtype=torch.float64),
    }


def _accumulate_branch_stats(a_mat: torch.Tensor, b_mat: torch.Tensor, stats: dict[str, torch.Tensor]) -> None:
    stats["aa"] += a_mat.T @ a_mat
    stats["ab"] += a_mat.T @ b_mat
    stats["bb"] += b_mat.T @ b_mat
    stats["sum_a"] += a_mat.sum(dim=0)
    stats["sum_b"] += b_mat.sum(dim=0)
    stats["count"] += float(a_mat.shape[0])


def _accumulate_group_for_chunk(
    arrays: dict[str, torch.Tensor],
    layers: list[int],
    stats: dict[str, dict[str, torch.Tensor]],
    rope_a,
    rope_b,
    *,
    k_de_rope: bool,
) -> None:
    a_k = pack_group(
        [
            (
                de_rope_k_torch(arrays[f"k_a.{layer}"], rope_a)
                if k_de_rope
                else arrays[f"k_a.{layer}"]
            ).to(dtype=torch.float32)
            for layer in layers
        ]
    )
    b_k = pack_group(
        [
            (
                de_rope_k_torch(arrays[f"k_b.{layer}"], rope_b)
                if k_de_rope
                else arrays[f"k_b.{layer}"]
            ).to(dtype=torch.float32)
            for layer in layers
        ]
    )
    a_v = pack_group([arrays[f"v_a.{layer}"].to(dtype=torch.float32) for layer in layers])
    b_v = pack_group([arrays[f"v_b.{layer}"].to(dtype=torch.float32) for layer in layers])
    _accumulate_branch_stats(a_k, b_k, stats["k"])
    _accumulate_branch_stats(a_v, b_v, stats["v"])


def _accumulate_groups_with_retry(
    arrays: dict[str, torch.Tensor],
    gids: list[int],
    layers_by_gid: dict[int, list[int]],
    stats_by_gid: dict[int, dict[str, dict[str, torch.Tensor]]],
    rope_a,
    rope_b,
    *,
    k_de_rope: bool,
    logger,
    depth: int = 0,
) -> None:
    try:
        for gid in gids:
            _accumulate_group_for_chunk(
                arrays,
                layers_by_gid[gid],
                stats_by_gid[gid],
                rope_a,
                rope_b,
                k_de_rope=k_de_rope,
            )
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
        logger.warning("reuse stats OOM, retrying by splitting groups depth=%d size=%d -> %d + %d", depth, len(gids), len(left), len(right))
        _accumulate_groups_with_retry(
            arrays,
            left,
            layers_by_gid,
            stats_by_gid,
            rope_a,
            rope_b,
            k_de_rope=k_de_rope,
            logger=logger,
            depth=depth + 1,
        )
        _accumulate_groups_with_retry(
            arrays,
            right,
            layers_by_gid,
            stats_by_gid,
            rope_a,
            rope_b,
            k_de_rope=k_de_rope,
            logger=logger,
            depth=depth + 1,
        )


def _solve_group_branch(
    branch_stats: dict[str, torch.Tensor],
    *,
    rank: int,
    ridge_lambda: float,
    center_before_svd: bool,
    cpu_fallback: bool,
    logger,
    gid: int,
    branch_name: str,
    device: torch.device,
):
    args = {
        "aa": branch_stats["aa"],
        "ab": branch_stats["ab"],
        "bb": branch_stats["bb"],
        "rank": rank,
        "ridge_lambda": ridge_lambda,
        "sum_a": branch_stats["sum_a"],
        "sum_b": branch_stats["sum_b"],
        "count": float(branch_stats["count"].item()),
        "center_before_svd": center_before_svd,
    }
    try:
        return train_reuse_group_from_moments_torch(**args)
    except RuntimeError:
        if not cpu_fallback:
            raise
        logger.warning("reuse solver failed on %s, retrying group %d branch %s on CPU", device, gid, branch_name)
        return train_reuse_group_from_moments_torch(
            aa=branch_stats["aa"].cpu(),
            ab=branch_stats["ab"].cpu(),
            bb=branch_stats["bb"].cpu(),
            rank=rank,
            ridge_lambda=ridge_lambda,
            sum_a=branch_stats["sum_a"].cpu(),
            sum_b=branch_stats["sum_b"].cpu(),
            count=float(branch_stats["count"].item()),
            center_before_svd=center_before_svd,
        )


def _group_compression_ratio(num_layers_in_group: int, head_dim: int, rank_k: int, rank_v: int) -> float:
    raw_dims = max(1, 2 * num_layers_in_group * head_dim)
    latent_dims = max(1, rank_k + rank_v)
    return float(raw_dims / latent_dims)


def main() -> None:
    args = build_parser().parse_args()
    logger = resolve_logger(args.log)
    set_global_seed(args.seed)

    meta = load_pairs_meta(args.pairs)
    train_cfg = load_cfg(args.train_config)
    xkv_cfg = resolve_xkv_group_config(train_cfg)
    accel = resolve_accel_options(train_cfg, args)
    configure_torch_acceleration(str(accel["accel_mode"]))

    compute_cfg = train_cfg.get("compute", {})
    runtime_cfg = train_cfg.get("runtime", {})
    train_device = str(args.train_device or compute_cfg.get("train_device") or "cuda:0")
    cpu_fallback = bool(runtime_cfg.get("cpu_fallback", True))
    device = resolve_device(train_device, cpu_fallback=cpu_fallback)

    requested_mode = str(xkv_cfg["mode"])
    requested_path = str(xkv_cfg["default_path"])
    if requested_mode != REUSE_MODE_XKV_GROUP or requested_path != TRANSLATOR_PATH_SHARED_Z:
        logger.warning(
            "legacy reuse config requested mode=%s path=%s; mainline stays on %s/%s",
            requested_mode,
            requested_path,
            REUSE_MODE_XKV_GROUP,
            TRANSLATOR_PATH_SHARED_Z,
        )
    if str(xkv_cfg["svd_backend"]) != SVD_BACKEND_MOMENTS_EIGH:
        raise ValueError(f"Only svd_backend={SVD_BACKEND_MOMENTS_EIGH} is supported, got {xkv_cfg['svd_backend']}")

    group_size = int(xkv_cfg["group_size"])
    rank_k = int(xkv_cfg["rank_k"])
    rank_v = int(xkv_cfg["rank_v"])
    ridge_lambda = float(xkv_cfg["ridge_lambda"])
    center_before_svd = bool(xkv_cfg["center_before_svd"])
    max_groups = int(xkv_cfg["max_groups"])
    translator_hidden_dim = int(xkv_cfg["translator_hidden_dim"])
    translator_activation = str(xkv_cfg["translator_activation"] or SHARED_Z_ACTIVATION_IDENTITY)
    rope_a, rope_b = resolve_rope_specs(meta, args.models_config or None)
    k_de_rope = not (rope_a == rope_b)
    k_re_rope = k_de_rope

    num_layers = int(meta["num_layers"])
    head_dim = int(meta["head_dim"])
    layers_by_gid = group_layers_from_size(num_layers, group_size, max_groups=max_groups)
    num_groups = len(layers_by_gid)
    group_block_size = int(accel["group_block_size"])
    logger.info(
        "xkv_group training start: device=%s accel=%s group_block_size=%d num_groups=%d group_size=%d rank_k=%d rank_v=%d center_before_svd=%s",
        device,
        accel["accel_mode"],
        group_block_size,
        num_groups,
        group_size,
        rank_k,
        rank_v,
        center_before_svd,
    )
    logger.info(
        "reuse mainline: mode=%s translator.default_path=%s svd_backend=%s max_groups=%d shared_z.hidden_dim=%d shared_z.activation=%s k_derope=%s k_rerope=%s",
        REUSE_MODE_XKV_GROUP,
        TRANSLATOR_PATH_SHARED_Z,
        SVD_BACKEND_MOMENTS_EIGH,
        max_groups,
        translator_hidden_dim,
        translator_activation,
        k_de_rope,
        k_re_rope,
    )

    stats_by_gid: dict[int, dict[str, dict[str, torch.Tensor]]] = {}
    for gid, layers in layers_by_gid.items():
        feat_dim = len(layers) * head_dim
        stats_by_gid[gid] = {
            "k": _create_branch_stats(feat_dim, device=device),
            "v": _create_branch_stats(feat_dim, device=device),
        }

    chunk_paths = list(_iter_chunk_paths(args.pairs))
    from ..io_utils import load_bundle

    start_all = time.perf_counter()
    for idx, chunk_path in enumerate(chunk_paths, start=1):
        bundle = load_bundle(chunk_path, device=str(device), dtype="float32", cpu_fallback=cpu_fallback)
        arrays = bundle.arrays

        gids = list(range(num_groups))
        for start in range(0, len(gids), group_block_size):
            block = gids[start : start + group_block_size]
            _accumulate_groups_with_retry(
                arrays,
                block,
                layers_by_gid,
                stats_by_gid,
                rope_a,
                rope_b,
                k_de_rope=k_de_rope,
                logger=logger,
            )

        del arrays, bundle
        clear_torch_cache()

        if idx == 1 or idx % 10 == 0 or idx == len(chunk_paths):
            elapsed = time.perf_counter() - start_all
            eta_s = (elapsed / idx) * (len(chunk_paths) - idx) if idx > 0 else 0.0
            logger.info("xkv_group stats progress: chunk %d/%d elapsed=%.1fs eta=%.1fs", idx, len(chunk_paths), elapsed, eta_s)

    group_weights: dict[int, ReuseGroupWeights] = {}
    group_summaries: list[dict[str, object]] = []
    for gid in range(num_groups):
        layers = layers_by_gid[gid]
        enc_k, dec_k_b, mean_a_k, mean_b_k = _solve_group_branch(
            stats_by_gid[gid]["k"],
            rank=rank_k,
            ridge_lambda=ridge_lambda,
            center_before_svd=center_before_svd,
            cpu_fallback=cpu_fallback,
            logger=logger,
            gid=gid,
            branch_name="k",
            device=device,
        )
        enc_v, dec_v_b, mean_a_v, mean_b_v = _solve_group_branch(
            stats_by_gid[gid]["v"],
            rank=rank_v,
            ridge_lambda=ridge_lambda,
            center_before_svd=center_before_svd,
            cpu_fallback=cpu_fallback,
            logger=logger,
            gid=gid,
            branch_name="v",
            device=device,
        )

        row = ReuseGroupWeights(
            enc_k=tensor_to_cpu(enc_k, dtype="float32"),
            dec_k_b=tensor_to_cpu(dec_k_b, dtype="float32"),
            enc_v=tensor_to_cpu(enc_v, dtype="float32"),
            dec_v_b=tensor_to_cpu(dec_v_b, dtype="float32"),
            mean_a_k=None if mean_a_k is None else tensor_to_cpu(mean_a_k, dtype="float32"),
            mean_b_k=None if mean_b_k is None else tensor_to_cpu(mean_b_k, dtype="float32"),
            mean_a_v=None if mean_a_v is None else tensor_to_cpu(mean_a_v, dtype="float32"),
            mean_b_v=None if mean_b_v is None else tensor_to_cpu(mean_b_v, dtype="float32"),
            rank_k=int(enc_k.shape[1]),
            rank_v=int(enc_v.shape[1]),
        )
        group_weights[gid] = row
        compression_ratio = _group_compression_ratio(len(layers), head_dim, row.rank_k, row.rank_v)
        group_summaries.append(
            {
                "group_id": gid,
                "layers": list(layers),
                "rank_k": row.rank_k,
                "rank_v": row.rank_v,
                "raw_feature_dims": int(2 * len(layers) * head_dim),
                "encoded_feature_dims": int(row.rank_k + row.rank_v),
                "compression_ratio": compression_ratio,
            }
        )
        logger.info(
            "xkv_group solved: gid=%d layers=%s rank_k=%d rank_v=%d compression_ratio=%.3f",
            gid,
            layers,
            row.rank_k,
            row.rank_v,
            compression_ratio,
        )

    weights = ReuseWeights(
        group_size=group_size,
        rank=max(rank_k, rank_v),
        rank_k=rank_k,
        rank_v=rank_v,
        num_layers=num_layers,
        num_groups=num_groups,
        weights=group_weights,
        mode=REUSE_MODE_XKV_GROUP,
        default_path=TRANSLATOR_PATH_SHARED_Z,
        group_layers=layers_by_gid,
        center_before_svd=center_before_svd,
        svd_backend=SVD_BACKEND_MOMENTS_EIGH,
        max_groups=max_groups,
        k_de_rope=k_de_rope,
        k_re_rope=k_re_rope,
        translator_hidden_dim=translator_hidden_dim,
        translator_activation=translator_activation,
    )
    save_reuse_weights(
        args.out,
        weights,
        extra_meta={
            "head_dim": head_dim,
            "num_key_value_heads": int(meta["num_key_value_heads"]),
            "batch_size": int(meta["batch_size"]),
            "prefix_tokens": int(meta["prefix_tokens"]),
            "k_space": "raw_derope" if k_de_rope else "rope_native",
            "rope": {"model_a": rope_a.to_dict(), "model_b": rope_b.to_dict()},
            "mode": REUSE_MODE_XKV_GROUP,
            "translator": {
                "default_path": TRANSLATOR_PATH_SHARED_Z,
                "shared_z": {
                    "hidden_dim": translator_hidden_dim,
                    "activation": translator_activation,
                },
            },
            "xkv_group": {
                "group_size": group_size,
                "rank_k": rank_k,
                "rank_v": rank_v,
                "center_before_svd": center_before_svd,
                "svd_backend": SVD_BACKEND_MOMENTS_EIGH,
                "max_groups": max_groups,
            },
            "groups": [
                {
                    **spec,
                    "rank_k": group_weights[spec["group_id"]].rank_k,
                    "rank_v": group_weights[spec["group_id"]].rank_v,
                    "center_before_svd": center_before_svd,
                    "k_de_rope": k_de_rope,
                    "k_re_rope": k_re_rope,
                }
                for spec in build_group_specs(layers_by_gid, head_dim)
            ],
            "group_summaries": group_summaries,
        },
    )
    validate_path(args.out)
    config_paths = {"train": args.train_config}
    if args.models_config:
        config_paths["models"] = args.models_config
    write_manifest(out_dir=args.out, command=command_from_argv(sys.argv), config_paths=config_paths, seed=args.seed)
    logger.info("xkv_group shared-z reuse checkpoint saved to %s", args.out)


if __name__ == "__main__":
    main()
