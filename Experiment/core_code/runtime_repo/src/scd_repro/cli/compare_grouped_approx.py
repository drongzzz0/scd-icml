from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..io_utils import save_json
from ..pipeline import iter_pair_chunks, load_pairs_meta, load_reuse_weights
from ..reuse import decode_group, encode_group, group_layers_for_gid, pack_group, unpack_group
from ..rope import de_rope_k_torch, resolve_rope_specs
from ..torch_utils import clear_torch_cache, require_torch, resolve_device


@dataclass
class ErrorAccumulator:
    sse: float = 0.0
    ref_sse: float = 0.0
    numel: int = 0

    def update(self, pred, target) -> None:
        t = require_torch()
        diff = pred.to(dtype=t.float32) - target.to(dtype=t.float32)
        self.sse += float(diff.square().sum().item())
        self.ref_sse += float(target.to(dtype=t.float32).square().sum().item())
        self.numel += int(target.numel())

    def to_dict(self) -> dict[str, float | int]:
        mse = self.sse / max(1, self.numel)
        rmse = math.sqrt(mse)
        rel_l2 = math.sqrt(self.sse / max(self.ref_sse, 1e-12))
        return {
            "mse": float(mse),
            "rmse": float(rmse),
            "rel_l2": float(rel_l2),
            "sse": float(self.sse),
            "ref_sse": float(self.ref_sse),
            "numel": int(self.numel),
        }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Compare internal grouped reuse ckpt vs xKV-style grouped fake-SVD on the same captured KV tensors")
    p.add_argument("--pairs", required=True)
    p.add_argument("--reuse-ckpt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--models-config", default="")
    p.add_argument("--device", default="cpu")
    p.add_argument("--xkv-rank-k", type=int, default=0)
    p.add_argument("--xkv-rank-v", type=int, default=0)
    p.add_argument("--xkv-k-spaces", choices=["rope_native", "raw_derope", "both"], default="both")
    p.add_argument("--limit-chunks", type=int, default=0)
    return p


def _xkv_fake_svd_approx(tensor, rank: int):
    t = require_torch()
    if int(rank) <= 0:
        raise ValueError(f"xKV rank 必须为正整数，实际为 {rank}")
    bs, nh, sl, hd = tensor.shape
    tensor_reshaped = tensor.transpose(1, 2).reshape(bs, sl, nh * hd)
    max_rank = min(int(sl), int(nh * hd))
    if int(rank) >= max_rank:
        return tensor

    u, s, v_h = t.linalg.svd(tensor_reshaped.to(dtype=t.float32), full_matrices=False)
    u_trunc = u[:, :, :rank]
    s_trunc = s[:, :rank]
    vt_trunc = v_h[:, :rank, :]
    sqrt_s = t.sqrt(s_trunc)
    u_scaled = u_trunc * sqrt_s.unsqueeze(1)
    vt_scaled = sqrt_s.unsqueeze(-1) * vt_trunc
    approx = t.matmul(u_scaled, vt_scaled)
    return approx.view(bs, sl, nh, hd).transpose(1, 2).to(dtype=tensor.dtype)


def _ensure_method_stats(
    container: dict[str, dict[str, ErrorAccumulator]],
    method: str,
) -> dict[str, ErrorAccumulator]:
    if method not in container:
        container[method] = {"k": ErrorAccumulator(), "v": ErrorAccumulator()}
    return container[method]


def _update_method_stats(
    container: dict[str, dict[str, ErrorAccumulator]],
    method: str,
    *,
    pred_k,
    target_k,
    pred_v,
    target_v,
) -> None:
    stats = _ensure_method_stats(container, method)
    stats["k"].update(pred_k, target_k)
    stats["v"].update(pred_v, target_v)


def _combine_branch_stats(stats: dict[str, ErrorAccumulator]) -> dict[str, Any]:
    total = ErrorAccumulator(
        sse=stats["k"].sse + stats["v"].sse,
        ref_sse=stats["k"].ref_sse + stats["v"].ref_sse,
        numel=stats["k"].numel + stats["v"].numel,
    )
    return {
        "k": stats["k"].to_dict(),
        "v": stats["v"].to_dict(),
        "total": total.to_dict(),
    }


def _k_layers_for_space(layers: list[Any], *, rope_spec, k_space: str) -> list[Any]:
    if k_space == "rope_native":
        return [layer.to(dtype=require_torch().float32) for layer in layers]
    if k_space != "raw_derope":
        raise ValueError(f"不支持的 k_space: {k_space}")
    if rope_spec is None:
        raise ValueError("raw_derope 需要可解析的 rope spec")
    return [de_rope_k_torch(layer.to(dtype=require_torch().float32), rope_spec) for layer in layers]


def _split_group_xkv(
    k_layers: list[Any],
    v_layers: list[Any],
    *,
    rank_k: int,
    rank_v: int,
) -> tuple[list[Any], list[Any]]:
    split_sizes = [int(layer.shape[1]) for layer in k_layers]
    combined_k = require_torch().cat(k_layers, dim=1)
    combined_v = require_torch().cat(v_layers, dim=1)
    approx_k = _xkv_fake_svd_approx(combined_k, rank_k)
    approx_v = _xkv_fake_svd_approx(combined_v, rank_v)
    return list(require_torch().split(approx_k, split_sizes, dim=1)), list(require_torch().split(approx_v, split_sizes, dim=1))


def _internal_group_reconstruction(
    arrays: dict[str, Any],
    layers: list[int],
    *,
    row,
    reuse_weights,
    rope_a,
    rope_b,
    prefix_tokens: int,
    head_dim: int,
    num_key_value_heads: int,
    batch_size: int,
) -> tuple[list[Any], list[Any], list[Any], list[Any]]:
    t = require_torch()
    source_k_layers = []
    target_k_layers = []
    source_v_layers = []
    target_v_layers = []
    for layer in layers:
        k_a = arrays[f"k_a.{layer}"]
        k_b = arrays[f"k_b.{layer}"]
        if bool(reuse_weights.k_de_rope):
            if rope_a is None or rope_b is None:
                raise ValueError("reuse ckpt 需要 de-rope，但缺少 rope 配置")
            source_k_layers.append(de_rope_k_torch(k_a.to(dtype=t.float32), rope_a))
            target_k_layers.append(de_rope_k_torch(k_b.to(dtype=t.float32), rope_b))
        else:
            source_k_layers.append(k_a.to(dtype=t.float32))
            target_k_layers.append(k_b.to(dtype=t.float32))
        source_v_layers.append(arrays[f"v_a.{layer}"].to(dtype=t.float32))
        target_v_layers.append(arrays[f"v_b.{layer}"].to(dtype=t.float32))

    a_k = pack_group(source_k_layers)
    a_v = pack_group(source_v_layers)
    z_k, z_v = encode_group(a_k, a_v, row)
    rec_k, rec_v = decode_group(z_k, z_v, row)
    rec_k_layers = unpack_group(rec_k, len(layers), head_dim, batch_size, num_key_value_heads, prefix_tokens)
    rec_v_layers = unpack_group(rec_v, len(layers), head_dim, batch_size, num_key_value_heads, prefix_tokens)
    return rec_k_layers, target_k_layers, rec_v_layers, target_v_layers


def _summarize_stats(
    stats_by_method: dict[str, dict[str, ErrorAccumulator]],
) -> dict[str, dict[str, Any]]:
    return {method: _combine_branch_stats(stats) for method, stats in stats_by_method.items()}


def main() -> None:
    args = build_parser().parse_args()
    device = resolve_device(args.device, cpu_fallback=True)
    pairs_meta = load_pairs_meta(args.pairs)
    reuse_weights, reuse_meta = load_reuse_weights(args.reuse_ckpt, device=str(device), dtype="float32", cpu_fallback=True)

    rope_a = rope_b = None
    try:
        rope_a, rope_b = resolve_rope_specs(pairs_meta, args.models_config or None)
    except ValueError:
        rope_a = rope_b = None

    xkv_rank_k = int(args.xkv_rank_k or reuse_meta.get("rank_k") or reuse_weights.rank_k)
    xkv_rank_v = int(args.xkv_rank_v or reuse_meta.get("rank_v") or reuse_weights.rank_v)
    xkv_k_spaces = ["rope_native", "raw_derope"] if args.xkv_k_spaces == "both" else [str(args.xkv_k_spaces)]
    if "raw_derope" in xkv_k_spaces and rope_a is None:
        raise ValueError("请求 xkv raw_derope 对比，但未能从 pairs/models-config 解析 rope spec")

    head_dim = int(pairs_meta["head_dim"])
    num_key_value_heads = int(pairs_meta["num_key_value_heads"])
    batch_size = int(pairs_meta["batch_size"])
    prefix_tokens = int(pairs_meta["prefix_tokens"])

    overall_stats: dict[str, dict[str, ErrorAccumulator]] = {}
    group_stats: dict[int, dict[str, dict[str, ErrorAccumulator]]] = {}
    layer_stats: dict[int, dict[str, dict[str, ErrorAccumulator]]] = {}

    chunks_seen = 0
    for arrays, chunk_meta in iter_pair_chunks(args.pairs, device=str(device), dtype="float32", cpu_fallback=True):
        chunks_seen += 1
        chunk_prefix_tokens = int(chunk_meta["prefix_tokens"])
        chunk_batch_size = int(chunk_meta["batch_size"])
        for gid in range(reuse_weights.num_groups):
            layers = group_layers_for_gid(reuse_weights, gid)
            row = reuse_weights.weights[gid]

            rec_k_layers, target_k_layers, rec_v_layers, target_v_layers = _internal_group_reconstruction(
                arrays,
                layers,
                row=row,
                reuse_weights=reuse_weights,
                rope_a=rope_a,
                rope_b=rope_b,
                prefix_tokens=chunk_prefix_tokens,
                head_dim=head_dim,
                num_key_value_heads=num_key_value_heads,
                batch_size=chunk_batch_size,
            )
            group_container = group_stats.setdefault(gid, {})
            for idx, layer in enumerate(layers):
                _update_method_stats(
                    overall_stats,
                    "internal_ckpt",
                    pred_k=rec_k_layers[idx],
                    target_k=target_k_layers[idx],
                    pred_v=rec_v_layers[idx],
                    target_v=target_v_layers[idx],
                )
                _update_method_stats(
                    group_container,
                    "internal_ckpt",
                    pred_k=rec_k_layers[idx],
                    target_k=target_k_layers[idx],
                    pred_v=rec_v_layers[idx],
                    target_v=target_v_layers[idx],
                )
                _update_method_stats(
                    layer_stats.setdefault(int(layer), {}),
                    "internal_ckpt",
                    pred_k=rec_k_layers[idx],
                    target_k=target_k_layers[idx],
                    pred_v=rec_v_layers[idx],
                    target_v=target_v_layers[idx],
                )

            base_k_layers = [arrays[f"k_a.{layer}"] for layer in layers]
            base_v_layers = [arrays[f"v_a.{layer}"].to(dtype=require_torch().float32) for layer in layers]
            for k_space in xkv_k_spaces:
                method = f"xkv_{k_space}"
                xkv_k_layers = _k_layers_for_space(base_k_layers, rope_spec=rope_a, k_space=k_space)
                xkv_v_layers = [layer.to(dtype=require_torch().float32) for layer in base_v_layers]
                pred_k_layers, pred_v_layers = _split_group_xkv(
                    xkv_k_layers,
                    xkv_v_layers,
                    rank_k=xkv_rank_k,
                    rank_v=xkv_rank_v,
                )
                for idx, layer in enumerate(layers):
                    _update_method_stats(
                        overall_stats,
                        method,
                        pred_k=pred_k_layers[idx],
                        target_k=xkv_k_layers[idx],
                        pred_v=pred_v_layers[idx],
                        target_v=xkv_v_layers[idx],
                    )
                    _update_method_stats(
                        group_container,
                        method,
                        pred_k=pred_k_layers[idx],
                        target_k=xkv_k_layers[idx],
                        pred_v=pred_v_layers[idx],
                        target_v=xkv_v_layers[idx],
                    )
                    _update_method_stats(
                        layer_stats.setdefault(int(layer), {}),
                        method,
                        pred_k=pred_k_layers[idx],
                        target_k=xkv_k_layers[idx],
                        pred_v=pred_v_layers[idx],
                        target_v=xkv_v_layers[idx],
                    )

        clear_torch_cache()
        if int(args.limit_chunks) > 0 and chunks_seen >= int(args.limit_chunks):
            break

    payload = {
        "meta": {
            "pairs_path": str(Path(args.pairs).resolve()),
            "reuse_ckpt_path": str(Path(args.reuse_ckpt).resolve()),
            "out_path": str(Path(args.out).resolve()),
            "device": str(device),
            "num_chunks_processed": int(chunks_seen),
            "num_layers": int(reuse_weights.num_layers),
            "num_groups": int(reuse_weights.num_groups),
            "group_size": int(reuse_weights.group_size),
            "rank_k": int(reuse_weights.rank_k),
            "rank_v": int(reuse_weights.rank_v),
            "xkv_rank_k": int(xkv_rank_k),
            "xkv_rank_v": int(xkv_rank_v),
            "xkv_k_spaces": list(xkv_k_spaces),
            "internal_k_de_rope": bool(reuse_weights.k_de_rope),
            "internal_k_re_rope": bool(reuse_weights.k_re_rope),
            "head_dim": int(head_dim),
            "num_key_value_heads": int(num_key_value_heads),
            "prefix_tokens": int(prefix_tokens),
            "layouts": {
                "internal_group_codec": {
                    "rows": "B*KVH*T",
                    "cols": "group_size*HD",
                },
                "xkv_fake_svd": {
                    "rows": "T",
                    "cols": "group_size*KVH*HD",
                    "batched_over": "B",
                },
            },
        },
        "summary": _summarize_stats(overall_stats),
        "groups": [
            {
                "group_id": int(gid),
                "layers": [int(layer) for layer in group_layers_for_gid(reuse_weights, gid)],
                "metrics": _summarize_stats(group_container),
            }
            for gid, group_container in sorted(group_stats.items())
        ],
        "layers": [
            {
                "layer": int(layer),
                "metrics": _summarize_stats(layer_container),
            }
            for layer, layer_container in sorted(layer_stats.items())
        ],
    }
    save_json(args.out, payload)


if __name__ == "__main__":
    main()
