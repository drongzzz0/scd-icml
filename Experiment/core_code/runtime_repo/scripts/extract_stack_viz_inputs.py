#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract KV distance and t-SNE/2D embedding inputs for the SCD "
            "stack visualization from pair artifacts and a Reuse checkpoint."
        )
    )
    parser.add_argument("--pairs", default="", help="Pair artifact directory containing manifest.json and chunks/")
    parser.add_argument("--reuse-ckpt", default="", help="Reuse checkpoint directory")
    parser.add_argument("--vectors-npz", default="", help="Optional sampled vector NPZ from a previous extraction; skips --pairs/--reuse-ckpt")
    parser.add_argument("--save-vectors-npz", default="", help="Optional path to save sampled source/target/translated vectors")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--dataset", default="cmrc2018", help="Dataset prefix used in output file names")
    parser.add_argument("--profile", default="", help="Optional profile JSON; first recommended_patch_layers entry is used when --layer is not set")
    parser.add_argument("--layer", type=int, default=-1, help="Layer to visualize; overrides --profile")
    parser.add_argument("--chunk-index", type=int, default=0, help="0-based pair chunk index")
    parser.add_argument("--points", type=int, default=160, help="Maximum points per series")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--perplexity", type=int, default=50)
    return parser


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"no rows to write: {path}")
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def choose_layer(num_layers: int, explicit_layer: int, profile_path: Path | None) -> int:
    if explicit_layer >= 0:
        layer = int(explicit_layer)
    elif profile_path is not None:
        profile = load_json(profile_path)
        recommended = profile.get("recommended_patch_layers") or []
        if not recommended:
            raise ValueError(f"profile has no recommended_patch_layers: {profile_path}")
        layer = int(recommended[0])
    else:
        layer = max(0, int(num_layers) // 2)
    if layer < 0 or layer >= int(num_layers):
        raise ValueError(f"layer out of range: {layer} for num_layers={num_layers}")
    return layer


def flatten_kv_for_plot(k_tensor: Any, v_tensor: Any, *, num_key_value_heads: int, head_dim: int) -> np.ndarray:
    import torch

    k = k_tensor
    v = v_tensor
    if not torch.is_tensor(k) or not torch.is_tensor(v):
        raise TypeError("k_tensor and v_tensor must be torch tensors")
    kk = k.permute(0, 2, 1, 3).reshape(-1, int(num_key_value_heads) * int(head_dim))
    vv = v.permute(0, 2, 1, 3).reshape(-1, int(num_key_value_heads) * int(head_dim))
    return torch.cat([kk, vv], dim=1).detach().cpu().numpy().astype("float32")


def sample_ids(total: int, n: int, seed: int) -> list[int]:
    if total <= 0:
        raise ValueError("cannot sample from empty point set")
    count = min(int(n), int(total))
    rng = random.Random(int(seed))
    ids = list(range(int(total)))
    rng.shuffle(ids)
    return sorted(ids[:count])


def relative_l2(values: np.ndarray, target: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    numerator = np.linalg.norm(values - target, axis=1)
    denominator = np.maximum(np.linalg.norm(target, axis=1), eps)
    return (numerator / denominator).astype("float32")


def cosine_distance(values: np.ndarray, target: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    dot = np.sum(values * target, axis=1)
    denom = np.maximum(np.linalg.norm(values, axis=1) * np.linalg.norm(target, axis=1), eps)
    return (1.0 - dot / denom).astype("float32")


def fractional_improvement(before: np.ndarray, after: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return ((before - after) / np.maximum(before, eps)).astype("float32")


def build_distance_rows(layer: int, point_ids: list[int], source: np.ndarray, target: np.ndarray, translated: np.ndarray) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source_rel = relative_l2(source, target)
    translated_rel = relative_l2(translated, target)
    source_cos = cosine_distance(source, target)
    translated_cos = cosine_distance(translated, target)
    rel_improvement = fractional_improvement(source_rel, translated_rel)
    cos_improvement = fractional_improvement(source_cos, translated_cos)
    rows: list[dict[str, Any]] = []
    for row_idx, point_id in enumerate(point_ids):
        rows.append(
            {
                "layer": int(layer),
                "point_id": int(point_id),
                "source_rel_l2": float(source_rel[row_idx]),
                "translated_rel_l2": float(translated_rel[row_idx]),
                "source_cos_dist": float(source_cos[row_idx]),
                "translated_cos_dist": float(translated_cos[row_idx]),
                "rel_l2_improvement": float(rel_improvement[row_idx]),
                "cos_improvement": float(cos_improvement[row_idx]),
            }
        )
    meta = {
        "layer": int(layer),
        "points": int(len(rows)),
        "source_rel_l2_mean": float(np.mean(source_rel)),
        "translated_rel_l2_mean": float(np.mean(translated_rel)),
        "source_rel_l2_median": float(np.median(source_rel)),
        "translated_rel_l2_median": float(np.median(translated_rel)),
        "mean_rel_l2_reduction": float(np.mean(rel_improvement)),
        "median_rel_l2_reduction": float((np.median(source_rel) - np.median(translated_rel)) / max(float(np.median(source_rel)), 1e-12)),
        "source_cos_dist_mean": float(np.mean(source_cos)),
        "translated_cos_dist_mean": float(np.mean(translated_cos)),
        "mean_cos_reduction": float(np.mean(cos_improvement)),
    }
    return rows, meta


def reduce_2d(x: np.ndarray, perplexity: int, seed: int) -> tuple[np.ndarray, str, int | None]:
    actual_perplexity = min(int(perplexity), max(2, int(x.shape[0]) - 1))
    try:
        from sklearn.manifold import TSNE

        coords = TSNE(
            n_components=2,
            perplexity=actual_perplexity,
            init="pca",
            learning_rate="auto",
            random_state=int(seed),
        ).fit_transform(x)
        return coords.astype("float32"), "tsne", actual_perplexity
    except Exception:
        x0 = x - x.mean(axis=0, keepdims=True)
        u, s, _vh = np.linalg.svd(x0, full_matrices=False)
        coords = u[:, :2] * s[:2]
        return coords.astype("float32"), "pca_fallback", None


def build_tsne_rows(layer: int, point_ids: list[int], source: np.ndarray, target: np.ndarray, translated: np.ndarray, *, perplexity: int, seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    n = len(point_ids)
    x = np.concatenate([source, target, translated], axis=0)
    labels = ["source"] * n + ["target"] * n + ["translated_reuse"] * n
    coords, reducer, actual_perplexity = reduce_2d(x, perplexity=perplexity, seed=seed)
    rows: list[dict[str, Any]] = []
    for idx, label in enumerate(labels):
        rows.append(
            {
                "layer": int(layer),
                "point_id": int(point_ids[idx % n]),
                "series": label,
                "x": float(coords[idx, 0]),
                "y": float(coords[idx, 1]),
                "reducer": reducer,
                "perplexity": "" if actual_perplexity is None else int(actual_perplexity),
            }
        )
    source_centroid = coords[0:n].mean(axis=0)
    target_centroid = coords[n : 2 * n].mean(axis=0)
    translated_centroid = coords[2 * n : 3 * n].mean(axis=0)
    raw_gap = float(np.linalg.norm(source_centroid - target_centroid))
    translated_gap = float(np.linalg.norm(translated_centroid - target_centroid))
    meta = {
        "layer": int(layer),
        "points_per_series": int(n),
        "reducer": reducer,
        "perplexity": actual_perplexity,
        "raw_centroid_gap": raw_gap,
        "translated_centroid_gap": translated_gap,
        "gap_ratio": translated_gap / max(raw_gap, 1e-12),
    }
    return rows, meta


def save_vectors_npz(path: Path, layer: int, point_ids: list[int], source: np.ndarray, target: np.ndarray, translated: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        layer=np.asarray([int(layer)], dtype=np.int64),
        point_ids=np.asarray(point_ids, dtype=np.int64),
        source=np.asarray(source, dtype=np.float32),
        target=np.asarray(target, dtype=np.float32),
        translated=np.asarray(translated, dtype=np.float32),
    )


def load_vectors_npz(path: Path) -> tuple[int, list[int], np.ndarray, np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path) as data:
        layer = int(np.asarray(data["layer"]).reshape(-1)[0])
        point_ids = [int(value) for value in np.asarray(data["point_ids"]).reshape(-1)]
        source = np.asarray(data["source"], dtype=np.float32)
        target = np.asarray(data["target"], dtype=np.float32)
        translated = np.asarray(data["translated"], dtype=np.float32)
    if not (source.shape == target.shape == translated.shape):
        raise ValueError(f"vector shapes do not match: source={source.shape} target={target.shape} translated={translated.shape}")
    if len(point_ids) != int(source.shape[0]):
        raise ValueError(f"point_ids length {len(point_ids)} does not match vector rows {source.shape[0]}")
    return layer, point_ids, source, target, translated


def load_first_chunk(pairs_dir: Path, chunk_index: int, *, device: str, dtype: str):
    from scd_repro.io_utils import load_bundle

    manifest = load_json(pairs_dir / "manifest.json")
    chunks = list(manifest.get("chunks", []))
    if not chunks:
        raise ValueError(f"pair manifest has no chunks: {pairs_dir}")
    if chunk_index < 0 or chunk_index >= len(chunks):
        raise ValueError(f"chunk index {chunk_index} out of range for {len(chunks)} chunks")
    chunk_path = pairs_dir / str(chunks[chunk_index]["path"])
    return load_bundle(chunk_path, device=device, dtype=dtype, cpu_fallback=True)


def extract_arrays(args: argparse.Namespace) -> tuple[int, list[int], np.ndarray, np.ndarray, np.ndarray]:
    import torch
    from scd_repro.pipeline import load_pairs_meta, load_reuse_weights
    from scd_repro.reuse import decode_group, encode_group, group_layers_for_gid, move_reuse_weights, pack_group, unpack_group
    from scd_repro.rope import de_rope_k_torch, re_rope_k_torch, resolve_rope_specs

    if not str(args.pairs or "").strip() or not str(args.reuse_ckpt or "").strip():
        raise ValueError("--pairs and --reuse-ckpt are required unless --vectors-npz is provided")
    pairs_dir = Path(args.pairs)
    meta = load_pairs_meta(pairs_dir)
    layer = choose_layer(
        int(meta["num_layers"]),
        int(args.layer),
        Path(args.profile) if str(args.profile or "").strip() else None,
    )
    reuse_cpu, reuse_meta = load_reuse_weights(args.reuse_ckpt, device="cpu", dtype=str(args.dtype), cpu_fallback=True)
    reuse = move_reuse_weights(reuse_cpu, str(args.device), dtype=str(args.dtype), cpu_fallback=True)
    rope_source = reuse_meta if isinstance(reuse_meta.get("rope"), dict) else meta
    rope_a, rope_b = resolve_rope_specs(rope_source, None)

    bundle = load_first_chunk(pairs_dir, int(args.chunk_index), device=str(args.device), dtype=str(args.dtype))
    arrays = bundle.arrays
    head_dim = int(meta["head_dim"])
    batch_size = int(bundle.meta.get("batch_size", meta.get("batch_size", 1)))
    num_key_value_heads = int(bundle.meta.get("num_key_value_heads", meta["num_key_value_heads"]))
    prefix_tokens = int(bundle.meta.get("prefix_tokens", meta["prefix_tokens"]))

    gid = None
    group_layers: list[int] = []
    for candidate_gid in range(reuse.num_groups):
        candidate_layers = group_layers_for_gid(reuse, candidate_gid)
        if layer in candidate_layers:
            gid = candidate_gid
            group_layers = candidate_layers
            break
    if gid is None:
        raise RuntimeError(f"cannot locate reuse group for layer {layer}")

    a_k = pack_group([de_rope_k_torch(arrays[f"k_a.{item}"], rope_a).to(dtype=torch.float32) for item in group_layers])
    a_v = pack_group([arrays[f"v_a.{item}"].to(dtype=torch.float32) for item in group_layers])
    with torch.no_grad():
        z_k, z_v = encode_group(a_k, a_v, reuse.weights[int(gid)])
        k_hat_raw, v_hat = decode_group(z_k, z_v, reuse.weights[int(gid)])
    k_layers_raw = unpack_group(k_hat_raw, len(group_layers), head_dim, batch_size, num_key_value_heads, prefix_tokens)
    v_layers = unpack_group(v_hat, len(group_layers), head_dim, batch_size, num_key_value_heads, prefix_tokens)
    layer_offset = group_layers.index(layer)
    k_trans = re_rope_k_torch(k_layers_raw[layer_offset], rope_b)
    v_trans = v_layers[layer_offset]

    source = flatten_kv_for_plot(arrays[f"k_a.{layer}"], arrays[f"v_a.{layer}"], num_key_value_heads=num_key_value_heads, head_dim=head_dim)
    target = flatten_kv_for_plot(arrays[f"k_b.{layer}"], arrays[f"v_b.{layer}"], num_key_value_heads=num_key_value_heads, head_dim=head_dim)
    translated = flatten_kv_for_plot(k_trans, v_trans, num_key_value_heads=num_key_value_heads, head_dim=head_dim)
    total = min(source.shape[0], target.shape[0], translated.shape[0])
    ids = sample_ids(total, int(args.points), int(args.seed))
    return layer, ids, source[ids], target[ids], translated[ids]


def main() -> None:
    args = build_parser().parse_args()
    if str(args.vectors_npz or "").strip():
        layer, ids, source, target, translated = load_vectors_npz(Path(args.vectors_npz))
    else:
        layer, ids, source, target, translated = extract_arrays(args)
        if str(args.save_vectors_npz or "").strip():
            save_vectors_npz(Path(args.save_vectors_npz), layer, ids, source, target, translated)
    out_dir = Path(args.out_dir)
    dataset = str(args.dataset).strip() or "dataset"
    distance_rows, distance_meta = build_distance_rows(layer, ids, source, target, translated)
    tsne_rows, tsne_meta = build_tsne_rows(
        layer,
        ids,
        source,
        target,
        translated,
        perplexity=int(args.perplexity),
        seed=int(args.seed),
    )
    distance_csv = out_dir / f"{dataset}_kv_distance_distribution.csv"
    distance_meta_path = out_dir / f"{dataset}_kv_distance_distribution_meta.json"
    tsne_csv = out_dir / f"{dataset}_tsne_kv_alignment.csv"
    tsne_meta_path = out_dir / f"{dataset}_tsne_kv_alignment_meta.json"
    write_csv(distance_csv, distance_rows)
    write_json(distance_meta_path, distance_meta)
    write_csv(tsne_csv, tsne_rows)
    write_json(tsne_meta_path, tsne_meta)
    for path in (distance_csv, distance_meta_path, tsne_csv, tsne_meta_path):
        print(path)


if __name__ == "__main__":
    main()
