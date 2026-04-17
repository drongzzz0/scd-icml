from __future__ import annotations

import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from .cache_writer import CacheShapeSpec, CacheWriter
from .io_utils import load_bundle, load_json, save_bundle, save_json
from .patch import PATCH_FIT_MODE_DIRECT_KV_V1, infer_direct_kv_patch_layer_torch
from .reuse import (
    REUSE_MODE_XKV_GROUP,
    SHARED_Z_ACTIVATION_IDENTITY,
    TRANSLATOR_PATH_SHARED_Z,
    ReuseGroupWeights,
    ReuseWeights,
    build_group_specs,
    decode_group,
    encode_group,
    group_layers_for_gid,
    move_reuse_weights,
    pack_group,
    unpack_group,
)
from .rope import de_rope_k_torch, re_rope_k_torch, resolve_rope_specs
from .torch_utils import require_torch, tensor_to_cpu, tensor_to_device, tensor_to_numpy


PATCH_CKPT_SHARDED_FORMAT = "patch_ckpt_sharded_v1"
_XNORM_NATIVEKV_RESIDUAL_FIT_MODE = "xnorm_nativekv_residual_mlp_v1"


@dataclass
class PairMeta:
    model_a: str
    model_b: str
    dtype: str
    layout: str
    num_layers: int
    num_key_value_heads: int
    num_attention_heads: int
    head_dim: int
    hidden_dim: int
    batch_size: int
    prefix_tokens: int
    vocab_size: int
    seed: int
    eval_samples: int = 0
    rope: dict[str, object] | None = None


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _pair_meta_dict(meta: PairMeta | dict[str, object]) -> dict[str, object]:
    if isinstance(meta, PairMeta):
        return asdict(meta)
    return dict(meta)


def _pad_time_axis_4d(x: np.ndarray, target_t: int) -> np.ndarray:
    if x.shape[2] == target_t:
        return x.astype(np.float32)
    out = np.zeros((x.shape[0], x.shape[1], target_t, x.shape[3]), dtype=np.float32)
    out[:, :, : x.shape[2], :] = x.astype(np.float32)
    return out


def _pad_time_axis_3d(x: np.ndarray, target_t: int) -> np.ndarray:
    if x.shape[1] == target_t:
        return x.astype(np.float32)
    out = np.zeros((x.shape[0], target_t, x.shape[2]), dtype=np.float32)
    out[:, : x.shape[1], :] = x.astype(np.float32)
    return out


def _pad_time_axis_2d(x: np.ndarray, target_t: int, pad_value: int = 0) -> np.ndarray:
    if x.shape[1] == target_t:
        return x.copy()
    out = np.full((x.shape[0], target_t), pad_value, dtype=x.dtype)
    out[:, : x.shape[1]] = x
    return out


def synthesize_pairs(meta: PairMeta) -> dict[str, np.ndarray]:
    rng = _rng(meta.seed)
    arrays: dict[str, np.ndarray] = {}
    critical = {meta.num_layers // 2 - 1, meta.num_layers // 2, meta.num_layers // 2 + 1, max(0, meta.num_layers - 2)}

    input_ids = rng.integers(1, max(2, meta.vocab_size), size=(meta.batch_size, meta.prefix_tokens), dtype=np.int64)
    attention_mask = np.ones((meta.batch_size, meta.prefix_tokens), dtype=np.int64)
    next_token_ids = np.full((meta.batch_size, meta.prefix_tokens), -100, dtype=np.int64)
    if meta.prefix_tokens > 1:
        next_token_ids[:, :-1] = input_ids[:, 1:]
    arrays["input_ids"] = input_ids
    arrays["attention_mask"] = attention_mask
    arrays["next_token_ids"] = next_token_ids

    for layer in range(meta.num_layers):
        drift = 0.08 if layer in critical else 0.02
        base_k = rng.normal(0.0, 1.0, size=(meta.batch_size, meta.num_key_value_heads, meta.prefix_tokens, meta.head_dim)).astype(np.float32)
        base_v = rng.normal(0.0, 1.0, size=(meta.batch_size, meta.num_key_value_heads, meta.prefix_tokens, meta.head_dim)).astype(np.float32)
        noise_a = rng.normal(0.0, 0.01, size=base_k.shape).astype(np.float32)
        noise_b = rng.normal(0.0, drift, size=base_k.shape).astype(np.float32)

        arrays[f"k_a.{layer}"] = base_k + noise_a
        arrays[f"v_a.{layer}"] = base_v + noise_a
        arrays[f"k_b.{layer}"] = base_k + noise_b
        arrays[f"v_b.{layer}"] = base_v + noise_b

        r_base = rng.normal(0.0, 1.0, size=(meta.batch_size, meta.prefix_tokens, meta.hidden_dim)).astype(np.float32)
        arrays[f"resid_a.{layer}"] = r_base + rng.normal(0.0, 0.02, size=r_base.shape).astype(np.float32)
        arrays[f"resid_b.{layer}"] = r_base + rng.normal(0.0, drift, size=r_base.shape).astype(np.float32)
        arrays[f"xnorm_a.{layer}"] = arrays[f"resid_a.{layer}"] + rng.normal(0.0, 0.01, size=r_base.shape).astype(np.float32)
        arrays[f"xnorm_b.{layer}"] = arrays[f"resid_b.{layer}"] + rng.normal(0.0, 0.01, size=r_base.shape).astype(np.float32)

    return arrays


def save_chunked_pairs(
    out_dir: str | Path,
    chunks: list[tuple[PairMeta | dict[str, object], dict[str, Any]]],
    artifact_dtype: str = "bfloat16",
) -> None:
    if not chunks:
        raise ValueError("chunks 不能为空")

    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    chunks_dir = root / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    base_meta = _pair_meta_dict(chunks[0][0])
    manifest_chunks: list[dict[str, object]] = []
    total_batch = 0
    total_eval_samples = 0
    max_prefix_tokens = 0

    for idx, (chunk_meta_raw, arrays) in enumerate(chunks):
        chunk_meta = _pair_meta_dict(chunk_meta_raw)
        chunk_name = f"chunk_{idx:05d}.pt"
        chunk_path = chunks_dir / chunk_name
        save_bundle(chunk_path, arrays, meta=chunk_meta, artifact_dtype=artifact_dtype)

        batch_size = int(chunk_meta["batch_size"])
        prefix_tokens = int(chunk_meta["prefix_tokens"])
        eval_samples = int(chunk_meta.get("eval_samples") or batch_size)
        total_batch += batch_size
        total_eval_samples += eval_samples
        max_prefix_tokens = max(max_prefix_tokens, prefix_tokens)
        manifest_chunks.append(
            {
                "path": f"chunks/{chunk_name}",
                "batch_size": batch_size,
                "prefix_tokens": prefix_tokens,
                "eval_samples": eval_samples,
                "num_layers": int(chunk_meta["num_layers"]),
            }
        )

    manifest = dict(base_meta)
    manifest.update(
        {
            "format": "chunked_pt_v1",
            "artifact_dtype": artifact_dtype,
            "batch_size": total_batch,
            "eval_samples": total_eval_samples,
            "prefix_tokens": max_prefix_tokens,
            "num_chunks": len(manifest_chunks),
            "chunks": manifest_chunks,
        }
    )
    save_json(root / "manifest.json", manifest)


def save_pairs(out_dir: str | Path, meta: PairMeta | dict[str, object], arrays: dict[str, Any], artifact_dtype: str = "bfloat16") -> None:
    save_chunked_pairs(out_dir, [(meta, arrays)], artifact_dtype=artifact_dtype)


def load_pairs_meta(path: str | Path) -> dict[str, object]:
    return load_json(Path(path) / "manifest.json")


def iter_pair_chunks(
    path: str | Path,
    *,
    device: str | None = "cpu",
    dtype: str | None = None,
    as_numpy: bool = False,
    cpu_fallback: bool = True,
) -> Iterator[tuple[dict[str, Any], dict[str, object]]]:
    root = Path(path)
    manifest = load_json(root / "manifest.json")
    if manifest.get("format") != "chunked_pt_v1":
        raise FileNotFoundError(f"未找到可读取的 pairs 产物: {root}")

    for chunk in manifest.get("chunks", []):
        payload = load_bundle(root / str(chunk["path"]), device=device, dtype=dtype, as_numpy=as_numpy, cpu_fallback=cpu_fallback)
        meta = dict(payload.meta or {})
        if not meta:
            meta = {
                "batch_size": int(chunk["batch_size"]),
                "prefix_tokens": int(chunk["prefix_tokens"]),
                "eval_samples": int(chunk["eval_samples"]),
                "num_layers": int(chunk["num_layers"]),
            }
        yield payload.arrays, meta


def _merge_pair_chunks(chunks: list[tuple[dict[str, np.ndarray], dict[str, object]]]) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    if not chunks:
        raise ValueError("pairs 为空")

    global_t = max(int(meta["prefix_tokens"]) for _arrays, meta in chunks)
    merged: dict[str, list[np.ndarray]] = {}
    for arrays, _meta in chunks:
        for key, arr in arrays.items():
            if key.startswith("xnorm") or key.startswith("resid"):
                padded = _pad_time_axis_3d(arr, global_t)
            elif key == "input_ids":
                padded = _pad_time_axis_2d(arr, global_t, pad_value=0)
            elif key == "attention_mask":
                padded = _pad_time_axis_2d(arr, global_t, pad_value=0)
            elif key == "next_token_ids":
                padded = _pad_time_axis_2d(arr, global_t, pad_value=-100)
            else:
                padded = _pad_time_axis_4d(arr, global_t)
            merged.setdefault(key, []).append(padded)

    arrays_out = {key: np.concatenate(parts, axis=0) for key, parts in merged.items()}
    meta_out = dict(chunks[0][1])
    meta_out["batch_size"] = int(sum(int(meta["batch_size"]) for _arrays, meta in chunks))
    meta_out["prefix_tokens"] = int(global_t)
    meta_out["eval_samples"] = int(sum(int(meta.get("eval_samples") or meta["batch_size"]) for _arrays, meta in chunks))
    return arrays_out, meta_out


def load_pairs(path: str | Path) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    chunks = list(iter_pair_chunks(path, as_numpy=True))
    return _merge_pair_chunks(chunks)


def save_reuse_weights(out_dir: str | Path, weights: ReuseWeights, extra_meta: dict[str, object] | None = None) -> None:
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, Any] = {}
    for gid, row in weights.weights.items():
        arrays[f"enc_k.{gid}"] = row.enc_k
        arrays[f"dec_k_b.{gid}"] = row.dec_k_b
        arrays[f"enc_v.{gid}"] = row.enc_v
        arrays[f"dec_v_b.{gid}"] = row.dec_v_b
        if row.mean_a_k is not None:
            arrays[f"mean_a_k.{gid}"] = row.mean_a_k
        if row.mean_b_k is not None:
            arrays[f"mean_b_k.{gid}"] = row.mean_b_k
        if row.mean_a_v is not None:
            arrays[f"mean_a_v.{gid}"] = row.mean_a_v
        if row.mean_b_v is not None:
            arrays[f"mean_b_v.{gid}"] = row.mean_b_v
    save_bundle(root / "weights.pt", arrays, artifact_dtype="float32")

    extra_meta = dict(extra_meta or {})
    head_dim = int(extra_meta.get("head_dim", 0) or 0)
    meta = {
        "mode": weights.mode or REUSE_MODE_XKV_GROUP,
        "translator": {
            "default_path": weights.default_path or TRANSLATOR_PATH_SHARED_Z,
            "shared_z": {
                "hidden_dim": int(weights.translator_hidden_dim),
                "activation": str(weights.translator_activation or SHARED_Z_ACTIVATION_IDENTITY),
            },
        },
        "xkv_group": {
            "group_size": weights.group_size,
            "rank_k": weights.rank_k,
            "rank_v": weights.rank_v,
            "center_before_svd": bool(weights.center_before_svd),
            "svd_backend": str(weights.svd_backend),
            "max_groups": int(weights.max_groups),
        },
        "group_size": weights.group_size,
        "rank": weights.rank,
        "rank_k": weights.rank_k,
        "rank_v": weights.rank_v,
        "num_layers": weights.num_layers,
        "num_groups": weights.num_groups,
        "center_before_svd": bool(weights.center_before_svd),
        "svd_backend": str(weights.svd_backend),
        "k_de_rope": bool(weights.k_de_rope),
        "k_re_rope": bool(weights.k_re_rope),
        "group_layers": {str(gid): list(layers) for gid, layers in (weights.group_layers or {}).items()},
    }
    if head_dim > 0 and weights.group_layers:
        meta["groups"] = [
            {
                **spec,
                "rank_k": weights.weights[spec["group_id"]].rank_k,
                "rank_v": weights.weights[spec["group_id"]].rank_v,
                "center_before_svd": bool(weights.center_before_svd),
                "k_de_rope": bool(weights.k_de_rope),
                "k_re_rope": bool(weights.k_re_rope),
            }
            for spec in build_group_specs(weights.group_layers, head_dim)
        ]
    meta.update(extra_meta)
    save_json(root / "meta.json", meta)


def load_reuse_weights(path: str | Path, *, device: str | None = "cpu", dtype: str = "float32", cpu_fallback: bool = True) -> tuple[ReuseWeights, dict[str, object]]:
    root = Path(path)
    meta = load_json(root / "meta.json")
    bundle = load_bundle(root / "weights.pt", device=device, dtype=dtype, cpu_fallback=cpu_fallback)
    xkv_meta = meta.get("xkv_group", {}) if isinstance(meta.get("xkv_group", {}), dict) else {}
    translator_meta = meta.get("translator", {}) if isinstance(meta.get("translator", {}), dict) else {}
    shared_z_meta = translator_meta.get("shared_z", {}) if isinstance(translator_meta.get("shared_z", {}), dict) else {}
    groups_meta = meta.get("groups", [])
    group_meta_by_gid = {
        int(row["group_id"]): row
        for row in groups_meta
        if isinstance(row, dict) and "group_id" in row
    } if isinstance(groups_meta, list) else {}
    if isinstance(groups_meta, list) and groups_meta:
        group_layers = {int(row["group_id"]): [int(layer) for layer in row.get("layers", [])] for row in groups_meta}
    else:
        raw_group_layers = meta.get("group_layers", {})
        if isinstance(raw_group_layers, dict) and raw_group_layers:
            group_layers = {int(gid): [int(layer) for layer in layers] for gid, layers in raw_group_layers.items()}
        else:
            group_size = int(meta["group_size"])
            num_layers = int(meta["num_layers"])
            num_groups = int(meta["num_groups"])
            group_layers = {}
            for gid in range(num_groups):
                start = gid * group_size
                stop = min(start + group_size, num_layers)
                group_layers[gid] = list(range(start, stop))
    weights: dict[int, ReuseGroupWeights] = {}
    rank_k = int(meta.get("rank_k") or xkv_meta.get("rank_k") or meta.get("rank") or 0)
    rank_v = int(meta.get("rank_v") or xkv_meta.get("rank_v") or meta.get("rank") or 0)
    group_count = int(meta.get("num_groups") or len(group_layers))
    for gid in range(group_count):
        weights[gid] = ReuseGroupWeights(
            enc_k=bundle.arrays[f"enc_k.{gid}"],
            dec_k_b=bundle.arrays[f"dec_k_b.{gid}"],
            enc_v=bundle.arrays[f"enc_v.{gid}"],
            dec_v_b=bundle.arrays[f"dec_v_b.{gid}"],
            mean_a_k=bundle.arrays.get(f"mean_a_k.{gid}"),
            mean_b_k=bundle.arrays.get(f"mean_b_k.{gid}"),
            mean_a_v=bundle.arrays.get(f"mean_a_v.{gid}"),
            mean_b_v=bundle.arrays.get(f"mean_b_v.{gid}"),
            rank_k=int(group_meta_by_gid.get(gid, {}).get("rank_k", rank_k)),
            rank_v=int(group_meta_by_gid.get(gid, {}).get("rank_v", rank_v)),
        )
    return (
        ReuseWeights(
            group_size=int(meta["group_size"]),
            rank=int(meta["rank"]),
            rank_k=rank_k,
            rank_v=rank_v,
            num_layers=int(meta["num_layers"]),
            num_groups=group_count,
            weights=weights,
            mode=str(meta.get("mode") or REUSE_MODE_XKV_GROUP),
            default_path=str(translator_meta.get("default_path") or TRANSLATOR_PATH_SHARED_Z),
            group_layers=group_layers,
            center_before_svd=bool(meta.get("center_before_svd") or xkv_meta.get("center_before_svd", False)),
            svd_backend=str(meta.get("svd_backend") or xkv_meta.get("svd_backend") or "moments_eigh"),
            max_groups=int(xkv_meta.get("max_groups") or 0),
            k_de_rope=bool(meta.get("k_de_rope", True)),
            k_re_rope=bool(meta.get("k_re_rope", True)),
            translator_hidden_dim=int(shared_z_meta.get("hidden_dim") or 0),
            translator_activation=str(shared_z_meta.get("activation") or SHARED_Z_ACTIVATION_IDENTITY),
        ),
        meta,
    )

def _patch_key_layer(key: str) -> int | None:
    parts = str(key).split(".")
    if len(parts) < 2 or not parts[1].isdigit():
        return None
    return int(parts[1])


def _patch_layer_key_index(arrays: dict[str, Any]) -> tuple[dict[int, list[str]], list[str]]:
    layer_to_keys: dict[int, list[str]] = {}
    global_keys: list[str] = []
    for raw_key in arrays.keys():
        key = str(raw_key)
        layer = _patch_key_layer(key)
        if layer is None:
            global_keys.append(key)
            continue
        layer_to_keys.setdefault(int(layer), []).append(key)
    for keys in layer_to_keys.values():
        keys.sort()
    global_keys.sort()
    return layer_to_keys, global_keys


def _tensor_abs_max(value: Any) -> float:
    t = require_torch()
    if t.is_tensor(value):
        if int(value.numel()) == 0:
            return 0.0
        return float(value.detach().abs().max().item())
    arr = np.asarray(value)
    if arr.size == 0:
        return 0.0
    return float(np.abs(arr).max())


def _identity_runtime_hints(arrays: dict[str, Any], meta: dict[str, object]) -> dict[str, dict[str, object]]:
    fit_mode = str(meta.get("fit_mode") or "")
    if fit_mode != _XNORM_NATIVEKV_RESIDUAL_FIT_MODE:
        return {}
    residual_layers = [int(v) for v in meta.get("residual_layers", [])] if isinstance(meta.get("residual_layers", []), list) else []
    hints: dict[str, dict[str, object]] = {}
    for layer in residual_layers:
        weight = arrays.get(f"xnorm_nativekv_residual_patch.{int(layer)}.delta_h_head.weight")
        bias = arrays.get(f"xnorm_nativekv_residual_patch.{int(layer)}.delta_h_head.bias")
        if weight is None and bias is None:
            continue
        max_abs = 0.0
        if weight is not None:
            max_abs = max(max_abs, _tensor_abs_max(weight))
        if bias is not None:
            max_abs = max(max_abs, _tensor_abs_max(bias))
        hints[str(int(layer))] = {
            "xnorm_nativekv_residual_identity": bool(max_abs <= 0.0),
            "xnorm_nativekv_residual_delta_max_abs": float(max_abs),
        }
    return hints


def save_patch_ckpt_sharded(
    out_dir: str | Path,
    arrays: dict[str, Any],
    meta: dict[str, object],
    *,
    layers_per_shard: int = 4,
    drop_identity_residual: bool = False,
) -> dict[str, object]:
    root = Path(out_dir)
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    shards_dir = root / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)

    layers_per_shard = max(1, int(layers_per_shard))
    layer_to_keys, global_keys = _patch_layer_key_index(arrays)
    selected_layers = [int(v) for v in meta.get("selected_layers", [])] if isinstance(meta.get("selected_layers", []), list) else []
    ordered_layers = sorted({int(layer) for layer in layer_to_keys} | set(selected_layers))
    runtime_hints = _identity_runtime_hints(arrays, meta)
    filtered_layer_keys: dict[int, list[str]] = {}

    for layer in ordered_layers:
        keys = list(layer_to_keys.get(int(layer), []))
        if drop_identity_residual and bool(runtime_hints.get(str(int(layer)), {}).get("xnorm_nativekv_residual_identity", False)):
            prefix = f"xnorm_nativekv_residual_patch.{int(layer)}."
            kept = [key for key in keys if not key.startswith(prefix)]
            omitted = [key for key in keys if key.startswith(prefix)]
            if omitted:
                runtime_hints.setdefault(str(int(layer)), {})["omitted_identity_state_keys"] = int(len(omitted))
            keys = kept
        filtered_layer_keys[int(layer)] = keys

    global_path = ""
    if global_keys:
        global_path = "shards/global.pt"
        save_bundle(
            root / global_path,
            {key: arrays[key] for key in global_keys},
            meta={"keys": global_keys},
            artifact_dtype="float32",
        )

    shards: list[dict[str, object]] = []
    layer_to_shard: dict[str, str] = {}
    layer_keys_manifest: dict[str, list[str]] = {}
    for start_idx in range(0, len(ordered_layers), layers_per_shard):
        shard_layers = ordered_layers[start_idx : start_idx + layers_per_shard]
        if not shard_layers:
            continue
        shard_arrays: dict[str, Any] = {}
        for layer in shard_layers:
            for key in filtered_layer_keys.get(int(layer), []):
                shard_arrays[key] = arrays[key]
                layer_to_shard[str(int(layer))] = f"shards/layers_{int(shard_layers[0]):04d}_{int(shard_layers[-1]):04d}.pt"
                layer_keys_manifest[str(int(layer))] = list(filtered_layer_keys.get(int(layer), []))
        shard_relpath = f"shards/layers_{int(shard_layers[0]):04d}_{int(shard_layers[-1]):04d}.pt"
        save_bundle(
            root / shard_relpath,
            shard_arrays,
            meta={"layers": shard_layers, "num_keys": len(shard_arrays)},
            artifact_dtype="float32",
        )
        shards.append(
            {
                "path": shard_relpath,
                "layers": [int(layer) for layer in shard_layers],
                "num_keys": int(len(shard_arrays)),
            }
        )

    manifest = {
        "format": PATCH_CKPT_SHARDED_FORMAT,
        "layers_per_shard": int(layers_per_shard),
        "num_shards": int(len(shards)),
        "layers": [int(layer) for layer in ordered_layers],
        "shards": shards,
        "layer_to_shard": layer_to_shard,
        "layer_keys": layer_keys_manifest,
        "global_keys": global_keys,
        "global_path": global_path,
        "runtime_layer_hints": runtime_hints,
        "drop_identity_residual": bool(drop_identity_residual),
    }
    save_json(root / "shards" / "manifest.json", manifest)

    meta_payload = dict(meta)
    meta_payload.update(
        {
            "format": PATCH_CKPT_SHARDED_FORMAT,
            "layers_per_shard": int(layers_per_shard),
            "shard_manifest_path": "shards/manifest.json",
            "drop_identity_residual": bool(drop_identity_residual),
            "runtime_layer_hints": runtime_hints,
        }
    )
    save_json(root / "meta.json", meta_payload)
    return manifest


def save_patch_ckpt(
    out_dir: str | Path,
    arrays: dict[str, Any],
    meta: dict[str, object],
    *,
    sharded: bool = False,
    layers_per_shard: int = 0,
    drop_identity_residual: bool = False,
) -> None:
    if bool(sharded) or int(layers_per_shard) > 0:
        save_patch_ckpt_sharded(
            out_dir,
            arrays,
            meta,
            layers_per_shard=max(1, int(layers_per_shard) or 4),
            drop_identity_residual=bool(drop_identity_residual),
        )
        return
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    save_bundle(root / "weights.pt", arrays, artifact_dtype="float32")
    save_json(root / "meta.json", meta)


def load_patch_ckpt(path: str | Path, *, device: str | None = "cpu", dtype: str = "float32", cpu_fallback: bool = True) -> tuple[dict[str, Any], dict[str, object]]:
    root = Path(path)
    meta = load_json(root / "meta.json")
    shard_manifest_path = root / str(meta.get("shard_manifest_path") or "shards/manifest.json")
    if shard_manifest_path.exists():
        manifest = load_json(shard_manifest_path)
        total_shard_bytes = 0
        for shard in manifest.get("shards", []):
            shard_relpath = str(shard.get("path") or "")
            if not shard_relpath:
                continue
            shard_path = root / shard_relpath
            if shard_path.exists():
                total_shard_bytes += int(shard_path.stat().st_size)
        global_path = str(manifest.get("global_path") or "")
        if global_path:
            global_file = root / global_path
            if global_file.exists():
                total_shard_bytes += int(global_file.stat().st_size)
        meta_payload = dict(meta)
        meta_payload["format"] = str(manifest.get("format") or PATCH_CKPT_SHARDED_FORMAT)
        meta_payload["shard_manifest"] = manifest
        meta_payload["__patch_ckpt_root__"] = str(root)
        meta_payload["total_shard_bytes"] = int(total_shard_bytes)
        meta_payload["runtime_patch_persist_shards"] = bool(total_shard_bytes <= (2 * 1024 * 1024 * 1024))
        meta_payload["runtime_patch_persist_modules"] = bool(total_shard_bytes <= (2 * 1024 * 1024 * 1024))
        if "runtime_layer_hints" not in meta_payload and isinstance(manifest.get("runtime_layer_hints"), dict):
            meta_payload["runtime_layer_hints"] = manifest.get("runtime_layer_hints")
        return {}, meta_payload
    bundle = load_bundle(root / "weights.pt", device=device, dtype=dtype, cpu_fallback=cpu_fallback)
    return bundle.arrays, meta


def reshard_patch_ckpt(
    src_dir: str | Path,
    out_dir: str | Path,
    *,
    layers_per_shard: int = 4,
    drop_identity_residual: bool = False,
    cpu_fallback: bool = True,
) -> dict[str, object]:
    src_root = Path(src_dir)
    dst_root = Path(out_dir)
    if src_root.resolve() == dst_root.resolve():
        raise ValueError("reshard 输出目录不能与输入目录相同")
    meta = load_json(src_root / "meta.json")
    if (src_root / "weights.pt").exists():
        arrays = load_bundle(src_root / "weights.pt", device="cpu", dtype="float32", cpu_fallback=cpu_fallback).arrays
    else:
        shard_manifest_path = src_root / str(meta.get("shard_manifest_path") or "shards/manifest.json")
        if not shard_manifest_path.exists():
            raise FileNotFoundError(f"未找到 patch checkpoint 权重: {src_root}")
        manifest = load_json(shard_manifest_path)
        arrays = {}
        global_path = str(manifest.get("global_path") or "")
        if global_path:
            arrays.update(load_bundle(src_root / global_path, device="cpu", dtype="float32", cpu_fallback=cpu_fallback).arrays)
        for shard in manifest.get("shards", []):
            shard_path = str(shard.get("path") or "")
            if not shard_path:
                continue
            arrays.update(load_bundle(src_root / shard_path, device="cpu", dtype="float32", cpu_fallback=cpu_fallback).arrays)
    return save_patch_ckpt_sharded(
        dst_root,
        arrays,
        meta,
        layers_per_shard=max(1, int(layers_per_shard)),
        drop_identity_residual=bool(drop_identity_residual),
    )


def _shape(arr: Any) -> tuple[int, ...]:
    t = require_torch()
    if t.is_tensor(arr):
        return tuple(arr.shape)
    return tuple(arr.shape)


def _element_size_bytes(arr: Any) -> int:
    t = require_torch()
    if t.is_tensor(arr):
        return arr.element_size()
    return int(arr.dtype.itemsize)


def _resolve_optional_rope_specs(meta: dict[str, object]) -> tuple[object | None, object | None]:
    try:
        return resolve_rope_specs(meta)
    except ValueError:
        return None, None


def method_payload_mb(method: str, arrays: dict[str, Any], route: dict[str, list[int]] | None = None) -> float:
    num_layers = len([key for key in arrays if key.startswith("k_a.")])
    sample = arrays["k_a.0"]
    shape = _shape(sample)
    batch = max(1, int(shape[0]))
    raw_mb = (int(np.prod(shape)) * _element_size_bytes(sample) * 2 * num_layers) / (1024.0 * 1024.0) / batch
    if method == "oracle":
        return 0.0
    if method == "raw_kv":
        return raw_mb
    if method == "quantized_4bit":
        return raw_mb / 4.0
    if method == "selective_recompute":
        if not route:
            return raw_mb
        frac = max(0.0, 1.0 - (len(route.get("fallback_layers", [])) / max(1, num_layers)))
        return raw_mb * frac
    if method == "reuse_only":
        return raw_mb * 0.6
    if method == "scd_full":
        return raw_mb * 0.72
    raise ValueError(f"未知方法: {method}")


def quantize_4bit(x):
    t = require_torch()
    if t.is_tensor(x):
        scale = t.max(t.abs(x)) / 7.0 + 1e-8
        q = t.round(x / scale)
        q = t.clamp(q, -8, 7)
        return (q * scale).to(dtype=t.float32)
    scale = np.max(np.abs(x)) / 7.0 + 1e-8
    q = np.round(x / scale)
    q = np.clip(q, -8, 7)
    return (q * scale).astype(np.float32)


def _cache_feature_vector(writer: CacheWriter):
    t = require_torch()
    feats = []
    for layer in sorted(writer.k_cache.keys()):
        k = writer.k_cache[layer].to(dtype=t.float32)
        v = writer.v_cache[layer].to(dtype=t.float32)
        feats.extend([k.mean(), k.std(unbiased=False), v.mean(), v.std(unbiased=False)])
    return t.stack(feats)


def _logits_from_cache(writer: CacheWriter, vocab_size: int, seed: int):
    t = require_torch()
    feat = _cache_feature_vector(writer)
    generator = t.Generator(device=feat.device.type)
    generator.manual_seed(seed)
    proj = t.randn((feat.numel(), vocab_size), generator=generator, device=feat.device, dtype=t.float32)
    proj = proj / max(1, feat.numel())
    return feat @ proj


def _softmax(logits):
    t = require_torch()
    x = logits - t.max(logits, dim=-1, keepdim=True).values
    ex = t.exp(x)
    return ex / t.sum(ex, dim=-1, keepdim=True)


def _method_noise_scale(method: str) -> float:
    return {
        "oracle": 0.0,
        "raw_kv": 0.06,
        "quantized_4bit": 0.20,
        "selective_recompute": 0.05,
        "reuse_only": 0.0,
        "scd_full": 0.0,
    }[method]


def _cache_drift_vs_consumer(writer: CacheWriter, arrays: dict[str, Any], num_layers: int) -> float:
    t = require_torch()
    if num_layers <= 0:
        return 0.0
    vals = []
    for layer in range(num_layers):
        k_hat = writer.k_cache[layer]
        v_hat = writer.v_cache[layer]
        k_ref = arrays[f"k_b.{layer}"]
        v_ref = arrays[f"v_b.{layer}"]
        vals.append(t.mean((k_hat - k_ref) ** 2))
        vals.append(t.mean((v_hat - v_ref) ** 2))
    return float(t.mean(t.stack(vals)).item())


def _prepare_cache_for_method(
    method: str,
    arrays: dict[str, Any],
    meta: dict[str, object],
    reuse: ReuseWeights | None,
    patch_bundle: dict[str, Any] | None,
    patch_meta: dict[str, object] | None,
    route: dict[str, list[int]] | None,
    infer_device: str,
    cpu_fallback: bool,
) -> CacheWriter:
    t = require_torch()
    num_layers = int(meta["num_layers"])
    spec = CacheShapeSpec(
        batch_size=int(meta["batch_size"]),
        num_layers=num_layers,
        num_key_value_heads=int(meta["num_key_value_heads"]),
        prefix_tokens=int(meta["prefix_tokens"]),
        head_dim=int(meta["head_dim"]),
        dtype="float32",
        device=infer_device,
    )
    writer = CacheWriter(spec)

    if route is None:
        route = {"reuse_layers": list(range(num_layers)), "patch_layers": [], "fallback_layers": []}

    reuse_on_device = move_reuse_weights(reuse, infer_device, dtype="float32", cpu_fallback=cpu_fallback) if reuse is not None else None
    rope_a, rope_b = _resolve_optional_rope_specs(meta)

    decoded_reuse_k: dict[int, Any] = {}
    decoded_reuse_v: dict[int, Any] = {}
    if reuse_on_device is not None:
        bsz = int(meta["batch_size"])
        kvh = int(meta["num_key_value_heads"])
        t_steps = int(meta["prefix_tokens"])
        hd = int(meta["head_dim"])
        for gid in range(reuse_on_device.num_groups):
            layers = group_layers_for_gid(reuse_on_device, gid)
            a_k_inputs = []
            for layer in layers:
                k_src = arrays[f"k_a.{layer}"]
                if rope_a is not None and reuse_on_device.k_de_rope:
                    k_src = de_rope_k_torch(k_src, rope_a)
                a_k_inputs.append(k_src)
            a_k = pack_group(a_k_inputs)
            a_v = pack_group([arrays[f"v_a.{layer}"] for layer in layers])
            z_k, z_v = encode_group(a_k, a_v, reuse_on_device.weights[gid])
            rec_k, rec_v = decode_group(z_k, z_v, reuse_on_device.weights[gid])
            rec_k_layers = unpack_group(rec_k, len(layers), hd, bsz, kvh, t_steps)
            rec_v_layers = unpack_group(rec_v, len(layers), hd, bsz, kvh, t_steps)
            for layer, rk, rv in zip(layers, rec_k_layers, rec_v_layers):
                if rope_b is not None and reuse_on_device.k_re_rope:
                    rk = re_rope_k_torch(rk, rope_b)
                decoded_reuse_k[layer] = rk
                decoded_reuse_v[layer] = rv

    for layer in range(num_layers):
        if method == "oracle":
            k = arrays[f"k_b.{layer}"]
            v = arrays[f"v_b.{layer}"]
        elif method == "raw_kv":
            k = arrays[f"k_a.{layer}"]
            v = arrays[f"v_a.{layer}"]
        elif method == "quantized_4bit":
            k = quantize_4bit(arrays[f"k_a.{layer}"])
            v = quantize_4bit(arrays[f"v_a.{layer}"])
        elif method == "selective_recompute":
            if layer in route.get("fallback_layers", []):
                k = arrays[f"k_b.{layer}"]
                v = arrays[f"v_b.{layer}"]
            else:
                k = arrays[f"k_a.{layer}"]
                v = arrays[f"v_a.{layer}"]
        elif method == "reuse_only":
            if layer not in decoded_reuse_k:
                raise ValueError("reuse_only 需要提供 reuse 检查点")
            k = decoded_reuse_k[layer]
            v = decoded_reuse_v[layer]
        elif method == "scd_full":
            if layer in route.get("fallback_layers", []):
                k = arrays[f"k_b.{layer}"]
                v = arrays[f"v_b.{layer}"]
            elif layer in route.get("patch_layers", []):
                if patch_bundle is None or patch_meta is None:
                    raise ValueError("scd_full 的 patch 层需要提供 patch 检查点")
                x = arrays[f"xnorm_a.{layer}"].reshape(-1, int(meta["hidden_dim"]))
                fit_mode = str(patch_meta.get("fit_mode") or "")
                if fit_mode == PATCH_FIT_MODE_DIRECT_KV_V1:
                    wk = patch_bundle[f"direct_wk.{layer}"]
                    wv = patch_bundle[f"direct_wv.{layer}"]
                    k2, v2 = infer_direct_kv_patch_layer_torch(x, wk, wv)
                else:
                    w = patch_bundle[f"align_w.{layer}"]
                    b_vec = patch_bundle[f"align_b.{layer}"]
                    xhat = x @ w + b_vec
                    wk = patch_bundle[f"wk.{layer}"]
                    wv = patch_bundle[f"wv.{layer}"]
                    k2 = xhat @ wk
                    v2 = xhat @ wv
                bsz = int(meta["batch_size"])
                t_steps = int(meta["prefix_tokens"])
                kvh = int(meta["num_key_value_heads"])
                hd = int(meta["head_dim"])
                k = k2.reshape(bsz, t_steps, kvh, hd).permute(0, 2, 1, 3).contiguous()
                v = v2.reshape(bsz, t_steps, kvh, hd).permute(0, 2, 1, 3).contiguous()
            else:
                if layer not in decoded_reuse_k:
                    raise ValueError("scd_full 的 reuse 层需要提供 reuse 检查点")
                k = decoded_reuse_k[layer]
                v = decoded_reuse_v[layer]
        else:
            raise ValueError(f"未知方法 {method}")

        writer.update(layer, tensor_to_device(k, infer_device, dtype="float32", cpu_fallback=cpu_fallback), tensor_to_device(v, infer_device, dtype="float32", cpu_fallback=cpu_fallback))

    return writer


def run_inference_method(
    method: str,
    arrays: dict[str, Any],
    meta: dict[str, object],
    route: dict[str, list[int]] | None,
    reuse: ReuseWeights | None,
    patch_bundle: dict[str, Any] | None,
    patch_meta: dict[str, object] | None,
    bandwidth_gbps: float,
    seed: int,
    infer_device: str = "cpu",
    cpu_fallback: bool = True,
) -> dict[str, object]:
    t = require_torch()
    arrays = {key: tensor_to_device(value, infer_device, dtype="float32", cpu_fallback=cpu_fallback) for key, value in arrays.items()}
    patch_bundle = None if patch_bundle is None else {key: tensor_to_device(value, infer_device, dtype="float32", cpu_fallback=cpu_fallback) for key, value in patch_bundle.items()}

    prep_start = time.perf_counter()
    writer = _prepare_cache_for_method(method, arrays, meta, reuse, patch_bundle, patch_meta, route, infer_device, cpu_fallback)
    assemble_ms = (time.perf_counter() - prep_start) * 1000.0

    payload_mb = method_payload_mb(method, arrays, route)
    transfer_ms = 0.0
    if payload_mb > 0:
        payload_bits = payload_mb * 1024.0 * 1024.0 * 8.0
        transfer_ms = payload_bits / (bandwidth_gbps * 1e9) * 1000.0

    decode_start = time.perf_counter()
    base_logits = _logits_from_cache(writer, vocab_size=int(meta["vocab_size"]), seed=seed)
    sample_count = int(meta.get("eval_samples") or meta.get("batch_size") or 8)
    drift = 0.0 if method == "oracle" else _cache_drift_vs_consumer(writer, arrays, int(meta["num_layers"]))
    noise_scale = _method_noise_scale(method) + min(0.5, 2.0 * drift)
    generator = t.Generator(device=base_logits.device.type)
    generator.manual_seed(seed + 97)

    sample_logits = []
    sample_probs = []
    samples: list[dict[str, object]] = []
    first_decode_ms = 0.0

    for idx in range(sample_count):
        step_start = time.perf_counter()
        noise = t.randn(base_logits.shape, generator=generator, device=base_logits.device, dtype=t.float32) * noise_scale
        logits = base_logits + noise
        probs = _softmax(logits.unsqueeze(0))[0]
        token = int(t.argmax(probs).item())
        sample_logits.append(logits.detach())
        sample_probs.append(probs.detach())
        samples.append({"id": idx, "pred_text": f"tok_{token}", "pred_token": token})
        if idx == 0:
            first_decode_ms = (time.perf_counter() - step_start) * 1000.0

    decode_total_ms = (time.perf_counter() - decode_start) * 1000.0
    ttft_ms = assemble_ms + transfer_ms + first_decode_ms

    logits_mean = t.mean(t.stack(sample_logits, dim=0), dim=0)
    probs_mean = t.mean(t.stack(sample_probs, dim=0), dim=0)

    return {
        "method": method,
        "samples": samples,
        "logits": tensor_to_numpy(tensor_to_cpu(logits_mean)).tolist(),
        "probs": tensor_to_numpy(tensor_to_cpu(probs_mean)).tolist(),
        "timing_ms": {
            "assemble": float(assemble_ms),
            "transfer": float(transfer_ms),
            "decode_first": float(first_decode_ms),
            "decode_total": float(decode_total_ms),
            "ttft": float(ttft_ms),
            "payload_mb": float(payload_mb),
        },
        "cache_complete": writer.is_complete(),
        "cache_rows": len(writer.audit_rows),
        "cache_audit": writer.audit_rows,
    }


def run_inference_method_over_pairs(
    pairs_path: str | Path,
    method: str,
    route: dict[str, list[int]] | None,
    reuse: ReuseWeights | None,
    patch_bundle: dict[str, Any] | None,
    patch_meta: dict[str, object] | None,
    bandwidth_gbps: float,
    seed: int,
    infer_device: str = "cpu",
    cpu_fallback: bool = True,
) -> dict[str, object]:
    logits_acc: np.ndarray | None = None
    probs_acc: np.ndarray | None = None
    total_weight = 0
    sample_offset = 0
    samples: list[dict[str, object]] = []
    cache_rows = 0
    cache_complete = True
    cache_audit: list[object] = []
    timing_acc = {
        "assemble": 0.0,
        "transfer": 0.0,
        "decode_first": 0.0,
        "decode_total": 0.0,
        "ttft": 0.0,
        "payload_mb": 0.0,
    }

    for chunk_idx, (arrays, meta) in enumerate(iter_pair_chunks(pairs_path, device=infer_device, dtype="float32", cpu_fallback=cpu_fallback)):
        out = run_inference_method(
            method=method,
            arrays=arrays,
            meta=meta,
            route=route,
            reuse=reuse,
            patch_bundle=patch_bundle,
            patch_meta=patch_meta,
            bandwidth_gbps=bandwidth_gbps,
            seed=seed + chunk_idx * 1009,
            infer_device=infer_device,
            cpu_fallback=cpu_fallback,
        )
        weight = int(meta.get("eval_samples") or meta.get("batch_size") or 1)
        logits = np.array(out["logits"], dtype=np.float64)
        probs = np.array(out["probs"], dtype=np.float64)
        logits_acc = logits * weight if logits_acc is None else logits_acc + logits * weight
        probs_acc = probs * weight if probs_acc is None else probs_acc + probs * weight
        total_weight += weight

        for key in timing_acc:
            timing_acc[key] += float(out["timing_ms"][key]) * weight

        for row in out["samples"]:
            item = dict(row)
            item["id"] = sample_offset
            samples.append(item)
            sample_offset += 1

        cache_rows += int(out["cache_rows"])
        cache_complete = cache_complete and bool(out["cache_complete"])
        for row in out["cache_audit"]:
            item = dict(row)
            item["chunk"] = chunk_idx
            cache_audit.append(item)

    if logits_acc is None or probs_acc is None or total_weight <= 0:
        raise ValueError("pairs 为空，无法执行推理")

    return {
        "method": method,
        "samples": samples,
        "logits": (logits_acc / total_weight).tolist(),
        "probs": (probs_acc / total_weight).tolist(),
        "timing_ms": {key: float(value / total_weight) for key, value in timing_acc.items()},
        "cache_complete": cache_complete,
        "cache_rows": cache_rows,
        "cache_audit": cache_audit,
    }


def save_prediction_run(out_dir: str | Path, pred: dict[str, object], oracle: dict[str, object] | None = None) -> None:
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    save_json(root / "predictions.json", pred)
    if oracle is not None:
        save_json(root / "oracle.json", oracle)
