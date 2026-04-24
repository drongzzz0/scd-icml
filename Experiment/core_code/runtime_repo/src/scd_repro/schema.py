from __future__ import annotations

from pathlib import Path
from typing import Any

from .io_utils import load_bundle, load_json


class ValidationError(ValueError):
    pass


def _expect_keys(obj: dict[str, Any], required: list[str], context: str) -> None:
    missing = [key for key in required if key not in obj]
    if missing:
        raise ValidationError(f"{context}: 缺少必填键 {missing}")


def validate_pairs(path: str | Path) -> dict[str, Any]:
    root = Path(path)
    manifest = load_json(root / "manifest.json")
    _expect_keys(
        manifest,
        [
            "format",
            "artifact_dtype",
            "num_layers",
            "num_key_value_heads",
            "head_dim",
            "hidden_dim",
            "layout",
            "chunks",
            "num_chunks",
        ],
        "pairs/manifest.json",
    )
    if manifest["format"] != "chunked_pt_v1":
        raise ValidationError(f"pairs/manifest.json: 不支持的格式 {manifest['format']}")
    if manifest["layout"] != "B,KVH,T,HD":
        raise ValidationError(f"pairs/manifest.json: layout 必须为 B,KVH,T,HD，实际为 {manifest['layout']}")

    chunks = manifest.get("chunks", [])
    if len(chunks) != int(manifest["num_chunks"]):
        raise ValidationError("pairs/manifest.json: chunks 数量与 num_chunks 不一致")
    if not chunks:
        raise ValidationError("pairs/manifest.json: chunks 不能为空")

    for idx, row in enumerate(chunks):
        _expect_keys(row, ["path", "batch_size", "prefix_tokens", "eval_samples", "num_layers"], f"pairs/chunks[{idx}]")
        bundle = load_bundle(root / str(row["path"]), as_numpy=True)
        meta = dict(bundle.meta)
        arrays = bundle.arrays
        if int(meta.get("num_layers", row["num_layers"])) != int(manifest["num_layers"]):
            raise ValidationError(f"pairs/chunks[{idx}]: num_layers 与根 manifest 不一致")
        expected_b = int(meta.get("batch_size", row["batch_size"]))
        expected_t = int(meta.get("prefix_tokens", row["prefix_tokens"]))
        expected_k = (expected_b, int(manifest["num_key_value_heads"]), expected_t, int(manifest["head_dim"]))
        expected_x = (expected_b, expected_t, int(manifest["hidden_dim"]))
        expected_ids = (expected_b, expected_t)
        for layer in range(int(manifest["num_layers"])):
            for key in (f"k_a.{layer}", f"v_a.{layer}", f"k_b.{layer}", f"v_b.{layer}"):
                if key not in arrays:
                    raise ValidationError(f"pairs/chunks[{idx}]: 缺少键 {key}")
                if tuple(arrays[key].shape) != expected_k:
                    raise ValidationError(f"pairs/chunks[{idx}]: {key} 形状错误，期望 {expected_k}，实际 {tuple(arrays[key].shape)}")
            for key in (f"xnorm_a.{layer}", f"xnorm_b.{layer}"):
                if key not in arrays:
                    raise ValidationError(f"pairs/chunks[{idx}]: 缺少键 {key}")
                if tuple(arrays[key].shape) != expected_x:
                    raise ValidationError(f"pairs/chunks[{idx}]: {key} 形状错误，期望 {expected_x}，实际 {tuple(arrays[key].shape)}")
            for key in (f"resid_a.{layer}", f"resid_b.{layer}"):
                if key in arrays and tuple(arrays[key].shape) != expected_x:
                    raise ValidationError(f"pairs/chunks[{idx}]: {key} 形状错误，期望 {expected_x}，实际 {tuple(arrays[key].shape)}")
        for key in ("input_ids", "attention_mask", "next_token_ids"):
            if key in arrays and tuple(arrays[key].shape) != expected_ids:
                raise ValidationError(f"pairs/chunks[{idx}]: {key} 形状错误，期望 {expected_ids}，实际 {tuple(arrays[key].shape)}")
    return manifest


def validate_reuse_ckpt(path: str | Path) -> dict[str, Any]:
    root = Path(path)
    meta = load_json(root / "meta.json")
    _expect_keys(meta, ["group_size", "rank", "num_layers", "num_groups", "head_dim", "num_key_value_heads"], "reuse/meta.json")
    if "mode" in meta and str(meta["mode"]) != "xkv_group":
        raise ValidationError(f"reuse/meta.json: 仅支持 mode=xkv_group，实际为 {meta['mode']}")
    translator = meta.get("translator", {})
    if isinstance(translator, dict) and translator and str(translator.get("default_path") or "shared_z") != "shared_z":
        raise ValidationError(f"reuse/meta.json: 仅支持 translator.default_path=shared_z，实际为 {translator.get('default_path')}")
    bundle = load_bundle(root / "weights.pt", as_numpy=True)
    for gid in range(int(meta["num_groups"])):
        for prefix in ("enc_k.", "dec_k_b.", "enc_v.", "dec_v_b."):
            key = f"{prefix}{gid}"
            if key not in bundle.arrays:
                raise ValidationError(f"reuse/weights.pt: 缺少键 {key}")
            if len(bundle.arrays[key].shape) != 2:
                raise ValidationError(f"reuse/weights.pt: {key} 必须是 2D 矩阵")
        for prefix in ("mean_a_k.", "mean_b_k.", "mean_a_v.", "mean_b_v."):
            key = f"{prefix}{gid}"
            if key in bundle.arrays and len(bundle.arrays[key].shape) != 1:
                raise ValidationError(f"reuse/weights.pt: {key} 必须是 1D 向量")
    return meta


def validate_patch_ckpt(path: str | Path) -> dict[str, Any]:
    root = Path(path)
    meta = load_json(root / "meta.json")
    _expect_keys(meta, ["selected_layers", "hidden_dim", "code_dim"], "patch/meta.json")
    bundle = load_bundle(root / "weights.pt", as_numpy=True)
    fit_mode = str(meta.get("fit_mode") or "")
    selected_layers = [int(layer) for layer in meta["selected_layers"]]
    if fit_mode == "xnorm_semantic_v1":
        _expect_keys(
            meta,
            [
                "mlp_hidden_dim",
                "num_key_value_heads",
                "head_dim",
                "k_space",
                "semantic_source",
                "semantic_probe_shared",
                "semantic_probe_frozen",
                "semantic_probe_arch",
                "semantic_probe_version",
                "semantic_target_space",
                "loss_weights",
                "tau",
                "enc_h_is_identity",
            ],
            "patch/meta.json",
        )
        enc_h_is_identity = bool(meta.get("enc_h_is_identity", False))
        for layer in selected_layers:
            if not enc_h_is_identity:
                key = f"enc_h.{layer}"
                if key not in bundle.arrays:
                    raise ValidationError(f"patch/weights.pt: 缺少键 {key}")
            for key in (
                f"patch_mlp.{layer}.fc1.weight",
                f"patch_mlp.{layer}.fc1.bias",
                f"patch_mlp.{layer}.fc2.weight",
                f"patch_mlp.{layer}.fc2.bias",
            ):
                if key not in bundle.arrays:
                    raise ValidationError(f"patch/weights.pt: 缺少键 {key}")
        if str(meta.get("semantic_source")) == "probe":
            if bool(meta.get("semantic_probe_shared", True)):
                probe_prefixes = (
                    "semantic_probe.fc1.weight",
                    "semantic_probe.fc1.bias",
                    "semantic_probe.fc2.weight",
                    "semantic_probe.fc2.bias",
                )
                for key in probe_prefixes:
                    if key not in bundle.arrays:
                        raise ValidationError(f"patch/weights.pt: 缺少键 {key}")
            else:
                for layer in selected_layers:
                    for key in (
                        f"semantic_probe.{layer}.fc1.weight",
                        f"semantic_probe.{layer}.fc1.bias",
                        f"semantic_probe.{layer}.fc2.weight",
                        f"semantic_probe.{layer}.fc2.bias",
                    ):
                        if key not in bundle.arrays:
                            raise ValidationError(f"patch/weights.pt: 缺少键 {key}")
        if str(meta.get("semantic_target_space") or "") == "next_token_logits":
            _expect_keys(meta, ["vocab_size"], "patch/meta.json")
            for key in ("semantic_vocab_head.weight", "semantic_vocab_head.bias"):
                if key not in bundle.arrays:
                    raise ValidationError(f"patch/weights.pt: 缺少键 {key}")
        return meta

    for layer in selected_layers:
        if fit_mode in {"direct_kv_v1", "direct_kv_knorm_v1"}:
            prefixes = ("direct_wk.", "direct_wv.")
        else:
            prefixes = ("align_w.", "align_b.", "wk.", "wv.")
        for prefix in prefixes:
            key = f"{prefix}{layer}"
            if key not in bundle.arrays:
                raise ValidationError(f"patch/weights.pt: 缺少键 {key}")
    return meta


def validate_route(path: str | Path) -> dict[str, Any]:
    route = load_json(path)
    _expect_keys(route, ["reuse_layers", "patch_layers"], "route.json")
    return route


def validate_predictions(path: str | Path) -> dict[str, Any]:
    pred = load_json(path)
    _expect_keys(pred, ["method", "samples", "timing_ms"], "predictions.json")
    return pred
