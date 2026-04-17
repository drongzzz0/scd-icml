from pathlib import Path


ROOT = Path("/disk_n/c101-3-1/projects/scd-repro-codex-mistral-instruct-20260323")


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if old not in text:
        raise RuntimeError(f"{label}: target block not found")
    return text.replace(old, new, 1)


def patch_train_patch() -> None:
    path = ROOT / "src/scd_repro/cli/train_patch.py"
    text = path.read_text(encoding="utf-8")
    text = replace_once(
        text,
        'from ..direct_kv_residual_patch import PATCH_FIT_MODE_DIRECT_KV_KNORM_RESIDUAL_MLP_V1, train_direct_kv_knorm_residual_patch\n',
        'from ..direct_kv_residual_patch import PATCH_FIT_MODE_DIRECT_KV_KNORM_RESIDUAL_MLP_V1, PATCH_FIT_MODE_XNORM_NATIVEKV_RESIDUAL_MLP_V1, train_direct_kv_knorm_residual_patch, train_xnorm_nativekv_residual_patch\n',
        "train_patch import xnorm nativekv residual",
    )
    text = replace_once(
        text,
        """    if fit_mode == PATCH_FIT_MODE_DIRECT_KV_KNORM_RESIDUAL_MLP_V1:\n        train_direct_kv_knorm_residual_patch(\n            args=args,\n            logger=logger,\n            cfg=cfg,\n            meta=meta,\n            profile=profile,\n            route=route,\n            rope_b=rope_b,\n            device=device,\n            accel=accel,\n        )\n        return\n""",
        """    if fit_mode == PATCH_FIT_MODE_DIRECT_KV_KNORM_RESIDUAL_MLP_V1:\n        train_direct_kv_knorm_residual_patch(\n            args=args,\n            logger=logger,\n            cfg=cfg,\n            meta=meta,\n            profile=profile,\n            route=route,\n            rope_b=rope_b,\n            device=device,\n            accel=accel,\n        )\n        return\n    if fit_mode == PATCH_FIT_MODE_XNORM_NATIVEKV_RESIDUAL_MLP_V1:\n        train_xnorm_nativekv_residual_patch(\n            args=args,\n            logger=logger,\n            cfg=cfg,\n            meta=meta,\n            profile=profile,\n            route=route,\n            rope_b=rope_b,\n            device=device,\n            accel=accel,\n        )\n        return\n""",
        "train_patch dispatch xnorm nativekv residual",
    )
    path.write_text(text, encoding="utf-8")


def patch_strict_runtime() -> None:
    path = ROOT / "src/scd_repro/strict_runtime.py"
    text = path.read_text(encoding="utf-8")
    text = replace_once(
        text,
        'from .direct_kv_residual_patch import PATCH_FIT_MODE_DIRECT_KV_KNORM_RESIDUAL_MLP_V1, infer_direct_kv_residual_patch_layer_torch, load_direct_kv_residual_patch_layer_from_ckpt\n',
        'from .direct_kv_residual_patch import PATCH_FIT_MODE_DIRECT_KV_KNORM_RESIDUAL_MLP_V1, PATCH_FIT_MODE_XNORM_NATIVEKV_RESIDUAL_MLP_V1, infer_direct_kv_residual_patch_layer_torch, infer_xnorm_nativekv_residual_patch_layer_torch, load_direct_kv_residual_patch_layer_from_ckpt, load_xnorm_nativekv_residual_patch_layer_from_ckpt\n',
        "strict_runtime import xnorm nativekv residual",
    )
    text = replace_once(
        text,
        """def _apply_semantic_patch_layer(\n    *,\n    layer: int,\n    xa: Any,\n    patch_bundle: dict[str, Any],\n    patch_meta: dict[str, object],\n    consumer_session: TransformersRuntimeSession,\n    kvh: int,\n    hd: int,\n) -> tuple[Any, Any, int]:\n""",
        """def _load_runtime_xnorm_nativekv_residual_module(\n    *,\n    layer: int,\n    patch_bundle: dict[str, Any],\n    patch_meta: dict[str, object],\n    consumer_session: TransformersRuntimeSession,\n):\n    runtime_modules = patch_meta.setdefault(\"__runtime_xnorm_nativekv_residual_modules__\", {})  # type: ignore[assignment]\n    if not isinstance(runtime_modules, dict):\n        runtime_modules = {}\n        patch_meta[\"__runtime_xnorm_nativekv_residual_modules__\"] = runtime_modules\n    if int(layer) not in runtime_modules:\n        runtime_modules[int(layer)] = load_xnorm_nativekv_residual_patch_layer_from_ckpt(\n            patch_bundle,\n            patch_meta,\n            layer=int(layer),\n            device=consumer_session.target_device,\n            dtype=\"float32\",\n        )\n    return runtime_modules[int(layer)]\n\n\ndef _apply_semantic_patch_layer(\n    *,\n    layer: int,\n    xa: Any,\n    patch_bundle: dict[str, Any],\n    patch_meta: dict[str, object],\n    consumer_session: TransformersRuntimeSession,\n    kvh: int,\n    hd: int,\n) -> tuple[Any, Any, int]:\n""",
        "strict_runtime helper xnorm nativekv residual",
    )
    text = replace_once(
        text,
        """        elif fit_mode == PATCH_FIT_MODE_DIRECT_KV_KNORM_RESIDUAL_MLP_V1:\n            wk = patch_bundle[f\"direct_wk.{layer}\"]\n            wv = patch_bundle[f\"direct_wv.{layer}\"]\n            residual_layers = [int(v) for v in patch_meta.get(\"residual_layers\", [])] if isinstance(patch_meta.get(\"residual_layers\", []), list) else []\n            if int(layer) in residual_layers:\n                residual_module = _load_runtime_direct_residual_module(\n                    layer=layer,\n                    patch_bundle=patch_bundle,\n                    patch_meta=patch_meta,\n                    consumer_session=consumer_session,\n                )\n                k_pre_flat, vb_flat = infer_direct_kv_residual_patch_layer_torch(xa, wk, wv, residual_module)\n            else:\n                k_pre_flat, vb_flat = infer_direct_kv_patch_layer_torch(xa, wk, wv)\n            _k_proj, _v_proj, k_norm = _get_consumer_attn_modules(consumer_session, int(layer))\n            kb_norm_flat = _apply_consumer_k_norm_to_flat_keys(k_pre_flat, k_norm, kvh, hd)\n            kb_norm = kb_norm_flat.reshape(1, prefix_tokens, kv_dim).reshape(1, prefix_tokens, kvh, hd).permute(0, 2, 1, 3).contiguous()\n            vb = vb_flat.reshape(1, prefix_tokens, kv_dim).reshape(1, prefix_tokens, kvh, hd).permute(0, 2, 1, 3).contiguous()\n            payload_bytes += int(payload_units) * _dtype_size_bytes(transfer_dtype)\n            kb = _apply_consumer_native_rotary_to_patch_keys(\n                k_states=kb_norm,\n                layer=layer,\n                consumer_session=consumer_session,\n                attention_mask_prefix=consumer_capture.attention_mask,\n                hidden_dim=hidden_dim,\n            )\n            caches[layer] = (kb.to(device=consumer_device, dtype=consumer_dtype), vb.to(device=consumer_device, dtype=consumer_dtype))\n            continue\n        elif fit_mode == PATCH_FIT_MODE_XNORM_SEMANTIC_V1:\n""",
        """        elif fit_mode == PATCH_FIT_MODE_DIRECT_KV_KNORM_RESIDUAL_MLP_V1:\n            wk = patch_bundle[f\"direct_wk.{layer}\"]\n            wv = patch_bundle[f\"direct_wv.{layer}\"]\n            residual_layers = [int(v) for v in patch_meta.get(\"residual_layers\", [])] if isinstance(patch_meta.get(\"residual_layers\", []), list) else []\n            if int(layer) in residual_layers:\n                residual_module = _load_runtime_direct_residual_module(\n                    layer=layer,\n                    patch_bundle=patch_bundle,\n                    patch_meta=patch_meta,\n                    consumer_session=consumer_session,\n                )\n                k_pre_flat, vb_flat = infer_direct_kv_residual_patch_layer_torch(xa, wk, wv, residual_module)\n            else:\n                k_pre_flat, vb_flat = infer_direct_kv_patch_layer_torch(xa, wk, wv)\n            _k_proj, _v_proj, k_norm = _get_consumer_attn_modules(consumer_session, int(layer))\n            kb_norm_flat = _apply_consumer_k_norm_to_flat_keys(k_pre_flat, k_norm, kvh, hd)\n            kb_norm = kb_norm_flat.reshape(1, prefix_tokens, kv_dim).reshape(1, prefix_tokens, kvh, hd).permute(0, 2, 1, 3).contiguous()\n            vb = vb_flat.reshape(1, prefix_tokens, kv_dim).reshape(1, prefix_tokens, kvh, hd).permute(0, 2, 1, 3).contiguous()\n            payload_bytes += int(payload_units) * _dtype_size_bytes(transfer_dtype)\n            kb = _apply_consumer_native_rotary_to_patch_keys(\n                k_states=kb_norm,\n                layer=layer,\n                consumer_session=consumer_session,\n                attention_mask_prefix=consumer_capture.attention_mask,\n                hidden_dim=hidden_dim,\n            )\n            caches[layer] = (kb.to(device=consumer_device, dtype=consumer_dtype), vb.to(device=consumer_device, dtype=consumer_dtype))\n            continue\n        elif fit_mode == PATCH_FIT_MODE_XNORM_NATIVEKV_RESIDUAL_MLP_V1:\n            wk = patch_bundle[f\"direct_wk.{layer}\"]\n            wv = patch_bundle[f\"direct_wv.{layer}\"]\n            residual_layers = [int(v) for v in patch_meta.get(\"residual_layers\", [])] if isinstance(patch_meta.get(\"residual_layers\", []), list) else []\n            _k_proj, _v_proj, k_norm = _get_consumer_attn_modules(consumer_session, int(layer))\n            if int(layer) in residual_layers:\n                residual_module = _load_runtime_xnorm_nativekv_residual_module(\n                    layer=layer,\n                    patch_bundle=patch_bundle,\n                    patch_meta=patch_meta,\n                    consumer_session=consumer_session,\n                )\n                xb_hat = infer_xnorm_nativekv_residual_patch_layer_torch(xa, residual_module)\n                proj_dtype = getattr(getattr(_k_proj, \"weight\", None), \"dtype\", xb_hat.dtype)\n                xb_proj = xb_hat.to(device=consumer_device, dtype=proj_dtype)\n                k_pre_flat = _k_proj(xb_proj).to(dtype=t.float32)\n                vb_flat = _v_proj(xb_proj).to(dtype=t.float32)\n            else:\n                k_pre_flat, vb_flat = infer_direct_kv_patch_layer_torch(xa, wk, wv)\n            kb_norm_flat = _apply_consumer_k_norm_to_flat_keys(k_pre_flat, k_norm, kvh, hd)\n            kb_norm = kb_norm_flat.reshape(1, prefix_tokens, kv_dim).reshape(1, prefix_tokens, kvh, hd).permute(0, 2, 1, 3).contiguous()\n            vb = vb_flat.reshape(1, prefix_tokens, kv_dim).reshape(1, prefix_tokens, kvh, hd).permute(0, 2, 1, 3).contiguous()\n            payload_bytes += int(payload_units) * _dtype_size_bytes(transfer_dtype)\n            kb = _apply_consumer_native_rotary_to_patch_keys(\n                k_states=kb_norm,\n                layer=layer,\n                consumer_session=consumer_session,\n                attention_mask_prefix=consumer_capture.attention_mask,\n                hidden_dim=hidden_dim,\n            )\n            caches[layer] = (kb.to(device=consumer_device, dtype=consumer_dtype), vb.to(device=consumer_device, dtype=consumer_dtype))\n            continue\n        elif fit_mode == PATCH_FIT_MODE_XNORM_SEMANTIC_V1:\n""",
        "strict_runtime branch xnorm nativekv residual",
    )
    path.write_text(text, encoding="utf-8")


def main() -> None:
    patch_train_patch()
    patch_strict_runtime()
    print("patched remote repo for xnorm_nativekv_residual_mlp_v1")


if __name__ == "__main__":
    main()
