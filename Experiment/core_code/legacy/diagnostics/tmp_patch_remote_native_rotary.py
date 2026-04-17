from pathlib import Path


ROOT = Path("/disk_n/c101-3-1/projects/scd-repro-codex-mistral-instruct-20260323")


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if old not in text:
        raise RuntimeError(f"{label}: target block not found")
    return text.replace(old, new, 1)


def main() -> None:
    path = ROOT / "src/scd_repro/strict_runtime.py"
    text = path.read_text(encoding="utf-8")
    text = replace_once(
        text,
        """def _apply_consumer_k_norm_to_flat_keys(k_pre_flat: Any, k_norm: Any | None, kvh: int, hd: int) -> Any:\n    t = require_torch()\n    if k_norm is None:\n        return k_pre_flat.to(dtype=t.float32)\n    k_pre_flat = k_pre_flat.to(dtype=t.float32)\n    norm_in = k_pre_flat.reshape(-1, kvh, hd)\n    norm_dtype = getattr(getattr(k_norm, "weight", None), "dtype", norm_in.dtype)\n    with t.no_grad():\n        normed = k_norm(norm_in.to(dtype=norm_dtype)).to(dtype=t.float32)\n    return normed.reshape(-1, kvh * hd).contiguous()\n\n\n""",
        """def _apply_consumer_k_norm_to_flat_keys(k_pre_flat: Any, k_norm: Any | None, kvh: int, hd: int) -> Any:\n    t = require_torch()\n    if k_norm is None:\n        return k_pre_flat.to(dtype=t.float32)\n    k_pre_flat = k_pre_flat.to(dtype=t.float32)\n    norm_in = k_pre_flat.reshape(-1, kvh, hd)\n    norm_dtype = getattr(getattr(k_norm, "weight", None), "dtype", norm_in.dtype)\n    with t.no_grad():\n        normed = k_norm(norm_in.to(dtype=norm_dtype)).to(dtype=t.float32)\n    return normed.reshape(-1, kvh * hd).contiguous()\n\n\ndef _apply_consumer_native_rotary_to_patch_keys(\n    *,\n    k_states: Any,\n    layer: int,\n    consumer_session: TransformersRuntimeSession,\n    attention_mask_prefix: Any,\n    hidden_dim: int,\n) -> Any:\n    t = require_torch()\n    layers = consumer_session.layers if hasattr(consumer_session, "layers") else _get_decoder_layers(consumer_session.model)\n    layer_module = layers[int(layer)]\n    attn = getattr(layer_module, "self_attn", None) or getattr(layer_module, "attn", None)\n    if attn is None:\n        return re_rope_k_torch(k_states, consumer_session.rope_spec)\n    try:\n        import importlib\n\n        attn_mod = importlib.import_module(attn.__class__.__module__)\n        apply_rotary_pos_emb = getattr(attn_mod, "apply_rotary_pos_emb", None)\n    except Exception:\n        apply_rotary_pos_emb = None\n    if apply_rotary_pos_emb is None:\n        return re_rope_k_torch(k_states, consumer_session.rope_spec)\n    seq_len = int(k_states.shape[2])\n    model_dtype = getattr(consumer_session.model, "dtype", t.float32)\n    dummy_hidden = t.zeros((1, seq_len, hidden_dim), device=consumer_session.target_device, dtype=model_dtype)\n    layer_inputs = _build_layer_forward_inputs(\n        consumer_session,\n        dummy_hidden,\n        attention_mask_prefix[:, :seq_len],\n    )\n    position_embeddings = layer_inputs.get("position_embeddings")\n    if not isinstance(position_embeddings, tuple) or len(position_embeddings) < 2:\n        return re_rope_k_torch(k_states, consumer_session.rope_spec)\n    cos, sin = position_embeddings\n    rotary_dtype = getattr(getattr(attn, "k_proj", None), "weight", k_states).dtype\n    k_states = k_states.to(device=consumer_session.target_device, dtype=rotary_dtype)\n    q_dummy = t.zeros_like(k_states)\n    with t.no_grad():\n        _q_rot, k_rot = apply_rotary_pos_emb(q_dummy, k_states, cos, sin)\n    return k_rot.to(dtype=t.float32)\n\n\n""",
        "strict_runtime helper insert",
    )
    text = replace_once(
        text,
        """        elif fit_mode == PATCH_FIT_MODE_DIRECT_KV_KNORM_V1:\n            wk = patch_bundle[f"direct_wk.{layer}"]\n            wv = patch_bundle[f"direct_wv.{layer}"]\n            k_pre_flat, vb_flat = infer_direct_kv_patch_layer_torch(xa, wk, wv)\n            _k_proj, _v_proj, k_norm = _get_consumer_attn_modules(consumer_session, int(layer))\n            kb_raw_flat = _apply_consumer_k_norm_to_flat_keys(k_pre_flat, k_norm, kvh, hd)\n            kb_raw = kb_raw_flat.reshape(1, prefix_tokens, kv_dim).reshape(1, prefix_tokens, kvh, hd).permute(0, 2, 1, 3).contiguous()\n            vb = vb_flat.reshape(1, prefix_tokens, kv_dim).reshape(1, prefix_tokens, kvh, hd).permute(0, 2, 1, 3).contiguous()\n        elif fit_mode == PATCH_FIT_MODE_XNORM_SEMANTIC_V1:\n""",
        """        elif fit_mode == PATCH_FIT_MODE_DIRECT_KV_KNORM_V1:\n            wk = patch_bundle[f"direct_wk.{layer}"]\n            wv = patch_bundle[f"direct_wv.{layer}"]\n            k_pre_flat, vb_flat = infer_direct_kv_patch_layer_torch(xa, wk, wv)\n            _k_proj, _v_proj, k_norm = _get_consumer_attn_modules(consumer_session, int(layer))\n            kb_norm_flat = _apply_consumer_k_norm_to_flat_keys(k_pre_flat, k_norm, kvh, hd)\n            kb_norm = kb_norm_flat.reshape(1, prefix_tokens, kv_dim).reshape(1, prefix_tokens, kvh, hd).permute(0, 2, 1, 3).contiguous()\n            vb = vb_flat.reshape(1, prefix_tokens, kv_dim).reshape(1, prefix_tokens, kvh, hd).permute(0, 2, 1, 3).contiguous()\n            payload_bytes += int(payload_units) * _dtype_size_bytes(transfer_dtype)\n            kb = _apply_consumer_native_rotary_to_patch_keys(\n                k_states=kb_norm,\n                layer=layer,\n                consumer_session=consumer_session,\n                attention_mask_prefix=consumer_capture.attention_mask,\n                hidden_dim=hidden_dim,\n            )\n            caches[layer] = (kb.to(device=consumer_device, dtype=consumer_dtype), vb.to(device=consumer_device, dtype=consumer_dtype))\n            continue\n        elif fit_mode == PATCH_FIT_MODE_XNORM_SEMANTIC_V1:\n""",
        "strict_runtime direct_kv_knorm branch",
    )
    path.write_text(text, encoding="utf-8")
    print("patched strict_runtime native rotary path")


if __name__ == "__main__":
    main()
