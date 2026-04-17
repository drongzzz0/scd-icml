from pathlib import Path
import re


ROOT = Path("/disk_n/c101-3-1/projects/scd-repro-codex-mistral-instruct-20260323")


def replace_once(text: str, old: str, new: str, *, label: str) -> str:
    if old not in text:
        raise RuntimeError(f"{label}: target block not found")
    return text.replace(old, new, 1)


def re_replace_once(text: str, pattern: str, repl: str, *, label: str) -> str:
    out, n = re.subn(pattern, repl, text, count=1, flags=re.S)
    if n != 1:
        raise RuntimeError(f"{label}: pattern replacement failed")
    return out


def patch_patch_py() -> None:
    path = ROOT / "src/scd_repro/patch.py"
    text = path.read_text(encoding="utf-8")
    text = replace_once(
        text,
        'PATCH_FIT_MODE_DIRECT_KV_V1 = "direct_kv_v1"\nPATCH_FIT_MODE_XNORM_SEMANTIC_V1 = "xnorm_semantic_v1"\n',
        'PATCH_FIT_MODE_DIRECT_KV_V1 = "direct_kv_v1"\nPATCH_FIT_MODE_DIRECT_KV_KNORM_V1 = "direct_kv_knorm_v1"\nPATCH_FIT_MODE_XNORM_SEMANTIC_V1 = "xnorm_semantic_v1"\n',
        label="patch.py constants",
    )
    path.write_text(text, encoding="utf-8")


def patch_train_patch_py() -> None:
    path = ROOT / "src/scd_repro/cli/train_patch.py"
    text = path.read_text(encoding="utf-8")
    text = replace_once(
        text,
        '    PATCH_FIT_MODE_DIRECT_KV_V1,\n    PATCH_FIT_MODE_XNORM_SEMANTIC_V1,\n',
        '    PATCH_FIT_MODE_DIRECT_KV_V1,\n    PATCH_FIT_MODE_DIRECT_KV_KNORM_V1,\n    PATCH_FIT_MODE_XNORM_SEMANTIC_V1,\n',
        label="train_patch.py imports",
    )
    text = replace_once(
        text,
        """def _coerce_consumer_native_components(value: ConsumerNativeComponents | dict[str, Any]) -> ConsumerNativeComponents:\n    if isinstance(value, ConsumerNativeComponents):\n        return value\n    return ConsumerNativeComponents(\n        k_proj_weight=value["k_proj_weight"],\n        k_proj_bias=value.get("k_proj_bias"),\n        v_proj_weight=value["v_proj_weight"],\n        v_proj_bias=value.get("v_proj_bias"),\n    )\n\n\n""",
        """def _coerce_consumer_native_components(value: ConsumerNativeComponents | dict[str, Any]) -> ConsumerNativeComponents:\n    if isinstance(value, ConsumerNativeComponents):\n        return value\n    return ConsumerNativeComponents(\n        k_proj_weight=value["k_proj_weight"],\n        k_proj_bias=value.get("k_proj_bias"),\n        v_proj_weight=value["v_proj_weight"],\n        v_proj_bias=value.get("v_proj_bias"),\n    )\n\n\ndef _project_consumer_native_kv_from_xnorm_b(\n    arrays: dict[str, torch.Tensor],\n    layer: int,\n    hidden_dim: int,\n    device: torch.device,\n    native_components: ConsumerNativeComponents | dict[str, Any],\n) -> tuple[torch.Tensor, torch.Tensor]:\n    components = _coerce_consumer_native_components(native_components)\n    xb = arrays[f"xnorm_b.{layer}"].reshape(-1, hidden_dim).to(device=device, dtype=torch.float32)\n    k_weight = components.k_proj_weight.to(device=device, dtype=torch.float32)\n    v_weight = components.v_proj_weight.to(device=device, dtype=torch.float32)\n    k_bias = None if components.k_proj_bias is None else components.k_proj_bias.to(device=device, dtype=torch.float32)\n    v_bias = None if components.v_proj_bias is None else components.v_proj_bias.to(device=device, dtype=torch.float32)\n    kb_raw = F.linear(xb, k_weight, k_bias)\n    vb = F.linear(xb, v_weight, v_bias)\n    return kb_raw, vb\n\n\n""",
        label="train_patch.py helper insert",
    )
    text = re_replace_once(
        text,
        r"def _accumulate_direct_kv_layer_stats\(.*?\n\n\ndef _accumulate_direct_kv_error_layer_stats\(",
        """def _accumulate_direct_kv_layer_stats(\n    arrays: dict[str, torch.Tensor],\n    layer: int,\n    hidden_dim: int,\n    kv_dim: int,\n    device: torch.device,\n    stats: dict[str, torch.Tensor],\n    rope_b,\n    native_components: ConsumerNativeComponents | None = None,\n) -> None:\n    from ..patch import accumulate_direct_kv_stats_torch\n\n    xa = arrays[f"xnorm_a.{layer}"].reshape(-1, hidden_dim).to(device=device, dtype=torch.float32)\n    if native_components is None:\n        kb_raw = de_rope_k_torch(arrays[f"k_b.{layer}"], rope_b).permute(0, 2, 1, 3).reshape(-1, kv_dim).to(device=device, dtype=torch.float32)\n        vb = arrays[f"v_b.{layer}"].permute(0, 2, 1, 3).reshape(-1, kv_dim).to(device=device, dtype=torch.float32)\n    else:\n        kb_raw, vb = _project_consumer_native_kv_from_xnorm_b(arrays, layer, hidden_dim, device, native_components)\n    accumulate_direct_kv_stats_torch(xa, kb_raw, vb, stats)\n\n\ndef _accumulate_direct_kv_error_layer_stats(""",
        label="train_patch.py layer stats",
    )
    text = re_replace_once(
        text,
        r"def _accumulate_direct_kv_error_layer_stats\(.*?\n\n\ndef _accumulate_direct_kv_layers_with_retry\(",
        """def _accumulate_direct_kv_error_layer_stats(\n    arrays: dict[str, torch.Tensor],\n    layer: int,\n    hidden_dim: int,\n    kv_dim: int,\n    device: torch.device,\n    stats: dict[str, torch.Tensor],\n    rope_b,\n    wk: torch.Tensor,\n    wv: torch.Tensor,\n    native_components: ConsumerNativeComponents | None = None,\n) -> None:\n    from ..patch import accumulate_direct_kv_error_stats_torch\n\n    xa = arrays[f"xnorm_a.{layer}"].reshape(-1, hidden_dim).to(device=device, dtype=torch.float32)\n    if native_components is None:\n        kb_raw = de_rope_k_torch(arrays[f"k_b.{layer}"], rope_b).permute(0, 2, 1, 3).reshape(-1, kv_dim).to(device=device, dtype=torch.float32)\n        vb = arrays[f"v_b.{layer}"].permute(0, 2, 1, 3).reshape(-1, kv_dim).to(device=device, dtype=torch.float32)\n    else:\n        kb_raw, vb = _project_consumer_native_kv_from_xnorm_b(arrays, layer, hidden_dim, device, native_components)\n    accumulate_direct_kv_error_stats_torch(xa, kb_raw, vb, wk, wv, stats)\n\n\ndef _accumulate_direct_kv_layers_with_retry(""",
        label="train_patch.py error stats",
    )
    text = re_replace_once(
        text,
        r"def _accumulate_direct_kv_layers_with_retry\(.*?\n\n\ndef _accumulate_direct_kv_error_layers_with_retry\(",
        """def _accumulate_direct_kv_layers_with_retry(\n    arrays: dict[str, torch.Tensor],\n    layers: list[int],\n    hidden_dim: int,\n    kv_dim: int,\n    device: torch.device,\n    stats_by_layer: dict[int, dict[str, torch.Tensor]],\n    rope_b,\n    logger,\n    depth: int = 0,\n    native_components_by_layer: dict[int, ConsumerNativeComponents] | None = None,\n) -> None:\n    try:\n        for layer in layers:\n            native_components = None if native_components_by_layer is None else native_components_by_layer[int(layer)]\n            _accumulate_direct_kv_layer_stats(arrays, layer, hidden_dim, kv_dim, device, stats_by_layer[layer], rope_b, native_components)\n    except RuntimeError as exc:\n        text = str(exc).lower()\n        if "out of memory" not in text and "cuda oom" not in text:\n            raise\n        clear_torch_cache()\n        if len(layers) <= 1:\n            raise\n        mid = len(layers) // 2\n        left = layers[:mid]\n        right = layers[mid:]\n        logger.warning("patch KV 统计触发 OOM，按层块二分重试 depth=%d size=%d -> %d + %d", depth, len(layers), len(left), len(right))\n        _accumulate_direct_kv_layers_with_retry(arrays, left, hidden_dim, kv_dim, device, stats_by_layer, rope_b, logger, depth + 1, native_components_by_layer)\n        _accumulate_direct_kv_layers_with_retry(arrays, right, hidden_dim, kv_dim, device, stats_by_layer, rope_b, logger, depth + 1, native_components_by_layer)\n\n\ndef _accumulate_direct_kv_error_layers_with_retry(""",
        label="train_patch.py stats retry",
    )
    text = re_replace_once(
        text,
        r"def _accumulate_direct_kv_error_layers_with_retry\(.*?\n\n\ndef _train_direct_kv_patch\(",
        """def _accumulate_direct_kv_error_layers_with_retry(\n    arrays: dict[str, torch.Tensor],\n    layers: list[int],\n    hidden_dim: int,\n    kv_dim: int,\n    device: torch.device,\n    stats_by_layer: dict[int, dict[str, torch.Tensor]],\n    rope_b,\n    wk_by_layer: dict[int, torch.Tensor],\n    wv_by_layer: dict[int, torch.Tensor],\n    logger,\n    depth: int = 0,\n    native_components_by_layer: dict[int, ConsumerNativeComponents] | None = None,\n) -> None:\n    try:\n        for layer in layers:\n            native_components = None if native_components_by_layer is None else native_components_by_layer[int(layer)]\n            _accumulate_direct_kv_error_layer_stats(\n                arrays,\n                layer,\n                hidden_dim,\n                kv_dim,\n                device,\n                stats_by_layer[layer],\n                rope_b,\n                wk_by_layer[layer],\n                wv_by_layer[layer],\n                native_components,\n            )\n    except RuntimeError as exc:\n        text = str(exc).lower()\n        if "out of memory" not in text and "cuda oom" not in text:\n            raise\n        clear_torch_cache()\n        if len(layers) <= 1:\n            raise\n        mid = len(layers) // 2\n        left = layers[:mid]\n        right = layers[mid:]\n        logger.warning("patch 误差统计触发 OOM，按层块二分重试 depth=%d size=%d -> %d + %d", depth, len(layers), len(left), len(right))\n        _accumulate_direct_kv_error_layers_with_retry(arrays, left, hidden_dim, kv_dim, device, stats_by_layer, rope_b, wk_by_layer, wv_by_layer, logger, depth + 1, native_components_by_layer)\n        _accumulate_direct_kv_error_layers_with_retry(arrays, right, hidden_dim, kv_dim, device, stats_by_layer, rope_b, wk_by_layer, wv_by_layer, logger, depth + 1, native_components_by_layer)\n\n\ndef _train_direct_kv_patch(""",
        label="train_patch.py error retry",
    )
    text = re_replace_once(
        text,
        r"def _train_direct_kv_patch\(\n    \*,\n    args,\n    logger,\n    cfg: dict\[str, Any\],\n    meta: dict\[str, Any\],\n    profile: dict\[str, Any\],\n    route: dict\[str, Any\] \| None,\n    rope_b,\n    device: torch.device,\n    accel: dict\[str, Any\],\n\) -> None:.*?logger\.info\(\"补丁检查点已保存到 %s\", args\.out\)\n",
        """def _train_direct_kv_patch(\n    *,\n    args,\n    logger,\n    cfg: dict[str, Any],\n    meta: dict[str, Any],\n    profile: dict[str, Any],\n    route: dict[str, Any] | None,\n    rope_b,\n    device: torch.device,\n    accel: dict[str, Any],\n    fit_mode: str,\n) -> None:\n    from ..patch import finalize_direct_kv_error_stats_torch\n\n    patch_cfg = cfg.get("patch", {})\n    patch_budget = int(patch_cfg["patch_budget"])\n    ridge_lambda = float(patch_cfg["ridge_lambda"])\n    selected = _select_layers(profile, patch_budget, route=route)\n    hidden_dim = int(meta["hidden_dim"])\n    kv_dim = int(meta["num_key_value_heads"]) * int(meta["head_dim"])\n    layer_block_size = int(accel["layer_block_size"])\n    use_consumer_native_targets = fit_mode == PATCH_FIT_MODE_DIRECT_KV_KNORM_V1\n\n    logger.info(\n        "patch 训练开始: fit_mode=%s device=%s accel=%s layer_block_size=%d selected=%s",\n        fit_mode,\n        device,\n        accel["accel_mode"],\n        layer_block_size,\n        selected,\n    )\n\n    native_components_by_layer: dict[int, ConsumerNativeComponents] | None = None\n    if use_consumer_native_targets:\n        native_components_by_layer = _load_consumer_native_components(\n            args.models_config,\n            selected,\n            hidden_dim,\n            kv_dim,\n            device,\n            "float32",\n        )\n        logger.info("direct-kv knorm-aware 训练目标: consumer pre-k_norm K + native V")\n\n    direct_stats_by_layer: dict[int, dict[str, torch.Tensor]] = {layer: create_direct_kv_stats_torch(hidden_dim, kv_dim, device) for layer in selected}\n    chunk_paths = list(_iter_chunk_paths(args.pairs))\n\n    start = time.perf_counter()\n    for idx, chunk_path in enumerate(chunk_paths, start=1):\n        bundle = load_bundle(chunk_path, device=str(device), dtype="float32")\n        arrays = bundle.arrays\n        for start_idx in range(0, len(selected), layer_block_size):\n            block_layers = selected[start_idx : start_idx + layer_block_size]\n            _accumulate_direct_kv_layers_with_retry(\n                arrays,\n                block_layers,\n                hidden_dim,\n                kv_dim,\n                device,\n                direct_stats_by_layer,\n                rope_b,\n                logger,\n                native_components_by_layer=native_components_by_layer,\n            )\n        del arrays, bundle\n        clear_torch_cache()\n        if idx == 1 or idx % 5 == 0 or idx == len(chunk_paths):\n            elapsed = time.perf_counter() - start\n            eta_s = (elapsed / idx) * (len(chunk_paths) - idx) if idx > 0 else 0.0\n            logger.info("patch KV 统计进度: chunk %d/%d, elapsed=%.1fs, eta=%.1fs", idx, len(chunk_paths), elapsed, eta_s)\n\n    out_arrays: dict[str, object] = {}\n    wk_by_layer: dict[int, torch.Tensor] = {}\n    wv_by_layer: dict[int, torch.Tensor] = {}\n    for layer in selected:\n        wk, wv, _patch_k_formula, _patch_v_formula, _patch_kv_formula = solve_direct_kv_from_stats_torch(\n            direct_stats_by_layer[layer],\n            ridge_lambda=ridge_lambda,\n        )\n        wk_by_layer[layer] = wk.to(device=device, dtype=torch.float32)\n        wv_by_layer[layer] = wv.to(device=device, dtype=torch.float32)\n\n    del direct_stats_by_layer\n    clear_torch_cache()\n\n    error_stats_by_layer: dict[int, dict[str, torch.Tensor]] = {layer: create_direct_kv_error_stats_torch(device) for layer in selected}\n    start = time.perf_counter()\n    for idx, chunk_path in enumerate(chunk_paths, start=1):\n        bundle = load_bundle(chunk_path, device=str(device), dtype="float32")\n        arrays = bundle.arrays\n        for start_idx in range(0, len(selected), layer_block_size):\n            block_layers = selected[start_idx : start_idx + layer_block_size]\n            _accumulate_direct_kv_error_layers_with_retry(\n                arrays,\n                block_layers,\n                hidden_dim,\n                kv_dim,\n                device,\n                error_stats_by_layer,\n                rope_b,\n                wk_by_layer,\n                wv_by_layer,\n                logger,\n                native_components_by_layer=native_components_by_layer,\n            )\n        del arrays, bundle\n        clear_torch_cache()\n        if idx == 1 or idx % 5 == 0 or idx == len(chunk_paths):\n            elapsed = time.perf_counter() - start\n            eta_s = (elapsed / idx) * (len(chunk_paths) - idx) if idx > 0 else 0.0\n            logger.info("patch 误差统计进度: chunk %d/%d, elapsed=%.1fs, eta=%.1fs", idx, len(chunk_paths), elapsed, eta_s)\n\n    for layer in selected:\n        wk = wk_by_layer[layer]\n        wv = wv_by_layer[layer]\n        patch_k_error, patch_v_error, patch_kv_error = finalize_direct_kv_error_stats_torch(error_stats_by_layer[layer], kv_dim=kv_dim)\n        out_arrays[f"direct_wk.{layer}"] = wk.detach().to(device="cpu", dtype=torch.float32).contiguous()\n        out_arrays[f"direct_wv.{layer}"] = wv.detach().to(device="cpu", dtype=torch.float32).contiguous()\n        logger.info(\n            "已完成补丁层训练: layer=%d patch_k_error=%.6f patch_v_error=%.6f patch_kv_error=%.6f",\n            layer,\n            patch_k_error,\n            patch_v_error,\n            patch_kv_error,\n        )\n\n    save_patch_ckpt(\n        args.out,\n        out_arrays,\n        {\n            "selected_layers": selected,\n            "hidden_dim": hidden_dim,\n            "code_dim": hidden_dim,\n            "num_key_value_heads": int(meta["num_key_value_heads"]),\n            "head_dim": int(meta["head_dim"]),\n            "k_space": "consumer_prenorm_k_native_v" if use_consumer_native_targets else "raw_derope",\n            "rope": {"model_b": rope_b.to_dict()},\n            "fit_mode": fit_mode,\n            "route_layers": [] if route is None else [int(layer) for layer in route.get("patch_layers", [])],\n        },\n    )\n    logger.info("补丁检查点已保存到 %s", args.out)\n""",
        label="train_patch.py direct kv trainer",
    )
    text = replace_once(
        text,
        """    fit_mode = str(patch_cfg.get("fit_mode") or PATCH_FIT_MODE_DIRECT_KV_V1)\n    if fit_mode == PATCH_FIT_MODE_DIRECT_KV_V1:\n        _train_direct_kv_patch(\n            args=args,\n            logger=logger,\n            cfg=cfg,\n            meta=meta,\n            profile=profile,\n            route=route,\n            rope_b=rope_b,\n            device=device,\n            accel=accel,\n        )\n        return\n""",
        """    fit_mode = str(patch_cfg.get("fit_mode") or PATCH_FIT_MODE_DIRECT_KV_V1)\n    if fit_mode in {PATCH_FIT_MODE_DIRECT_KV_V1, PATCH_FIT_MODE_DIRECT_KV_KNORM_V1}:\n        _train_direct_kv_patch(\n            args=args,\n            logger=logger,\n            cfg=cfg,\n            meta=meta,\n            profile=profile,\n            route=route,\n            rope_b=rope_b,\n            device=device,\n            accel=accel,\n            fit_mode=fit_mode,\n        )\n        return\n""",
        label="train_patch.py main dispatch",
    )
    path.write_text(text, encoding="utf-8")


def patch_strict_runtime_py() -> None:
    path = ROOT / "src/scd_repro/strict_runtime.py"
    text = path.read_text(encoding="utf-8")
    text = replace_once(
        text,
        '    PATCH_FIT_MODE_DIRECT_KV_V1,\n    PATCH_FIT_MODE_XNORM_SEMANTIC_V1,\n',
        '    PATCH_FIT_MODE_DIRECT_KV_V1,\n    PATCH_FIT_MODE_DIRECT_KV_KNORM_V1,\n    PATCH_FIT_MODE_XNORM_SEMANTIC_V1,\n',
        label="strict_runtime.py imports",
    )
    text = replace_once(
        text,
        """def _get_consumer_attn_modules(consumer_session: TransformersRuntimeSession, layer: int) -> tuple[Any, Any]:\n    layers = consumer_session.layers if hasattr(consumer_session, "layers") else _get_decoder_layers(consumer_session.model)\n    layer_module = layers[int(layer)]\n    attn = getattr(layer_module, "self_attn", None) or getattr(layer_module, "attn", None)\n    if attn is None or not hasattr(attn, "k_proj") or not hasattr(attn, "v_proj"):\n        raise ValueError(f"layer={layer} 缺少 consumer native k_proj/v_proj")\n    return attn.k_proj, attn.v_proj\n\n\n""",
        """def _get_consumer_attn_modules(consumer_session: TransformersRuntimeSession, layer: int) -> tuple[Any, Any, Any | None]:\n    layers = consumer_session.layers if hasattr(consumer_session, "layers") else _get_decoder_layers(consumer_session.model)\n    layer_module = layers[int(layer)]\n    attn = getattr(layer_module, "self_attn", None) or getattr(layer_module, "attn", None)\n    if attn is None or not hasattr(attn, "k_proj") or not hasattr(attn, "v_proj"):\n        raise ValueError(f"layer={layer} 缺少 consumer native k_proj/v_proj")\n    return attn.k_proj, attn.v_proj, getattr(attn, "k_norm", None)\n\n\ndef _apply_consumer_k_norm_to_flat_keys(k_pre_flat: Any, k_norm: Any | None, kvh: int, hd: int) -> Any:\n    t = require_torch()\n    if k_norm is None:\n        return k_pre_flat.to(dtype=t.float32)\n    k_pre_flat = k_pre_flat.to(dtype=t.float32)\n    norm_in = k_pre_flat.reshape(-1, kvh, hd)\n    norm_dtype = getattr(getattr(k_norm, "weight", None), "dtype", norm_in.dtype)\n    with t.no_grad():\n        normed = k_norm(norm_in.to(dtype=norm_dtype)).to(dtype=t.float32)\n    return normed.reshape(-1, kvh * hd).contiguous()\n\n\n""",
        label="strict_runtime.py consumer attn helper",
    )
    text = replace_once(
        text,
        '    k_proj, v_proj = _get_consumer_attn_modules(consumer_session, int(layer))\n',
        '    k_proj, v_proj, _k_norm = _get_consumer_attn_modules(consumer_session, int(layer))\n',
        label="strict_runtime.py semantic unpack",
    )
    text = replace_once(
        text,
        """        if fit_mode == PATCH_FIT_MODE_DIRECT_KV_V1:\n            wk = patch_bundle[f"direct_wk.{layer}"]\n            wv = patch_bundle[f"direct_wv.{layer}"]\n            kb_raw_flat, vb_flat = infer_direct_kv_patch_layer_torch(xa, wk, wv)\n            kb_raw = kb_raw_flat.reshape(1, prefix_tokens, kv_dim).reshape(1, prefix_tokens, kvh, hd).permute(0, 2, 1, 3).contiguous()\n            vb = vb_flat.reshape(1, prefix_tokens, kv_dim).reshape(1, prefix_tokens, kvh, hd).permute(0, 2, 1, 3).contiguous()\n        elif fit_mode == PATCH_FIT_MODE_XNORM_SEMANTIC_V1:\n""",
        """        if fit_mode == PATCH_FIT_MODE_DIRECT_KV_V1:\n            wk = patch_bundle[f"direct_wk.{layer}"]\n            wv = patch_bundle[f"direct_wv.{layer}"]\n            kb_raw_flat, vb_flat = infer_direct_kv_patch_layer_torch(xa, wk, wv)\n            kb_raw = kb_raw_flat.reshape(1, prefix_tokens, kv_dim).reshape(1, prefix_tokens, kvh, hd).permute(0, 2, 1, 3).contiguous()\n            vb = vb_flat.reshape(1, prefix_tokens, kv_dim).reshape(1, prefix_tokens, kvh, hd).permute(0, 2, 1, 3).contiguous()\n        elif fit_mode == PATCH_FIT_MODE_DIRECT_KV_KNORM_V1:\n            wk = patch_bundle[f"direct_wk.{layer}"]\n            wv = patch_bundle[f"direct_wv.{layer}"]\n            k_pre_flat, vb_flat = infer_direct_kv_patch_layer_torch(xa, wk, wv)\n            _k_proj, _v_proj, k_norm = _get_consumer_attn_modules(consumer_session, int(layer))\n            kb_raw_flat = _apply_consumer_k_norm_to_flat_keys(k_pre_flat, k_norm, kvh, hd)\n            kb_raw = kb_raw_flat.reshape(1, prefix_tokens, kv_dim).reshape(1, prefix_tokens, kvh, hd).permute(0, 2, 1, 3).contiguous()\n            vb = vb_flat.reshape(1, prefix_tokens, kv_dim).reshape(1, prefix_tokens, kvh, hd).permute(0, 2, 1, 3).contiguous()\n        elif fit_mode == PATCH_FIT_MODE_XNORM_SEMANTIC_V1:\n""",
        label="strict_runtime.py direct kv branch",
    )
    path.write_text(text, encoding="utf-8")


def patch_schema_py() -> None:
    path = ROOT / "src/scd_repro/schema.py"
    text = path.read_text(encoding="utf-8")
    text = replace_once(
        text,
        '        if fit_mode == "direct_kv_v1":\n            prefixes = ("direct_wk.", "direct_wv.")\n',
        '        if fit_mode in {"direct_kv_v1", "direct_kv_knorm_v1"}:\n            prefixes = ("direct_wk.", "direct_wv.")\n',
        label="schema.py direct kv validation",
    )
    path.write_text(text, encoding="utf-8")


def main() -> None:
    patch_patch_py()
    patch_train_patch_py()
    patch_strict_runtime_py()
    patch_schema_py()
    print("patched remote repo for direct_kv_knorm_v1")


if __name__ == "__main__":
    main()
