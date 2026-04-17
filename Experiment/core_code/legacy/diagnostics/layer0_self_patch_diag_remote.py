from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"count": 0.0, "mean": 0.0, "max": 0.0, "min": 0.0}
    values_sorted = sorted(float(v) for v in values)
    n = len(values_sorted)
    return {
        "count": float(n),
        "mean": float(sum(values_sorted) / n),
        "min": float(values_sorted[0]),
        "p50": float(values_sorted[n // 2]),
        "p90": float(values_sorted[min(n - 1, int(round(0.9 * (n - 1))))]),
        "max": float(values_sorted[-1]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose self-self layer-0 patch behavior on remote SCD repo.")
    parser.add_argument("--pairs-dir", required=True)
    parser.add_argument("--patch-ckpt", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-samples", type=int, default=50)
    args = parser.parse_args()

    import torch
    import torch.nn.functional as F

    from scd_repro.io_utils import save_json
    from scd_repro.patch import infer_direct_kv_patch_layer_torch
    from scd_repro.pipeline import iter_pair_chunks, load_patch_ckpt
    from scd_repro.strict_runtime import (
        TransformersRuntimeSession,
        _apply_consumer_k_norm_to_flat_keys,
        _apply_consumer_native_rotary_to_patch_keys,
        _get_consumer_attn_modules,
    )

    layer = 0
    session = TransformersRuntimeSession(args.model_path, device=args.device, dtype=args.dtype, cpu_fallback=True)
    patch_bundle, patch_meta = load_patch_ckpt(args.patch_ckpt, device=args.device, dtype="float32")
    k_proj, v_proj, k_norm_module = _get_consumer_attn_modules(session, layer)
    k_weight = k_proj.weight.detach().to(device=session.target_device, dtype=torch.float32)
    v_weight = v_proj.weight.detach().to(device=session.target_device, dtype=torch.float32)
    k_bias = None if k_proj.bias is None else k_proj.bias.detach().to(device=session.target_device, dtype=torch.float32)
    v_bias = None if v_proj.bias is None else v_proj.bias.detach().to(device=session.target_device, dtype=torch.float32)
    wk = patch_bundle[f"direct_wk.{layer}"].to(device=session.target_device, dtype=torch.float32)
    wv = patch_bundle[f"direct_wv.{layer}"].to(device=session.target_device, dtype=torch.float32)

    xa_xb_mse: list[float] = []
    xa_xb_rel: list[float] = []
    native_k_mse: list[float] = []
    native_k_rel: list[float] = []
    native_v_mse: list[float] = []
    native_v_rel: list[float] = []
    learned_k_mse: list[float] = []
    learned_k_rel: list[float] = []
    learned_v_mse: list[float] = []
    learned_v_rel: list[float] = []
    sample_rows: list[dict[str, Any]] = []

    kvh = int(session.model.config.num_key_value_heads)
    hd = int(session.model.config.hidden_size // session.model.config.num_attention_heads)
    hidden_dim = int(session.model.config.hidden_size)

    for idx, (arrays, meta) in enumerate(
        iter_pair_chunks(args.pairs_dir, device=str(session.target_device), dtype="float32", cpu_fallback=True),
        start=1,
    ):
        if idx > int(args.max_samples):
            break
        prefix_tokens = int(meta["prefix_tokens"])
        xa = arrays[f"xnorm_a.{layer}"].to(device=session.target_device, dtype=torch.float32)
        xb = arrays[f"xnorm_b.{layer}"].to(device=session.target_device, dtype=torch.float32)
        attn_mask = arrays["attention_mask"].to(device=session.target_device)
        kb_true = arrays[f"k_b.{layer}"].to(device=session.target_device, dtype=torch.float32)
        vb_true = arrays[f"v_b.{layer}"].to(device=session.target_device, dtype=torch.float32)

        xa_flat = xa.reshape(-1, hidden_dim)
        xb_flat = xb.reshape(-1, hidden_dim)

        with torch.no_grad():
            k_pre_native = F.linear(xa_flat, k_weight, k_bias)
            v_native_flat = F.linear(xa_flat, v_weight, v_bias)
            k_norm_flat = _apply_consumer_k_norm_to_flat_keys(k_pre_native, k_norm_module, kvh, hd)
            k_native_norm = (
                k_norm_flat.reshape(1, prefix_tokens, kvh * hd)
                .reshape(1, prefix_tokens, kvh, hd)
                .permute(0, 2, 1, 3)
                .contiguous()
            )
            k_native = _apply_consumer_native_rotary_to_patch_keys(
                k_states=k_native_norm,
                layer=layer,
                consumer_session=session,
                attention_mask_prefix=attn_mask,
                hidden_dim=hidden_dim,
            )
            v_native = (
                v_native_flat.reshape(1, prefix_tokens, kvh * hd)
                .reshape(1, prefix_tokens, kvh, hd)
                .permute(0, 2, 1, 3)
                .contiguous()
            )

            k_pre_learned, v_learned_flat = infer_direct_kv_patch_layer_torch(xa_flat, wk, wv)
            k_learned_norm_flat = _apply_consumer_k_norm_to_flat_keys(k_pre_learned, k_norm_module, kvh, hd)
            k_learned_norm = (
                k_learned_norm_flat.reshape(1, prefix_tokens, kvh * hd)
                .reshape(1, prefix_tokens, kvh, hd)
                .permute(0, 2, 1, 3)
                .contiguous()
            )
            k_learned = _apply_consumer_native_rotary_to_patch_keys(
                k_states=k_learned_norm,
                layer=layer,
                consumer_session=session,
                attention_mask_prefix=attn_mask,
                hidden_dim=hidden_dim,
            )
            v_learned = (
                v_learned_flat.reshape(1, prefix_tokens, kvh * hd)
                .reshape(1, prefix_tokens, kvh, hd)
                .permute(0, 2, 1, 3)
                .contiguous()
            )

        xa_err = xa - xb
        native_k_err = k_native - kb_true
        native_v_err = v_native - vb_true
        learned_k_err = k_learned - kb_true
        learned_v_err = v_learned - vb_true

        def mse(err: torch.Tensor) -> float:
            return float(torch.mean(err.square()).item())

        def rel(err: torch.Tensor, ref: torch.Tensor) -> float:
            denom = float(torch.mean(ref.square()).item()) + 1e-12
            return float(torch.mean(err.square()).item() / denom)

        xa_xb_mse.append(mse(xa_err))
        xa_xb_rel.append(rel(xa_err, xb))
        native_k_mse.append(mse(native_k_err))
        native_k_rel.append(rel(native_k_err, kb_true))
        native_v_mse.append(mse(native_v_err))
        native_v_rel.append(rel(native_v_err, vb_true))
        learned_k_mse.append(mse(learned_k_err))
        learned_k_rel.append(rel(learned_k_err, kb_true))
        learned_v_mse.append(mse(learned_v_err))
        learned_v_rel.append(rel(learned_v_err, vb_true))

        sample_rows.append(
            {
                "sample_index": idx - 1,
                "prefix_tokens": prefix_tokens,
                "xa_vs_xb_mse": xa_xb_mse[-1],
                "xa_vs_xb_rel_mse": xa_xb_rel[-1],
                "native_k_mse": native_k_mse[-1],
                "native_k_rel_mse": native_k_rel[-1],
                "native_v_mse": native_v_mse[-1],
                "native_v_rel_mse": native_v_rel[-1],
                "learned_k_mse": learned_k_mse[-1],
                "learned_k_rel_mse": learned_k_rel[-1],
                "learned_v_mse": learned_v_mse[-1],
                "learned_v_rel_mse": learned_v_rel[-1],
            }
        )

    out = {
        "meta": {
            "pairs_dir": args.pairs_dir,
            "patch_ckpt": args.patch_ckpt,
            "model_path": args.model_path,
            "device": str(session.target_device),
            "dtype": args.dtype,
            "fit_mode": patch_meta.get("fit_mode"),
            "layer": layer,
            "max_samples": int(args.max_samples),
        },
        "aggregate": {
            "xa_vs_xb_mse": _stats(xa_xb_mse),
            "xa_vs_xb_rel_mse": _stats(xa_xb_rel),
            "native_k_mse": _stats(native_k_mse),
            "native_k_rel_mse": _stats(native_k_rel),
            "native_v_mse": _stats(native_v_mse),
            "native_v_rel_mse": _stats(native_v_rel),
            "learned_k_mse": _stats(learned_k_mse),
            "learned_k_rel_mse": _stats(learned_k_rel),
            "learned_v_mse": _stats(learned_v_mse),
            "learned_v_rel_mse": _stats(learned_v_rel),
        },
        "samples": sample_rows,
    }
    save_json(Path(args.out), out)
    print(json.dumps(out["aggregate"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
