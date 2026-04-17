from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from .io_utils import load_bundle, save_json
from .patch import (
    accumulate_direct_kv_error_stats_torch,
    accumulate_direct_kv_stats_torch,
    create_direct_kv_error_stats_torch,
    create_direct_kv_stats_torch,
    finalize_direct_kv_error_stats_torch,
    pack_module_state,
    solve_direct_kv_from_stats_torch,
    unpack_module_state,
)
from .pipeline import save_patch_ckpt
from .rope import de_rope_k_torch
from .torch_utils import clear_torch_cache


PATCH_FIT_MODE_DIRECT_KV_KNORM_RESIDUAL_MLP_V1 = "direct_kv_knorm_residual_mlp_v1"
PATCH_FIT_MODE_XNORM_NATIVEKV_RESIDUAL_MLP_V1 = "xnorm_nativekv_residual_mlp_v1"


@dataclass
class ConsumerNativeComponents:
    k_proj_weight: torch.Tensor
    k_proj_bias: torch.Tensor | None
    v_proj_weight: torch.Tensor
    v_proj_bias: torch.Tensor | None


class SimpleRMSNorm(torch.nn.Module):
    def __init__(self, width: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.width = int(width)
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms


class ResidualSwiGLUBlock(torch.nn.Module):
    def __init__(self, width: int, ffn_multiplier: float = 2.0) -> None:
        super().__init__()
        hidden = max(width, int(width * float(ffn_multiplier)))
        self.norm = SimpleRMSNorm(width)
        self.gate_proj = torch.nn.Linear(width, hidden, bias=False)
        self.up_proj = torch.nn.Linear(width, hidden, bias=False)
        self.down_proj = torch.nn.Linear(hidden, width, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = F.silu(self.gate_proj(h)) * self.up_proj(h)
        return x + self.down_proj(h)


class DirectKVResidualPatchLayer(torch.nn.Module):
    def __init__(
        self,
        *,
        hidden_dim: int,
        kv_dim: int,
        trunk_dim: int,
        num_blocks: int,
        ffn_multiplier: float,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.kv_dim = int(kv_dim)
        self.trunk_dim = int(trunk_dim)
        self.num_blocks = int(num_blocks)
        self.ffn_multiplier = float(ffn_multiplier)

        self.input_norm = SimpleRMSNorm(self.hidden_dim)
        self.input_proj = torch.nn.Linear(self.hidden_dim, self.trunk_dim)
        self.blocks = torch.nn.ModuleList(
            [ResidualSwiGLUBlock(self.trunk_dim, ffn_multiplier=self.ffn_multiplier) for _ in range(self.num_blocks)]
        )
        self.output_norm = SimpleRMSNorm(self.trunk_dim)
        self.k_head = torch.nn.Linear(self.trunk_dim, self.kv_dim)
        self.v_head = torch.nn.Linear(self.trunk_dim, self.kv_dim)

        # Start exactly from the linear baseline.
        torch.nn.init.zeros_(self.k_head.weight)
        torch.nn.init.zeros_(self.k_head.bias)
        torch.nn.init.zeros_(self.v_head.weight)
        torch.nn.init.zeros_(self.v_head.bias)

    def _base_forward(self, xa: torch.Tensor, wk: torch.Tensor, wv: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        xa_aug = torch.cat(
            [xa, torch.ones((xa.shape[0], 1), device=xa.device, dtype=xa.dtype)],
            dim=1,
        )
        return xa_aug @ wk, xa_aug @ wv

    def _trunk_forward(self, xa: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(self.input_norm(xa))
        for block in self.blocks:
            h = block(h)
        return self.output_norm(h)

    def forward(self, xa: torch.Tensor, wk: torch.Tensor, wv: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        base_k, base_v = self._base_forward(xa, wk, wv)
        trunk = self._trunk_forward(xa)
        delta_k = self.k_head(trunk)
        delta_v = self.v_head(trunk)
        pred_k = base_k + delta_k
        pred_v = base_v + delta_v
        aux = {
            "base_k": base_k,
            "base_v": base_v,
            "delta_k": delta_k,
            "delta_v": delta_v,
        }
        return pred_k, pred_v, aux


class XNormNativeKVResidualPatchLayer(torch.nn.Module):
    def __init__(
        self,
        *,
        hidden_dim: int,
        trunk_dim: int,
        num_blocks: int,
        ffn_multiplier: float,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.trunk_dim = int(trunk_dim)
        self.num_blocks = int(num_blocks)
        self.ffn_multiplier = float(ffn_multiplier)

        self.input_norm = SimpleRMSNorm(self.hidden_dim)
        self.input_proj = torch.nn.Linear(self.hidden_dim, self.trunk_dim)
        self.blocks = torch.nn.ModuleList(
            [ResidualSwiGLUBlock(self.trunk_dim, ffn_multiplier=self.ffn_multiplier) for _ in range(self.num_blocks)]
        )
        self.output_norm = SimpleRMSNorm(self.trunk_dim)
        self.delta_h_head = torch.nn.Linear(self.trunk_dim, self.hidden_dim)

        # Start exactly from identity xb_hat = xa.
        torch.nn.init.zeros_(self.delta_h_head.weight)
        torch.nn.init.zeros_(self.delta_h_head.bias)

    def _trunk_forward(self, xa: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(self.input_norm(xa))
        for block in self.blocks:
            h = block(h)
        return self.output_norm(h)

    def forward(self, xa: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        trunk = self._trunk_forward(xa)
        delta_h = self.delta_h_head(trunk)
        xb_hat = xa + delta_h
        return xb_hat, {"delta_h": delta_h}


def pack_direct_kv_residual_patch_layer(layer: int, module: DirectKVResidualPatchLayer) -> dict[str, Any]:
    return pack_module_state(f"direct_residual_patch.{int(layer)}", module)


def pack_xnorm_nativekv_residual_patch_layer(layer: int, module: XNormNativeKVResidualPatchLayer) -> dict[str, Any]:
    return pack_module_state(f"xnorm_nativekv_residual_patch.{int(layer)}", module)


def load_direct_kv_residual_patch_layer_from_ckpt(
    arrays: dict[str, Any],
    meta: dict[str, Any],
    *,
    layer: int,
    device: str | Any = "cpu",
    dtype: str = "float32",
) -> DirectKVResidualPatchLayer:
    fit_mode = str(meta.get("fit_mode") or "")
    if fit_mode != PATCH_FIT_MODE_DIRECT_KV_KNORM_RESIDUAL_MLP_V1:
        raise ValueError(f"expected {PATCH_FIT_MODE_DIRECT_KV_KNORM_RESIDUAL_MLP_V1}, got {fit_mode or '<empty>'}")
    hidden_dim = int(meta["hidden_dim"])
    kv_dim = int(meta["num_key_value_heads"]) * int(meta["head_dim"])
    module = DirectKVResidualPatchLayer(
        hidden_dim=hidden_dim,
        kv_dim=kv_dim,
        trunk_dim=int(meta.get("residual_hidden_dim") or hidden_dim),
        num_blocks=int(meta.get("residual_num_blocks") or 4),
        ffn_multiplier=float(meta.get("residual_ffn_multiplier") or 2.0),
    )
    unpack_module_state(module, arrays, f"direct_residual_patch.{int(layer)}")
    target_dtype = torch.zeros((), dtype=torch.float32, device=device).to(dtype=getattr(torch, dtype)).dtype if isinstance(dtype, str) and hasattr(torch, dtype) else torch.float32
    return module.to(device=device, dtype=target_dtype).eval()


def load_xnorm_nativekv_residual_patch_layer_from_ckpt(
    arrays: dict[str, Any],
    meta: dict[str, Any],
    *,
    layer: int,
    device: str | Any = "cpu",
    dtype: str = "float32",
) -> XNormNativeKVResidualPatchLayer:
    fit_mode = str(meta.get("fit_mode") or "")
    if fit_mode != PATCH_FIT_MODE_XNORM_NATIVEKV_RESIDUAL_MLP_V1:
        raise ValueError(f"expected {PATCH_FIT_MODE_XNORM_NATIVEKV_RESIDUAL_MLP_V1}, got {fit_mode or '<empty>'}")
    hidden_dim = int(meta["hidden_dim"])
    module = XNormNativeKVResidualPatchLayer(
        hidden_dim=hidden_dim,
        trunk_dim=int(meta.get("residual_hidden_dim") or hidden_dim),
        num_blocks=int(meta.get("residual_num_blocks") or 4),
        ffn_multiplier=float(meta.get("residual_ffn_multiplier") or 2.0),
    )
    unpack_module_state(module, arrays, f"xnorm_nativekv_residual_patch.{int(layer)}")
    target_dtype = torch.zeros((), dtype=torch.float32, device=device).to(dtype=getattr(torch, dtype)).dtype if isinstance(dtype, str) and hasattr(torch, dtype) else torch.float32
    return module.to(device=device, dtype=target_dtype).eval()


def infer_direct_kv_residual_patch_layer_torch(
    xa: torch.Tensor,
    wk: torch.Tensor,
    wv: torch.Tensor,
    module: DirectKVResidualPatchLayer,
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        pred_k, pred_v, _aux = module(xa, wk, wv)
    return pred_k, pred_v


def infer_xnorm_nativekv_residual_patch_layer_torch(
    xa: torch.Tensor,
    module: XNormNativeKVResidualPatchLayer,
) -> torch.Tensor:
    with torch.no_grad():
        xb_hat, _aux = module(xa)
    return xb_hat


def _iter_chunk_paths(pairs_dir: str):
    root = Path(pairs_dir)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    for row in manifest.get("chunks", []):
        yield root / str(row["path"])


def _select_layers(profile: dict[str, Any], patch_budget: int, route: dict[str, Any] | None = None) -> list[int]:
    if route is not None:
        route_layers = [int(layer) for layer in route.get("patch_layers", [])]
        if route_layers:
            return route_layers[:patch_budget]
    if "recommended_patch_layers" in profile:
        return list(profile["recommended_patch_layers"][:patch_budget])
    scored = sorted(profile["layers"], key=lambda item: float(item["expected_gain"]), reverse=True)
    return [int(row["layer"]) for row in scored[:patch_budget]]


def _resolve_decoder_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise RuntimeError("cannot resolve decoder layers")


def _resolve_attn_module(layer_module):
    if hasattr(layer_module, "self_attn"):
        return layer_module.self_attn
    if hasattr(layer_module, "attn"):
        return layer_module.attn
    raise RuntimeError("cannot resolve attention module")


def _tensor_or_none(value: Any, *, device: torch.device) -> torch.Tensor | None:
    if value is None:
        return None
    return value.detach().to(device=device, dtype=torch.float32).contiguous()


def _load_consumer_native_components(
    models_config: str,
    selected_layers: list[int],
    hidden_dim: int,
    kv_dim: int,
    device: torch.device,
) -> dict[int, ConsumerNativeComponents]:
    if not models_config:
        raise ValueError("models_config is required for knorm residual patch")
    try:
        from transformers import AutoModelForCausalLM
    except Exception as exc:
        raise RuntimeError("missing dependency: transformers") from exc

    models_cfg = json.loads(Path(models_config).read_text(encoding="utf-8"))
    model_b_path = str(models_cfg.get("model_b", {}).get("path") or "").strip()
    if not model_b_path:
        raise ValueError("models_config missing model_b.path")

    model = AutoModelForCausalLM.from_pretrained(
        model_b_path,
        torch_dtype=torch.float32,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.eval()
    layers = _resolve_decoder_layers(model)
    out: dict[int, ConsumerNativeComponents] = {}
    try:
        for layer in selected_layers:
            attn = _resolve_attn_module(layers[int(layer)])
            k_proj = attn.k_proj
            v_proj = attn.v_proj
            if int(k_proj.weight.shape[1]) != int(hidden_dim):
                raise ValueError(f"layer={layer}: hidden_dim mismatch")
            if int(k_proj.weight.shape[0]) != int(kv_dim) or int(v_proj.weight.shape[0]) != int(kv_dim):
                raise ValueError(f"layer={layer}: kv_dim mismatch")
            out[int(layer)] = ConsumerNativeComponents(
                k_proj_weight=k_proj.weight.detach().to(device=device, dtype=torch.float32).contiguous(),
                k_proj_bias=_tensor_or_none(k_proj.bias, device=device),
                v_proj_weight=v_proj.weight.detach().to(device=device, dtype=torch.float32).contiguous(),
                v_proj_bias=_tensor_or_none(v_proj.bias, device=device),
            )
    finally:
        del model
        clear_torch_cache()
    return out


def _project_consumer_native_kv_from_xnorm_b(
    arrays: dict[str, torch.Tensor],
    layer: int,
    hidden_dim: int,
    device: torch.device,
    native_components: ConsumerNativeComponents,
) -> tuple[torch.Tensor, torch.Tensor]:
    xb = arrays[f"xnorm_b.{int(layer)}"].reshape(-1, hidden_dim).to(device=device, dtype=torch.float32)
    k_proj_weight = native_components.k_proj_weight.to(device=device, dtype=torch.float32)
    k_proj_bias = _tensor_or_none(native_components.k_proj_bias, device=device)
    v_proj_weight = native_components.v_proj_weight.to(device=device, dtype=torch.float32)
    v_proj_bias = _tensor_or_none(native_components.v_proj_bias, device=device)
    kb_raw = F.linear(xb, k_proj_weight, k_proj_bias)
    vb = F.linear(xb, v_proj_weight, v_proj_bias)
    return kb_raw, vb


def _project_consumer_native_kv_from_hidden_tensor(
    xb: torch.Tensor,
    device: torch.device,
    native_components: ConsumerNativeComponents,
) -> tuple[torch.Tensor, torch.Tensor]:
    xb = xb.to(device=device, dtype=torch.float32)
    k_proj_weight = native_components.k_proj_weight.to(device=device, dtype=torch.float32)
    k_proj_bias = _tensor_or_none(native_components.k_proj_bias, device=device)
    v_proj_weight = native_components.v_proj_weight.to(device=device, dtype=torch.float32)
    v_proj_bias = _tensor_or_none(native_components.v_proj_bias, device=device)
    kb_raw = F.linear(xb, k_proj_weight, k_proj_bias)
    vb = F.linear(xb, v_proj_weight, v_proj_bias)
    return kb_raw, vb


def _relative_mse(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    denom = target.pow(2).mean().detach()
    return F.mse_loss(pred, target) / (denom + float(eps))


def _preload_residual_dataset_for_layer(
    *,
    chunk_paths: list[Path],
    layer: int,
    hidden_dim: int,
    device: torch.device,
    native_components: ConsumerNativeComponents,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    xa_parts: list[torch.Tensor] = []
    k_parts: list[torch.Tensor] = []
    v_parts: list[torch.Tensor] = []
    for chunk_path in chunk_paths:
        bundle = load_bundle(chunk_path, device="cpu", dtype="float32")
        arrays = bundle.arrays
        xa = arrays[f"xnorm_a.{int(layer)}"].reshape(-1, hidden_dim).to(device="cpu", dtype=torch.float32).contiguous()
        kb_raw, vb = _project_consumer_native_kv_from_xnorm_b(arrays, layer, hidden_dim, torch.device("cpu"), native_components)
        xa_parts.append(xa)
        k_parts.append(kb_raw.to(device="cpu", dtype=torch.float32).contiguous())
        v_parts.append(vb.to(device="cpu", dtype=torch.float32).contiguous())
        del arrays, bundle
        clear_torch_cache()
    return torch.cat(xa_parts, dim=0), torch.cat(k_parts, dim=0), torch.cat(v_parts, dim=0)


def _preload_xnorm_nativekv_residual_dataset_for_layer(
    *,
    chunk_paths: list[Path],
    layer: int,
    hidden_dim: int,
    device: torch.device,
    native_components: ConsumerNativeComponents,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    xa_parts: list[torch.Tensor] = []
    xb_parts: list[torch.Tensor] = []
    k_parts: list[torch.Tensor] = []
    v_parts: list[torch.Tensor] = []
    for chunk_path in chunk_paths:
        bundle = load_bundle(chunk_path, device="cpu", dtype="float32")
        arrays = bundle.arrays
        xa = arrays[f"xnorm_a.{int(layer)}"].reshape(-1, hidden_dim).to(device="cpu", dtype=torch.float32).contiguous()
        xb = arrays[f"xnorm_b.{int(layer)}"].reshape(-1, hidden_dim).to(device="cpu", dtype=torch.float32).contiguous()
        kb_raw, vb = _project_consumer_native_kv_from_hidden_tensor(xb, torch.device("cpu"), native_components)
        xa_parts.append(xa)
        xb_parts.append(xb)
        k_parts.append(kb_raw.to(device="cpu", dtype=torch.float32).contiguous())
        v_parts.append(vb.to(device="cpu", dtype=torch.float32).contiguous())
        del arrays, bundle
        clear_torch_cache()
    return (
        torch.cat(xa_parts, dim=0),
        torch.cat(xb_parts, dim=0),
        torch.cat(k_parts, dim=0),
        torch.cat(v_parts, dim=0),
    )


def _train_residual_layer(
    *,
    layer: int,
    xa: torch.Tensor,
    k_target: torch.Tensor,
    v_target: torch.Tensor,
    wk: torch.Tensor,
    wv: torch.Tensor,
    patch_cfg: dict[str, Any],
    device: torch.device,
    logger,
) -> tuple[dict[str, Any], dict[str, float]]:
    kv_dim = int(k_target.shape[-1])
    hidden_dim = int(xa.shape[-1])
    trunk_dim = int(patch_cfg.get("patch_residual_hidden_dim") or 4096)
    num_blocks = int(patch_cfg.get("patch_residual_num_blocks") or 4)
    ffn_multiplier = float(patch_cfg.get("patch_residual_ffn_multiplier") or 2.0)
    batch_size = max(1, int(patch_cfg.get("patch_batch_size") or 512))
    num_epochs = int(patch_cfg.get("patch_num_epochs") or 8)
    lr = float(patch_cfg.get("lr") or 3e-4)
    weight_decay = float(patch_cfg.get("weight_decay") or 0.01)
    lambda_k = float(patch_cfg.get("lambda_k") or 1.0)
    lambda_v = float(patch_cfg.get("lambda_v") or 1.0)
    lambda_delta = float(patch_cfg.get("lambda_delta") or 1e-3)

    module = DirectKVResidualPatchLayer(
        hidden_dim=hidden_dim,
        kv_dim=kv_dim,
        trunk_dim=trunk_dim,
        num_blocks=num_blocks,
        ffn_multiplier=ffn_multiplier,
    ).to(device=device, dtype=torch.float32)
    optimizer = torch.optim.AdamW(module.parameters(), lr=lr, weight_decay=weight_decay)

    wk = wk.to(device=device, dtype=torch.float32)
    wv = wv.to(device=device, dtype=torch.float32)
    num_rows = int(xa.shape[0])

    best_state: dict[str, torch.Tensor] | None = None
    best_metrics: dict[str, float] | None = None
    best_loss = float("inf")

    base_xa = xa.to(device=device, dtype=torch.float32)
    base_k = torch.cat(
        [base_xa, torch.ones((base_xa.shape[0], 1), device=device, dtype=torch.float32)],
        dim=1,
    ) @ wk
    base_v = torch.cat(
        [base_xa, torch.ones((base_xa.shape[0], 1), device=device, dtype=torch.float32)],
        dim=1,
    ) @ wv
    base_k_rel = float(_relative_mse(base_k, k_target.to(device=device, dtype=torch.float32)).item())
    base_v_rel = float(_relative_mse(base_v, v_target.to(device=device, dtype=torch.float32)).item())
    del base_xa, base_k, base_v
    clear_torch_cache()

    for epoch in range(1, num_epochs + 1):
        module.train()
        running = {
            "rows": 0.0,
            "k_rel_mse": 0.0,
            "v_rel_mse": 0.0,
            "delta_reg": 0.0,
            "total_loss": 0.0,
        }
        for start in range(0, num_rows, batch_size):
            stop = min(start + batch_size, num_rows)
            batch_xa = xa[start:stop].to(device=device, dtype=torch.float32, non_blocking=True)
            batch_k = k_target[start:stop].to(device=device, dtype=torch.float32, non_blocking=True)
            batch_v = v_target[start:stop].to(device=device, dtype=torch.float32, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            pred_k, pred_v, aux = module(batch_xa, wk, wv)
            loss_k = _relative_mse(pred_k, batch_k)
            loss_v = _relative_mse(pred_v, batch_v)
            delta_reg = _relative_mse(aux["delta_k"], torch.zeros_like(aux["delta_k"])) + _relative_mse(aux["delta_v"], torch.zeros_like(aux["delta_v"]))
            total_loss = (lambda_k * loss_k) + (lambda_v * loss_v) + (lambda_delta * delta_reg)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(module.parameters(), max_norm=1.0)
            optimizer.step()

            rows = int(batch_xa.shape[0])
            running["rows"] += float(rows)
            running["k_rel_mse"] += float(loss_k.item()) * rows
            running["v_rel_mse"] += float(loss_v.item()) * rows
            running["delta_reg"] += float(delta_reg.item()) * rows
            running["total_loss"] += float(total_loss.item()) * rows

        denom = max(1.0, running["rows"])
        epoch_metrics = {
            "k_rel_mse": running["k_rel_mse"] / denom,
            "v_rel_mse": running["v_rel_mse"] / denom,
            "delta_reg": running["delta_reg"] / denom,
            "total_loss": running["total_loss"] / denom,
        }
        logger.info(
            "direct residual patch epoch: layer=%d epoch=%d/%d k_rel_mse=%.6f v_rel_mse=%.6f delta_reg=%.6f total_loss=%.6f",
            layer,
            epoch,
            num_epochs,
            epoch_metrics["k_rel_mse"],
            epoch_metrics["v_rel_mse"],
            epoch_metrics["delta_reg"],
            epoch_metrics["total_loss"],
        )
        if epoch_metrics["total_loss"] < best_loss:
            best_loss = epoch_metrics["total_loss"]
            best_metrics = dict(epoch_metrics)
            best_state = {k: v.detach().to(device="cpu", dtype=torch.float32).clone() for k, v in module.state_dict().items()}

    if best_state is None or best_metrics is None:
        raise RuntimeError(f"layer={layer}: residual patch training produced no checkpoint")

    module.load_state_dict(best_state)
    final_metrics = {
        "base_k_rel_mse": base_k_rel,
        "base_v_rel_mse": base_v_rel,
        "k_rel_mse": best_metrics["k_rel_mse"],
        "v_rel_mse": best_metrics["v_rel_mse"],
        "delta_reg": best_metrics["delta_reg"],
        "total_loss": best_metrics["total_loss"],
    }
    return pack_direct_kv_residual_patch_layer(layer, module), final_metrics


def _train_xnorm_nativekv_residual_layer(
    *,
    layer: int,
    xa: torch.Tensor,
    xb_target: torch.Tensor,
    k_target: torch.Tensor,
    v_target: torch.Tensor,
    patch_cfg: dict[str, Any],
    device: torch.device,
    native_components: ConsumerNativeComponents,
    logger,
) -> tuple[dict[str, Any], dict[str, float]]:
    hidden_dim = int(xa.shape[-1])
    trunk_dim = int(patch_cfg.get("patch_residual_hidden_dim") or 4096)
    num_blocks = int(patch_cfg.get("patch_residual_num_blocks") or 4)
    ffn_multiplier = float(patch_cfg.get("patch_residual_ffn_multiplier") or 2.0)
    batch_size = max(1, int(patch_cfg.get("patch_batch_size") or 512))
    num_epochs = int(patch_cfg.get("patch_num_epochs") or 8)
    lr = float(patch_cfg.get("lr") or 3e-4)
    weight_decay = float(patch_cfg.get("weight_decay") or 0.01)
    lambda_h = float(patch_cfg.get("lambda_h") or 1.0)
    lambda_k = float(patch_cfg.get("lambda_k") or 1.0)
    lambda_v = float(patch_cfg.get("lambda_v") or 1.0)
    lambda_delta = float(patch_cfg.get("lambda_delta") or 1e-3)

    module = XNormNativeKVResidualPatchLayer(
        hidden_dim=hidden_dim,
        trunk_dim=trunk_dim,
        num_blocks=num_blocks,
        ffn_multiplier=ffn_multiplier,
    ).to(device=device, dtype=torch.float32)
    optimizer = torch.optim.AdamW(module.parameters(), lr=lr, weight_decay=weight_decay)
    num_rows = int(xa.shape[0])

    best_state = {k: v.detach().to(device="cpu", dtype=torch.float32).clone() for k, v in module.state_dict().items()}
    best_metrics: dict[str, float] | None = None
    best_loss = float("inf")

    init_metrics: dict[str, float] | None = None
    with torch.no_grad():
        base_xa = xa.to(device=device, dtype=torch.float32)
        base_xb = xb_target.to(device=device, dtype=torch.float32)
        pred_xb, aux = module(base_xa)
        pred_k, pred_v = _project_consumer_native_kv_from_hidden_tensor(pred_xb, device, native_components)
        init_h_rel = float(_relative_mse(pred_xb, base_xb).item())
        init_k_rel = float(_relative_mse(pred_k, k_target.to(device=device, dtype=torch.float32)).item())
        init_v_rel = float(_relative_mse(pred_v, v_target.to(device=device, dtype=torch.float32)).item())
        init_delta_reg = float(_relative_mse(aux["delta_h"], torch.zeros_like(aux["delta_h"])).item())
        init_total = (lambda_h * init_h_rel) + (lambda_k * init_k_rel) + (lambda_v * init_v_rel) + (lambda_delta * init_delta_reg)
        init_metrics = {
            "h_rel_mse": init_h_rel,
            "k_rel_mse": init_k_rel,
            "v_rel_mse": init_v_rel,
            "delta_reg": init_delta_reg,
            "total_loss": init_total,
        }
        best_metrics = dict(init_metrics)
        best_loss = init_total
        logger.info(
            "xnorm native-kv residual init: layer=%d h_rel_mse=%.6f k_rel_mse=%.6f v_rel_mse=%.6f delta_reg=%.6f total_loss=%.6f",
            layer,
            init_h_rel,
            init_k_rel,
            init_v_rel,
            init_delta_reg,
            init_total,
        )
        del base_xa, base_xb, pred_xb, pred_k, pred_v, aux
        clear_torch_cache()

    for epoch in range(1, num_epochs + 1):
        module.train()
        running = {
            "rows": 0.0,
            "h_rel_mse": 0.0,
            "k_rel_mse": 0.0,
            "v_rel_mse": 0.0,
            "delta_reg": 0.0,
            "total_loss": 0.0,
        }
        for start in range(0, num_rows, batch_size):
            stop = min(start + batch_size, num_rows)
            batch_xa = xa[start:stop].to(device=device, dtype=torch.float32, non_blocking=True)
            batch_xb = xb_target[start:stop].to(device=device, dtype=torch.float32, non_blocking=True)
            batch_k = k_target[start:stop].to(device=device, dtype=torch.float32, non_blocking=True)
            batch_v = v_target[start:stop].to(device=device, dtype=torch.float32, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            pred_xb, aux = module(batch_xa)
            pred_k, pred_v = _project_consumer_native_kv_from_hidden_tensor(pred_xb, device, native_components)
            loss_h = _relative_mse(pred_xb, batch_xb)
            loss_k = _relative_mse(pred_k, batch_k)
            loss_v = _relative_mse(pred_v, batch_v)
            delta_reg = _relative_mse(aux["delta_h"], torch.zeros_like(aux["delta_h"]))
            total_loss = (lambda_h * loss_h) + (lambda_k * loss_k) + (lambda_v * loss_v) + (lambda_delta * delta_reg)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(module.parameters(), max_norm=1.0)
            optimizer.step()

            rows = int(batch_xa.shape[0])
            running["rows"] += float(rows)
            running["h_rel_mse"] += float(loss_h.item()) * rows
            running["k_rel_mse"] += float(loss_k.item()) * rows
            running["v_rel_mse"] += float(loss_v.item()) * rows
            running["delta_reg"] += float(delta_reg.item()) * rows
            running["total_loss"] += float(total_loss.item()) * rows

        denom = max(1.0, running["rows"])
        epoch_metrics = {
            "h_rel_mse": running["h_rel_mse"] / denom,
            "k_rel_mse": running["k_rel_mse"] / denom,
            "v_rel_mse": running["v_rel_mse"] / denom,
            "delta_reg": running["delta_reg"] / denom,
            "total_loss": running["total_loss"] / denom,
        }
        logger.info(
            "xnorm native-kv residual epoch: layer=%d epoch=%d/%d h_rel_mse=%.6f k_rel_mse=%.6f v_rel_mse=%.6f delta_reg=%.6f total_loss=%.6f",
            layer,
            epoch,
            num_epochs,
            epoch_metrics["h_rel_mse"],
            epoch_metrics["k_rel_mse"],
            epoch_metrics["v_rel_mse"],
            epoch_metrics["delta_reg"],
            epoch_metrics["total_loss"],
        )
        if epoch_metrics["total_loss"] < best_loss:
            best_loss = epoch_metrics["total_loss"]
            best_metrics = dict(epoch_metrics)
            best_state = {k: v.detach().to(device="cpu", dtype=torch.float32).clone() for k, v in module.state_dict().items()}

    if best_metrics is None or init_metrics is None:
        raise RuntimeError(f"layer={layer}: xnorm native-kv residual patch training produced no checkpoint")

    module.load_state_dict(best_state)
    final_metrics = {
        "base_h_rel_mse": init_metrics["h_rel_mse"],
        "base_k_rel_mse": init_metrics["k_rel_mse"],
        "base_v_rel_mse": init_metrics["v_rel_mse"],
        "h_rel_mse": best_metrics["h_rel_mse"],
        "k_rel_mse": best_metrics["k_rel_mse"],
        "v_rel_mse": best_metrics["v_rel_mse"],
        "delta_reg": best_metrics["delta_reg"],
        "total_loss": best_metrics["total_loss"],
    }
    return pack_xnorm_nativekv_residual_patch_layer(layer, module), final_metrics


def train_direct_kv_knorm_residual_patch(
    *,
    args,
    logger,
    cfg: dict[str, Any],
    meta: dict[str, Any],
    profile: dict[str, Any],
    route: dict[str, Any] | None,
    rope_b,
    device: torch.device,
    accel: dict[str, Any],
) -> None:
    patch_cfg = cfg.get("patch", {})
    patch_budget = int(patch_cfg["patch_budget"])
    ridge_lambda = float(patch_cfg["ridge_lambda"])
    selected = _select_layers(profile, patch_budget, route=route)
    hidden_dim = int(meta["hidden_dim"])
    kv_dim = int(meta["num_key_value_heads"]) * int(meta["head_dim"])
    layer_block_size = int(accel["layer_block_size"])

    residual_layers_cfg = patch_cfg.get("residual_layers")
    if isinstance(residual_layers_cfg, list) and residual_layers_cfg:
        residual_layers = [int(layer) for layer in residual_layers_cfg]
    else:
        residual_layers = list(selected)
    residual_layers = [layer for layer in residual_layers if layer in selected]
    if not residual_layers:
        raise ValueError("residual_layers resolved to empty set")

    logger.info(
        "residual direct-kv patch training start: fit_mode=%s device=%s accel=%s selected=%s residual_layers=%s",
        PATCH_FIT_MODE_DIRECT_KV_KNORM_RESIDUAL_MLP_V1,
        device,
        accel["accel_mode"],
        selected,
        residual_layers,
    )

    native_components_by_layer = _load_consumer_native_components(
        args.models_config,
        selected,
        hidden_dim,
        kv_dim,
        device,
    )

    direct_stats_by_layer: dict[int, dict[str, torch.Tensor]] = {
        layer: create_direct_kv_stats_torch(hidden_dim, kv_dim, device) for layer in selected
    }
    chunk_paths = list(_iter_chunk_paths(args.pairs))

    start = time.perf_counter()
    for idx, chunk_path in enumerate(chunk_paths, start=1):
        bundle = load_bundle(chunk_path, device=str(device), dtype="float32")
        arrays = bundle.arrays
        for start_idx in range(0, len(selected), layer_block_size):
            block_layers = selected[start_idx : start_idx + layer_block_size]
            for layer in block_layers:
                xa = arrays[f"xnorm_a.{int(layer)}"].reshape(-1, hidden_dim).to(device=device, dtype=torch.float32)
                kb_raw, vb = _project_consumer_native_kv_from_xnorm_b(arrays, layer, hidden_dim, device, native_components_by_layer[int(layer)])
                accumulate_direct_kv_stats_torch(xa, kb_raw, vb, direct_stats_by_layer[int(layer)])
        del arrays, bundle
        clear_torch_cache()
        if idx == 1 or idx % 5 == 0 or idx == len(chunk_paths):
            elapsed = time.perf_counter() - start
            eta_s = (elapsed / idx) * (len(chunk_paths) - idx) if idx > 0 else 0.0
            logger.info("residual direct-kv stats progress: chunk %d/%d elapsed=%.1fs eta=%.1fs", idx, len(chunk_paths), elapsed, eta_s)

    out_arrays: dict[str, object] = {}
    metrics: dict[str, dict[str, float]] = {}
    wk_by_layer: dict[int, torch.Tensor] = {}
    wv_by_layer: dict[int, torch.Tensor] = {}

    for layer in selected:
        wk, wv, _pk, _pv, _pkv = solve_direct_kv_from_stats_torch(direct_stats_by_layer[int(layer)], ridge_lambda=ridge_lambda)
        wk_by_layer[int(layer)] = wk.to(device=device, dtype=torch.float32)
        wv_by_layer[int(layer)] = wv.to(device=device, dtype=torch.float32)
        out_arrays[f"direct_wk.{int(layer)}"] = wk.detach().to(device="cpu", dtype=torch.float32).contiguous()
        out_arrays[f"direct_wv.{int(layer)}"] = wv.detach().to(device="cpu", dtype=torch.float32).contiguous()

    del direct_stats_by_layer
    clear_torch_cache()

    for layer in selected:
        error_stats = create_direct_kv_error_stats_torch(device)
        for chunk_path in chunk_paths:
            bundle = load_bundle(chunk_path, device=str(device), dtype="float32")
            arrays = bundle.arrays
            xa = arrays[f"xnorm_a.{int(layer)}"].reshape(-1, hidden_dim).to(device=device, dtype=torch.float32)
            kb_raw, vb = _project_consumer_native_kv_from_xnorm_b(arrays, layer, hidden_dim, device, native_components_by_layer[int(layer)])
            accumulate_direct_kv_error_stats_torch(xa, kb_raw, vb, wk_by_layer[int(layer)], wv_by_layer[int(layer)], error_stats)
            del arrays, bundle
            clear_torch_cache()
        patch_k_error, patch_v_error, patch_kv_error = finalize_direct_kv_error_stats_torch(error_stats, kv_dim=kv_dim)
        metrics[str(int(layer))] = {
            "base_k_rel_mse": float(patch_k_error),
            "base_v_rel_mse": float(patch_v_error),
            "base_kv_rel_mse": float(patch_kv_error),
        }
        logger.info(
            "linear base solved: layer=%d base_k_rel_mse=%.6f base_v_rel_mse=%.6f base_kv_rel_mse=%.6f",
            int(layer),
            patch_k_error,
            patch_v_error,
            patch_kv_error,
        )

    for layer in residual_layers:
        xa_all, k_all, v_all = _preload_residual_dataset_for_layer(
            chunk_paths=chunk_paths,
            layer=int(layer),
            hidden_dim=hidden_dim,
            device=device,
            native_components=native_components_by_layer[int(layer)],
        )
        residual_arrays, residual_metrics = _train_residual_layer(
            layer=int(layer),
            xa=xa_all,
            k_target=k_all,
            v_target=v_all,
            wk=wk_by_layer[int(layer)],
            wv=wv_by_layer[int(layer)],
            patch_cfg=patch_cfg,
            device=device,
            logger=logger,
        )
        out_arrays.update(residual_arrays)
        metrics[str(int(layer))].update(residual_metrics)
        logger.info(
            "residual layer complete: layer=%d base_k_rel_mse=%.6f -> %.6f, base_v_rel_mse=%.6f -> %.6f",
            int(layer),
            metrics[str(int(layer))]["base_k_rel_mse"],
            metrics[str(int(layer))]["k_rel_mse"],
            metrics[str(int(layer))]["base_v_rel_mse"],
            metrics[str(int(layer))]["v_rel_mse"],
        )
        del xa_all, k_all, v_all
        clear_torch_cache()

    meta_payload = {
        "selected_layers": selected,
        "residual_layers": residual_layers,
        "hidden_dim": hidden_dim,
        "code_dim": hidden_dim,
        "num_key_value_heads": int(meta["num_key_value_heads"]),
        "head_dim": int(meta["head_dim"]),
        "k_space": "consumer_prenorm_k_native_v",
        "rope": {"model_b": rope_b.to_dict()},
        "fit_mode": PATCH_FIT_MODE_DIRECT_KV_KNORM_RESIDUAL_MLP_V1,
        "route_layers": [] if route is None else [int(layer) for layer in route.get("patch_layers", [])],
        "residual_hidden_dim": int(patch_cfg.get("patch_residual_hidden_dim") or 4096),
        "residual_num_blocks": int(patch_cfg.get("patch_residual_num_blocks") or 4),
        "residual_ffn_multiplier": float(patch_cfg.get("patch_residual_ffn_multiplier") or 2.0),
        "loss_weights": {
            "lambda_k": float(patch_cfg.get("lambda_k") or 1.0),
            "lambda_v": float(patch_cfg.get("lambda_v") or 1.0),
            "lambda_delta": float(patch_cfg.get("lambda_delta") or 1e-3),
        },
    }
    save_patch_ckpt(args.out, out_arrays, meta_payload)
    save_json(Path(args.out) / "metrics.json", {"fit_mode": PATCH_FIT_MODE_DIRECT_KV_KNORM_RESIDUAL_MLP_V1, "layers": metrics})
    logger.info("residual direct-kv patch checkpoint saved to %s", args.out)


def train_xnorm_nativekv_residual_patch(
    *,
    args,
    logger,
    cfg: dict[str, Any],
    meta: dict[str, Any],
    profile: dict[str, Any],
    route: dict[str, Any] | None,
    rope_b,
    device: torch.device,
    accel: dict[str, Any],
) -> None:
    patch_cfg = cfg.get("patch", {})
    patch_budget = int(patch_cfg["patch_budget"])
    ridge_lambda = float(patch_cfg["ridge_lambda"])
    selected = _select_layers(profile, patch_budget, route=route)
    hidden_dim = int(meta["hidden_dim"])
    kv_dim = int(meta["num_key_value_heads"]) * int(meta["head_dim"])
    layer_block_size = int(accel["layer_block_size"])

    residual_layers_cfg = patch_cfg.get("residual_layers")
    if isinstance(residual_layers_cfg, list) and residual_layers_cfg:
        residual_layers = [int(layer) for layer in residual_layers_cfg]
    else:
        residual_layers = list(selected)
    residual_layers = [layer for layer in residual_layers if layer in selected]
    if not residual_layers:
        raise ValueError("residual_layers resolved to empty set")

    logger.info(
        "xnorm native-kv residual patch training start: fit_mode=%s device=%s accel=%s selected=%s residual_layers=%s",
        PATCH_FIT_MODE_XNORM_NATIVEKV_RESIDUAL_MLP_V1,
        device,
        accel["accel_mode"],
        selected,
        residual_layers,
    )

    native_components_by_layer = _load_consumer_native_components(
        args.models_config,
        selected,
        hidden_dim,
        kv_dim,
        device,
    )

    direct_stats_by_layer: dict[int, dict[str, torch.Tensor]] = {
        layer: create_direct_kv_stats_torch(hidden_dim, kv_dim, device) for layer in selected
    }
    chunk_paths = list(_iter_chunk_paths(args.pairs))

    start = time.perf_counter()
    for idx, chunk_path in enumerate(chunk_paths, start=1):
        bundle = load_bundle(chunk_path, device=str(device), dtype="float32")
        arrays = bundle.arrays
        for start_idx in range(0, len(selected), layer_block_size):
            block_layers = selected[start_idx : start_idx + layer_block_size]
            for layer in block_layers:
                xa = arrays[f"xnorm_a.{int(layer)}"].reshape(-1, hidden_dim).to(device=device, dtype=torch.float32)
                kb_raw, vb = _project_consumer_native_kv_from_xnorm_b(arrays, layer, hidden_dim, device, native_components_by_layer[int(layer)])
                accumulate_direct_kv_stats_torch(xa, kb_raw, vb, direct_stats_by_layer[int(layer)])
        del arrays, bundle
        clear_torch_cache()
        if idx == 1 or idx % 5 == 0 or idx == len(chunk_paths):
            elapsed = time.perf_counter() - start
            eta_s = (elapsed / idx) * (len(chunk_paths) - idx) if idx > 0 else 0.0
            logger.info("xnorm native-kv stats progress: chunk %d/%d elapsed=%.1fs eta=%.1fs", idx, len(chunk_paths), elapsed, eta_s)

    out_arrays: dict[str, object] = {}
    metrics: dict[str, dict[str, float]] = {}
    wk_by_layer: dict[int, torch.Tensor] = {}
    wv_by_layer: dict[int, torch.Tensor] = {}

    for layer in selected:
        wk, wv, _pk, _pv, _pkv = solve_direct_kv_from_stats_torch(direct_stats_by_layer[int(layer)], ridge_lambda=ridge_lambda)
        wk_by_layer[int(layer)] = wk.to(device=device, dtype=torch.float32)
        wv_by_layer[int(layer)] = wv.to(device=device, dtype=torch.float32)
        out_arrays[f"direct_wk.{int(layer)}"] = wk.detach().to(device="cpu", dtype=torch.float32).contiguous()
        out_arrays[f"direct_wv.{int(layer)}"] = wv.detach().to(device="cpu", dtype=torch.float32).contiguous()

    del direct_stats_by_layer
    clear_torch_cache()

    for layer in selected:
        error_stats = create_direct_kv_error_stats_torch(device)
        for chunk_path in chunk_paths:
            bundle = load_bundle(chunk_path, device=str(device), dtype="float32")
            arrays = bundle.arrays
            xa = arrays[f"xnorm_a.{int(layer)}"].reshape(-1, hidden_dim).to(device=device, dtype=torch.float32)
            kb_raw, vb = _project_consumer_native_kv_from_xnorm_b(arrays, layer, hidden_dim, device, native_components_by_layer[int(layer)])
            accumulate_direct_kv_error_stats_torch(xa, kb_raw, vb, wk_by_layer[int(layer)], wv_by_layer[int(layer)], error_stats)
            del arrays, bundle
            clear_torch_cache()
        patch_k_error, patch_v_error, patch_kv_error = finalize_direct_kv_error_stats_torch(error_stats, kv_dim=kv_dim)
        metrics[str(int(layer))] = {
            "linear_base_k_rel_mse": float(patch_k_error),
            "linear_base_v_rel_mse": float(patch_v_error),
            "linear_base_kv_rel_mse": float(patch_kv_error),
        }
        logger.info(
            "linear base solved: layer=%d linear_base_k_rel_mse=%.6f linear_base_v_rel_mse=%.6f linear_base_kv_rel_mse=%.6f",
            int(layer),
            patch_k_error,
            patch_v_error,
            patch_kv_error,
        )

    for layer in residual_layers:
        xa_all, xb_all, k_all, v_all = _preload_xnorm_nativekv_residual_dataset_for_layer(
            chunk_paths=chunk_paths,
            layer=int(layer),
            hidden_dim=hidden_dim,
            device=device,
            native_components=native_components_by_layer[int(layer)],
        )
        residual_arrays, residual_metrics = _train_xnorm_nativekv_residual_layer(
            layer=int(layer),
            xa=xa_all,
            xb_target=xb_all,
            k_target=k_all,
            v_target=v_all,
            patch_cfg=patch_cfg,
            device=device,
            native_components=native_components_by_layer[int(layer)],
            logger=logger,
        )
        out_arrays.update(residual_arrays)
        metrics[str(int(layer))].update(residual_metrics)
        logger.info(
            "xnorm native-kv residual layer complete: layer=%d base_h_rel_mse=%.6f -> %.6f, base_k_rel_mse=%.6f -> %.6f, base_v_rel_mse=%.6f -> %.6f",
            int(layer),
            metrics[str(int(layer))]["base_h_rel_mse"],
            metrics[str(int(layer))]["h_rel_mse"],
            metrics[str(int(layer))]["base_k_rel_mse"],
            metrics[str(int(layer))]["k_rel_mse"],
            metrics[str(int(layer))]["base_v_rel_mse"],
            metrics[str(int(layer))]["v_rel_mse"],
        )
        del xa_all, xb_all, k_all, v_all
        clear_torch_cache()

    meta_payload = {
        "selected_layers": selected,
        "residual_layers": residual_layers,
        "hidden_dim": hidden_dim,
        "code_dim": hidden_dim,
        "num_key_value_heads": int(meta["num_key_value_heads"]),
        "head_dim": int(meta["head_dim"]),
        "k_space": "consumer_prenorm_k_native_v",
        "rope": {"model_b": rope_b.to_dict()},
        "fit_mode": PATCH_FIT_MODE_XNORM_NATIVEKV_RESIDUAL_MLP_V1,
        "route_layers": [] if route is None else [int(layer) for layer in route.get("patch_layers", [])],
        "residual_hidden_dim": int(patch_cfg.get("patch_residual_hidden_dim") or 4096),
        "residual_num_blocks": int(patch_cfg.get("patch_residual_num_blocks") or 4),
        "residual_ffn_multiplier": float(patch_cfg.get("patch_residual_ffn_multiplier") or 2.0),
        "loss_weights": {
            "lambda_h": float(patch_cfg.get("lambda_h") or 1.0),
            "lambda_k": float(patch_cfg.get("lambda_k") or 1.0),
            "lambda_v": float(patch_cfg.get("lambda_v") or 1.0),
            "lambda_delta": float(patch_cfg.get("lambda_delta") or 1e-3),
        },
    }
    save_patch_ckpt(args.out, out_arrays, meta_payload)
    save_json(Path(args.out) / "metrics.json", {"fit_mode": PATCH_FIT_MODE_XNORM_NATIVEKV_RESIDUAL_MLP_V1, "layers": metrics})
    logger.info("xnorm native-kv residual patch checkpoint saved to %s", args.out)
