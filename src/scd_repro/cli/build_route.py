from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from ..io_utils import load_bundle, load_json, save_json
from ..patch import (
    PATCH_FIT_MODE_XNORM_SEMANTIC_V1,
    load_semantic_patch_layer_from_ckpt,
    load_semantic_vocab_head_from_ckpt,
    load_semantic_probe_from_ckpt,
)
from ..pipeline import load_patch_ckpt, load_pairs_meta
from ..routing import (
    CandidateRouteDecision,
    LayerProfile,
    RouteSearchConfig,
    TaskLayerProfile,
    build_route,
    compute_adaptive_thresholds,
    greedy_search_task_route,
)
from ..torch_utils import clear_torch_cache, resolve_device
from .common import load_cfg, resolve_logger


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="根据层分析结果或 task-level 搜索结果构建复用、补丁和回退路由")
    p.add_argument("--profile", default="")
    p.add_argument("--pairs", default="")
    p.add_argument("--patch-ckpt", default="")
    p.add_argument("--route-input", default="")
    p.add_argument("--eval-config", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--log", default="")
    p.add_argument("--search-device", default="")
    return p


def _iter_chunk_paths(pairs_dir: str):
    root = Path(pairs_dir)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    for row in manifest.get("chunks", []):
        yield root / str(row["path"])


def _legacy_build_route(profile: dict, eval_cfg: dict[str, Any]) -> dict[str, Any]:
    route_cfg = eval_cfg.get("route", {})
    threshold_mode = str(route_cfg.get("threshold_mode") or "fixed")
    patch_budget = int(route_cfg["patch_budget"])
    patch_metric = str(profile.get("patch_metric") or "patch_error")

    rows = [
        LayerProfile(
            layer=int(x["layer"]),
            reuse_error=float(x["reuse_error"]),
            patch_error=float(x.get("patch_kv_error", x["patch_error"])),
        )
        for x in profile["layers"]
    ]
    reuse_errors = [float(row.reuse_error) for row in rows]

    if threshold_mode == "adaptive_p20_p80":
        reuse_threshold, fallback_threshold = compute_adaptive_thresholds(reuse_errors, reuse_percentile=20.0, fallback_percentile=80.0)
    else:
        reuse_threshold = float(route_cfg["reuse_threshold"])
        fallback_threshold = float(route_cfg["fallback_threshold"])

    route = build_route(rows, reuse_threshold=reuse_threshold, fallback_threshold=fallback_threshold, patch_budget=patch_budget)
    route.update(
        {
            "threshold_mode": threshold_mode,
            "reuse_threshold": float(reuse_threshold),
            "fallback_threshold": float(fallback_threshold),
            "patch_budget": int(patch_budget),
            "num_layers": len(rows),
            "patch_metric": patch_metric,
            "patch_gains": {str(row.layer): float(max(0.0, row.reuse_error - row.patch_error)) for row in rows},
        }
    )
    return route


def _teacher_forced_logp_sum(student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> tuple[float, int]:
    teacher_targets = torch.argmax(teacher_logits.detach(), dim=-1)
    student_log_probs = torch.log_softmax(student_logits, dim=-1)
    gathered = student_log_probs.gather(dim=-1, index=teacher_targets.unsqueeze(-1)).squeeze(-1)
    return float(gathered.sum().item()), int(gathered.numel())


def _resolve_candidate_layers(
    *,
    num_layers: int,
    patch_meta: dict[str, Any],
    profile: dict[str, Any] | None,
    route_input: dict[str, Any] | None,
) -> list[int]:
    if route_input:
        pool = [int(layer) for layer in route_input.get("patch_layers", [])] + [int(layer) for layer in route_input.get("fallback_layers", [])]
        if pool:
            return sorted(set(pool))
    patch_selected = [int(layer) for layer in patch_meta.get("selected_layers", [])]
    if patch_selected:
        return sorted(set(patch_selected))
    if profile and "recommended_patch_layers" in profile:
        return sorted({int(layer) for layer in profile["recommended_patch_layers"]})
    return list(range(int(num_layers)))


def _route_config_from_eval(eval_cfg: dict[str, Any]) -> RouteSearchConfig:
    route_cfg = eval_cfg.get("route", {}) if isinstance(eval_cfg.get("route", {}), dict) else {}
    network_cfg = eval_cfg.get("network", {}) if isinstance(eval_cfg.get("network", {}), dict) else {}
    lengths = route_cfg.get("recompute_segment_lengths", [1, 2, 4])
    if not isinstance(lengths, list):
        lengths = [lengths]
    return RouteSearchConfig(
        patch_budget=int(route_cfg.get("patch_budget", 0)),
        top_m_candidates=int(route_cfg.get("top_m_candidates", 0)),
        quality_metric=str(route_cfg.get("quality_metric") or "delta_logp"),
        cost_metric=str(route_cfg.get("cost_metric") or "delta_ttft_ms"),
        route_objective=str(route_cfg.get("route_objective") or "gain_per_cost"),
        latency_budget_ms=None if route_cfg.get("latency_budget_ms", None) in {"", None} else float(route_cfg.get("latency_budget_ms")),
        min_positive_gain=float(route_cfg.get("min_positive_gain", 0.0)),
        allow_patch_only=bool(route_cfg.get("allow_patch_only", True)),
        allow_patch_plus_recompute=bool(route_cfg.get("allow_patch_plus_recompute", True)),
        enable_short_recompute=bool(route_cfg.get("enable_short_recompute", False)),
        recompute_segment_lengths=[int(x) for x in lengths if int(x) > 0],
        recompute_eval_mode=str(route_cfg.get("recompute_eval_mode") or "best_of_lengths"),
        bandwidth_gbps=float(network_cfg.get("bandwidth_gbps", 200.0)),
        cache_search_results=bool(route_cfg.get("cache_search_results", True)),
        max_evals=int(route_cfg.get("max_evals", 0)),
        recompute_cost_per_layer_ms=float(route_cfg.get("recompute_cost_per_layer_ms", 1.0)),
        patch_compute_cost_ms_per_layer=float(route_cfg.get("patch_compute_cost_ms_per_layer", 0.0)),
        weighted_cost_ttft_coeff=float(route_cfg.get("weighted_cost_ttft_coeff", 1.0)),
        weighted_cost_bytes_coeff=float(route_cfg.get("weighted_cost_bytes_coeff", 0.0)),
    )


def _build_task_profiles_from_pairs(
    *,
    pairs_path: str,
    patch_bundle: dict[str, Any],
    patch_meta: dict[str, Any],
    candidate_layers: list[int],
    device: torch.device,
) -> dict[int, TaskLayerProfile]:
    if str(patch_meta.get("fit_mode") or "") != PATCH_FIT_MODE_XNORM_SEMANTIC_V1:
        raise ValueError(f"task-level route search 当前仅支持 fit_mode={PATCH_FIT_MODE_XNORM_SEMANTIC_V1}")

    pairs_meta = load_pairs_meta(pairs_path)
    num_layers = int(pairs_meta["num_layers"])
    hidden_dim = int(pairs_meta["hidden_dim"])
    patch_layers = {int(layer) for layer in patch_meta.get("selected_layers", [])}

    semantic_probe_shared = bool(patch_meta.get("semantic_probe_shared", True))
    shared_probe = load_semantic_probe_from_ckpt(patch_bundle, patch_meta, device=device) if semantic_probe_shared else None
    shared_vocab_head = None
    if str(patch_meta.get("semantic_target_space") or "") == "next_token_logits" and "semantic_vocab_head.weight" in patch_bundle:
        shared_vocab_head = load_semantic_vocab_head_from_ckpt(patch_bundle, patch_meta, device=device)
    state_target_space = str(patch_meta.get("state_target_space") or "xnorm")
    patch_modules = {
        int(layer): load_semantic_patch_layer_from_ckpt(patch_bundle, patch_meta, layer=int(layer), device=device)
        for layer in candidate_layers
        if int(layer) in patch_layers
    }

    logp_sums = {
        layer: {"reuse": 0.0, "patch": 0.0, "recompute": 0.0, "tokens": 0, "patch_bytes": 0}
        for layer in range(num_layers)
    }

    for chunk_path in _iter_chunk_paths(pairs_path):
        bundle = load_bundle(chunk_path, device=str(device), dtype="float32")
        arrays = bundle.arrays
        for layer in range(num_layers):
            probe = shared_probe if semantic_probe_shared else load_semantic_probe_from_ckpt(patch_bundle, patch_meta, layer=layer, device=device)
            if state_target_space == "residual_pre_ln" and f"resid_a.{layer}" in arrays and f"resid_b.{layer}" in arrays:
                xa = arrays[f"resid_a.{layer}"].reshape(-1, hidden_dim).to(device=device, dtype=torch.float32)
                xb = arrays[f"resid_b.{layer}"].reshape(-1, hidden_dim).to(device=device, dtype=torch.float32)
            else:
                xa = arrays[f"xnorm_a.{layer}"].reshape(-1, hidden_dim).to(device=device, dtype=torch.float32)
                xb = arrays[f"xnorm_b.{layer}"].reshape(-1, hidden_dim).to(device=device, dtype=torch.float32)
            teacher_logits = probe(xb).detach()
            reuse_logits = probe(xa)
            if shared_vocab_head is not None:
                teacher_logits = shared_vocab_head(teacher_logits).detach()
                reuse_logits = shared_vocab_head(reuse_logits)
            reuse_logp_sum, token_count = _teacher_forced_logp_sum(reuse_logits, teacher_logits)
            patch_logp_sum: float | None = None
            if layer in patch_modules:
                xb_hat = patch_modules[layer](xa)
                patch_logits = probe(xb_hat)
                if shared_vocab_head is not None:
                    patch_logits = shared_vocab_head(patch_logits)
                patch_logp_sum, _ = _teacher_forced_logp_sum(patch_logits, teacher_logits)
                logp_sums[layer]["patch_bytes"] += int(arrays[f"xnorm_a.{layer}"].numel()) * int(arrays[f"xnorm_a.{layer}"].element_size())
            recompute_logp_sum, _ = _teacher_forced_logp_sum(teacher_logits, teacher_logits)
            logp_sums[layer]["reuse"] += float(reuse_logp_sum)
            if patch_logp_sum is not None:
                logp_sums[layer]["patch"] += float(patch_logp_sum)
            logp_sums[layer]["recompute"] += float(recompute_logp_sum)
            logp_sums[layer]["tokens"] += int(token_count)
        del arrays, bundle
        clear_torch_cache()

    profiles: dict[int, TaskLayerProfile] = {}
    for layer in range(num_layers):
        profiles[layer] = TaskLayerProfile(
            layer=layer,
            token_count=int(logp_sums[layer]["tokens"]),
            reuse_logp_sum=float(logp_sums[layer]["reuse"]),
            patch_logp_sum=None if int(layer) not in patch_modules else float(logp_sums[layer]["patch"]),
            recompute_logp_sum=float(logp_sums[layer]["recompute"]),
            patch_bytes=int(logp_sums[layer]["patch_bytes"]),
        )
    return profiles


def _task_level_build_route(
    *,
    args,
    logger,
    eval_cfg: dict[str, Any],
    profile: dict[str, Any] | None,
) -> dict[str, Any]:
    if not args.pairs or not args.patch_ckpt:
        raise ValueError("task-level route search 需要同时提供 --pairs 和 --patch-ckpt")
    route_input = load_json(args.route_input) if args.route_input else None
    search_device = resolve_device(args.search_device or eval_cfg.get("compute", {}).get("infer_device") or "cpu", cpu_fallback=True)
    patch_bundle, patch_meta = load_patch_ckpt(args.patch_ckpt, device=str(search_device), dtype="float32")
    pairs_meta = load_pairs_meta(args.pairs)
    candidate_layers = _resolve_candidate_layers(num_layers=int(pairs_meta["num_layers"]), patch_meta=patch_meta, profile=profile, route_input=route_input)
    config = _route_config_from_eval(eval_cfg)

    logger.info(
        "task-level route search: device=%s fit_mode=%s semantic_source=%s num_layers=%d candidates=%s",
        search_device,
        patch_meta.get("fit_mode"),
        patch_meta.get("semantic_source"),
        pairs_meta.get("num_layers"),
        candidate_layers,
    )

    task_profiles = _build_task_profiles_from_pairs(
        pairs_path=args.pairs,
        patch_bundle=patch_bundle,
        patch_meta=patch_meta,
        candidate_layers=candidate_layers,
        device=search_device,
    )
    artifact = greedy_search_task_route(task_profiles, candidate_layers=candidate_layers, config=config)
    artifact.update(
        {
            "fit_mode": str(patch_meta.get("fit_mode") or ""),
            "semantic_source": str(patch_meta.get("semantic_source") or ""),
            "route_objective": config.route_objective,
            "quality_metric": config.quality_metric,
            "cost_metric": config.cost_metric,
            "patch_budget": int(config.patch_budget),
            "latency_budget_ms": None if config.latency_budget_ms is None else float(config.latency_budget_ms),
            "top_m_candidates": int(config.top_m_candidates),
            "candidate_layers": candidate_layers,
            "recompute_eval_mode": config.recompute_eval_mode,
            "enable_short_recompute": bool(config.enable_short_recompute),
            "task_level_profile": True,
            "ttft_estimate_kind": "extra_over_full_reuse_surrogate",
            "recompute_eval_kind": "teacher_hidden_shim",
            "allow_patch_only": bool(config.allow_patch_only),
            "allow_patch_plus_recompute": bool(config.allow_patch_plus_recompute),
            "max_evals": int(config.max_evals),
            "cache_search_results": bool(config.cache_search_results),
            "restore_one_batch_size": int(eval_cfg.get("route", {}).get("restore_one_batch_size", 0) or 0),
            "greedy_batch_size": int(eval_cfg.get("route", {}).get("greedy_batch_size", 0) or 0),
        }
    )
    if route_input:
        artifact["route_input"] = route_input
    if profile:
        artifact["legacy_profile_diagnostics"] = {
            "patch_metric": profile.get("patch_metric"),
            "patch_fit_mode": profile.get("patch_fit_mode"),
            "recommended_patch_layers": profile.get("recommended_patch_layers", []),
        }
    return artifact


def main() -> None:
    args = build_parser().parse_args()
    logger = resolve_logger(args.log)
    eval_cfg = load_cfg(args.eval_config)
    route_cfg = eval_cfg.get("route", {}) if isinstance(eval_cfg.get("route", {}), dict) else {}
    profile = load_json(args.profile) if args.profile else None

    use_task_level = bool(route_cfg.get("enable_task_level_profile", False)) or bool(args.pairs and args.patch_ckpt)
    if use_task_level:
        route = _task_level_build_route(args=args, logger=logger, eval_cfg=eval_cfg, profile=profile)
        save_json(args.out, route)
        logger.info(
            "task-level 路由已保存到 %s，selected_patch_layers=%s，selected_recompute_layers=%s，stop_reason=%s",
            args.out,
            route.get("selected_patch_layers", []),
            route.get("selected_recompute_layers", []),
            route.get("stop_reason"),
        )
        return

    if profile is None:
        raise ValueError("legacy build_route 模式需要提供 --profile")
    route = _legacy_build_route(profile, eval_cfg)
    save_json(args.out, route)
    logger.info(
        "路由已保存到 %s，mode=%s，reuse_threshold=%.6f，fallback_threshold=%.6f",
        args.out,
        route["threshold_mode"],
        route["reuse_threshold"],
        route["fallback_threshold"],
    )


if __name__ == "__main__":
    main()
