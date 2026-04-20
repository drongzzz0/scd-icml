from __future__ import annotations

import argparse
import sys

from ..io_utils import load_json, save_json
from ..pipeline import load_patch_ckpt, load_reuse_weights, run_inference_method_over_pairs, save_prediction_run
from ..seed import set_global_seed
from ..tracking import command_from_argv, write_manifest
from ..torch_utils import configure_torch_acceleration
from .common import add_accel_args, load_cfg, resolve_accel_options, resolve_logger


SUPPORTED_METHODS = [
    "oracle",
    "raw_kv",
    "quantized_4bit",
    "selective_recompute",
    "reuse_only",
    "scd_full",
]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="执行基线或 SCD 推理方法")
    p.add_argument("--pairs", required=True)
    p.add_argument("--reuse-ckpt", default="")
    p.add_argument("--patch-ckpt", default="")
    p.add_argument("--route", default="")
    p.add_argument("--models-config", default="")
    p.add_argument("--eval-config", required=True)
    p.add_argument("--method", choices=SUPPORTED_METHODS, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log", default="")
    p.add_argument("--infer-device", default="")
    add_accel_args(p)
    return p


def main() -> None:
    args = build_parser().parse_args()
    logger = resolve_logger(args.log)
    set_global_seed(args.seed)

    eval_cfg = load_cfg(args.eval_config)
    accel = resolve_accel_options(eval_cfg, args)
    configure_torch_acceleration(str(accel["accel_mode"]))
    logger.info("计算加速模式: %s", accel["accel_mode"])
    bandwidth = float(eval_cfg["network"]["bandwidth_gbps"])
    infer_device = str(args.infer_device or eval_cfg.get("compute", {}).get("infer_device") or "cuda:0")
    cpu_fallback = bool(eval_cfg.get("runtime", {}).get("cpu_fallback", True))

    reuse = None
    if args.reuse_ckpt:
        reuse, reuse_meta = load_reuse_weights(args.reuse_ckpt, device=infer_device, dtype="float32", cpu_fallback=cpu_fallback)
        logger.info(
            "reuse mainline: mode=%s translator.default_path=%s num_groups=%s group_size=%s rank_k=%s rank_v=%s k_derope=%s k_rerope=%s",
            reuse_meta.get("mode", "xkv_group"),
            reuse_meta.get("translator", {}).get("default_path", "shared_z"),
            reuse_meta.get("num_groups"),
            reuse_meta.get("group_size"),
            reuse_meta.get("rank_k", reuse_meta.get("rank")),
            reuse_meta.get("rank_v", reuse_meta.get("rank")),
            reuse_meta.get("k_de_rope", True),
            reuse_meta.get("k_re_rope", True),
        )

    patch_bundle = None
    patch_meta = None
    if args.patch_ckpt:
        patch_bundle, patch_meta = load_patch_ckpt(args.patch_ckpt, device=infer_device, dtype="float32", cpu_fallback=cpu_fallback)

    route = None
    if args.route:
        route = load_json(args.route)

    pred = run_inference_method_over_pairs(
        pairs_path=args.pairs,
        method=args.method,
        route=route,
        reuse=reuse,
        patch_bundle=patch_bundle,
        patch_meta=patch_meta,
        models_config=args.models_config or None,
        bandwidth_gbps=bandwidth,
        seed=args.seed,
        infer_device=infer_device,
        cpu_fallback=cpu_fallback,
    )
    oracle = run_inference_method_over_pairs(
        pairs_path=args.pairs,
        method="oracle",
        route=route,
        reuse=reuse,
        patch_bundle=patch_bundle,
        patch_meta=patch_meta,
        models_config=args.models_config or None,
        bandwidth_gbps=bandwidth,
        seed=args.seed,
        infer_device=infer_device,
        cpu_fallback=cpu_fallback,
    )

    save_prediction_run(args.out, pred=pred, oracle=oracle)
    save_json(f"{args.out}/cache_audit.json", {"pred": pred["cache_audit"], "oracle": oracle["cache_audit"]})
    write_manifest(out_dir=args.out, command=command_from_argv(sys.argv), config_paths={"eval": args.eval_config}, seed=args.seed)
    logger.info("推理结果已保存到 %s", args.out)


if __name__ == "__main__":
    main()

