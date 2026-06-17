from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from scd_vllm import batch_benchmark, injection, reconstructed_export

from . import doctor
from .engine import SCDVLLMConfig, SCDVLLMEngine


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help"}:
        _print_help()
        return 0
    command, rest = args[0], args[1:]
    if command == "export-bundle":
        return reconstructed_export.main(rest)
    if command == "convert-bundle":
        return injection.main(["convert", *rest])
    if command == "smoke-test":
        return injection.main(["run", *rest])
    if command == "batch-benchmark":
        return batch_benchmark.main(rest)
    if command == "doctor":
        return doctor.main(rest)
    if command == "run":
        return _run_pipeline(rest)
    print(f"unknown command: {command}", file=sys.stderr)
    _print_help(file=sys.stderr)
    return 2


def _run_pipeline(argv: list[str]) -> int:
    args = _build_run_parser().parse_args(argv)
    cfg = SCDVLLMConfig(
        producer_model=args.producer_model,
        consumer_model=args.consumer_model,
        tokenizer=args.tokenizer,
        reuse_ckpt=args.reuse_ckpt,
        patch_ckpt=args.patch_ckpt,
        route=args.route,
        dtype=args.dtype,
        transfer_dtype=args.transfer_dtype,
        producer_device=args.producer_device,
        consumer_device=args.consumer_device,
        patch_ckpt_device=args.patch_ckpt_device,
        max_prefix_tokens=args.max_prefix_tokens,
        bandwidth_gbps=args.bandwidth_gbps,
        quant_group_size=args.quant_group_size,
        runtime_patch_window_size=args.runtime_patch_window_size,
        runtime_patch_persist_shards=args.runtime_patch_persist_shards,
        runtime_patch_persist_modules=args.runtime_patch_persist_modules,
    )
    engine = SCDVLLMEngine(cfg)
    metrics = engine.run_smoke_pipeline(
        out_dir=args.out_dir,
        prompt=args.prompt,
        prefix_file=args.prefix_file,
        bundle_name=args.bundle_name,
        block_size=args.block_size,
        overwrite=args.overwrite,
        max_model_len=args.max_model_len,
        max_new_tokens=args.max_new_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
        enable_prefix_caching=args.enable_prefix_caching,
        enable_chunked_prefill=args.enable_chunked_prefill,
        disable_hybrid_kv_cache_manager=args.disable_hybrid_kv_cache_manager,
    )
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "plugin_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0 if metrics.get("pass") else 1


def _build_run_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scd-vllm run",
        description="Export an SCD reuse_patch bundle and smoke-test it through vLLM connector storage.",
    )
    parser.add_argument("--producer-model", required=True)
    parser.add_argument("--consumer-model", required=True)
    parser.add_argument("--tokenizer", default="")
    parser.add_argument("--reuse-ckpt", required=True)
    parser.add_argument("--patch-ckpt", required=True)
    parser.add_argument("--route", required=True)
    parser.add_argument("--prompt", default="")
    parser.add_argument("--prefix-file", default="")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--bundle-name", default="reuse_patch_bundle")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--transfer-dtype", default="")
    parser.add_argument("--producer-device", default="cuda:0")
    parser.add_argument("--consumer-device", default="cuda:1")
    parser.add_argument("--patch-ckpt-device", default="cpu")
    parser.add_argument("--max-prefix-tokens", type=int, default=256)
    parser.add_argument("--bandwidth-gbps", type=float, default=200.0)
    parser.add_argument("--quant-group-size", type=int, default=128)
    parser.add_argument("--runtime-patch-window-size", type=int, default=8)
    parser.add_argument("--runtime-patch-persist-shards", action="store_true")
    parser.add_argument("--runtime-patch-persist-modules", action="store_true")
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--enable-prefix-caching", action="store_true")
    parser.add_argument("--enable-chunked-prefill", action="store_true")
    parser.add_argument("--disable-hybrid-kv-cache-manager", action="store_true")
    return parser


def _print_help(file=None) -> None:
    stream = sys.stdout if file is None else file
    stream.write(
        "\n".join(
            [
                "usage: scd-vllm <command> [options]",
                "",
                "Commands:",
                "  export-bundle   Forward to scd_vllm.reconstructed_export",
                "  convert-bundle  Convert an SCD bundle to vLLM ExampleConnector storage",
                "  smoke-test      Run vLLM baseline/load smoke test for an existing bundle",
                "  batch-benchmark Run repeated-prefix batched vLLM load benchmark for an existing bundle/storage pair",
                "  doctor          Check plugin prerequisites and strict route coverage",
                "  run             Export a bundle, convert it, and run the smoke test",
                "",
                "Examples:",
                "  scd-vllm doctor --reuse-ckpt /path/to/reuse --patch-ckpt /path/to/patch --route configs/routes/my_route.json --skip-vllm-import",
                "  scd-vllm doctor --bundle-dir runs/scd_vllm/my_bundle --route runs/scd_vllm/my_bundle/route.json",
                "  scd-vllm export-bundle --producer-model ... --consumer-model ... --reuse-ckpt ... --patch-ckpt ... --route ... --out runs/scd_vllm/my_bundle",
                "  scd-vllm convert-bundle --bundle-dir runs/scd_vllm/my_bundle --storage-dir runs/scd_vllm/my_storage --overwrite",
                "  scd-vllm smoke-test --bundle-dir runs/scd_vllm/my_bundle --storage-dir runs/scd_vllm/my_storage --model /path/to/consumer --out runs/scd_vllm/my_smoke --overwrite",
                "  scd-vllm batch-benchmark --bundle-dir runs/scd_vllm/my_bundle --storage-dir runs/scd_vllm/my_storage --model /path/to/consumer --batch-size 64 --out runs/scd_vllm/my_batch_bench",
            ]
        )
        + "\n"
    )


if __name__ == "__main__":
    raise SystemExit(main())
