"""``scd-vllm`` 命令行入口。

这个 facade 把 bundle 导出、ExampleConnector 转换、doctor、launch-spec、
launch-smoke 和一键 smoke pipeline 收在一个命令下。每个子命令仍委托
给更小的实现模块，便于测试和单独复用。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from scd_vllm import batch_benchmark, injection, reconstructed_export

from . import doctor
from .engine import (
    DEFAULT_PRELOADED_CONNECTOR,
    DEFAULT_PRELOADED_CONNECTOR_MODULE,
    SCDVLLMConfig,
    SCDVLLMEngine,
    build_preloaded_launch_spec,
    run_launch_smoke,
)


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
    if command == "launch-spec":
        return _launch_spec(rest)
    if command == "launch-smoke":
        return _launch_smoke(rest)
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


def _launch_spec(argv: list[str]) -> int:
    args = _build_launch_spec_parser().parse_args(argv)
    payload = build_preloaded_launch_spec(
        model=args.model,
        tokenizer=args.tokenizer,
        bundle_dir=args.bundle_dir,
        storage_dir=args.storage_dir,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        seed=args.seed,
        enforce_eager=args.enforce_eager,
        enable_prefix_caching=args.enable_prefix_caching,
        enable_chunked_prefill=args.enable_chunked_prefill,
        kv_connector=args.kv_connector,
        kv_connector_module_path=args.kv_connector_module_path,
        connector_cache_device=args.connector_cache_device,
        stats_dir=args.stats_dir,
        require_storage=not args.allow_missing_storage,
    )
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload.get("launch_spec_pass") else 1


def _launch_smoke(argv: list[str]) -> int:
    args = _build_launch_smoke_parser().parse_args(argv)
    if not args.child and args.out:
        return _launch_smoke_parent(args)
    metrics = run_launch_smoke(
        launch_spec=args.launch_spec,
        prompt=args.prompt,
        prefix_file=args.prefix_file,
        out=args.out,
        batch_size=args.batch_size,
        warmup_batches=args.warmup_batches,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
        use_tqdm=args.use_tqdm,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0 if metrics.get("launch_smoke_pass") else 1


def _launch_smoke_parent(args: argparse.Namespace) -> int:
    cmd = [
        sys.executable,
        "-m",
        "scd_vllm_plugin.cli",
        "launch-smoke",
        *_launch_smoke_child_argv(args),
        "--child",
    ]
    completed = subprocess.run(cmd, check=False, capture_output=True, text=True)
    out_path = Path(args.out)
    if completed.returncode != 0 and not out_path.is_file():
        if completed.stdout:
            print(completed.stdout, end="")
        if completed.stderr:
            print(completed.stderr, end="", file=sys.stderr)
        return completed.returncode
    metrics = json.loads(out_path.read_text(encoding="utf-8-sig"))
    stats_dir = metrics.get("preloaded_connector_stats_dir") or ""
    if stats_dir:
        metrics["preloaded_connector_stats"] = batch_benchmark._collect_preloaded_connector_stats(Path(stats_dir))
        metrics["preloaded_connector_final_stats_collected_after_child_exit"] = True
        out_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if completed.returncode != 0:
        metrics["child_returncode"] = completed.returncode
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0 if metrics.get("launch_smoke_pass") else 1


def _launch_smoke_child_argv(args: argparse.Namespace) -> list[str]:
    argv = [
        "--launch-spec",
        args.launch_spec,
        "--out",
        args.out,
        "--batch-size",
        str(args.batch_size),
        "--warmup-batches",
        str(args.warmup_batches),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--temperature",
        str(args.temperature),
        "--top-p",
        str(args.top_p),
        "--seed",
        str(args.seed),
    ]
    if args.prompt:
        argv.extend(["--prompt", args.prompt])
    if args.prefix_file:
        argv.extend(["--prefix-file", args.prefix_file])
    if args.use_tqdm:
        argv.append("--use-tqdm")
    return argv


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


def _build_launch_spec_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scd-vllm launch-spec",
        description="Emit official-vLLM LLM kwargs for loading a trusted SCD bundle with the external connector.",
    )
    parser.add_argument("--model", required=True, help="vLLM consumer model id or local path")
    parser.add_argument("--tokenizer", default="", help="Tokenizer id/path; defaults to --model")
    parser.add_argument("--bundle-dir", required=True, help="SCD bundle directory with manifest.json")
    parser.add_argument("--storage-dir", required=True, help="ExampleConnector-compatible storage directory")
    parser.add_argument("--out", default="", help="Optional JSON file to write the launch spec")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--enable-prefix-caching", action="store_true")
    parser.add_argument("--enable-chunked-prefill", action="store_true")
    parser.add_argument("--kv-connector", default=DEFAULT_PRELOADED_CONNECTOR)
    parser.add_argument("--kv-connector-module-path", default=DEFAULT_PRELOADED_CONNECTOR_MODULE)
    parser.add_argument("--connector-cache-device", default="cuda", choices=["cuda", "gpu", "cpu"])
    parser.add_argument("--stats-dir", default="", help="Optional SCD_PRELOADED_STATS_DIR for connector stats")
    parser.add_argument(
        "--allow-missing-storage",
        action="store_true",
        help="Emit a spec without requiring storage safetensors; intended only for templates/docs.",
    )
    return parser


def _build_launch_smoke_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scd-vllm launch-smoke",
        description="Instantiate official vLLM from a launch-spec JSON and run one trusted-prefix smoke batch.",
    )
    parser.add_argument("--launch-spec", required=True, help="JSON emitted by scd-vllm launch-spec")
    parser.add_argument("--prompt", default="", help="Prompt text; defaults to bundle prefix.txt")
    parser.add_argument("--prefix-file", default="", help="File containing prompt text")
    parser.add_argument("--out", default="", help="Optional JSON file to write launch-smoke metrics")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup-batches", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use-tqdm", action="store_true")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
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
                "  launch-spec     Emit official-vLLM LLM kwargs for a trusted SCD bundle/storage pair",
                "  launch-smoke    Consume a launch-spec JSON through official vLLM LLM and generate",
                "  run             Export a bundle, convert it, and run the smoke test",
                "",
                "Examples:",
                "  scd-vllm doctor --reuse-ckpt /path/to/reuse --patch-ckpt /path/to/patch --route configs/routes/my_route.json --skip-vllm-import",
                "  scd-vllm doctor --bundle-dir runs/scd_vllm/my_bundle --route runs/scd_vllm/my_bundle/route.json",
                "  scd-vllm export-bundle --producer-model ... --consumer-model ... --reuse-ckpt ... --patch-ckpt ... --route ... --out runs/scd_vllm/my_bundle",
                "  scd-vllm convert-bundle --bundle-dir runs/scd_vllm/my_bundle --storage-dir runs/scd_vllm/my_storage --overwrite",
                "  scd-vllm smoke-test --bundle-dir runs/scd_vllm/my_bundle --storage-dir runs/scd_vllm/my_storage --model /path/to/consumer --out runs/scd_vllm/my_smoke --overwrite",
                "  scd-vllm batch-benchmark --bundle-dir runs/scd_vllm/my_bundle --storage-dir runs/scd_vllm/my_storage --model /path/to/consumer --batch-size 64 --out runs/scd_vllm/my_batch_bench",
                "  scd-vllm launch-spec --bundle-dir runs/scd_vllm/my_bundle --storage-dir runs/scd_vllm/my_storage --model /path/to/consumer --out runs/scd_vllm/launch_spec.json",
                "  scd-vllm launch-smoke --launch-spec runs/scd_vllm/launch_spec.json --out runs/scd_vllm/launch_smoke.json",
            ]
        )
        + "\n"
    )


if __name__ == "__main__":
    raise SystemExit(main())
