from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load an scd-vllm launch-spec JSON and either dry-run validate it "
            "or execute a trusted-prefix official-vLLM generation smoke."
        )
    )
    parser.add_argument("--launch-spec", required=True, help="JSON emitted by `scd-vllm launch-spec`")
    parser.add_argument("--prompt", default="", help="Optional prompt override for the smoke run")
    parser.add_argument("--prefix-file", default="", help="Optional prompt file override for the smoke run")
    parser.add_argument("--out", default="", help="Optional JSON output path for dry-run or smoke metrics")
    parser.add_argument("--batch-size", type=int, default=1, help="Number of identical trusted-prefix requests")
    parser.add_argument("--warmup-batches", type=int, default=0, help="Warmup batches before measured generation")
    parser.add_argument("--max-new-tokens", type=int, default=16, help="Generated tokens per request")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use-tqdm", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and summarize the launch spec without importing vLLM or using a GPU",
    )
    return parser.parse_args(argv)


def load_launch_spec(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"launch spec must be a JSON object: {path}")
    return payload


def summarize_launch_spec(payload: dict[str, Any], *, launch_spec: str | Path, dry_run: bool) -> dict[str, Any]:
    llm_kwargs = payload.get("llm_kwargs") or {}
    kv_transfer_config = llm_kwargs.get("kv_transfer_config") or {}
    bundle = payload.get("bundle") or {}
    storage = payload.get("storage") or {}
    safety = payload.get("safety") or {}
    environment = payload.get("environment") or {}
    missing: list[str] = []
    if not llm_kwargs:
        missing.append("llm_kwargs")
    if not kv_transfer_config:
        missing.append("llm_kwargs.kv_transfer_config")
    if not payload.get("bundle_dir"):
        missing.append("bundle_dir")
    if not payload.get("storage_dir"):
        missing.append("storage_dir")
    if storage.get("exists") is not True or int(storage.get("safetensor_files", 0) or 0) <= 0:
        missing.append("storage.safetensor_files")
    if kv_transfer_config.get("kv_connector") != "SCDPreloadedConnector":
        missing.append("kv_transfer_config.kv_connector=SCDPreloadedConnector")
    if kv_transfer_config.get("kv_connector_module_path") != "scd_vllm.preloaded_connector":
        missing.append("kv_transfer_config.kv_connector_module_path=scd_vllm.preloaded_connector")
    if llm_kwargs.get("disable_hybrid_kv_cache_manager") is not True:
        missing.append("llm_kwargs.disable_hybrid_kv_cache_manager=True")
    if safety.get("trusted_bundle_required") is not True:
        missing.append("safety.trusted_bundle_required=True")

    return {
        "mode": "scd_vllm_official_launch_example",
        "dry_run": bool(dry_run),
        "ready": not missing,
        "missing": missing,
        "launch_spec": str(launch_spec),
        "model": llm_kwargs.get("model"),
        "tokenizer": llm_kwargs.get("tokenizer"),
        "bundle_dir": payload.get("bundle_dir"),
        "storage_dir": payload.get("storage_dir"),
        "route_id": bundle.get("route_id"),
        "route_hash": bundle.get("route_hash"),
        "token_hash": bundle.get("token_hash"),
        "kv_connector": kv_transfer_config.get("kv_connector"),
        "kv_connector_module_path": kv_transfer_config.get("kv_connector_module_path"),
        "disable_hybrid_kv_cache_manager": bool(llm_kwargs.get("disable_hybrid_kv_cache_manager")),
        "environment": environment,
        "storage": storage,
        "safety": safety,
        "claim_boundary": (
            "single-tenant trusted-bundle official-vLLM launch helper; "
            "not a multi-tenant cache-sharing or dynamic-routing server"
        ),
    }


def write_or_print(metrics: dict[str, Any], out: str | Path = "") -> None:
    text = json.dumps(metrics, ensure_ascii=False, indent=2) + "\n"
    if out:
        out_path = Path(out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
    else:
        print(text, end="")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = load_launch_spec(args.launch_spec)
    if args.dry_run:
        metrics = summarize_launch_spec(payload, launch_spec=args.launch_spec, dry_run=True)
        write_or_print(metrics, args.out)
        return 0 if metrics["ready"] else 2

    from scd_vllm_plugin.engine import run_launch_smoke

    metrics = run_launch_smoke(
        launch_spec=args.launch_spec,
        prompt=args.prompt,
        prefix_file=args.prefix_file,
        out="",
        batch_size=args.batch_size,
        warmup_batches=args.warmup_batches,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
        use_tqdm=args.use_tqdm,
    )
    metrics["example_mode"] = "scd_vllm_official_launch_example"
    write_or_print(metrics, args.out)
    return 0 if metrics.get("launch_smoke_pass") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
