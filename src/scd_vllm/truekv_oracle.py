"""TrueKV oracle 检查路径。

TrueKV oracle 用消费者模型真实 KV 构造上界/连通性检查，确认 vLLM
connector、token 对齐和 KV shape 在目标版本上可行。它不是 SCD 方法
本身，只是服务端接入前的安全校验。
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .bundle import array_sha256, text_sha256
from .hf_export import load_prompt


class TrueKVOracleError(RuntimeError):
    """Raised when the TrueKV oracle cannot proceed safely."""


CONNECTOR_METHODS = [
    "start_load_kv",
    "wait_for_layer_load",
    "save_kv_layer",
    "wait_for_save",
    "get_num_new_matched_tokens",
    "update_state_after_alloc",
    "build_connector_meta",
    "request_finished",
    "register_kv_caches",
    "get_finished",
]

CONNECTOR_CLASSES = {
    "KVConnectorBase_V1": "vllm.distributed.kv_transfer.kv_connector.v1.base:KVConnectorBase_V1",
    "ExampleConnector": "vllm.distributed.kv_transfer.kv_connector.v1.example_connector:ExampleConnector",
    "LMCacheConnectorV1": "vllm.distributed.kv_transfer.kv_connector.v1.lmcache_connector:LMCacheConnectorV1",
}

SOURCE_EVIDENCE_TERMS = [
    "slot_mapping",
    "_block_size",
    "dst_kv_cache_layer",
    "extract_kv_from_layer",
    "save_kv_layer",
    "start_load_kv",
]

DEFAULT_ORACLE_PROMPT = (
    "Semantic Cache Distillation transfers approximate semantic key value caches between compatible agents. " * 12
).strip()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect or run the SCD-vLLM TrueKV oracle")
    sub = parser.add_subparsers(dest="command", required=True)
    inspect_parser = sub.add_parser("inspect", help="Inspect installed vLLM KVConnector APIs")
    inspect_parser.add_argument("--out", default="runs/scd_vllm/truekv_oracle_inspection/metrics.json")
    inspect_parser.add_argument("--report-out", default="", help="Optional markdown inspection report")
    run_parser = sub.add_parser("run", help="Run a minimal TrueKV save/load oracle with vLLM's ExampleConnector")
    add_oracle_runtime_args(run_parser)
    child_parser = sub.add_parser("_generate", help=argparse.SUPPRESS)
    add_oracle_runtime_args(child_parser)
    child_parser.add_argument("--stage", required=True, choices=["baseline", "save", "load"])
    for name in ("save", "load"):
        p = sub.add_parser(name, help=f"{name} is available through the run orchestrator")
        add_oracle_runtime_args(p)
    return parser.parse_args(argv)


def add_oracle_runtime_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default="", help="vLLM model id or local path")
    parser.add_argument("--tokenizer", default="", help="Tokenizer id/path; defaults to --model")
    parser.add_argument("--prompt", default="", help="Prefix prompt text; defaults to an internal long static prompt")
    parser.add_argument("--prefix-file", default="", help="File containing prefix prompt text")
    parser.add_argument("--out", required=True, help="Output directory or JSON path")
    parser.add_argument("--storage-dir", default="", help="Directory used by the vLLM ExampleConnector")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-model-len", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--enable-prefix-caching", action="store_true")
    parser.add_argument("--enable-chunked-prefill", action="store_true")
    parser.add_argument("--disable-hybrid-kv-cache-manager", action="store_true")


def inspect_vllm_kv_api(out: str | Path, report_out: str | Path | None = None) -> dict[str, Any]:
    vllm = importlib.import_module("vllm")
    payload: dict[str, Any] = {
        "mode": "truekv_oracle_inspection",
        "oracle_pass": False,
        "vllm_version": getattr(vllm, "__version__", None),
        "vllm_root": str(Path(vllm.__file__).resolve().parent),
        "connector_classes": {},
        "slot_mapping_evidence": {},
        "required_next_step": "implement true KV save/load connector using inspected APIs; do not run SCD injection yet",
        "blocker": "TrueKV save/load oracle is not implemented yet; only vLLM API inspection has completed.",
    }
    for label, target in CONNECTOR_CLASSES.items():
        module_name, attr_name = target.split(":", 1)
        try:
            module = importlib.import_module(module_name)
            cls = getattr(module, attr_name)
        except Exception as exc:
            payload["connector_classes"][label] = {"available": False, "error": str(exc)}
            continue
        payload["connector_classes"][label] = describe_class(cls, CONNECTOR_METHODS)
        payload["slot_mapping_evidence"][label] = extract_source_evidence_from_object(cls, SOURCE_EVIDENCE_TERMS)

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if report_out:
        report_path = Path(report_out)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(render_inspection_markdown(payload), encoding="utf-8")
    return payload


def run_truekv_oracle(args: argparse.Namespace) -> dict[str, Any]:
    if not args.model:
        raise TrueKVOracleError("--model is required for truekv_oracle run")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    storage_dir = Path(args.storage_dir) if args.storage_dir else out_dir / "example_connector_storage"
    if storage_dir.exists():
        shutil.rmtree(storage_dir)
    storage_dir.mkdir(parents=True, exist_ok=True)

    prompt = _resolve_prompt(args.prompt, args.prefix_file)
    common = [
        "--model",
        args.model,
        "--tokenizer",
        args.tokenizer or args.model,
        "--prompt",
        prompt,
        "--storage-dir",
        str(storage_dir),
        "--dtype",
        args.dtype,
        "--max-model-len",
        str(args.max_model_len),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--seed",
        str(args.seed),
        "--temperature",
        str(args.temperature),
        "--top-p",
        str(args.top_p),
    ]
    for flag, enabled in (
        ("--enforce-eager", args.enforce_eager),
        ("--enable-prefix-caching", args.enable_prefix_caching),
        ("--enable-chunked-prefill", args.enable_chunked_prefill),
        ("--disable-hybrid-kv-cache-manager", args.disable_hybrid_kv_cache_manager),
    ):
        if enabled:
            common.append(flag)

    records: dict[str, dict[str, Any]] = {}
    for stage in ("baseline", "save", "load"):
        stage_out = out_dir / f"{stage}.json"
        cmd = [
            sys.executable,
            "-m",
            "scd_vllm.truekv_oracle",
            "_generate",
            "--stage",
            stage,
            "--out",
            str(stage_out),
            *common,
        ]
        subprocess.run(cmd, check=True)
        records[stage] = json.loads(stage_out.read_text(encoding="utf-8"))

    metrics = build_truekv_oracle_metrics(
        model=args.model,
        tokenizer=args.tokenizer or args.model,
        prompt=prompt,
        storage_dir=storage_dir,
        records=records,
        max_new_tokens=args.max_new_tokens,
    )
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out_dir / "prompt.txt").write_text(prompt + "\n", encoding="utf-8")
    return metrics


def run_generate_stage(args: argparse.Namespace) -> dict[str, Any]:
    if not args.model:
        raise TrueKVOracleError("--model is required for _generate")
    prompt = _resolve_prompt(args.prompt, args.prefix_file)
    vllm = importlib.import_module("vllm")
    torch = _load_optional("torch")
    sampling_params_cls = getattr(vllm, "SamplingParams")
    llm_cls = getattr(vllm, "LLM")
    llm_kwargs: dict[str, Any] = {
        "model": args.model,
        "tokenizer": args.tokenizer or args.model,
        "dtype": args.dtype,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "seed": args.seed,
        "enforce_eager": args.enforce_eager,
        "enable_prefix_caching": args.enable_prefix_caching,
        "enable_chunked_prefill": args.enable_chunked_prefill,
        "disable_hybrid_kv_cache_manager": args.disable_hybrid_kv_cache_manager,
    }
    if args.stage in {"save", "load"}:
        if not args.storage_dir:
            raise TrueKVOracleError("--storage-dir is required for save/load stages")
        llm_kwargs["kv_transfer_config"] = {
            "kv_connector": "ExampleConnector",
            "kv_role": "kv_both",
            "kv_load_failure_policy": "fail",
            "kv_connector_extra_config": {"shared_storage_path": args.storage_dir},
        }
    memory_before = _cuda_memory_info(torch)
    start_load = time.perf_counter()
    llm = llm_cls(**llm_kwargs)
    load_seconds = time.perf_counter() - start_load
    memory_after_load = _cuda_memory_info(torch)
    output_1tok = _generate_with_max_tokens(
        llm,
        sampling_params_cls,
        prompt,
        max_tokens=1,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
        torch=torch,
    )
    output_ntok = _generate_with_max_tokens(
        llm,
        sampling_params_cls,
        prompt,
        max_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
        torch=torch,
    )
    token_ids = output_ntok["prompt_token_ids"]
    storage_summary = summarize_storage(args.storage_dir) if args.storage_dir else {"exists": False}
    record = {
        "stage": args.stage,
        "mode": "truekv_oracle_generate_stage",
        "vllm_version": getattr(vllm, "__version__", None),
        "model": args.model,
        "tokenizer": args.tokenizer or args.model,
        "prompt_token_count": len(token_ids),
        "prompt_token_ids": token_ids,
        "prompt_token_hash": array_sha256(_np_array_2d(token_ids)),
        "prompt_text_hash": text_sha256(prompt),
        "load_seconds": load_seconds,
        "memory_before": memory_before,
        "memory_after_load": memory_after_load,
        "storage": storage_summary,
        "one_token": output_1tok,
        "n_token": output_ntok,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return record


def build_truekv_oracle_metrics(
    *,
    model: str,
    tokenizer: str,
    prompt: str,
    storage_dir: Path,
    records: dict[str, dict[str, Any]],
    max_new_tokens: int,
) -> dict[str, Any]:
    baseline = records["baseline"]
    save = records["save"]
    load = records["load"]
    same_token_ids = (
        baseline.get("prompt_token_ids") == save.get("prompt_token_ids") == load.get("prompt_token_ids")
    )
    first_token_baseline = _first_token(baseline, "one_token")
    first_token_loaded = _first_token(load, "one_token")
    baseline_16 = baseline.get("n_token", {}).get("output_token_ids", [])
    load_16 = load.get("n_token", {}).get("output_token_ids", [])
    save_files = int(save.get("storage", {}).get("safetensor_files", 0) or 0)
    load_files = int(load.get("storage", {}).get("safetensor_files", 0) or 0)
    first_token_top1_match = first_token_baseline is not None and first_token_baseline == first_token_loaded
    greedy_16tok_match = bool(baseline_16) and baseline_16 == load_16
    greedy_1tok_match = first_token_top1_match
    oracle_pass = bool(
        same_token_ids and save_files > 0 and load_files > 0 and greedy_1tok_match and greedy_16tok_match
    )
    return {
        "mode": "truekv_oracle",
        "oracle_pass": oracle_pass,
        "connector": "ExampleConnector",
        "model": model,
        "tokenizer": tokenizer,
        "prompt_text_hash": text_sha256(prompt),
        "prompt_token_count": baseline.get("prompt_token_count"),
        "prompt_token_hash": baseline.get("prompt_token_hash"),
        "max_new_tokens": max_new_tokens,
        "same_token_ids": same_token_ids,
        "same_position_ids": None,
        "kv_shape_match": None,
        "first_token_top1_match": first_token_top1_match,
        "first_token_id_baseline": first_token_baseline,
        "first_token_id_loaded": first_token_loaded,
        "greedy_text_match_1tok": baseline.get("one_token", {}).get("text") == load.get("one_token", {}).get("text"),
        "greedy_text_match_16tok": baseline.get("n_token", {}).get("text") == load.get("n_token", {}).get("text"),
        "greedy_1tok_match": greedy_1tok_match,
        "greedy_16tok_match": greedy_16tok_match,
        "logit_max_abs_diff": None,
        "logit_rel_l2": None,
        "logit_kl": None,
        "storage_dir": str(storage_dir),
        "save_safetensor_files": save_files,
        "load_safetensor_files": load_files,
        "records": records,
        "notes": [
            "Uses vLLM ExampleConnector with disk-backed vLLM-native KV save/load.",
            "Logits are unavailable from this offline vLLM path, so correctness gate uses deterministic greedy token/text match.",
            "SCD reconstructed KV injection remains skipped until this oracle is preserved as a regression test.",
        ],
    }


def describe_class(cls: type[Any], method_names: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "available": True,
        "module": getattr(cls, "__module__", ""),
        "qualname": getattr(cls, "__qualname__", ""),
        "source_file": _safe_source_file(cls),
        "methods": {},
    }
    for name in method_names:
        attr = getattr(cls, name, None)
        if attr is None:
            out["methods"][name] = {"available": False}
            continue
        out["methods"][name] = {
            "available": True,
            "signature": str(inspect.signature(attr)),
            "defined_in": _safe_source_file(attr),
        }
    return out


def extract_source_evidence_from_object(obj: Any, terms: list[str]) -> list[dict[str, Any]]:
    try:
        source_lines, start_line = inspect.getsourcelines(obj)
    except Exception:
        return []
    return extract_source_evidence(source_lines, terms, start_line=start_line)


def extract_source_evidence(source_lines: list[str], terms: list[str], *, start_line: int = 1) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for offset, line in enumerate(source_lines):
        if any(term in line for term in terms):
            evidence.append({"line": int(start_line + offset), "text": line.strip()})
    return evidence[:80]


def render_inspection_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# SCD-vLLM TrueKV API Inspection",
        "",
        f"- vLLM version: `{payload.get('vllm_version')}`",
        f"- vLLM root: `{payload.get('vllm_root')}`",
        f"- oracle_pass: `{payload.get('oracle_pass')}`",
        f"- blocker: {payload.get('blocker')}",
        "",
        "## Connector Classes",
        "",
    ]
    for label, info in payload.get("connector_classes", {}).items():
        lines.append(f"### {label}")
        lines.append("")
        if not info.get("available"):
            lines.append(f"- unavailable: `{info.get('error')}`")
            lines.append("")
            continue
        lines.append(f"- module: `{info.get('module')}`")
        lines.append(f"- source_file: `{info.get('source_file')}`")
        lines.append("")
        lines.append("| Method | Available | Signature |")
        lines.append("|---|---:|---|")
        for name, method in info.get("methods", {}).items():
            signature = str(method.get("signature") or "")
            lines.append(f"| `{name}` | `{method.get('available')}` | `{signature}` |")
        lines.append("")
    lines.append("## Slot Mapping Evidence")
    lines.append("")
    for label, entries in payload.get("slot_mapping_evidence", {}).items():
        if not entries:
            continue
        lines.append(f"### {label}")
        lines.append("")
        for entry in entries[:30]:
            lines.append(f"- line {entry['line']}: `{entry['text']}`")
        lines.append("")
    lines.append("## Next Step")
    lines.append("")
    lines.append("- Implement a TrueKV connector that first saves and reloads vLLM-native full-consumer KV, then compares deterministic decode output before any SCD injection.")
    lines.append("")
    return "\n".join(lines)


def _generate_with_max_tokens(
    llm: Any,
    sampling_params_cls: Any,
    prompt: str,
    *,
    max_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
    torch: Any,
) -> dict[str, Any]:
    _cuda_synchronize(torch)
    start = time.perf_counter()
    outputs = llm.generate(
        [prompt],
        sampling_params_cls(max_tokens=max_tokens, temperature=temperature, top_p=top_p, seed=seed),
        use_tqdm=False,
    )
    _cuda_synchronize(torch)
    elapsed = time.perf_counter() - start
    if len(outputs) != 1:
        raise TrueKVOracleError(f"expected one output, got {len(outputs)}")
    request_output = outputs[0]
    completion = request_output.outputs[0] if request_output.outputs else None
    return {
        "generation_seconds": elapsed,
        "text": getattr(completion, "text", "") if completion is not None else "",
        "output_token_ids": [int(x) for x in (getattr(completion, "token_ids", []) or [])],
        "finish_reason": getattr(completion, "finish_reason", None),
        "prompt_token_ids": [int(x) for x in (getattr(request_output, "prompt_token_ids", []) or [])],
    }


def summarize_storage(path: str | Path) -> dict[str, Any]:
    root = Path(path)
    if not root.exists():
        return {"exists": False}
    files = [p for p in root.rglob("*") if p.is_file()]
    return {
        "exists": True,
        "dirs": sum(1 for p in root.rglob("*") if p.is_dir()),
        "files": len(files),
        "safetensor_files": sum(1 for p in files if p.suffix == ".safetensors"),
        "bytes": sum(int(p.stat().st_size) for p in files),
    }


def _resolve_prompt(prompt: str, prefix_file: str) -> str:
    if prompt or prefix_file:
        return load_prompt(prompt, prefix_file)
    return DEFAULT_ORACLE_PROMPT


def _load_optional(module: str) -> Any:
    try:
        return importlib.import_module(module)
    except Exception:
        return None


def _cuda_synchronize(torch: Any) -> None:
    if torch is not None and getattr(torch, "cuda", None) is not None and torch.cuda.is_available():
        torch.cuda.synchronize()


def _cuda_memory_info(torch: Any) -> dict[str, Any]:
    if (
        torch is None
        or getattr(torch, "cuda", None) is None
        or not torch.cuda.is_available()
        or torch.cuda.device_count() <= 0
    ):
        return {"cuda_available": False}
    try:
        free, total = torch.cuda.mem_get_info()
        dev = int(torch.cuda.current_device())
        return {
            "cuda_available": True,
            "device": dev,
            "device_name": torch.cuda.get_device_name(dev),
            "free_bytes": int(free),
            "total_bytes": int(total),
            "used_bytes": int(total - free),
        }
    except Exception:
        return {"cuda_available": False}


def _np_array_2d(token_ids: list[int]) -> Any:
    np = importlib.import_module("numpy")
    return np.asarray([token_ids], dtype=np.int64)


def _first_token(record: dict[str, Any], key: str) -> int | None:
    token_ids = record.get(key, {}).get("output_token_ids", [])
    return int(token_ids[0]) if token_ids else None


def _safe_source_file(obj: Any) -> str | None:
    try:
        path = inspect.getsourcefile(obj)
    except Exception:
        return None
    return str(path) if path else None


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "inspect":
        payload = inspect_vllm_kv_api(args.out, args.report_out or None)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
    if args.command == "run":
        payload = run_truekv_oracle(args)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
    if args.command == "_generate":
        payload = run_generate_stage(args)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
    if args.command in {"save", "load"}:
        raise TrueKVOracleError(
            f"truekv_oracle {args.command!r} is handled by the run orchestrator; use `python -m scd_vllm.truekv_oracle run`"
        )
    raise TrueKVOracleError(f"unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
