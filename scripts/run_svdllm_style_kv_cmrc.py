from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run an SVD-LLM-style adapted KV-transfer baseline on the Mistral->Instruct CMRC strict runtime."
    )
    p.add_argument("--project-root", required=True, help="Remote scd_repro project root.")
    p.add_argument("--models-config", required=True, help="Path to models yaml/json config.")
    p.add_argument("--data-root", required=True, help="Data root containing cmrc2018 dev json.")
    p.add_argument("--out", required=True, help="Output directory.")
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--bandwidth-gbps", type=float, default=200.0)
    p.add_argument("--producer-device", default="cuda:1")
    p.add_argument("--consumer-device", default="cuda:0")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--rank", type=int, default=16)
    return p


def _write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_summary_md(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        "# SVD-LLM-style KV Strict Runtime Comparison",
        "",
        "| mode | f1 | em | avg_nll | bytes | compression | ttft_ms | total_ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['method_label']} | {float(row['f1']):.4f} | {float(row['em']):.4f} | "
            f"{float(row['avg_nll']):.4f} | {float(row['bytes_transferred']):.1f} | "
            f"{0.0 if row['compression_ratio'] is None else float(row['compression_ratio']):.3f} | "
            f"{float(row['ttft_ms']):.2f} | {float(row['total_ms']):.2f} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _sample_result_to_dict(row: Any) -> dict[str, object]:
    return {
        "qid": row.qid,
        "method": row.method,
        "pred_text": row.pred_text,
        "gold_answers": row.gold_answers,
        "f1": row.f1,
        "em": row.em,
        "avg_logp": row.avg_logp,
        "avg_nll": row.avg_nll,
        "teacher_token_count": row.teacher_token_count,
        "payload_mb": row.payload_mb,
        "prompt_tokens": row.prompt_tokens,
        "prefix_tokens": row.prefix_tokens,
        "first_token_id": row.first_token_id,
        "timing_ms": {
            "producer_prefill": row.timing.producer_prefill_ms,
            "consumer_prefill": row.timing.consumer_prefill_ms,
            "encode": row.timing.encode_ms,
            "transfer": row.timing.transfer_ms,
            "rebuild": row.timing.rebuild_ms,
            "patch_infer": row.timing.patch_infer_ms,
            "recompute": row.timing.recompute_ms,
            "decode_first": row.timing.decode_first_ms,
            "decode_rest": row.timing.decode_rest_ms,
            "decode_total": row.timing.decode_total_ms,
            "ttft": row.timing.ttft_ms,
            "total": row.timing.total_ms,
        },
        "bytes_transferred": row.timing.bytes_transferred,
        "raw_cache_bytes": row.timing.raw_cache_bytes,
        "compression_ratio": row.timing.compression_ratio,
    }


def _dtype_size_bytes(torch_mod: Any, dtype: Any) -> int:
    return int(torch_mod.tensor([], dtype=dtype).element_size())


def _flatten_kv_tensor(x: Any) -> tuple[Any, tuple[int, ...]]:
    shape = tuple(int(dim) for dim in x.shape)
    return x.reshape(-1, shape[-1]).to(dtype=x.dtype), shape


def _low_rank_pack(torch_mod: Any, x: Any, rank: int, transfer_dtype: Any) -> tuple[Any, Any, Any, tuple[int, ...]]:
    matrix, shape = _flatten_kv_tensor(x.to(dtype=torch_mod.float32))
    eff_rank = max(1, min(int(rank), int(matrix.shape[0]), int(matrix.shape[1])))
    u, s, vh = torch_mod.linalg.svd(matrix, full_matrices=False)
    u_r = u[:, :eff_rank].contiguous().to(dtype=transfer_dtype)
    s_r = s[:eff_rank].contiguous().to(dtype=transfer_dtype)
    vh_r = vh[:eff_rank, :].contiguous().to(dtype=transfer_dtype)
    return u_r, s_r, vh_r, shape


def _low_rank_rebuild(torch_mod: Any, pack: tuple[Any, Any, Any, tuple[int, ...]], device: str, dtype: Any) -> Any:
    u_r, s_r, vh_r, shape = pack
    u = u_r.to(device=device, dtype=torch_mod.float32)
    s = s_r.to(device=device, dtype=torch_mod.float32)
    vh = vh_r.to(device=device, dtype=torch_mod.float32)
    rec = (u * s.unsqueeze(0)) @ vh
    return rec.reshape(shape).to(device=device, dtype=dtype).contiguous()


def _payload_bytes(torch_mod: Any, transfer_dtype: Any, tensors: list[Any]) -> int:
    elem_size = _dtype_size_bytes(torch_mod, transfer_dtype)
    return int(sum(int(x.numel()) * elem_size for x in tensors))


def main() -> None:
    args = build_parser().parse_args()
    project_root = Path(args.project_root).resolve()
    sys.path.insert(0, str(project_root / "src"))

    from scd_repro.cli.common import load_cfg
    import scd_repro.strict_runtime as sr
    from scd_repro.cmrc import load_cmrc2018_samples
    from scd_repro.seed import set_global_seed
    from scd_repro.torch_utils import configure_torch_acceleration, resolve_torch_dtype

    set_global_seed(42)
    torch_mod = sr.require_torch()
    transfer_dtype = resolve_torch_dtype(args.dtype)

    sr._METHOD_LABELS["svdllm_style_kv"] = "SVD-LLM-style KV"
    old_assemble = sr._assemble_method_cache

    def patched_assemble(
        *,
        method: str,
        producer_capture: Any,
        consumer_capture: Any,
        producer_session: Any,
        consumer_session: Any,
        route: dict[str, list[int]],
        reuse_prod: Any,
        reuse_cons: Any,
        patch_bundle: dict[str, Any] | None,
        patch_meta: dict[str, object] | None,
        bandwidth_gbps: float,
        quant_group_size: int,
        transfer_dtype: Any,
        default_recompute_segment_length: int = 0,
        override_route_recompute_length: int | None = None,
        fail_on_missing_route_fields: bool = False,
        fail_on_missing_patch_fields: bool = False,
    ) -> tuple[list[tuple[Any, Any]], Any, float]:
        if method != "svdllm_style_kv":
            return old_assemble(
                method=method,
                producer_capture=producer_capture,
                consumer_capture=consumer_capture,
                producer_session=producer_session,
                consumer_session=consumer_session,
                route=route,
                reuse_prod=reuse_prod,
                reuse_cons=reuse_cons,
                patch_bundle=patch_bundle,
                patch_meta=patch_meta,
                bandwidth_gbps=bandwidth_gbps,
                quant_group_size=quant_group_size,
                transfer_dtype=transfer_dtype,
                default_recompute_segment_length=default_recompute_segment_length,
                override_route_recompute_length=override_route_recompute_length,
                fail_on_missing_route_fields=fail_on_missing_route_fields,
                fail_on_missing_patch_fields=fail_on_missing_patch_fields,
            )

        prefix_tokens = int(producer_capture.input_ids.shape[1]) - 1
        consumer_device = str(consumer_session.target_device)
        consumer_dtype = consumer_session.model.dtype
        num_layers = len(producer_capture.past_key_values)
        raw_cache_bytes = sr._raw_cache_payload_bytes(producer_capture, prefix_tokens, transfer_dtype)
        timing = sr.MethodTiming(raw_cache_bytes=int(raw_cache_bytes))
        caches: list[tuple[Any, Any]] = [None for _ in range(num_layers)]  # type: ignore[list-item]
        payload_bytes = 0

        sr._synchronize_for_device(producer_capture.past_key_values[0][0].device)
        enc_start = time.perf_counter()
        payloads: list[tuple[tuple[Any, Any, Any, tuple[int, ...]], tuple[Any, Any, Any, tuple[int, ...]]]] = []
        for layer in range(num_layers):
            mapped_k = sr._map_k_between_sessions(
                producer_capture.past_key_values[layer][0],
                prefix_tokens,
                producer_session,
                consumer_session,
            )
            mapped_v = sr._slice_prefix_tensor(producer_capture.past_key_values[layer][1], prefix_tokens)
            k_pack = _low_rank_pack(torch_mod, mapped_k, int(args.rank), transfer_dtype)
            v_pack = _low_rank_pack(torch_mod, mapped_v, int(args.rank), transfer_dtype)
            payloads.append((k_pack, v_pack))
            payload_bytes += _payload_bytes(torch_mod, transfer_dtype, [k_pack[0], k_pack[1], k_pack[2], v_pack[0], v_pack[1], v_pack[2]])
        sr._synchronize_for_device(producer_capture.past_key_values[0][0].device)
        timing.encode_ms = (time.perf_counter() - enc_start) * 1000.0

        sr._synchronize_for_device(consumer_device)
        rebuild_start = time.perf_counter()
        for layer, (k_pack, v_pack) in enumerate(payloads):
            caches[layer] = (
                _low_rank_rebuild(torch_mod, k_pack, consumer_device, consumer_dtype),
                _low_rank_rebuild(torch_mod, v_pack, consumer_device, consumer_dtype),
            )
        sr._synchronize_for_device(consumer_device)
        timing.rebuild_ms = (time.perf_counter() - rebuild_start) * 1000.0

        timing.transfer_ms = 0.0 if payload_bytes <= 0 else (float(payload_bytes) * 8.0 / (float(bandwidth_gbps) * 1e9) * 1000.0)
        timing.bytes_transferred = int(payload_bytes)
        timing.compression_ratio = sr._compression_ratio(raw_cache_bytes, payload_bytes)
        payload_mb = float(payload_bytes) / (1024.0 * 1024.0)
        return caches, timing, payload_mb

    sr._assemble_method_cache = patched_assemble

    configure_torch_acceleration("balanced")
    models_cfg = load_cfg(args.models_config)
    model_a_path = str(models_cfg["model_a"]["path"])
    model_b_path = str(models_cfg["model_b"]["path"])
    runtime_cfg = models_cfg.get("runtime", {})
    cpu_fallback = bool(runtime_cfg.get("cpu_fallback", True))

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    samples = load_cmrc2018_samples(args.data_root, limit=int(args.limit))
    methods = ["full_consumer", "raw_kv", "svdllm_style_kv"]
    per_method_results: dict[str, list[Any]] = {method: [] for method in methods}

    producer_session = sr.TransformersRuntimeSession(
        model_path=model_a_path,
        device=args.producer_device,
        dtype=args.dtype,
        cpu_fallback=cpu_fallback,
    )
    consumer_session = sr.TransformersRuntimeSession(
        model_path=model_b_path,
        device=args.consumer_device,
        dtype=args.dtype,
        cpu_fallback=cpu_fallback,
    )

    try:
        for idx, sample in enumerate(samples, start=1):
            producer_capture = producer_session.prefill(sample.prompt, max_length=args.max_length)
            consumer_capture = consumer_session.prefill(sample.prompt, max_length=args.max_length)
            sr.assert_same_tokenization(producer_capture, consumer_capture)

            for method in methods:
                result = sr.evaluate_cmrc_method(
                    method=method,
                    sample=sample,
                    producer_capture=producer_capture,
                    consumer_capture=consumer_capture,
                    producer_session=producer_session,
                    consumer_session=consumer_session,
                    route={},
                    reuse_prod=None,
                    reuse_cons=None,
                    patch_bundle=None,
                    patch_meta=None,
                    bandwidth_gbps=float(args.bandwidth_gbps),
                    max_new_tokens=int(args.max_new_tokens),
                    quant_group_size=128,
                    transfer_dtype=transfer_dtype,
                    measure_teacher_forced_metrics=True,
                )
                per_method_results[method].append(result)

            if idx % 5 == 0 or idx == len(samples):
                print(f"[progress] completed {idx}/{len(samples)} samples", flush=True)

        summary_rows: list[dict[str, object]] = []
        all_sample_rows: list[dict[str, object]] = []
        for method in methods:
            rows = per_method_results[method]
            summary = sr.summarize_method_results(rows)
            summary_rows.append(summary)
            for row in rows:
                all_sample_rows.append(_sample_result_to_dict(row))

        full_row = next((row for row in summary_rows if row["method"] == "full_consumer"), None)
        raw_row = next((row for row in summary_rows if row["method"] == "raw_kv"), None)
        for row in summary_rows:
            if full_row is not None:
                row["delta_f1_vs_full"] = float(row["f1"]) - float(full_row["f1"])
                row["delta_ttft_vs_full"] = float(row["ttft_ms"]) - float(full_row["ttft_ms"])
            if raw_row is not None:
                row["delta_f1_vs_raw_kv"] = float(row["f1"]) - float(raw_row["f1"])
                row["delta_ttft_vs_raw_kv"] = float(row["ttft_ms"]) - float(raw_row["ttft_ms"])

        meta = {
            "project_root": str(project_root),
            "models_config": args.models_config,
            "producer_model": model_a_path,
            "consumer_model": model_b_path,
            "producer_device": args.producer_device,
            "consumer_device": args.consumer_device,
            "dtype": args.dtype,
            "bandwidth_gbps": float(args.bandwidth_gbps),
            "limit": int(args.limit),
            "max_length": int(args.max_length),
            "max_new_tokens": int(args.max_new_tokens),
            "rank": int(args.rank),
            "methods": methods,
        }
        result_bundle = {"meta": meta, "summary": summary_rows}

        _write_json(out_root / "summary.json", result_bundle)
        _write_jsonl(out_root / "per_sample_metrics.jsonl", all_sample_rows)
        _write_summary_md(out_root / "summary.md", summary_rows)
        print(json.dumps(result_bundle, ensure_ascii=False, indent=2))
    finally:
        producer_session.close()
        consumer_session.close()


if __name__ == "__main__":
    main()
