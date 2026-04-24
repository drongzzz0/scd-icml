from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from pathlib import Path
from typing import Any

from ..adapters import CaptureRequest, TransformersCaptureSession, is_cuda_oom
from ..io_utils import save_bundle, save_json
from ..pipeline import PairMeta, save_pairs, synthesize_pairs
from ..rope import load_rope_spec_from_model_path, rope_spec_from_config_dict
from ..seed import set_global_seed
from ..torch_utils import clear_torch_cache, configure_torch_acceleration, current_rss_mb, gpu_memory_allocated_mb, require_torch, resolve_device, tensor_to_cpu
from ..tracking import command_from_argv, write_manifest
from ..validate import validate_path
from .common import add_accel_args, load_cfg, resolve_accel_options, resolve_logger


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="抓取 A/B 模型的 KV 对数据")
    p.add_argument("--models-config", required=True)
    p.add_argument("--data-config", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log", default="")
    p.add_argument("--backend", choices=["synthetic", "transformers"], default="synthetic")
    p.add_argument("--prompt", default="")
    p.add_argument("--prompt-file", default="", help="可选的 prompt 文件，支持 JSON 或纯文本")
    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument("--device", default="", help="兼容旧接口，未指定新设备参数时同时覆盖 A/B 设备")
    p.add_argument("--dtype", default="", help="覆盖模型推理 dtype")
    p.add_argument("--model-a-device", default="")
    p.add_argument("--model-b-device", default="")
    p.add_argument("--artifact-dtype", default="")
    p.add_argument("--chunk-samples", type=int, default=0)
    p.add_argument("--max-chunk-tokens", type=int, default=0)
    p.add_argument("--skip-validate", action="store_true", help="写完 manifest 后跳过全量 pairs schema 校验")
    add_accel_args(p)
    return p


def load_prompt_file(path: str) -> list[str]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"找不到 prompt 文件: {p}")

    text = p.read_text(encoding="utf-8-sig").strip()
    if not text:
        return []

    if p.suffix.lower() in {".json", ".jsonl"}:
        try:
            obj = json.loads(text)
            if isinstance(obj, list):
                return [str(x).strip() for x in obj if str(x).strip()]
            if isinstance(obj, dict) and isinstance(obj.get("prompts"), list):
                return [str(x).strip() for x in obj["prompts"] if str(x).strip()]
        except json.JSONDecodeError:
            pass

    return [line.strip() for line in text.splitlines() if line.strip()]


def resolve_prompt_payload(data: dict, args: argparse.Namespace) -> str | list[str]:
    capture = data.get("capture", {})

    if args.prompt_file:
        prompts = load_prompt_file(args.prompt_file)
        if prompts:
            return prompts

    cfg_prompt_file = str(capture.get("prompt_file", "") or "").strip()
    if cfg_prompt_file:
        prompts = load_prompt_file(cfg_prompt_file)
        if prompts:
            return prompts

    cfg_prompts = capture.get("prompts")
    if isinstance(cfg_prompts, list):
        prompts = [str(x).strip() for x in cfg_prompts if str(x).strip()]
        if prompts:
            return prompts

    return str(capture.get("prompt", "") or args.prompt)


def _normalize_prompt_list(prompt_payload: str | list[str]) -> list[str]:
    if isinstance(prompt_payload, list):
        prompts = [str(x).strip() for x in prompt_payload if str(x).strip()]
        if prompts:
            return prompts
    else:
        prompt = str(prompt_payload).strip()
        if prompt:
            return [prompt]
    return ["请解释语义缓存蒸馏。"]


def _token_supervision_arrays(prompts: list[str], tokenizer, global_t: int, max_length: int | None):
    import torch

    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=bool(max_length is not None and int(max_length) > 0),
        max_length=None if max_length is None or int(max_length) <= 0 else int(max_length),
    )
    input_ids = encoded["input_ids"][:, :global_t].to(dtype=torch.long)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)
    else:
        attention_mask = attention_mask[:, :global_t].to(dtype=torch.long)
    if int(input_ids.shape[1]) < int(global_t):
        pad_t = int(global_t) - int(input_ids.shape[1])
        input_ids = torch.nn.functional.pad(input_ids, (0, pad_t), value=0)
        attention_mask = torch.nn.functional.pad(attention_mask, (0, pad_t), value=0)
    next_token_ids = torch.full_like(input_ids, fill_value=-100)
    if int(global_t) > 1:
        shifted = input_ids[:, 1:global_t]
        valid = (attention_mask[:, : global_t - 1] > 0) & (attention_mask[:, 1:global_t] > 0)
        next_token_ids[:, : global_t - 1] = torch.where(valid, shifted, torch.full_like(shifted, -100))
    return {
        "input_ids": input_ids.contiguous(),
        "attention_mask": attention_mask.contiguous(),
        "next_token_ids": next_token_ids.contiguous(),
    }


def _build_pair_chunk(models: dict, capture_cfg: dict, seed: int, eval_samples: int, cap_a: dict, cap_b: dict, prompts: list[str], tokenizer, max_length: int | None):
    num_layers = min(len(cap_a), len(cap_b))
    if num_layers <= 0:
        raise RuntimeError("A/B 都没有成功抓到任何层")

    arrays = {}
    global_t = 0
    layer_rows = {}
    for layer in range(num_layers):
        a = cap_a[layer]
        b = cap_b[layer]
        for key in ("k", "v", "xnorm", "resid"):
            if key not in a or key not in b:
                raise RuntimeError(f"层 {layer} 缺少字段: {key}")
        t_steps = min(a["k"].shape[2], b["k"].shape[2], a["xnorm"].shape[1], b["xnorm"].shape[1], a["resid"].shape[1], b["resid"].shape[1])
        global_t = max(global_t, t_steps)
        layer_rows[layer] = {
            "k_a": a["k"][:, :, :t_steps, :],
            "v_a": a["v"][:, :, :t_steps, :],
            "k_b": b["k"][:, :, :t_steps, :],
            "v_b": b["v"][:, :, :t_steps, :],
            "xnorm_a": a["xnorm"][:, :t_steps, :],
            "xnorm_b": b["xnorm"][:, :t_steps, :],
            "resid_a": a["resid"][:, :t_steps, :],
            "resid_b": b["resid"][:, :t_steps, :],
        }

    torch = __import__("torch")
    for layer in range(num_layers):
        row = layer_rows[layer]
        k_a = row["k_a"]
        v_a = row["v_a"]
        k_b = row["k_b"]
        v_b = row["v_b"]
        x_a = row["xnorm_a"]
        x_b = row["xnorm_b"]
        r_a = row["resid_a"]
        r_b = row["resid_b"]
        if k_a.shape[2] != global_t:
            pad = (0, 0, 0, global_t - k_a.shape[2])
            k_a = torch.nn.functional.pad(k_a, pad)
            v_a = torch.nn.functional.pad(v_a, pad)
            k_b = torch.nn.functional.pad(k_b, pad)
            v_b = torch.nn.functional.pad(v_b, pad)
        if x_a.shape[1] != global_t:
            pad = (0, 0, 0, global_t - x_a.shape[1])
            x_a = torch.nn.functional.pad(x_a, pad)
            x_b = torch.nn.functional.pad(x_b, pad)
            r_a = torch.nn.functional.pad(r_a, pad)
            r_b = torch.nn.functional.pad(r_b, pad)
        arrays[f"k_a.{layer}"] = k_a.contiguous()
        arrays[f"v_a.{layer}"] = v_a.contiguous()
        arrays[f"k_b.{layer}"] = k_b.contiguous()
        arrays[f"v_b.{layer}"] = v_b.contiguous()
        arrays[f"xnorm_a.{layer}"] = x_a.contiguous()
        arrays[f"xnorm_b.{layer}"] = x_b.contiguous()
        arrays[f"resid_a.{layer}"] = r_a.contiguous()
        arrays[f"resid_b.{layer}"] = r_b.contiguous()
    arrays.update(_token_supervision_arrays(prompts, tokenizer, global_t, max_length))

    first_k = arrays["k_a.0"]
    first_x = arrays["xnorm_a.0"]
    arch = models.get("architecture", {})
    meta = PairMeta(
        model_a=str(models["model_a"]["name"]),
        model_b=str(models["model_b"]["name"]),
        dtype="float32",
        layout="B,KVH,T,HD",
        num_layers=num_layers,
        num_key_value_heads=int(first_k.shape[1]),
        num_attention_heads=int(arch.get("num_attention_heads", first_k.shape[1])),
        head_dim=int(first_k.shape[3]),
        hidden_dim=int(first_x.shape[2]),
        batch_size=int(first_k.shape[0]),
        prefix_tokens=int(global_t),
        vocab_size=int(capture_cfg.get("vocab_size", 32000)),
        seed=seed,
        eval_samples=int(eval_samples),
        rope=_resolve_rope_payload(models),
    )
    return meta, arrays


def _resolve_rope_payload(models: dict) -> dict[str, object]:
    try:
        rope_a = load_rope_spec_from_model_path(str(models["model_a"]["path"]))
        rope_b = load_rope_spec_from_model_path(str(models["model_b"]["path"]))
    except Exception:
        arch = dict(models.get("architecture", {}))
        rope_a = rope_spec_from_config_dict(arch)
        rope_b = rope_spec_from_config_dict(arch)
    return {"model_a": rope_a.to_dict(), "model_b": rope_b.to_dict()}


def _resolve_capture_settings(models: dict, data: dict, args: argparse.Namespace) -> dict[str, object]:
    runtime = models.get("runtime", {})
    compute = models.get("compute", {})
    capture_cfg = data.get("capture", {})
    artifact_cfg = data.get("artifact", {})

    base_device = str(args.device or runtime.get("device", "cuda:0"))
    model_a_device = str(args.model_a_device or compute.get("model_a_device") or base_device)
    model_b_device = str(args.model_b_device or compute.get("model_b_device") or ("cuda:1" if model_a_device in {"cuda", "cuda:0", "gpu0"} else model_a_device))
    return {
        "dtype": str(args.dtype or runtime.get("dtype", "bfloat16")),
        "artifact_dtype": str(args.artifact_dtype or artifact_cfg.get("dtype", "bfloat16")),
        "model_a_device": model_a_device,
        "model_b_device": model_b_device,
        "chunk_samples": max(1, int(args.chunk_samples or capture_cfg.get("chunk_samples") or capture_cfg.get("batch_size") or 1)),
        "max_chunk_tokens": max(1, int(args.max_chunk_tokens or capture_cfg.get("max_chunk_tokens") or capture_cfg.get("prefix_tokens") or 4096)),
        "cpu_fallback": bool(runtime.get("cpu_fallback", True)),
    }


def _split_prompt_chunks(prompts: list[str], estimator, chunk_samples: int, max_chunk_tokens: int, max_length: int | None) -> list[list[str]]:
    chunks: list[list[str]] = []
    current: list[str] = []
    current_tokens = 0

    for prompt in prompts:
        token_count = max(1, int(estimator(prompt, max_length=max_length)))
        if current and (len(current) >= chunk_samples or current_tokens + token_count > max_chunk_tokens):
            chunks.append(current)
            current = []
            current_tokens = 0
        current.append(prompt)
        current_tokens += token_count
        if len(current) >= chunk_samples or current_tokens >= max_chunk_tokens:
            chunks.append(current)
            current = []
            current_tokens = 0

    if current:
        chunks.append(current)
    return chunks


def _capture_chunk_with_retry(session_a: TransformersCaptureSession, session_b: TransformersCaptureSession, prompts: list[str], max_length: int | None, logger, depth: int = 0):
    try:
        return [(prompts, session_a.capture(prompts, max_length=max_length), session_b.capture(prompts, max_length=max_length))]
    except RuntimeError as exc:
        if not is_cuda_oom(exc):
            raise
        clear_torch_cache()
        if len(prompts) <= 1:
            raise RuntimeError(f"单样本 chunk 仍然 OOM，无法继续切分。prompt 长度={len(prompts[0])}") from exc
        mid = len(prompts) // 2
        logger.warning("chunk 触发 OOM，按样本二分重试，depth=%d, size=%d -> %d + %d", depth, len(prompts), mid, len(prompts) - mid)
        left = _capture_chunk_with_retry(session_a, session_b, prompts[:mid], max_length, logger, depth + 1)
        right = _capture_chunk_with_retry(session_a, session_b, prompts[mid:], max_length, logger, depth + 1)
        return left + right


def _capture_single_session_with_retry(session: TransformersCaptureSession, prompts: list[str], max_length: int | None, logger, depth: int = 0):
    try:
        return [(prompts, session.capture(prompts, max_length=max_length))]
    except RuntimeError as exc:
        if not is_cuda_oom(exc):
            raise
        clear_torch_cache()
        if len(prompts) <= 1:
            raise RuntimeError(f"鍗曟牱鏈?chunk 浠嶇劧 OOM锛屾棤娉曠户缁垏鍒嗐€俻rompt 闀垮害={len(prompts[0])}") from exc
        mid = len(prompts) // 2
        logger.warning("single-model chunk 瑙﹀彂 OOM锛屾寜鏍锋湰浜屽垎閲嶈瘯锛宒epth=%d, size=%d -> %d + %d", depth, len(prompts), mid, len(prompts) - mid)
        left = _capture_single_session_with_retry(session, prompts[:mid], max_length, logger, depth + 1)
        right = _capture_single_session_with_retry(session, prompts[mid:], max_length, logger, depth + 1)
        return left + right


def _normalize_runtime_device(device: str, cpu_fallback: bool) -> str:
    return str(resolve_device(device, cpu_fallback=cpu_fallback))


def _should_capture_sequentially(settings: dict[str, object]) -> bool:
    return _normalize_runtime_device(str(settings["model_a_device"]), bool(settings["cpu_fallback"])) == _normalize_runtime_device(
        str(settings["model_b_device"]), bool(settings["cpu_fallback"])
    )


def _tensor_tree_to_cpu(value: Any, artifact_dtype: str | None):
    if isinstance(value, dict):
        return {key: _tensor_tree_to_cpu(sub_value, artifact_dtype) for key, sub_value in value.items()}
    if isinstance(value, list):
        return [_tensor_tree_to_cpu(sub_value, artifact_dtype) for sub_value in value]
    if isinstance(value, tuple):
        return tuple(_tensor_tree_to_cpu(sub_value, artifact_dtype) for sub_value in value)
    if isinstance(value, (str, bytes, int, float, bool)) or value is None:
        return value
    return tensor_to_cpu(value, dtype=artifact_dtype)


def _save_capture_payload(path: Path, prompts: list[str], payload: dict[int, dict[str, object]], artifact_dtype: str) -> None:
    torch = require_torch()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"prompts": list(prompts), "payload": _tensor_tree_to_cpu(payload, artifact_dtype)}, path)


def _load_capture_payload(path: Path) -> tuple[list[str], dict[int, dict[str, object]]]:
    torch = require_torch()
    saved = torch.load(path, map_location="cpu", weights_only=False)
    prompts = [str(prompt) for prompt in saved.get("prompts", [])]
    payload = dict(saved.get("payload", {}))
    return prompts, payload


def _write_pairs_manifest(out_dir: Path, base_meta: PairMeta, chunk_entries: list[dict[str, object]], artifact_dtype: str) -> None:
    manifest = base_meta.__dict__.copy()
    manifest.update(
        {
            "format": "chunked_pt_v1",
            "artifact_dtype": artifact_dtype,
            "batch_size": int(sum(int(row["batch_size"]) for row in chunk_entries)),
            "eval_samples": int(sum(int(row["eval_samples"]) for row in chunk_entries)),
            "prefix_tokens": int(max(int(row["prefix_tokens"]) for row in chunk_entries)),
            "num_chunks": len(chunk_entries),
            "chunks": chunk_entries,
        }
    )
    save_json(out_dir / "manifest.json", manifest)


def _capture_with_live_sessions(models: dict, data: dict, args: argparse.Namespace, logger) -> None:
    settings = _resolve_capture_settings(models, data, args)
    prompts = _normalize_prompt_list(resolve_prompt_payload(data, args))
    capture_cfg = data.get("capture", {})
    cli_has_max_length = any(arg == "--max-length" or arg.startswith("--max-length=") for arg in __import__("sys").argv[1:])
    max_length_raw = args.max_length if cli_has_max_length else capture_cfg.get("max_length", args.max_length)
    max_length = None if max_length_raw is None or int(max_length_raw) <= 0 else int(max_length_raw)

    out_dir = Path(args.out)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    (out_dir / "chunks").mkdir(parents=True, exist_ok=True)

    session_a = TransformersCaptureSession(str(models["model_a"]["path"]), str(settings["model_a_device"]), str(settings["dtype"]), cpu_fallback=bool(settings["cpu_fallback"]))
    session_b = TransformersCaptureSession(str(models["model_b"]["path"]), str(settings["model_b_device"]), str(settings["dtype"]), cpu_fallback=bool(settings["cpu_fallback"]))

    try:
        prompt_chunks = _split_prompt_chunks(
            prompts,
            session_a.estimate_prompt_tokens,
            chunk_samples=int(settings["chunk_samples"]),
            max_chunk_tokens=int(settings["max_chunk_tokens"]),
            max_length=max_length,
        )
        logger.info("采集开始：总样本=%d，初始 chunk=%d，A=%s，B=%s", len(prompts), len(prompt_chunks), settings["model_a_device"], settings["model_b_device"])

        chunk_entries: list[dict[str, object]] = []
        base_meta: PairMeta | None = None
        durations: list[float] = []
        chunk_id = 0

        for original_chunk in prompt_chunks:
            sub_chunks = _capture_chunk_with_retry(session_a, session_b, original_chunk, max_length, logger)
            for prompt_chunk, cap_a, cap_b in sub_chunks:
                start = time.perf_counter()
                meta, arrays = _build_pair_chunk(models, capture_cfg, args.seed, len(prompt_chunk), cap_a, cap_b, prompt_chunk, session_b.tokenizer, max_length)
                if base_meta is None:
                    base_meta = meta
                chunk_name = f"chunk_{chunk_id:05d}.pt"
                save_bundle(out_dir / "chunks" / chunk_name, arrays, meta=meta.__dict__, artifact_dtype=str(settings["artifact_dtype"]))
                write_ms = (time.perf_counter() - start) * 1000.0
                token_count = sum(session_a.estimate_prompt_tokens(prompt, max_length=max_length) for prompt in prompt_chunk)
                durations.append(write_ms)
                remain = max(0, len(prompt_chunks) - (len(chunk_entries) + 1))
                avg_ms = sum(durations) / max(1, len(durations))
                eta_s = (avg_ms * remain) / 1000.0
                logger.info(
                    "chunk=%05d samples=%d tokens=%d gpu_a=%.1fMB gpu_b=%.1fMB cpu_rss=%.1fMB write_ms=%.1f eta_s=%.1f",
                    chunk_id,
                    len(prompt_chunk),
                    token_count,
                    gpu_memory_allocated_mb(str(settings["model_a_device"])),
                    gpu_memory_allocated_mb(str(settings["model_b_device"])),
                    current_rss_mb(),
                    write_ms,
                    eta_s,
                )
                chunk_entries.append(
                    {
                        "path": f"chunks/{chunk_name}",
                        "batch_size": int(meta.batch_size),
                        "prefix_tokens": int(meta.prefix_tokens),
                        "eval_samples": int(meta.eval_samples),
                        "num_layers": int(meta.num_layers),
                    }
                )
                del arrays, cap_a, cap_b
                clear_torch_cache()
                chunk_id += 1

        if base_meta is None:
            raise RuntimeError("没有生成任何 chunk")
        _write_pairs_manifest(out_dir, base_meta, chunk_entries, str(settings["artifact_dtype"]))
    finally:
        session_a.close()
        session_b.close()


def _capture_with_sequential_session_swap(models: dict, data: dict, args: argparse.Namespace, logger) -> None:
    settings = _resolve_capture_settings(models, data, args)
    prompts = _normalize_prompt_list(resolve_prompt_payload(data, args))
    capture_cfg = data.get("capture", {})
    cli_has_max_length = any(arg == "--max-length" or arg.startswith("--max-length=") for arg in __import__("sys").argv[1:])
    max_length_raw = args.max_length if cli_has_max_length else capture_cfg.get("max_length", args.max_length)
    max_length = None if max_length_raw is None or int(max_length_raw) <= 0 else int(max_length_raw)

    out_dir = Path(args.out)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    (out_dir / "chunks").mkdir(parents=True, exist_ok=True)

    staged_dir = out_dir / ".model_a_chunks"
    staged_dir.mkdir(parents=True, exist_ok=True)

    session_a = TransformersCaptureSession(str(models["model_a"]["path"]), str(settings["model_a_device"]), str(settings["dtype"]), cpu_fallback=bool(settings["cpu_fallback"]))
    try:
        prompt_chunks = _split_prompt_chunks(
            prompts,
            session_a.estimate_prompt_tokens,
            chunk_samples=int(settings["chunk_samples"]),
            max_chunk_tokens=int(settings["max_chunk_tokens"]),
            max_length=max_length,
        )
        logger.info(
            "閲囬泦寮€濮嬶紙鍚屽崱椤哄簭妯″紡锛夛細鎬绘牱鏈?%d锛屽垵濮?chunk=%d锛孉=%s锛孊=%s",
            len(prompts),
            len(prompt_chunks),
            settings["model_a_device"],
            settings["model_b_device"],
        )
        staged_entries: list[str] = []
        stage_chunk_id = 0
        for original_chunk in prompt_chunks:
            sub_chunks = _capture_single_session_with_retry(session_a, original_chunk, max_length, logger)
            for prompt_chunk, cap_a in sub_chunks:
                chunk_name = f"chunk_{stage_chunk_id:05d}.pt"
                _save_capture_payload(staged_dir / chunk_name, prompt_chunk, cap_a, str(settings["artifact_dtype"]))
                staged_entries.append(chunk_name)
                logger.info(
                    "stage_a chunk=%05d samples=%d gpu_a=%.1fMB cpu_rss=%.1fMB",
                    stage_chunk_id,
                    len(prompt_chunk),
                    gpu_memory_allocated_mb(str(settings["model_a_device"])),
                    current_rss_mb(),
                )
                del cap_a
                clear_torch_cache()
                stage_chunk_id += 1
    finally:
        session_a.close()

    session_b = TransformersCaptureSession(str(models["model_b"]["path"]), str(settings["model_b_device"]), str(settings["dtype"]), cpu_fallback=bool(settings["cpu_fallback"]))
    try:
        chunk_entries: list[dict[str, object]] = []
        base_meta: PairMeta | None = None
        durations: list[float] = []
        total_chunks = len(staged_entries)

        for chunk_id, chunk_name in enumerate(staged_entries):
            stage_path = staged_dir / chunk_name
            prompt_chunk, cap_a = _load_capture_payload(stage_path)
            cap_b = session_b.capture(prompt_chunk, max_length=max_length)
            start = time.perf_counter()
            meta, arrays = _build_pair_chunk(models, capture_cfg, args.seed, len(prompt_chunk), cap_a, cap_b, prompt_chunk, session_b.tokenizer, max_length)
            if base_meta is None:
                base_meta = meta
            output_chunk_name = f"chunk_{chunk_id:05d}.pt"
            save_bundle(out_dir / "chunks" / output_chunk_name, arrays, meta=meta.__dict__, artifact_dtype=str(settings["artifact_dtype"]))
            write_ms = (time.perf_counter() - start) * 1000.0
            token_count = sum(session_b.estimate_prompt_tokens(prompt, max_length=max_length) for prompt in prompt_chunk)
            durations.append(write_ms)
            remain = max(0, total_chunks - (chunk_id + 1))
            avg_ms = sum(durations) / max(1, len(durations))
            eta_s = (avg_ms * remain) / 1000.0
            logger.info(
                "merge chunk=%05d samples=%d tokens=%d gpu_b=%.1fMB cpu_rss=%.1fMB write_ms=%.1f eta_s=%.1f",
                chunk_id,
                len(prompt_chunk),
                token_count,
                gpu_memory_allocated_mb(str(settings["model_b_device"])),
                current_rss_mb(),
                write_ms,
                eta_s,
            )
            chunk_entries.append(
                {
                    "path": f"chunks/{output_chunk_name}",
                    "batch_size": int(meta.batch_size),
                    "prefix_tokens": int(meta.prefix_tokens),
                    "eval_samples": int(meta.eval_samples),
                    "num_layers": int(meta.num_layers),
                }
            )
            stage_path.unlink(missing_ok=True)
            del arrays, cap_a, cap_b
            clear_torch_cache()

        if base_meta is None:
            raise RuntimeError("娌℃湁鐢熸垚浠讳綍 chunk")
        _write_pairs_manifest(out_dir, base_meta, chunk_entries, str(settings["artifact_dtype"]))
    finally:
        session_b.close()
        shutil.rmtree(staged_dir, ignore_errors=True)


def capture_with_backend_transformers(models: dict, data: dict, args: argparse.Namespace, logger) -> None:
    settings = _resolve_capture_settings(models, data, args)
    if _should_capture_sequentially(settings):
        _capture_with_sequential_session_swap(models, data, args, logger)
        return
    _capture_with_live_sessions(models, data, args, logger)


def main() -> None:
    args = build_parser().parse_args()
    logger = resolve_logger(args.log)
    set_global_seed(args.seed)

    models = load_cfg(args.models_config)
    data = load_cfg(args.data_config)
    accel = resolve_accel_options(models, args)
    configure_torch_acceleration(str(accel["accel_mode"]))
    logger.info("计算加速模式: %s", accel["accel_mode"])

    if args.backend == "synthetic":
        meta = PairMeta(
            model_a=str(models["model_a"]["name"]),
            model_b=str(models["model_b"]["name"]),
            dtype=str(models.get("runtime", {}).get("dtype", "float32")),
            layout="B,KVH,T,HD",
            num_layers=int(models["architecture"]["num_layers"]),
            num_key_value_heads=int(models["architecture"]["num_key_value_heads"]),
            num_attention_heads=int(models["architecture"]["num_attention_heads"]),
            head_dim=int(models["architecture"]["head_dim"]),
            hidden_dim=int(models["architecture"]["hidden_dim"]),
            batch_size=int(data["capture"].get("chunk_samples") or data["capture"].get("batch_size") or 1),
            prefix_tokens=int(data["capture"].get("prefix_tokens", 256)),
            vocab_size=int(data["capture"].get("vocab_size", 32000)),
            seed=args.seed,
            eval_samples=int(data["capture"].get("chunk_samples") or data["capture"].get("batch_size") or 1),
            rope=_resolve_rope_payload(models),
        )
        artifact_dtype = str(data.get("artifact", {}).get("dtype", "bfloat16"))
        save_pairs(args.out, meta, synthesize_pairs(meta), artifact_dtype=artifact_dtype)
    else:
        capture_with_backend_transformers(models, data, args, logger)

    if not bool(args.skip_validate):
        validate_path(args.out)
    write_manifest(
        out_dir=args.out,
        command=command_from_argv(sys.argv),
        config_paths={"models": args.models_config, "data": args.data_config},
        seed=args.seed,
    )
    logger.info("成对数据已保存到 %s (skip_validate=%s)", Path(args.out).resolve(), bool(args.skip_validate))


if __name__ == "__main__":
    main()
