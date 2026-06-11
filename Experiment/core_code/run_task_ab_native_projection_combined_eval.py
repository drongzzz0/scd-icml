from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import statistics
import time
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import run_task_aa_native_projection_patch as aa
import run_task_af_layer27_decode_adapter as af
import run_task_p_long_context as p
import run_task_t2_layer_reduction_f1 as t2
import run_task_v_200x16k_qa_complete_eval as v
import run_task_y_xnorm_ridge_quick as y
from run_online_prefix_policy_eval import run_prefix_layers
from run_real_mapping_patch import select_small_by_mapping


PROJECT_ROOT = pathlib.Path("/disk_n/c101-3-mql/projects/scd-tpami")
OUT = PROJECT_ROOT / "Experiment/analysis/long_context"
BASE_PATCH = OUT / "W_qa_per_block_ridge_train40_B8.pt"
BASELINE_CSV = OUT / "V_200x16k_eval_per_sample.csv"
ALL_LAYERS = list(range(28))


def mean(xs: Iterable[float]) -> float:
    vals = [float(x) for x in xs if x == x]
    return sum(vals) / len(vals) if vals else float("nan")


def stdev(xs: Iterable[float]) -> float:
    vals = [float(x) for x in xs if x == x]
    return float(statistics.pstdev(vals)) if len(vals) > 1 else 0.0


def parse_layers(text: str) -> List[int]:
    if not text.strip():
        return []
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def write_csv(path: pathlib.Path, rows: List[Dict[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def write_json(path: pathlib.Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def load_baseline(path: pathlib.Path) -> Dict[int, Dict[str, Any]]:
    return aa.load_baseline(path)


def merge_native_payload(paths: Sequence[pathlib.Path]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    sources: List[Dict[str, Any]] = []
    target_layers: List[int] = []
    weights: Dict[Tuple[int, int], torch.Tensor] = {}
    biases: Dict[Tuple[int, int], torch.Tensor] = {}
    mapping: Dict[str, int] = {}
    counts: Dict[str, int] = {}
    for path in paths:
        obj = torch.load(path, map_location="cpu", weights_only=False)
        if obj.get("kind") != "small_xnorm_to_large_xnorm_native_projection_per_block_ridge":
            raise ValueError(f"unsupported native payload kind in {path}: {obj.get('kind')}")
        if not merged:
            for key in ["kind", "blocks", "seq_len"]:
                merged[key] = obj[key]
        else:
            for key in ["kind", "blocks", "seq_len"]:
                if merged[key] != obj[key]:
                    raise ValueError(f"metadata mismatch for {key}: {merged[key]} != {obj[key]}")
        sources.append(
            {
                "path": str(path),
                "target_layers": [int(x) for x in obj["target_layers"]],
                "ridge_lambda": float(obj.get("ridge_lambda", float("nan"))),
                "train_start": int(obj.get("train_start", -1)),
                "train_n": int(obj.get("train_n", -1)),
                "tokens_per_block": int(obj.get("tokens_per_block", -1)),
            }
        )
        target_layers.extend(int(x) for x in obj["target_layers"])
        weights.update(obj["weights"])
        biases.update(obj["biases"])
        mapping.update({str(k): int(v) for k, v in obj["mapping"].items()})
        counts.update({str(k): int(v) for k, v in obj["counts"].items()})
    merged["target_layers"] = sorted(set(target_layers))
    merged["weights"] = weights
    merged["biases"] = biases
    merged["mapping"] = mapping
    merged["counts"] = counts
    merged["sources"] = sources
    return merged


def load_af_adapters(paths_text: str, device: torch.device) -> Dict[int, af.XNormResidualAdapter]:
    adapters: Dict[int, af.XNormResidualAdapter] = {}
    for raw in paths_text.split(","):
        raw = raw.strip()
        if not raw:
            continue
        payload = torch.load(pathlib.Path(raw), map_location="cpu", weights_only=False)
        if payload.get("kind") != "layer27_decode_aware_xnorm_residual":
            raise ValueError(f"unsupported AF adapter kind in {raw}: {payload.get('kind')}")
        layer = int(payload["layer"])
        adapter = af.XNormResidualAdapter(
            int(payload["input_dim"]),
            int(payload["output_dim"]),
            int(payload["hidden"]),
        ).to(device).eval()
        adapter.load_state_dict(payload["state_dict"])
        for param in adapter.parameters():
            param.requires_grad_(False)
        adapters[layer] = adapter
    return adapters


def load_base_patch(device: torch.device):
    base_payload = torch.load(BASE_PATCH, map_location="cpu", weights_only=False)
    base = p.load_ridge_patch(device)
    return t2.AllLayerPerBlockRidgePatch(
        base, base_payload["weights"], base_payload["biases"], base_payload["seq_len"], base_payload["blocks"]
    ).to(device).eval()


def route_specs(text: str) -> List[Dict[str, Any]]:
    specs = []
    for part in text.split(";"):
        part = part.strip()
        if not part:
            continue
        name, rest = part.split(":", 1)
        fields: Dict[str, str] = {}
        for item in rest.split("|"):
            if not item:
                continue
            key, val = item.split("=", 1)
            fields[key.strip()] = val.strip()
        specs.append(
            {
                "name": name.strip(),
                "native_layers": parse_layers(fields.get("native", "")),
                "exact_layers": parse_layers(fields.get("exact", "")),
            }
        )
    return specs


def evaluate(args: argparse.Namespace) -> None:
    dtype = p.dtype_from_name(args.dtype)
    large_device = torch.device(args.device)
    small_device = torch.device(args.small_device)
    native_payload = merge_native_payload([pathlib.Path(x) for x in args.native_patches.split(",") if x.strip()])
    native_weights = native_payload["weights"]
    native_biases = native_payload["biases"]
    native_all = set(int(x) for x in native_payload["target_layers"])
    af_adapters = load_af_adapters(args.af_adapters, small_device)
    routes = route_specs(args.routes)
    for route in routes:
        missing = [x for x in route["native_layers"] if x not in native_all]
        if missing:
            raise ValueError(f"route {route['name']} requests native layers not in payload: {missing}")

    tokenizer = AutoTokenizer.from_pretrained(str(p.SMALL_MODEL), trust_remote_code=True, local_files_only=True)
    records = v.load_qa_records(tokenizer, args.start + args.n)[args.start : args.start + args.n]
    baseline = load_baseline(pathlib.Path(args.baseline_csv))

    large_model = AutoModelForCausalLM.from_pretrained(
        str(p.LARGE_MODEL), trust_remote_code=True, local_files_only=True, torch_dtype=dtype, low_cpu_mem_usage=True
    ).to(large_device).eval()
    small_model = AutoModelForCausalLM.from_pretrained(
        str(p.SMALL_MODEL), trust_remote_code=True, local_files_only=True, torch_dtype=dtype, low_cpu_mem_usage=True
    ).to(small_device).eval()
    base_patch = load_base_patch(small_device)
    mapping = p.load_mapping(small_device)
    mapping_cpu_tensor = mapping.detach().cpu()
    mapping_cpu = [int(x) for x in (mapping_cpu_tensor[1].tolist() if mapping_cpu_tensor.ndim == 2 else mapping_cpu_tensor.tolist())]
    target_ids = torch.arange(28, device=small_device)

    rows: List[Dict[str, Any]] = []
    print(json.dumps({"mode": "native_combined_eval_start", "tag": args.tag, "n": len(records), "routes": routes}), flush=True)
    with torch.inference_mode():
        for rec in records:
            sid = int(rec["sample_id"])
            base_row = baseline[sid]
            ids_cpu = torch.tensor(rec["input_ids"], dtype=torch.long)
            seq_len = int(ids_cpu.numel())
            ids_large = ids_cpu.to(large_device)
            large_attn = torch.ones(1, seq_len, dtype=torch.long, device=large_device)
            large_out = large_model(input_ids=ids_large.unsqueeze(0), attention_mask=large_attn, use_cache=True, logits_to_keep=1)
            large_all = v.cache_to_features(large_out.past_key_values).to(device=large_device, dtype=dtype)
            del large_out
            if large_device.type == "cuda":
                torch.cuda.empty_cache()

            ids_small = ids_cpu.to(small_device)
            small_attn = torch.ones(1, seq_len, dtype=torch.long, device=small_device)
            t0 = time.perf_counter()
            small_out = small_model(
                input_ids=ids_small.unsqueeze(0),
                attention_mask=small_attn,
                use_cache=True,
                output_hidden_states=True,
                logits_to_keep=1,
            )
            if small_device.type == "cuda":
                torch.cuda.synchronize(small_device)
            small_ms = (time.perf_counter() - t0) * 1000.0
            small_features = v.cache_to_features(small_out.past_key_values).to(device=small_device, dtype=dtype)
            selected = select_small_by_mapping(small_features, mapping)
            t0 = time.perf_counter()
            pred = base_patch(selected, target_ids).to(dtype=dtype).to(device=large_device, dtype=dtype)
            # Compute every native layer needed by any route once for this prompt.
            for layer in sorted({x for route in routes for x in route["native_layers"]}):
                small_layer = mapping_cpu[layer]
                x_small = y.get_small_xnorm(small_model, small_out.hidden_states, small_layer).to(dtype=dtype)
                adapter = af_adapters.get(int(layer))
                if adapter is not None:
                    x_large_pred = af.block_predict_xnorm(
                        x_small,
                        native_payload,
                        adapter,
                        layer,
                        seq_len,
                        small_device,
                        dtype,
                    )
                else:
                    x_large_pred = aa.predict_large_xnorm(
                        x_small,
                        native_weights,
                        native_biases,
                        layer,
                        seq_len,
                        int(native_payload["blocks"]),
                        small_device,
                        dtype,
                    )
                pred[:, layer, :, :] = aa.native_project_features(
                    large_model, layer, x_large_pred.to(device=large_device), seq_len, dtype, large_device
                )
                del x_small, x_large_pred
            if small_device.type == "cuda":
                torch.cuda.synchronize(small_device)
            if large_device.type == "cuda":
                torch.cuda.synchronize(large_device)
            patch_ms_all = (time.perf_counter() - t0) * 1000.0
            del selected, small_features, small_out, ids_small, small_attn
            if small_device.type == "cuda":
                torch.cuda.empty_cache()

            for route in routes:
                exact = sorted(set(route["exact_layers"]))
                patch_layers = [x for x in ALL_LAYERS if x not in set(exact)]
                t0 = time.perf_counter()
                mixed = pred.clone()
                exact_tensor = torch.tensor(exact, device=large_device, dtype=torch.long)
                if exact:
                    mixed[:, exact_tensor, :, :] = large_all[:, exact_tensor, :, :]
                if large_device.type == "cuda":
                    torch.cuda.synchronize(large_device)
                materialize_ms = (time.perf_counter() - t0) * 1000.0

                count_depth = len(exact)
                deploy_depth = (max(exact) + 1) if exact else 0
                t0 = time.perf_counter()
                tmp = run_prefix_layers(large_model, ids_large.unsqueeze(0), large_attn, count_depth)
                if large_device.type == "cuda":
                    torch.cuda.synchronize(large_device)
                count_recompute_ms = (time.perf_counter() - t0) * 1000.0
                del tmp

                if deploy_depth == count_depth:
                    deploy_recompute_ms = count_recompute_ms
                else:
                    t0 = time.perf_counter()
                    tmp = run_prefix_layers(large_model, ids_large.unsqueeze(0), large_attn, deploy_depth)
                    if large_device.type == "cuda":
                        torch.cuda.synchronize(large_device)
                    deploy_recompute_ms = (time.perf_counter() - t0) * 1000.0
                    del tmp

                mx = v.decode_from_mixed_cache(large_model, tokenizer, ids_large, mixed, dtype, large_device, args.max_new_tokens)
                score = v.score_text("mixed", mx["text"], rec["gold_answers"])
                full_total = float(base_row["full_large_total_ms"])
                total_count = small_ms + patch_ms_all + materialize_ms + count_recompute_ms + float(mx["decode_ms"])
                total_deploy = small_ms + patch_ms_all + materialize_ms + deploy_recompute_ms + float(mx["decode_ms"])
                row = {
                    "sample_id": sid,
                    "route_name": route["name"],
                    "native_layers_json": json.dumps(route["native_layers"], separators=(",", ":")),
                    "exact_layers_json": json.dumps(exact, separators=(",", ":")),
                    "patch_layers_json": json.dumps(patch_layers, separators=(",", ":")),
                    "exact_layer_count": len(exact),
                    "patch_layer_count": len(patch_layers),
                    "count_proxy_depth": count_depth,
                    "deploy_prefix_depth": deploy_depth,
                    "full_large_f1": float(base_row["full_large_f1"]),
                    "full_large_em": float(base_row["full_large_em"]),
                    "full_large_total_ms": full_total,
                    "mixed_f1": float(score["mixed_f1"]),
                    "mixed_em": float(score["mixed_em"]),
                    "mixed_raw_f1": float(score["mixed_raw_f1"]),
                    "mixed_raw_em": float(score["mixed_raw_em"]),
                    "small_prefill_ms": small_ms,
                    "patch_compute_ms": patch_ms_all,
                    "kv_materialize_ms": materialize_ms,
                    "count_proxy_recompute_ms": count_recompute_ms,
                    "deploy_recompute_ms": deploy_recompute_ms,
                    "decode_ms": float(mx["decode_ms"]),
                    "mixed_total_count_proxy_ms": total_count,
                    "mixed_total_deploy_ms": total_deploy,
                    "saving_count_proxy_vs_full_large": 1.0 - total_count / full_total,
                    "saving_deploy_vs_full_large": 1.0 - total_deploy / full_total,
                    "decode_length": int(mx["decode_length"]),
                    "stop_reason": mx["stop_reason"],
                    "mixed_output": mx["text"],
                }
                rows.append(row)
                print(json.dumps({"mode": "native_combined_sample", "sample": sid, "route": route["name"], "f1": row["mixed_f1"], "deploy_saving": row["saving_deploy_vs_full_large"]}), flush=True)
                del mixed, mx
                if large_device.type == "cuda":
                    torch.cuda.empty_cache()
            del pred, large_all, ids_large, large_attn
            if large_device.type == "cuda":
                torch.cuda.empty_cache()

    per_path = OUT / f"{args.tag}_per_sample.csv"
    fields = [
        "sample_id", "route_name", "native_layers_json", "exact_layers_json", "patch_layers_json",
        "exact_layer_count", "patch_layer_count", "count_proxy_depth", "deploy_prefix_depth",
        "full_large_f1", "full_large_em", "full_large_total_ms", "mixed_f1", "mixed_em",
        "mixed_raw_f1", "mixed_raw_em", "small_prefill_ms", "patch_compute_ms",
        "kv_materialize_ms", "count_proxy_recompute_ms", "deploy_recompute_ms", "decode_ms",
        "mixed_total_count_proxy_ms", "mixed_total_deploy_ms", "saving_count_proxy_vs_full_large",
        "saving_deploy_vs_full_large", "decode_length", "stop_reason", "mixed_output",
    ]
    write_csv(per_path, rows, fields)
    summary = []
    for route in routes:
        sub = [r for r in rows if r["route_name"] == route["name"]]
        full_f1 = mean(r["full_large_f1"] for r in sub)
        mixed_f1 = mean(r["mixed_f1"] for r in sub)
        summary.append({
            "route_name": route["name"],
            "count": len(sub),
            "native_layers_json": json.dumps(route["native_layers"], separators=(",", ":")),
            "exact_layers_json": json.dumps(route["exact_layers"], separators=(",", ":")),
            "exact_layer_count": len(route["exact_layers"]),
            "patch_layer_count": 28 - len(route["exact_layers"]),
            "count_proxy_depth": len(route["exact_layers"]),
            "deploy_prefix_depth": (max(route["exact_layers"]) + 1) if route["exact_layers"] else 0,
            "full_large_f1": full_f1,
            "mixed_f1": mixed_f1,
            "mixed_em": mean(r["mixed_em"] for r in sub),
            "f1_retention": mixed_f1 / full_f1 if full_f1 else float("nan"),
            "full_large_total_ms": mean(r["full_large_total_ms"] for r in sub),
            "mixed_total_count_proxy_ms": mean(r["mixed_total_count_proxy_ms"] for r in sub),
            "mixed_total_deploy_ms": mean(r["mixed_total_deploy_ms"] for r in sub),
            "saving_count_proxy_vs_full_large": mean(r["saving_count_proxy_vs_full_large"] for r in sub),
            "saving_deploy_vs_full_large": mean(r["saving_deploy_vs_full_large"] for r in sub),
            "small_prefill_ms": mean(r["small_prefill_ms"] for r in sub),
            "patch_compute_ms": mean(r["patch_compute_ms"] for r in sub),
            "kv_materialize_ms": mean(r["kv_materialize_ms"] for r in sub),
            "count_proxy_recompute_ms": mean(r["count_proxy_recompute_ms"] for r in sub),
            "deploy_recompute_ms": mean(r["deploy_recompute_ms"] for r in sub),
            "decode_ms": mean(r["decode_ms"] for r in sub),
            "mixed_total_deploy_ms_std": stdev(r["mixed_total_deploy_ms"] for r in sub),
        })
    sum_path = OUT / f"{args.tag}_summary.csv"
    write_csv(sum_path, summary, list(summary[0].keys()) if summary else [])
    write_json(OUT / f"{args.tag}_summary.json", {"status": "case_resolved", "summary_csv": str(sum_path), "per_sample_csv": str(per_path), "routes": summary})
    print(json.dumps({"status": "case_resolved", "summary_csv": str(sum_path), "per_sample_csv": str(per_path)}, ensure_ascii=False), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--small-device", default="cuda:1")
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--native-patches", required=True)
    ap.add_argument("--af-adapters", default="")
    ap.add_argument("--routes", required=True)
    ap.add_argument("--baseline-csv", default=str(BASELINE_CSV))
    ap.add_argument("--tag", default="AB_native_combined_n40")
    evaluate(ap.parse_args())


if __name__ == "__main__":
    main()
