from __future__ import annotations

import argparse
import csv
import json
import pathlib
import statistics
import time
from typing import Any, Dict, Iterable, List, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import run_task_aa_native_projection_patch as aa
import run_task_ab_native_projection_combined_eval as ab
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


def write_csv(path: pathlib.Path, rows: List[Dict[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def parse_routes(text: str) -> List[Dict[str, Any]]:
    routes = []
    for spec in text.split(";"):
        spec = spec.strip()
        if not spec:
            continue
        name, rest = spec.split(":", 1)
        fields = {}
        for part in rest.split("|"):
            key, val = part.split("=", 1)
            fields[key] = [int(x) for x in val.split(",") if x.strip()]
        exact = sorted(set(fields.get("exact", [])))
        patch = sorted(set(fields.get("patch", [x for x in ALL_LAYERS if x not in set(exact)])))
        native = sorted(set(fields.get("native", [])))
        routes.append({"name": name, "exact_layers": exact, "patch_layers": patch, "native_layers": native})
    return routes


def load_sparse_patch(device: torch.device):
    payload = torch.load(BASE_PATCH, map_location="cpu", weights_only=False)
    base = p.load_ridge_patch(device)
    return SparsePerBlockRidgePatch(
        base=base,
        weights=payload["weights"],
        biases=payload["biases"],
        seq_len=int(payload["seq_len"]),
        blocks=int(payload["blocks"]),
    ).to(device).eval()


class SparsePerBlockRidgePatch(torch.nn.Module):
    def __init__(
        self,
        *,
        base: torch.nn.Module,
        weights: Dict[tuple[int, int], torch.Tensor],
        biases: Dict[tuple[int, int], torch.Tensor],
        seq_len: int,
        blocks: int,
    ) -> None:
        super().__init__()
        self.base = base
        self.weights = {k: v.float() for k, v in weights.items()}
        self.biases = {k: v.float() for k, v in biases.items()}
        self.seq_len = int(seq_len)
        self.blocks = int(blocks)
        self.block_len = int((seq_len + blocks - 1) // blocks)

    def block_slice(self, block: int) -> slice:
        start = block * self.block_len
        stop = min(self.seq_len, (block + 1) * self.block_len)
        return slice(start, stop)

    def forward_sparse(self, selected_full: torch.Tensor, target_layer_ids: torch.Tensor) -> torch.Tensor:
        ids = target_layer_ids.to(device=selected_full.device, dtype=torch.long)
        selected = selected_full.index_select(dim=1, index=ids)
        out = self.base(selected, ids).clone()
        layer_to_pos = {int(layer): pos for pos, layer in enumerate(ids.detach().cpu().tolist())}
        for (block, layer), w in self.weights.items():
            pos = layer_to_pos.get(int(layer))
            if pos is None:
                continue
            sl = self.block_slice(int(block))
            out[:, pos, sl] = (
                selected[:, pos, sl].float() @ w.to(selected_full.device)
                + self.biases[(block, layer)].to(selected_full.device)
            )
        return out


def evaluate(args: argparse.Namespace) -> None:
    dtype = p.dtype_from_name(args.dtype)
    large_device = torch.device(args.device)
    small_device = torch.device(args.small_device)
    tokenizer = AutoTokenizer.from_pretrained(str(p.SMALL_MODEL), trust_remote_code=True, local_files_only=True)
    records = v.load_qa_records(tokenizer, args.start + args.n)[args.start : args.start + args.n]
    baseline = {}
    with pathlib.Path(args.baseline_csv).open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            baseline[int(row["sample_id"])] = row
    routes = parse_routes(args.routes)
    native_payload = None
    native_weights = None
    native_biases = None
    af_adapters: Dict[int, torch.nn.Module] = {}
    if args.native_patches:
        native_payload = ab.merge_native_payload([pathlib.Path(x) for x in args.native_patches.split(",") if x.strip()])
        native_weights = native_payload["weights"]
        native_biases = native_payload["biases"]
        af_adapters = ab.load_af_adapters(args.af_adapters, small_device)
        native_all = set(int(x) for x in native_payload["target_layers"])
        for route in routes:
            missing = [x for x in route["native_layers"] if x not in native_all]
            if missing:
                raise ValueError(f"route {route['name']} requests native layers not in payload: {missing}")
    large_model = AutoModelForCausalLM.from_pretrained(
        str(p.LARGE_MODEL), trust_remote_code=True, local_files_only=True, torch_dtype=dtype, low_cpu_mem_usage=True
    ).to(large_device).eval()
    small_model = AutoModelForCausalLM.from_pretrained(
        str(p.SMALL_MODEL), trust_remote_code=True, local_files_only=True, torch_dtype=dtype, low_cpu_mem_usage=True
    ).to(small_device).eval()
    patch = load_sparse_patch(small_device)
    mapping = p.load_mapping(small_device)
    rows: List[Dict[str, Any]] = []
    print(json.dumps({"mode": "aj_sparse_eval_start", "tag": args.tag, "n": len(records), "routes": routes}), flush=True)
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
            need_hidden = any(route["native_layers"] for route in routes)
            small_out = small_model(
                input_ids=ids_small.unsqueeze(0),
                attention_mask=small_attn,
                use_cache=True,
                output_hidden_states=need_hidden,
                logits_to_keep=1,
            )
            if small_device.type == "cuda":
                torch.cuda.synchronize(small_device)
            small_ms = (time.perf_counter() - t0) * 1000.0
            small_features = v.cache_to_features(small_out.past_key_values).to(device=small_device, dtype=dtype)
            selected = select_small_by_mapping(small_features, mapping)
            del small_features, ids_small, small_attn
            if small_device.type == "cuda":
                torch.cuda.empty_cache()

            for route in routes:
                exact = sorted(route["exact_layers"])
                patch_layers = sorted(route["patch_layers"])
                native_layers = sorted(route["native_layers"])
                base_patch_layers = [x for x in patch_layers if x not in set(native_layers)]
                patch_ids = torch.tensor(base_patch_layers, device=small_device, dtype=torch.long)
                t0 = time.perf_counter()
                pred_patch = patch.forward_sparse(selected, patch_ids).to(device=large_device, dtype=dtype) if base_patch_layers else None
                pred_native: Dict[int, torch.Tensor] = {}
                for layer in native_layers:
                    if native_payload is None or native_weights is None or native_biases is None:
                        raise ValueError("native route requested but --native-patches was not provided")
                    small_layer = int(native_payload["mapping"][str(layer)])
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
                    pred_native[layer] = aa.native_project_features(
                        large_model, layer, x_large_pred.to(device=large_device), seq_len, dtype, large_device
                    )
                    del x_small, x_large_pred
                if small_device.type == "cuda":
                    torch.cuda.synchronize(small_device)
                if large_device.type == "cuda":
                    torch.cuda.synchronize(large_device)
                patch_ms = (time.perf_counter() - t0) * 1000.0

                t0 = time.perf_counter()
                mixed = large_all.clone()
                if base_patch_layers:
                    patch_tensor = torch.tensor(base_patch_layers, device=large_device, dtype=torch.long)
                    mixed[:, patch_tensor, :, :] = pred_patch
                for layer, feat in pred_native.items():
                    mixed[:, int(layer), :, :] = feat
                if large_device.type == "cuda":
                    torch.cuda.synchronize(large_device)
                materialize_ms = (time.perf_counter() - t0) * 1000.0
                deploy_depth = (max(exact) + 1) if exact else 0
                t0 = time.perf_counter()
                tmp = run_prefix_layers(large_model, ids_large.unsqueeze(0), large_attn, deploy_depth)
                if large_device.type == "cuda":
                    torch.cuda.synchronize(large_device)
                recompute_ms = (time.perf_counter() - t0) * 1000.0
                del tmp
                mx = v.decode_from_mixed_cache(large_model, tokenizer, ids_large, mixed, dtype, large_device, args.max_new_tokens)
                score = v.score_text("mixed", mx["text"], rec["gold_answers"])
                full_total = float(base_row["full_large_total_ms"])
                total = small_ms + patch_ms + materialize_ms + recompute_ms + float(mx["decode_ms"])
                row = {
                    "sample_id": sid,
                    "route_name": route["name"],
                    "exact_layers_json": json.dumps(exact, separators=(",", ":")),
                    "patch_layers_json": json.dumps(patch_layers, separators=(",", ":")),
                    "native_layers_json": json.dumps(native_layers, separators=(",", ":")),
                    "exact_layer_count": len(exact),
                    "patch_layer_count": len(patch_layers),
                    "deploy_prefix_depth": deploy_depth,
                    "full_large_f1": float(base_row["full_large_f1"]),
                    "full_large_em": float(base_row["full_large_em"]),
                    "full_large_total_ms": full_total,
                    "mixed_f1": float(score["mixed_f1"]),
                    "mixed_em": float(score["mixed_em"]),
                    "mixed_raw_f1": float(score["mixed_raw_f1"]),
                    "mixed_raw_em": float(score["mixed_raw_em"]),
                    "small_prefill_ms": small_ms,
                    "patch_compute_ms": patch_ms,
                    "kv_materialize_ms": materialize_ms,
                    "deploy_recompute_ms": recompute_ms,
                    "decode_ms": float(mx["decode_ms"]),
                    "mixed_total_deploy_ms": total,
                    "saving_deploy_vs_full_large": 1.0 - total / full_total,
                    "decode_length": int(mx["decode_length"]),
                    "stop_reason": mx["stop_reason"],
                    "mixed_output": mx["text"],
                }
                rows.append(row)
                print(json.dumps({"mode": "aj_sparse_eval_sample", "sample": sid, "route": route["name"], "f1": row["mixed_f1"], "saving": row["saving_deploy_vs_full_large"], "patch_ms": patch_ms}), flush=True)
                del pred_patch, pred_native, mixed, mx
                if large_device.type == "cuda":
                    torch.cuda.empty_cache()
            del selected, small_out, large_all, ids_large, large_attn
            if large_device.type == "cuda":
                torch.cuda.empty_cache()

    fields = [
        "sample_id", "route_name", "exact_layers_json", "patch_layers_json", "exact_layer_count",
        "patch_layer_count", "native_layers_json", "deploy_prefix_depth", "full_large_f1", "full_large_em",
        "full_large_total_ms", "mixed_f1", "mixed_em", "mixed_raw_f1", "mixed_raw_em",
        "small_prefill_ms", "patch_compute_ms", "kv_materialize_ms", "deploy_recompute_ms",
        "decode_ms", "mixed_total_deploy_ms", "saving_deploy_vs_full_large", "decode_length",
        "stop_reason", "mixed_output",
    ]
    per_path = OUT / f"{args.tag}_per_sample.csv"
    write_csv(per_path, rows, fields)
    summary = []
    for route in routes:
        sub = [r for r in rows if r["route_name"] == route["name"]]
        full_f1 = mean(r["full_large_f1"] for r in sub)
        mixed_f1 = mean(r["mixed_f1"] for r in sub)
        summary.append({
            "route_name": route["name"],
            "count": len(sub),
            "exact_layers_json": json.dumps(route["exact_layers"], separators=(",", ":")),
            "patch_layers_json": json.dumps(route["patch_layers"], separators=(",", ":")),
            "native_layers_json": json.dumps(route["native_layers"], separators=(",", ":")),
            "exact_layer_count": len(route["exact_layers"]),
            "patch_layer_count": len(route["patch_layers"]),
            "deploy_prefix_depth": (max(route["exact_layers"]) + 1) if route["exact_layers"] else 0,
            "full_large_f1": full_f1,
            "mixed_f1": mixed_f1,
            "mixed_em": mean(r["mixed_em"] for r in sub),
            "f1_retention": mixed_f1 / full_f1 if full_f1 else float("nan"),
            "full_large_total_ms": mean(r["full_large_total_ms"] for r in sub),
            "mixed_total_deploy_ms": mean(r["mixed_total_deploy_ms"] for r in sub),
            "saving_deploy_vs_full_large": mean(r["saving_deploy_vs_full_large"] for r in sub),
            "small_prefill_ms": mean(r["small_prefill_ms"] for r in sub),
            "patch_compute_ms": mean(r["patch_compute_ms"] for r in sub),
            "kv_materialize_ms": mean(r["kv_materialize_ms"] for r in sub),
            "deploy_recompute_ms": mean(r["deploy_recompute_ms"] for r in sub),
            "decode_ms": mean(r["decode_ms"] for r in sub),
            "mixed_total_deploy_ms_std": stdev(r["mixed_total_deploy_ms"] for r in sub),
        })
    sum_path = OUT / f"{args.tag}_summary.csv"
    write_csv(sum_path, summary, list(summary[0].keys()) if summary else [])
    print(json.dumps({"status": "case_resolved", "summary_csv": str(sum_path), "per_sample_csv": str(per_path)}, ensure_ascii=False), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--small-device", default="cuda:1")
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--baseline-csv", default=str(BASELINE_CSV))
    ap.add_argument("--routes", required=True)
    ap.add_argument("--native-patches", default="")
    ap.add_argument("--af-adapters", default="")
    ap.add_argument("--tag", default="AJ_sparse_patch_n40")
    args = ap.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
