from __future__ import annotations

from pathlib import Path

from .common import read_jsonl
from .inventory import inventory_index


DEFAULT_TRACE_TENSORS = [
    "x_norm",
    "k_content_deroped",
    "v",
    "logits",
]


def build_trace_capture_plan(inventory: dict, request: dict) -> dict:
    inventory_idx = inventory_index(inventory)
    producer = inventory_idx[request["producer_model"]]
    consumer = inventory_idx[request["consumer_model"]]
    prompt_manifest_path = Path(request["prompt_manifest_path"])
    prompt_rows = read_jsonl(prompt_manifest_path)
    layers = request.get("layers")
    if layers in (None, "all"):
        layers = list(range(int(producer["architecture"]["num_layers"])))
    tensors = request.get("trace_tensors", DEFAULT_TRACE_TENSORS)
    output_root = Path(request["output_root"])

    operations = []
    for role, model in [("producer", producer), ("consumer", consumer)]:
        operations.append(
            {
                "role": role,
                "model_id": model["model_id"],
                "model_path": model.get("model_path", ""),
                "repo_id": model.get("repo_id", ""),
                "layers": layers,
                "trace_tensors": tensors,
                "prompt_count": len(prompt_rows),
                "output_dir": str(output_root / request["pair_id"] / role),
            }
        )

    return {
        "pair_id": request["pair_id"],
        "producer_model": producer["model_id"],
        "consumer_model": consumer["model_id"],
        "task": request["task"],
        "split": request["split"],
        "context_len": request["context_len"],
        "decode_len": request["decode_len"],
        "prompt_manifest_path": str(prompt_manifest_path),
        "operations": operations,
        "notes": [
            "This is a trace-capture plan only. It does not imply traces were actually collected.",
            "For the original SCD baseline, suffix recomputation remains excluded at this stage.",
        ],
    }

