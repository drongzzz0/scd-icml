from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def same_index_mapping(producer_layers: int, consumer_layers: int) -> list[int]:
    if producer_layers != consumer_layers:
        raise ValueError("same_index mapping requires equal producer and consumer depth")
    return list(range(int(consumer_layers)))


def proportional_mapping(producer_layers: int, consumer_layers: int) -> list[int]:
    producer_layers = int(producer_layers)
    consumer_layers = int(consumer_layers)
    if producer_layers <= 0 or consumer_layers <= 0:
        raise ValueError("layer counts must be positive")
    if consumer_layers == 1:
        return [0]
    return [
        int(round(layer * float(producer_layers - 1) / float(consumer_layers - 1)))
        for layer in range(consumer_layers)
    ]


def validate_monotonic_mapping(mapping: list[int], producer_layers: int) -> None:
    if not mapping:
        raise ValueError("layer mapping is empty")
    last = -1
    for idx, source in enumerate(mapping):
        source = int(source)
        if source < 0 or source >= int(producer_layers):
            raise ValueError(f"source layer out of range at consumer layer {idx}: {source}")
        if source < last:
            raise ValueError(f"mapping must be monotonic: layer {idx} maps to {source} after {last}")
        last = source


def estimate_layer_mapping(
    *,
    strategy: str,
    producer_layers: int,
    consumer_layers: int,
    mapping_path: str | Path | None = None,
) -> list[int]:
    strategy = str(strategy or "proportional")
    if strategy == "same_index":
        mapping = same_index_mapping(producer_layers, consumer_layers)
    elif strategy == "proportional":
        mapping = proportional_mapping(producer_layers, consumer_layers)
    elif strategy in {"cka", "dtw"}:
        if mapping_path:
            mapping = load_mapping(mapping_path)
        else:
            raise NotImplementedError(
                f"{strategy} mapping needs a precomputed mapping_path in this release; "
                "use proportional as the safe fallback."
            )
    else:
        raise ValueError(f"unknown layer mapping strategy: {strategy}")
    validate_monotonic_mapping(mapping, producer_layers)
    if len(mapping) != int(consumer_layers):
        raise ValueError(f"mapping length {len(mapping)} != consumer_layers {consumer_layers}")
    return mapping


def load_mapping(path: str | Path) -> list[int]:
    payload: Any = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if isinstance(payload, dict):
        if "mapping" in payload:
            payload = payload["mapping"]
        elif "source_layer_by_consumer_layer" in payload:
            payload = payload["source_layer_by_consumer_layer"]
    if not isinstance(payload, list):
        raise ValueError(f"mapping file must contain a list, got {type(payload).__name__}")
    return [int(value) for value in payload]


def mapping_artifact(
    *,
    strategy: str,
    producer_layers: int,
    consumer_layers: int,
    mapping: list[int],
) -> dict[str, Any]:
    validate_monotonic_mapping(mapping, producer_layers)
    return {
        "mapping_version": "source_target_layer_mapping_v1",
        "strategy": str(strategy),
        "producer_num_layers": int(producer_layers),
        "consumer_num_layers": int(consumer_layers),
        "source_layer_by_consumer_layer": [int(value) for value in mapping],
        "notes": [
            "same_index and proportional are implemented fallbacks.",
            "CKA/DTW scoring should write the same source_layer_by_consumer_layer schema.",
        ],
    }
