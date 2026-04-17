from __future__ import annotations

from itertools import combinations


INVENTORY_REQUIRED_FIELDS = [
    "model_id",
    "family",
    "base_model",
    "tokenizer_id",
    "architecture",
]

ARCHITECTURE_FIELDS = [
    "num_layers",
    "hidden_size",
    "num_attention_heads",
    "num_key_value_heads",
    "rope_type",
]


def _require_keys(data: dict, required: list[str], *, where: str) -> None:
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"missing keys in {where}: {missing}")


def validate_inventory(inventory: dict) -> dict:
    models = inventory.get("models", [])
    if not models:
        raise ValueError("inventory must contain a non-empty 'models' list")
    seen: set[str] = set()
    for idx, model in enumerate(models):
        _require_keys(model, INVENTORY_REQUIRED_FIELDS, where=f"models[{idx}]")
        _require_keys(model["architecture"], ARCHITECTURE_FIELDS, where=f"models[{idx}].architecture")
        model_id = model["model_id"]
        if model_id in seen:
            raise ValueError(f"duplicate model_id in inventory: {model_id}")
        seen.add(model_id)
    return inventory


def architecture_signature(model: dict) -> tuple:
    arch = model["architecture"]
    return tuple(arch[field] for field in ARCHITECTURE_FIELDS)


def inventory_index(inventory: dict) -> dict[str, dict]:
    validate_inventory(inventory)
    return {model["model_id"]: model for model in inventory["models"]}


def pair_id_for(producer_model_id: str, consumer_model_id: str) -> str:
    return f"{producer_model_id}__to__{consumer_model_id}"


def build_candidate_pairs(inventory: dict, pair_policy: dict | None = None) -> list[dict]:
    validate_inventory(inventory)
    pair_policy = pair_policy or {}
    require_same_family = bool(pair_policy.get("require_same_family", True))
    require_same_base_model = bool(pair_policy.get("require_same_base_model", True))
    require_shared_tokenizer = bool(pair_policy.get("require_shared_tokenizer", True))
    require_architecture_match = bool(pair_policy.get("require_architecture_match", True))

    candidates: list[dict] = []
    for producer, consumer in combinations(inventory["models"], 2):
        same_family = producer["family"] == consumer["family"]
        same_base_model = producer["base_model"] == consumer["base_model"]
        shared_tokenizer = producer["tokenizer_id"] == consumer["tokenizer_id"]
        architecture_compatible = architecture_signature(producer) == architecture_signature(consumer)
        if require_same_family and not same_family:
            continue
        if require_same_base_model and not same_base_model:
            continue
        if require_shared_tokenizer and not shared_tokenizer:
            continue
        if require_architecture_match and not architecture_compatible:
            continue

        notes = []
        producer_domain = producer.get("finetune_domain")
        consumer_domain = consumer.get("finetune_domain")
        if producer_domain or consumer_domain:
            notes.append(f"domains={producer_domain}->{consumer_domain}")
        candidates.append(
            {
                "pair_id": pair_id_for(producer["model_id"], consumer["model_id"]),
                "producer_model": producer["model_id"],
                "consumer_model": consumer["model_id"],
                "producer_spec": producer,
                "consumer_spec": consumer,
                "tokenizer_shared": shared_tokenizer,
                "architecture_compatible": architecture_compatible,
                "same_family": same_family,
                "same_base_model": same_base_model,
                "notes": "; ".join(notes),
            }
        )
    return candidates


def resolve_pair(inventory: dict, pair_id: str) -> dict:
    for pair in build_candidate_pairs(inventory, inventory.get("pair_policy", {})):
        if pair["pair_id"] == pair_id:
            return pair
    raise KeyError(f"pair_id not found in inventory pair candidates: {pair_id}")

