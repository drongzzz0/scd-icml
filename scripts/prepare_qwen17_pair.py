"""Bind the archived, train-only Qwen3-1.7B/Gray converter to the local models."""

import argparse
import json
from pathlib import Path

from benchmark_a800_cmrc_quality import load_verified_cases
from benchmark_a800_cmrc_raw_apc import retokenize_cases
from scd_runtime.assets.legacy_import import import_legacy
from scd_runtime.assets.manifest import file_sha256
from scd_runtime.contracts.identity import canonical_bytes, sha256
from scd_runtime.contracts.types import MethodProfile, ModelIdentity, ReaderIdentity, to_data


BASE_WEIGHTS = {
    "model-00001-of-00002.safetensors": "169ad53ec313c3a34b06c0809216e4fc072cce444a5d4ff2b59690d064130ed5",
    "model-00002-of-00002.safetensors": "912becff8d60672aa8628ef08c05898d9adf17c2ad4ae3caf99b065622fdeff9",
}
GRAY_WEIGHTS = {
    "model.safetensors": "2d9916f765badb30a246f614800f5de984ec0919e2c86525a96576a1cd3b069d",
}
TOKENIZER_SHA256 = "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"
ROUTE_SHA256 = "0890ae30ae18edea897cf97861d3a34ec94db1ab1fd49fc13f74846262d08375"
BANKS = {
    "reuse": ("ckpt_reuse_layerwise_seed42", "aeaf2989c6b3a722ca4405c39f0d1aa670388c06adbec3be95e5a3f7c8542fa8"),
    "patch": ("ckpt_patch_xnorm_alllayers_seed42", "c8cd12a751a32076caa3a91c9fd47cde2dacf1ec67df0b263da5c11257a7ebd3"),
}


def bind_model(root, weights, tokenizer, manifest):
    config = json.loads((root / "config.json").read_bytes())
    geometry = (config["num_hidden_layers"], config["num_key_value_heads"], config["head_dim"])
    if (config["model_type"] != "qwen3" or geometry != (28, 8, 128)
            or config["hidden_size"] != 2048 or config.get("sliding_window") is not None):
        raise ValueError("unexpected Qwen3 model/cache geometry")
    names = set(weights) | {
        "config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
        "special_tokens_map.json", "vocab.json", "merges.txt", "chat_template.jinja",
        "model.safetensors.index.json",
    }
    files = {}
    for name in sorted(names):
        path = root / name
        if path.is_file():
            files[name] = {"path": name, "size_bytes": path.stat().st_size, "sha256": file_sha256(path)}
    if (any(files[name]["sha256"] != expected for name, expected in weights.items())
            or files["tokenizer.json"]["sha256"] != TOKENIZER_SHA256):
        raise ValueError(f"model weights or tokenizer differ from the archived pair: {root}")
    manifest.write_bytes(canonical_bytes({"files": list(files.values())}))
    vocabulary = {
        name: value["sha256"] for name, value in files.items()
        if name.startswith("tokenizer") or name in ("vocab.json", "merges.txt", "special_tokens_map.json")
    }
    return ModelIdentity(
        manifest_sha256=file_sha256(manifest), config_sha256=files["config.json"]["sha256"],
        tokenizer_sha256=sha256(vocabulary), chat_template_sha256=sha256(tokenizer.chat_template),
        num_layers=geometry[0], num_kv_heads=geometry[1], head_dim=geometry[2], dtype="bfloat16",
        rope={"theta": config["rope_theta"], "rope_type": "default"},
        attention={"layer_types": ["full"] * geometry[0], "sliding_window": None},
    )


def main():
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "target", "archive", "cases", "provenance", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    root = args.provenance
    root.mkdir(parents=True, exist_ok=False)
    source_tokenizer, target_tokenizer = (
        AutoTokenizer.from_pretrained(str(model), local_files_only=True, trust_remote_code=False)
        for model in (args.source, args.target)
    )
    cases = load_verified_cases(args.cases.resolve(), None, 0, 1, keep_contexts=True)
    retokenize_cases(cases, source_tokenizer, target_tokenizer)
    cohort = {
        "questions": len(cases), "max_prompt_tokens": max(len(case["prompt_ids"]) for case in cases),
        "input_sha256": sha256([[case["qid"], case["prompt_text"], case["prompt_ids"]] for case in cases]),
        "target_stop_ids": sorted(target_tokenizer.all_special_ids),
    }
    (root / "retokenization.json").write_bytes(canonical_bytes(cohort))
    source_manifest, target_manifest = (root / f"{arm}-model-manifest.json" for arm in ("source", "target"))
    source = bind_model(args.source, BASE_WEIGHTS, source_tokenizer, source_manifest)
    target = bind_model(args.target, GRAY_WEIGHTS, target_tokenizer, target_manifest)
    route_path = args.archive / "route_qwen17_gray_patch6_seed42.json"
    if file_sha256(route_path) != ROUTE_SHA256:
        raise ValueError("archived train-only route has changed")
    route = json.loads(route_path.read_bytes())
    profile = MethodProfile(
        "icml_qwen17_gray_cmrc", "icml_reuse_patch", source, target,
        ReaderIdentity("none", target.manifest_sha256),
        {**{k: route[k] for k in ("patch_layers", "reuse_layers", "fallback_layers", "selected_recompute_layers")},
         "reuse_key_space": "post_rope", "fit_mode": "xnorm_nativekv_shared_residual_mlp_v1"},
        tuple(range(target.num_layers)), "icml-pinned-v3-hf-native-rope", "0" * 64,
    )
    profile_path = root / "unbound-profile.json"
    profile_path.write_bytes(canonical_bytes(to_data(profile)))
    inputs = {}
    expected = {}
    for arm, (subdirectory, digest) in BANKS.items():
        bank = args.archive / subdirectory
        inputs[arm] = bank / "weights.pt"
        inputs[arm + "_meta"] = bank / "meta.json"
        expected[arm] = digest
        expected[arm + "_meta"] = file_sha256(inputs[arm + "_meta"])
    metadata_path = root / "import-metadata.json"
    metadata_path.write_bytes(canonical_bytes({
        "legacy_format": "icml_banks", "expected_input_sha256": expected,
        "archive_run": "extra_model_qwen17_gray_full_strict_20260617_223703",
        "route_sha256": ROUTE_SHA256,
        "models": {
            "source": {"root": str(args.source), "manifest": str(source_manifest)},
            "target": {"root": str(args.target), "manifest": str(target_manifest)},
        },
    }))
    import_legacy(profile.profile_id, {**inputs, "profile": profile_path, "metadata": metadata_path},
                  args.out, trusted_legacy=True)
    (root / "profile.json").write_bytes((args.out / "profile.json").read_bytes())
    (root / "asset-manifest.json").write_bytes((args.out / "manifest.json").read_bytes())
    print(json.dumps({"status": "bound", **cohort, "asset_bytes": (args.out / "weights.safetensors").stat().st_size}))


if __name__ == "__main__":
    main()
