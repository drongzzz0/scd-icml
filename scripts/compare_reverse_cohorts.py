"""Join the independently replayed reverse CMRC cohorts on the verified QIDs."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
from statistics import mean

from benchmark_a800_cmrc_quality import DATASET_SIZE, read_json, write_json
from scd_runtime.assets.manifest import asset_binding
from scd_runtime.contracts.types import profile_from_dict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root
    arms = ("raw", "scd")
    reports = {arm: read_json(root / f"report-{arm}/metrics.json") for arm in arms}
    profiles = {arm: (root / "a800-results/provenance" / f"{arm}-profile.json").read_bytes() for arm in arms}
    if len({asset_binding(profile_from_dict(json.loads(data))) for data in profiles.values()}) != 1:
        raise ValueError("reverse model identities or route differ")
    cohorts = {arm: {} for arm in arms}
    for shard in range(2):
        gpus = []
        shard_ids = []
        for arm in arms:
            directory = root / f"a800-results/{arm}-shard-{shard}"
            gpus.append(read_json(directory / "launch.json")["gpu"][:2])
            header = reports[arm]["shards"][shard]
            if (header["shard_index"] != shard or header["profile_sha256"] != sha256(profiles[arm]).hexdigest()
                    or (header["source_model"], header["target_model"]) != ("Mistral-7B-v0.1", "MistralLite")):
                raise ValueError("unexpected reverse model or profile identity")
            with (directory / "attempts.jsonl").open(encoding="utf-8") as stream:
                rows = {row["qid"]: row for row in (json.loads(line) for line in stream)}
            shard_ids.append(set(rows))
            cohorts[arm].update(rows)
        if gpus[0] != gpus[1] or shard_ids[0] != shard_ids[1]:
            raise ValueError("reverse replay changed the physical GPU or QIDs in a shard")

    raw, scd = (cohorts[arm] for arm in arms)
    if len(raw) != DATASET_SIZE or raw.keys() != scd.keys():
        raise ValueError("reverse replay coverage differs")
    geometry = ("prompt_tokens", "loaded_tokens", "loaded_layers", "transfer_bytes")
    if any(raw[qid][key] != scd[qid][key] for qid in raw for key in geometry):
        raise ValueError("import or transfer geometry differs on the same prompt")
    same_length = [qid for qid in raw if len(raw[qid]["token_ids"]) == len(scd[qid]["token_ids"])]
    quality = {arm: reports[arm]["quality"][arm] for arm in arms}
    quality["delta"] = {k: quality["scd"][k] - quality["raw"][k] for k in ("F1", "EM")}
    timing = {arm: reports[arm]["timing"][arm] for arm in arms}
    out = root / "report"
    out.mkdir(exist_ok=True)
    result = {
        "source_model": "Mistral-7B-v0.1",
        "target_model": "MistralLite",
        "N": len(raw),
        "quality": quality,
        "timing": timing,
        "source_apc": {arm: reports[arm]["source_apc"] for arm in arms},
        "qidwise_identical_apc_tokens": sum(raw[q]["source_cached_tokens"] == scd[q]["source_cached_tokens"] for q in raw),
        "equal_output_length": {"N": len(same_length)},
    }
    if same_length:
        result["equal_output_length"].update({
            arm: {stage: mean(cohorts[arm][q][stage] for q in same_length)
                  for stage in ("ttft_ms", "request_wall_ms")}
            for arm in arms
        })
    write_json(out / "metrics.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
