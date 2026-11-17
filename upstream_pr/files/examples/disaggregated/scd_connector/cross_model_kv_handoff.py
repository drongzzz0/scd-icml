# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Minimal example of cross-model KV cache handoff with SCDConnectorV1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run cross-model SCD KV cache handoff into vLLM V1."
    )
    parser.add_argument("--profile", required=True, help="Path to SCD profile JSON")
    parser.add_argument("--assets", required=True, help="Path to imported SCD assets")
    parser.add_argument("--target-model", required=True, help="Target model path")
    parser.add_argument("--target-manifest", required=True, help="Target model manifest JSON")
    parser.add_argument("--prefix-ids", required=True, help="JSON list of prefix token IDs")
    parser.add_argument("--prompt-ids", required=True, help="JSON list of full prompt token IDs")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Disable CUDA graphs (default: CUDA graphs enabled for minimum latency).",
    )
    return parser.parse_args()


def main() -> None:
    from scd_runtime.contracts.types import PrefixContract, profile_from_dict
    from scd_runtime.resident_runner import GPUResidentConsumer

    args = parse_args()
    profile = profile_from_dict(json.loads(Path(args.profile).read_text(encoding="utf-8")))
    prefix_ids = tuple(json.loads(Path(args.prefix_ids).read_text(encoding="utf-8")))
    prompt_ids = tuple(json.loads(Path(args.prompt_ids).read_text(encoding="utf-8")))

    prefix = PrefixContract(
        prefix_ids,
        0,
        profile.source.tokenizer_sha256,
        profile.source.chat_template_sha256,
    )
    engine_config = {
        "model": args.target_model,
        "model_manifest": args.target_manifest,
        "block_size": 16,
        "max_model_len": 4096,
        "gpu_memory_utilization": 0.45,
        "enforce_eager": bool(args.enforce_eager),
        "attention_config": {"backend": "FLASH_ATTN"},
    }

    consumer = GPUResidentConsumer(profile, args.assets, engine_config)
    try:
        receipt = consumer.run(
            prefix,
            prompt_ids,
            {"max_new_tokens": args.max_new_tokens, "temperature": 0.0},
            mode="E",
            import_policy="fullprefix",
            full_prompt_ids=prompt_ids if len(prompt_ids) == len(prefix_ids) + 1 else None,
        )
        print(
            json.dumps(
                {
                    "token_ids": list(receipt.generation.token_ids),
                    "ttft_ms": receipt.metrics["pipeline_first_token_ms"],
                    "generation_ms": receipt.metrics["pipeline_generation_ms"],
                    "imported_tokens": receipt.generation.metadata["imported_tokens"],
                },
                indent=2,
            )
        )
    finally:
        consumer.close()


if __name__ == "__main__":
    main()
