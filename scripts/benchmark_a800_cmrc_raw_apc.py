"""Replay both cross-model KV paths on all verified questions of CMRC2018 dev."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import runpy
import time

from benchmark_a800_cmrc_quality import (
    CMRC_DEV_SHA256, DATASET_SIZE, DEV26_PARSER_SHA256, MAX_NEW_TOKENS,
    load_verified_cases, read_json, write_json,
)
from cmrc_answer_stop import CMRCAnswerStop


ARMS = ("raw", "scd")
MAX_MODEL_LEN = 1312
STAGE_KEYS = (
    "prefill_ms", "conversion_ms", "transfer_ms", "target_setup_ms", "kv_load_ms",
    "decode_ms", "ttft_ms", "completion_ms", "cleanup_ms", "request_wall_ms",
)


def paired_request_timings(prefill_ns, started, converted, received, load_start, load_end,
                           first_token, stop, returned):
    stamps = (started, converted, received, load_start, load_end, first_token, stop, returned)
    if prefill_ns <= 0 or any(not isinstance(x, int) for x in stamps) or list(stamps) != sorted(stamps):
        raise ValueError(f"invalid paired request timestamps: {prefill_ns=}, {stamps=}")
    durations = (
        prefill_ns, converted - started, received - converted, load_start - received,
        load_end - load_start, stop - load_end, prefill_ns + first_token - started,
        prefill_ns + stop - started, returned - stop, prefill_ns + returned - started,
    )
    return {key: value / 1e6 for key, value in zip(STAGE_KEYS, durations, strict=True)}


def validate_paired_rows(rows, qids, *, arms=ARMS):
    pairs = {}
    for row in rows:
        pair = pairs.setdefault(row["qid"], {})
        arm = row["arm"]
        if arm not in arms or arm in pair:
            raise ValueError(f"invalid or duplicate arm {arm!r} for {row['qid']}")
        pair[arm] = row
    if set(pairs) != qids or any(set(pair) != set(arms) for pair in pairs.values()):
        raise ValueError("paired CMRC QID coverage does not match")
    if len(arms) == 1:
        return pairs
    for qid, pair in pairs.items():
        raw, scd = (pair[arm] for arm in ARMS)
        keys = ("source_cached_tokens", "prompt_tokens", "loaded_tokens", "prefill_ms")
        if any(raw[key] != scd[key] for key in keys):
            raise ValueError(f"source or imported prefix differs between arms for {qid}")
    return pairs


def retokenize_cases(cases, source_tokenizer, target_tokenizer):
    """Rebind the exact frozen prompt text to a pair sharing a tokenizer."""
    for case in cases:
        prompt = case["prompt_text"]
        source_ids = source_tokenizer.encode(prompt, add_special_tokens=True)
        target_ids = target_tokenizer.encode(prompt, add_special_tokens=True)
        if source_ids != target_ids or not 2 <= len(source_ids) <= MAX_MODEL_LEN - MAX_NEW_TOKENS:
            raise ValueError(f"unmatched or out-of-range prompt tokens for {case['qid']}")
        case.update(prompt_ids=target_ids, source_input_ids=source_ids, prefix_ids=target_ids[:-1])
        case["generation"] = {**case["generation"], "stop_token_ids": list(target_tokenizer.all_special_ids)}


def _source_rpc(llm, method, **kwargs):
    (result,) = llm.collective_rpc(method, timeout=120, kwargs=kwargs)
    return result


def evaluate_case(source, target, case, profile, tokenizer, parser, sampling, arms, *, verify_bits):
    from scd_runtime.contracts.types import PrefixContract, to_data

    source_ids = case["source_input_ids"]
    prompt_ids = case["prompt_ids"]
    if source_ids != prompt_ids or source_ids[:-1] != case["prefix_ids"]:
        raise ValueError(f"prepared prompt contract differs for {case['qid']}")
    prefix = PrefixContract(
        tuple(source_ids[:-1]), 0, profile.source.tokenizer_sha256, profile.source.chat_template_sha256
    )
    prefix_data = to_data(prefix)
    stop_ids = set(case["generation"]["stop_token_ids"])
    stop_ids.update(tokenizer.all_special_ids)
    settings = {
        "max_new_tokens": MAX_NEW_TOKENS, "temperature": 0.0,
        "eos_token_id": sorted(stop_ids), "ignore_eos": False,
    }
    source_started = time.perf_counter_ns()
    (output,) = source.generate([{"prompt_token_ids": source_ids}], sampling, use_tqdm=False)
    prefill_ns = time.perf_counter_ns() - source_started
    cached = output.num_cached_tokens
    if type(cached) is not int or not 0 <= cached < len(source_ids) or len(output.outputs[0].token_ids) != 1:
        raise RuntimeError(f"invalid source prefill receipt for {case['qid']}")

    for arm in arms:
        stop = CMRCAnswerStop(
            tokenizer, parser["extract_answer_text"], case["context"],
            parser["ANSWER_CUTOFF_SEPARATORS"], stop_ids, MAX_NEW_TOKENS,
        )
        first_ns = stop_ns = None

        def observe(token_ids):
            nonlocal first_ns, stop_ns
            if first_ns is None:
                first_ns = time.perf_counter_ns()
            reason = stop(token_ids)
            if reason is not None:
                stop_ns = time.perf_counter_ns()
            return reason

        arm_start = time.perf_counter_ns()
        if arm == "scd":
            _source_rpc(source, "cmrc_convert", prefix_data=prefix_data, source_prompt_ids=source_ids)
            converted = time.perf_counter_ns()
        else:
            converted = arm_start
        exported = _source_rpc(source, "cmrc_export", arm=arm, source_prompt_ids=source_ids, verify_bits=verify_bits)
        try:
            transfer = target._rpc("cmrc_receive", descriptor=exported["ipc"], expected_digest=exported["digest"])
            received = time.perf_counter_ns()
            prepared = target._rpc("cmrc_register", session_id=target.session_id, prefix_data=prefix_data)
            result = target.consume_prepared(
                prepared, prompt_ids, settings, mode=arm, import_policy="fullprefix", token_observer=observe,
            )
        finally:
            _source_rpc(source, "cmrc_release_export")
        returned = time.perf_counter_ns()
        receipt = result.generation.metadata["injection_receipt"]
        if (result.status != "success" or receipt["loaded_tokens"] != len(prefix.token_ids)
                or receipt["loaded_layers"] != profile.target.num_layers or receipt["native_fallback"]
                or tuple(result.generation.token_ids) != stop.observed or stop.reason is None):
            raise RuntimeError(f"{arm} did not decode with the full imported prefix on {case['qid']}")
        if prepared["snapshot_bytes"] != transfer["transfer_bytes"]:
            raise RuntimeError("received and imported KV sizes differ")
        timings = paired_request_timings(
            prefill_ns, arm_start, converted, received, receipt["kv_load_start_ns"],
            receipt["kv_load_end_ns"], first_ns, stop_ns, returned,
        )
        yield {
            "qid": case["qid"], "arm": arm, "prediction": stop.prediction,
            "stop_reason": stop.reason, "token_ids": list(stop.observed),
            "prompt_tokens": len(prompt_ids), "source_cached_tokens": cached,
            "loaded_tokens": receipt["loaded_tokens"], "loaded_layers": receipt["loaded_layers"],
            "kv_load_gpu_ms": receipt["kv_load_gpu_ms"], "smoke_payload_sha256": exported["digest"],
            **transfer, **timings,
        }


def run(args):
    from importlib.metadata import version

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    from scd_raw_apc_source import APC_SOURCE_CONNECTOR_NAME
    from scd_runtime.contracts.types import profile_from_dict, to_data
    from scd_runtime.providers.hf.loading import verify_model
    from scd_runtime.resident_runner import GPUResidentConsumer

    class TargetConsumer(GPUResidentConsumer):
        _worker_extension_cls = "scd_raw_apc_workers.TargetWorker"

    parser_digest = sha256(args.legacy_cmrc_parser.read_bytes()).hexdigest()
    if parser_digest != DEV26_PARSER_SHA256:
        raise ValueError("expected the frozen dev26 answer parser")
    parser = runpy.run_path(str(args.legacy_cmrc_parser))
    cases = load_verified_cases(
        args.cases_root.resolve(), args.qid, args.shard_index, args.num_shards, keep_contexts=True,
    )
    profile = profile_from_dict(read_json(args.profile))
    verify_model(args.source_root, args.source_manifest, profile.source)
    tokenizer = AutoTokenizer.from_pretrained(str(args.target_root), local_files_only=True, trust_remote_code=False)
    if args.retokenize:
        source_tokenizer = AutoTokenizer.from_pretrained(
            str(args.source_root), local_files_only=True, trust_remote_code=False,
        )
        retokenize_cases(cases, source_tokenizer, tokenizer)
    arms = (args.only_arm,) if args.only_arm else ARMS
    common = {
        "block_size": 16, "attention_config": {"backend": "FLASH_ATTN"},
        "max_model_len": MAX_MODEL_LEN, "max_num_batched_tokens": MAX_MODEL_LEN, "max_num_seqs": 1,
        "gpu_memory_utilization": 0.43, "kv_cache_memory_bytes": 2 * 1024**3,
        "seed": 20261023, "enforce_eager": True, "async_scheduling": False,
    }
    cfg = {
        **common, "model": str(args.target_root), "model_manifest": str(args.target_manifest),
        "enable_prefix_caching": False, "request_reader_routing": False, "async_loading": False,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    target = TargetConsumer(profile, args.assets_root or args.profile.parent, cfg)
    source = None
    rows = []
    try:
        source = LLM(
            **common, model=str(args.source_root), enable_prefix_caching=True, dtype="bfloat16",
            tensor_parallel_size=1, pipeline_parallel_size=1, disable_hybrid_kv_cache_manager=True,
            enable_chunked_prefill=False, worker_extension_cls="scd_raw_apc_workers.SourceWorker",
            kv_transfer_config={
                "kv_connector": APC_SOURCE_CONNECTOR_NAME, "kv_connector_module_path": "scd_raw_apc_source",
                "kv_role": "kv_producer",
            },
        )
        engine = source.llm_engine
        if not engine.vllm_config.cache_config.enable_prefix_caching or not engine.reset_prefix_cache():
            raise RuntimeError("MistralLite source did not start with native APC")
        patch_layers = sorted({profile.layer_map[i] for i in profile.route["patch_layers"]})
        _source_rpc(source, "scd_apc_source_initialize", patch_layers=patch_layers)
        _source_rpc(source, "cmrc_conversion_setup", profile_data=to_data(profile),
                    assets_root=None if arms == ("raw",) else str(args.assets_root))
        sampling = SamplingParams(max_tokens=1, ignore_eos=True, temperature=0.0)
        for case in cases[:3]:
            list(evaluate_case(source, target, case, profile, tokenizer, parser, sampling, arms,
                               verify_bits=bool(args.qid)))
        if not engine.reset_prefix_cache():
            raise RuntimeError("failed to clear the source APC before measurement")
        with (args.out / "attempts.jsonl").open("x", encoding="utf-8", buffering=1) as log:
            for i, case in enumerate(cases, 1):
                arm_order = arms if (i + args.shard_index) % 2 else arms[::-1]
                for row in evaluate_case(source, target, case, profile, tokenizer, parser, sampling, arm_order,
                                         verify_bits=bool(args.qid)):
                    log.write(json.dumps(row, ensure_ascii=False) + "\n")
                    rows.append(row)
                if i == 1 or i % 100 == 0 or i == len(cases):
                    print(f"METRIC paired_cmrc_cases_done={i} paired_cmrc_cases_total={len(cases)}", flush=True)
    finally:
        try:
            if source is not None:
                source.llm_engine.engine_core.shutdown(timeout=5)
        finally:
            target.close()

    pairs = validate_paired_rows(rows, {case["qid"] for case in cases}, arms=arms)
    if not any(pair[arms[0]]["source_cached_tokens"] for pair in pairs.values()) and (not args.qid or len(cases) > 1):
        raise RuntimeError("no native source APC hits on the measured cases")
    for arm in arms:
        write_json(args.out / f"{arm}_predictions.json", {qid: pair[arm]["prediction"] for qid, pair in pairs.items()})
    write_json(args.out / "summary.json", {
        "status": "completed", "pairs": len(pairs), "arms": list(arms), "shard_index": args.shard_index,
        "num_shards": args.num_shards, "smoke_qids": args.qid, "dataset_sha256": CMRC_DEV_SHA256,
        "profile_sha256": sha256(args.profile.read_bytes()).hexdigest(), "parser_sha256": parser_digest,
        "vllm": version("vllm"), "source_apc": True, "target_apc": False,
        "source_model": args.source_root.name, "target_model": args.target_root.name,
        "geometry": [profile.target.num_layers, profile.target.num_kv_heads, profile.target.head_dim],
        "retokenize": args.retokenize,
        "transport": "cross_process_same_gpu_cuda_ipc_explicit_d2d_copy",
        "elapsed_seconds": time.perf_counter() - started,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("profile", "source-root", "source-manifest", "target-root", "target-manifest",
                 "cases-root", "legacy-cmrc-parser", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--assets-root", type=Path)
    parser.add_argument("--retokenize", action="store_true")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--raw-only", action="store_const", dest="only_arm", const="raw")
    selection.add_argument("--scd-only", action="store_const", dest="only_arm", const="scd")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--qid", action="append")
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.num_shards <= DATASET_SIZE or (args.qid and args.num_shards != 1):
        parser.error("use a valid shard index or select smoke QIDs without sharding")
    if args.only_arm != "raw" and args.assets_root is None:
        parser.error("SCD evaluation requires --assets-root")
    run(args)


if __name__ == "__main__":
    main()
