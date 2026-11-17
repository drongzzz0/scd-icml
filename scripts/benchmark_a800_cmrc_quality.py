"""Replay the paired CMRC dev26 evaluation with the repaired answer stop rule."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import runpy
import time

from cmrc_answer_stop import CMRCAnswerStop, measure_answer_request


CMRC_DEV_SHA256 = "5cfe4414c28a8ecbb51670f78c0dc7d1049f286c2d5769b52f1f94bcc0752cf1"
DEV26_PARSER_SHA256 = "450a9a05618a0288de91397976c070f158d9475a89d84996e066a155d541e49c"
MAX_NEW_TOKENS = 32
DATASET_SIZE = 3219
LATENCY_SCOPE = "outer_request_to_target_submit_first_token_stop_decision_and_return"


def read_json(path):
    return json.loads(path.read_bytes())


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_verified_cases(root, qids, shard_index, num_shards, *, keep_contexts=False):
    prepared = read_json(root / "prepared.json")
    members = prepared["members"]
    all_ids = {member["qid"] for member in members}
    if prepared["dataset_sha256"] != CMRC_DEV_SHA256 or len(members) != len(all_ids) or len(all_ids) != DATASET_SIZE:
        raise ValueError("expected the original complete 3,219-question CMRC dev split")
    if qids:
        if set(qids) - all_ids:
            raise ValueError("requested smoke-test QID does not occur in CMRC dev")
        selected = [member for member in members if member["qid"] in qids]
    elif keep_contexts:
        contexts = {}
        for member in members:
            context = member["qid"].rsplit("_QUERY_", 1)[0]
            contexts.setdefault(context, []).append(member)
        selected = [member for group in list(contexts.values())[shard_index::num_shards] for member in group]
    else:
        selected = members[shard_index::num_shards]
    cases = []
    for member in selected:
        path = (root / member["path"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError("case path is outside the prepared input directory")
        raw = path.read_bytes()
        if sha256(raw).hexdigest() != member["sha256"]:
            raise ValueError(f"prepared case has changed: {member['qid']}")
        case = json.loads(raw)
        if case["qid"] != member["qid"]:
            raise ValueError(f"prepared case ID differs: {member['qid']}")
        cases.append(case)
    return cases


def evaluate_case(backend, case, arm, profile, tokenizer, parser, *, native_apc):
    from scd_runtime.contracts.types import PrefixContract

    stop_ids = set(case["generation"]["stop_token_ids"])
    stop_ids.update(x for x in (tokenizer.eos_token_id, tokenizer.pad_token_id) if x is not None)
    settings = {
        "max_new_tokens": MAX_NEW_TOKENS,
        "temperature": 0.0,
        "eos_token_id": sorted(stop_ids),
        "ignore_eos": False,
    }
    observer = CMRCAnswerStop(
        tokenizer, parser["extract_answer_text"], case["context"],
        parser["ANSWER_CUTOFF_SEPARATORS"], stop_ids, MAX_NEW_TOKENS,
    )
    is_apc = arm == "native" and native_apc
    if arm == "scd":
        prefix = PrefixContract(
            tuple(case["prefix_ids"]), 0, profile.source.tokenizer_sha256, profile.source.chat_template_sha256
        )
    if not is_apc:
        backend.synchronize()
    cached_tokens = 0
    scd_stage_ms = None
    if is_apc:
        (actual_ids, cached_tokens), timing = measure_answer_request(
            backend.llm_engine, lambda observe: backend.generate(case["prompt_ids"], settings, observe), observer,
        )
    elif arm == "native":
        native, timing = measure_answer_request(
            backend.engine.llm_engine,
            lambda observe: backend.native(case["prompt_ids"], settings, token_observer=observe), observer,
        )
        actual_ids = native["token_ids"]
    else:
        scd, timing = measure_answer_request(
            backend.engine.llm_engine,
            lambda observe: backend.run(
                prefix, case["prompt_ids"], settings, mode="E", import_policy="fullprefix",
                full_prompt_ids=case["source_input_ids"], token_observer=observe,
            ), observer,
        )
        receipt = scd.generation.metadata["injection_receipt"]
        if (scd.status != "success" or receipt["loaded_tokens"] != len(case["prefix_ids"])
                or receipt["loaded_layers"] != profile.target.num_layers):
            raise RuntimeError(f"SCD did not admit the complete prefix on {case['qid']}")
        actual_ids = scd.generation.token_ids
        scd_stage_ms = {
            key: scd.metrics[key] for key in ("source_prefill_ms", "convert_ms", "publish_ms", "prepare_rpc_ms", "admission_rpc_ms")
        }
    if tuple(actual_ids) != observer.observed or observer.reason is None:
        raise RuntimeError(f"{arm} did not finish the observed generation on {case['qid']}")
    if not is_apc:
        state = backend._rpc("scd_gpu_clear", session_id=backend.session_id)
        if state["entries"] or state["source_states"] or state["poisoned"]:
            raise RuntimeError(f"resident state did not clear after {arm} on {case['qid']}")
    return {
        "qid": case["qid"], "arm": arm, "prediction": observer.prediction,
        "stop_reason": observer.reason, "token_ids": list(observer.observed),
        "leading_newline": tokenizer.decode(observer.observed[:1], skip_special_tokens=True).startswith(("\n", "\r")),
        **timing, "prompt_tokens": len(case["prompt_ids"]),
        "native_cached_tokens": cached_tokens if arm == "native" else None,
        "scd_stage_ms": scd_stage_ms,
    }


def run_quality(args):
    from importlib.metadata import version

    from transformers import AutoTokenizer

    from scd_runtime.contracts.types import profile_from_dict
    from scd_runtime.resident_runner import GPUResidentConsumer

    parser_digest = sha256(args.legacy_cmrc_parser.read_bytes()).hexdigest()
    if parser_digest != DEV26_PARSER_SHA256:
        raise ValueError("expected the frozen dev26 answer parser")
    parser = runpy.run_path(str(args.legacy_cmrc_parser))
    cases = load_verified_cases(
        args.cases_root.resolve(), args.qid, args.shard_index, args.num_shards, keep_contexts=args.native_apc
    )
    runtime_version, vllm_version = version("scd-unified-runtime"), version("vllm")
    profile = profile_from_dict(read_json(args.profile))
    tokenizer = AutoTokenizer.from_pretrained(str(args.target_root), local_files_only=True, trust_remote_code=False)
    engine_config = {
        "model": str(args.target_root),
        "model_manifest": str(args.target_manifest),
        "block_size": 16,
        "attention_config": {"backend": "FLASH_ATTN"},
        "max_model_len": 1312,
        "max_num_batched_tokens": 1312,
        "max_num_seqs": 1,
        "gpu_memory_utilization": 0.43,
        "kv_cache_memory_bytes": 2 * 1024**3,
        "seed": 20261023,
        "enable_prefix_caching": False,
        "enforce_eager": True,
        "request_reader_routing": False,
        "async_loading": False,
        "async_scheduling": False,
    }
    arms = ("native", "scd") if args.arm is None else (args.arm,)
    predictions = {arm: {} for arm in arms}
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with (args.out / "attempts.jsonl").open("x", encoding="utf-8", buffering=1) as rows:
        def replay(backend, batch_arms):
            if args.native_apc:
                for case in cases[:3]:
                    for arm in batch_arms:
                        evaluate_case(backend, case, arm, profile, tokenizer, parser, native_apc=True)
                if batch_arms == ("native",):
                    backend.reset_prefix_cache()
            for i, case in enumerate(cases, 1):
                for arm in batch_arms:
                    row = evaluate_case(
                        backend, case, arm, profile, tokenizer, parser, native_apc=args.native_apc,
                    )
                    predictions[arm][case["qid"]] = row["prediction"]
                    rows.write(json.dumps(row, ensure_ascii=False) + "\n")
                if i == 1 or i % 50 == 0 or i == len(cases):
                    metric = "quality_cases" if args.native_apc else "quality_pairs"
                    print(f"METRIC {metric}_done={i} {metric}_total={len(cases)} arm={'+'.join(batch_arms)}", flush=True)

        sessions = [(arm,) for arm in arms] if args.native_apc else [arms]
        if args.native_apc and args.shard_index % 2:
            sessions.reverse()
        for session_arms in sessions:
            if args.native_apc and session_arms == ("native",):
                from cmrc_native_apc import NativeAPC

                backend = NativeAPC(engine_config)
            else:
                backend = GPUResidentConsumer(profile, args.assets_root, engine_config, enable_output_completion=True)
            try:
                replay(backend, session_arms)
            finally:
                backend.close()

    for arm, values in predictions.items():
        write_json(args.out / f"{arm}_predictions.json", values)
    write_json(args.out / "summary.json", {
        "status": "completed",
        "pairs": len(cases),
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "smoke_qids": args.qid,
        "dataset_sha256": CMRC_DEV_SHA256,
        "profile_sha256": sha256(args.profile.read_bytes()).hexdigest(),
        "parser_sha256": parser_digest,
        "runtime": runtime_version,
        "vllm": vllm_version,
        "arms": arms,
        "native_apc": args.native_apc,
        "latency_scope": LATENCY_SCOPE,
        "warmups_per_arm": min(3, len(cases)) if args.native_apc else 0,
        "elapsed_seconds": time.perf_counter() - started,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("profile", "assets-root", "target-root", "target-manifest", "cases-root", "legacy-cmrc-parser", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--qid", action="append", help="Restrict a smoke test to a named prepared question")
    parser.add_argument("--native-apc", action="store_true", help="Use stock native APC and retain cache across questions")
    parser.add_argument("--arm", choices=("native", "scd"), help="Run only one arm of the APC comparison")
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.num_shards <= DATASET_SIZE or (args.qid and args.num_shards != 1):
        parser.error("use a valid shard index or select smoke QIDs without sharding")
    if args.arm and not args.native_apc:
        parser.error("--arm is supported for the APC comparison")
    run_quality(args)


if __name__ == "__main__":
    main()
