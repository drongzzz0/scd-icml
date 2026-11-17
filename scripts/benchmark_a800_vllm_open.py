#!/usr/bin/env python3
"""Comprehensive A800 GPU verification & benchmark for scd-icml-vllm-open.

Verifies:
1. Real-asset A800 CUDA bitwise parity (64/64 BF16 KV tensors) between reference
   uncompiled ICMLBackend.convert and CompiledICMLPlan._convert_compiled(return_packed=True),
   plus standalone conversion kernel latency.
2. Paired same-engine vLLM V1 latency (TTFT & full completion with real AsyncOutput
   copy_event verification) under both:
   - enforce_eager=True  (apples-to-apples against dev26's 1.012x H-mode baseline)
   - enforce_eager=False (CUDA Graphs enabled - new open-source default)
   across real CMRC cases and long-prefix scaling (512, 1024, 2048, 3584 tokens),
   including H (source-ready), E (fresh source), and Memoized-E (repeated prefix).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
from pathlib import Path
import statistics
import time
from typing import Any
import uuid

import torch


def _median(values: list[float]) -> float:
    return float(statistics.median(values))


def _mean(values: list[float]) -> float:
    return float(statistics.mean(values))


def _make_stop_observer(max_tokens: int, eos_id: int | None):
    def observer(token_ids: tuple[int, ...]) -> str | None:
        if eos_id is not None and token_ids[-1] == eos_id:
            return 'stop_token'
        if len(token_ids) >= max_tokens:
            return 'max_new_tokens'
        return None
    return observer


def run_convert_microbench_and_parity(profile, assets_root: Path, cases: list[dict], device: torch.device) -> dict[str, Any]:
    from scd_runtime.assets.manifest import load_assets
    from scd_runtime.contracts.types import MethodAssets, PrefixContract
    from scd_runtime.methods.icml.backend import ICMLBackend
    from scd_runtime.methods.icml.reuse import reconstruct_layerwise
    from scd_runtime.methods.icml.shared_patch import shared_hidden
    from scd_runtime.methods.common import rope_tables, rotate
    from scd_runtime.providers.hf.loading import load_source_provider
    from scd_runtime.providers.hf.projections import TargetProjectionProvider

    assets = load_assets(assets_root, profile)
    provider = load_source_provider(assets, device=device)
    provider.model.requires_grad_(False)
    backend = ICMLBackend()
    provider.enable_persistent_hooks(backend.required_features(profile).hidden_hooks)

    values = {k: v for k, v in assets.tensors.items() if not k.startswith('lora.')}
    with torch.inference_mode(False), torch.no_grad():
        gpu_assets = MethodAssets(
            profile,
            assets.root,
            {k: v.to(device=device).clone() for k, v in values.items()},
            assets.metadata,
        )
    projections_ref = TargetProjectionProvider.from_assets(gpu_assets)
    projections_fast = TargetProjectionProvider.from_assets(gpu_assets)

    def uncompiled_reference_convert(state):
        p = gpu_assets.profile
        n = len(state.prefix.token_ids)
        reuse = tuple(p.route['reuse_layers'])
        patch = tuple(p.route['patch_layers'])
        native_tables = (
            rope_tables(p.target, n, device, torch.bfloat16, native_cpu_init=True)
            if p.numeric_version == 'icml-pinned-v3-hf-native-rope'
            else None
        )
        batched = reconstruct_layerwise(state, gpu_assets, reuse)
        indices = sorted(patch)
        identity = set(backend.shared_identity_layers(gpu_assets))
        active = [i for i in indices if i not in identity]
        shared = {i: state.hidden[(p.layer_map[i], 'attention_input')].float() for i in indices if i in identity}
        if active:
            inputs = torch.stack([state.hidden[(p.layer_map[i], 'attention_input')][0].float() for i in active])
            predicted = shared_hidden(inputs, torch.tensor(active, device=device, dtype=torch.long), gpu_assets)
            shared.update({i: predicted[j:j + 1] for j, i in enumerate(active)})
        out_layers = []
        for i in range(p.target.num_layers):
            if i in reuse:
                k, v = batched[i]
            else:
                k, v = projections_ref.project(i, shared[i])
                k = rotate(k.to(torch.bfloat16), native_tables if native_tables is not None else rope_tables(p.target, n, device, torch.bfloat16))
            out_layers.append((k.to(torch.bfloat16).contiguous(), v.to(torch.bfloat16).contiguous()))
        return out_layers

    backend.compile_plan(gpu_assets, projections_fast, device=device, max_seq_len=4096)

    parity_rows = []
    for case in cases:
        prefix = PrefixContract(
            tuple(case['prefix_ids']),
            0,
            profile.source.tokenizer_sha256,
            profile.source.chat_template_sha256,
        )
        with torch.inference_mode(), torch.autocast(device_type='cuda', enabled=False):
            state = provider.capture(
                prefix,
                backend.required_features(profile),
                full_prompt_ids=case['source_input_ids'],
            )
            torch.cuda.synchronize(device)

            # Warmup
            _ = uncompiled_reference_convert(state)
            _ = backend.convert(state, gpu_assets, projections_fast, return_packed=True)
            torch.cuda.synchronize(device)

            uncompiled_times = []
            for _ in range(5):
                t0 = time.perf_counter()
                ref_layers = uncompiled_reference_convert(state)
                torch.cuda.synchronize(device)
                uncompiled_times.append((time.perf_counter() - t0) * 1000)

            compiled_times = []
            for _ in range(5):
                t0 = time.perf_counter()
                fast_kv = backend.convert(state, gpu_assets, projections_fast, return_packed=True)
                torch.cuda.synchronize(device)
                compiled_times.append((time.perf_counter() - t0) * 1000)

            exact_tensors = 0
            max_abs_diff = 0.0
            for layer_idx in range(profile.target.num_layers):
                rk, rv = ref_layers[layer_idx]
                fk, fv = fast_kv.layers[layer_idx]
                if torch.equal(rk, fk.contiguous()):
                    exact_tensors += 1
                else:
                    max_abs_diff = max(max_abs_diff, float((rk.float() - fk.float()).abs().max().item()))
                if torch.equal(rv, fv.contiguous()):
                    exact_tensors += 1
                else:
                    max_abs_diff = max(max_abs_diff, float((rv.float() - fv.float()).abs().max().item()))

            parity_rows.append({
                'case_id': case['case_id'],
                'prefix_tokens': len(case['prefix_ids']),
                'exact_bf16_tensors': exact_tensors,
                'total_bf16_tensors': 2 * profile.target.num_layers,
                'all_64_bitwise_equal': exact_tensors == 2 * profile.target.num_layers,
                'max_abs_diff': max_abs_diff,
                'uncompiled_convert_median_ms': _median(uncompiled_times),
                'compiled_packed_convert_median_ms': _median(compiled_times),
                'convert_speedup_x': _median(uncompiled_times) / _median(compiled_times),
            })

    provider.close()
    del provider, gpu_assets, projections_ref, projections_fast
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    return {
        'all_cases_bitwise_equal': all(r['all_64_bitwise_equal'] for r in parity_rows),
        'cases': parity_rows,
    }


def run_engine_suite(
    *,
    label: str,
    profile,
    assets_root: Path,
    target_root: str,
    target_manifest: str,
    cases: list[dict],
    scaling_lengths: list[int],
    enforce_eager: bool,
    memoize_entries: int = 0,
    max_new_tokens: int = 16,
    reps: int = 3,
) -> dict[str, Any]:
    from scd_runtime.contracts.types import PrefixContract
    from scd_runtime.resident_runner import GPUResidentConsumer

    engine_cfg = {
        'model': target_root,
        'model_manifest': target_manifest,
        'block_size': 16,
        'attention_config': {'backend': 'FLASH_ATTN'},
        'max_model_len': 4096,
        'max_num_batched_tokens': 4096,
        'max_num_seqs': 1,
        'gpu_memory_utilization': 0.42,
        'kv_cache_memory_bytes': 2 * 1024**3,
        'seed': 20261023,
        'enable_prefix_caching': False,
        'enforce_eager': enforce_eager,
        'async_loading': False,
        'async_scheduling': False,
        'memoize_prefix_entries': memoize_entries,
    }
    generation = {'max_new_tokens': max_new_tokens, 'temperature': 0.0, 'ignore_eos': True, 'eos_token_id': None}

    t_init = time.perf_counter()
    consumer = GPUResidentConsumer(
        profile,
        assets_root,
        engine_cfg,
        enable_diagnostics=False,
        enable_output_completion=True,
    )
    init_s = time.perf_counter() - t_init

    try:
        # Warmup both native and SCD paths (especially CUDA graph replay!)
        warm_case = cases[0]
        warm_prefix = PrefixContract(
            tuple(warm_case['prefix_ids']),
            0,
            profile.source.tokenizer_sha256,
            profile.source.chat_template_sha256,
        )
        for _ in range(2):
            consumer.native(warm_case['prompt_ids'], generation, token_observer=_make_stop_observer(max_new_tokens, None))
            cap = consumer.capture_source(warm_prefix, full_prompt_ids=warm_case['source_input_ids'])
            consumer.run(
                warm_prefix,
                warm_case['prompt_ids'],
                generation,
                mode='H',
                source_handle=cap['source_handle'],
                full_prompt_ids=warm_case['source_input_ids'],
                import_policy='fullprefix',
                token_observer=_make_stop_observer(max_new_tokens, None),
            )
            consumer.run(
                warm_prefix,
                warm_case['prompt_ids'],
                generation,
                mode='E',
                full_prompt_ids=warm_case['source_input_ids'],
                import_policy='fullprefix',
                token_observer=_make_stop_observer(max_new_tokens, None),
            )
        consumer.synchronize()

        case_results = []
        for case in cases:
            prefix = PrefixContract(
                tuple(case['prefix_ids']),
                0,
                profile.source.tokenizer_sha256,
                profile.source.chat_template_sha256,
            )
            native_ttfts, native_totals = [], []
            h_ttfts, h_totals, h_gpu_converts = [], [], []
            e_ttfts, e_totals, e_gpu_sources = [], [], []
            native_tokens = None
            scd_tokens = None

            for _ in range(reps):
                # ABBA style pairing: Native -> H -> E
                consumer.synchronize()
                nat = consumer.native(
                    case['prompt_ids'],
                    generation,
                    token_observer=_make_stop_observer(max_new_tokens, None),
                )
                native_ttfts.append(nat['first_token_ms'])
                native_totals.append(nat['protocol_completion_ms'])
                native_tokens = nat['token_ids']

                cap = consumer.capture_source(prefix, full_prompt_ids=case['source_input_ids'])
                consumer.synchronize()
                res_h = consumer.run(
                    prefix,
                    case['prompt_ids'],
                    generation,
                    mode='H',
                    source_handle=cap['source_handle'],
                    full_prompt_ids=case['source_input_ids'],
                    import_policy='fullprefix',
                    token_observer=_make_stop_observer(max_new_tokens, None),
                )
                h_ttfts.append(res_h.metrics['pipeline_first_token_ms'])
                h_totals.append(res_h.generation.metadata['protocol_completion_ms'])
                if 'gpu_convert_ms' in res_h.metrics:
                    h_gpu_converts.append(res_h.metrics['gpu_convert_ms'])
                scd_tokens = list(res_h.generation.token_ids)

                consumer.synchronize()
                res_e = consumer.run(
                    prefix,
                    case['prompt_ids'],
                    generation,
                    mode='E',
                    full_prompt_ids=case['source_input_ids'],
                    import_policy='fullprefix',
                    token_observer=_make_stop_observer(max_new_tokens, None),
                )
                e_ttfts.append(res_e.metrics['pipeline_first_token_ms'])
                e_totals.append(res_e.generation.metadata['protocol_completion_ms'])
                if 'gpu_source_ms' in res_e.metrics:
                    e_gpu_sources.append(res_e.metrics['gpu_source_ms'])
                assert list(res_e.generation.token_ids) == scd_tokens, 'H and E token sequences diverged!'

            nat_ttft_med = _median(native_ttfts)
            h_ttft_med = _median(h_ttfts)
            e_ttft_med = _median(e_ttfts)
            nat_tot_med = _median(native_totals)
            h_tot_med = _median(h_totals)
            e_tot_med = _median(e_totals)

            case_results.append({
                'case_id': case['case_id'],
                'prompt_tokens': len(case['prompt_ids']),
                'prefix_tokens': len(case['prefix_ids']),
                'native_ttft_ms': nat_ttft_med,
                'scd_H_ttft_ms': h_ttft_med,
                'scd_E_ttft_ms': e_ttft_med,
                'H_ttft_speedup_x': nat_ttft_med / h_ttft_med,
                'E_ttft_speedup_x': nat_ttft_med / e_ttft_med,
                'native_completion_ms': nat_tot_med,
                'scd_H_completion_ms': h_tot_med,
                'scd_E_completion_ms': e_tot_med,
                'H_completion_speedup_x': nat_tot_med / h_tot_med,
                'E_completion_speedup_x': nat_tot_med / e_tot_med,
                'gpu_convert_median_ms': _median(h_gpu_converts) if h_gpu_converts else None,
                'gpu_source_median_ms': _median(e_gpu_sources) if e_gpu_sources else None,
                'scd_tokens': scd_tokens,
                'native_tokens': native_tokens,
            })

        # Context-length scaling sweep using real token sequences repeated cleanly up to target length
        scaling_results = []
        base_tokens = list(cases[0]['prefix_ids'])
        for length in scaling_lengths:
            repeats = (length // len(base_tokens)) + 2
            synth_prefix_ids = tuple((base_tokens * repeats)[:length])
            synth_prompt_ids = synth_prefix_ids + (cases[0]['prompt_ids'][-1],)
            synth_prefix = PrefixContract(
                synth_prefix_ids,
                0,
                profile.source.tokenizer_sha256,
                profile.source.chat_template_sha256,
            )
            # Warmup once at this length
            consumer.native(synth_prompt_ids, generation, token_observer=_make_stop_observer(max_new_tokens, None))
            cap_w = consumer.capture_source(synth_prefix, full_prompt_ids=synth_prompt_ids)
            consumer.run(
                synth_prefix,
                synth_prompt_ids,
                generation,
                mode='H',
                source_handle=cap_w['source_handle'],
                full_prompt_ids=synth_prompt_ids,
                import_policy='fullprefix',
                token_observer=_make_stop_observer(max_new_tokens, None),
            )

            nat_t, h_t, e_t, nat_c, h_c = [], [], [], [], []
            for _ in range(reps):
                consumer.synchronize()
                n_out = consumer.native(synth_prompt_ids, generation, token_observer=_make_stop_observer(max_new_tokens, None))
                nat_t.append(n_out['first_token_ms'])
                nat_c.append(n_out['protocol_completion_ms'])

                cap_s = consumer.capture_source(synth_prefix, full_prompt_ids=synth_prompt_ids)
                consumer.synchronize()
                h_out = consumer.run(
                    synth_prefix,
                    synth_prompt_ids,
                    generation,
                    mode='H',
                    source_handle=cap_s['source_handle'],
                    full_prompt_ids=synth_prompt_ids,
                    import_policy='fullprefix',
                    token_observer=_make_stop_observer(max_new_tokens, None),
                )
                h_t.append(h_out.metrics['pipeline_first_token_ms'])
                h_c.append(h_out.generation.metadata['protocol_completion_ms'])

                consumer.synchronize()
                e_out = consumer.run(
                    synth_prefix,
                    synth_prompt_ids,
                    generation,
                    mode='E',
                    full_prompt_ids=synth_prompt_ids,
                    import_policy='fullprefix',
                    token_observer=_make_stop_observer(max_new_tokens, None),
                )
                e_t.append(e_out.metrics['pipeline_first_token_ms'])

            scaling_results.append({
                'prefix_tokens': length,
                'native_ttft_ms': _median(nat_t),
                'scd_H_ttft_ms': _median(h_t),
                'scd_E_ttft_ms': _median(e_t),
                'H_ttft_speedup_x': _median(nat_t) / _median(h_t),
                'E_ttft_speedup_x': _median(nat_t) / _median(e_t),
                'native_completion_16tok_ms': _median(nat_c),
                'scd_H_completion_16tok_ms': _median(h_c),
                'H_completion_speedup_x': _median(nat_c) / _median(h_c),
                'memoized': memoize_entries > 0,
            })

        return {
            'label': label,
            'enforce_eager': enforce_eager,
            'memoize_prefix_entries': memoize_entries,
            'engine_init_seconds': init_s,
            'compiled_fast_path': consumer.setup_receipt.get('compiled_fast_path'),
            'cmrc_cases': case_results,
            'cmrc_summary': {
                'mean_native_ttft_ms': _mean([r['native_ttft_ms'] for r in case_results]),
                'mean_scd_H_ttft_ms': _mean([r['scd_H_ttft_ms'] for r in case_results]),
                'mean_scd_E_ttft_ms': _mean([r['scd_E_ttft_ms'] for r in case_results]),
                'median_H_ttft_speedup_x': _median([r['H_ttft_speedup_x'] for r in case_results]),
                'median_H_completion_speedup_x': _median([r['H_completion_speedup_x'] for r in case_results]),
            },
            'scaling': scaling_results,
        }
    finally:
        consumer.close()
        torch.cuda.empty_cache()


def _time_batch(run):
    started = time.perf_counter()
    result = run(started)
    return started, (time.perf_counter() - started) * 1000, result


def _batch_trial(wall_ms, first_token_ms, output_lengths, max_new_tokens, **evidence):
    if any(length != max_new_tokens for length in output_lengths):
        raise RuntimeError(f'fixed-length batch returned unexpected output counts: {output_lengths}')
    if not first_token_ms or any(not 0 <= value <= wall_ms for value in first_token_ms):
        raise RuntimeError('first-token timestamps are outside the timed request')
    return {
        'wall_ms': wall_ms,
        'mean_first_token_ms': _mean(first_token_ms),
        'first_token_ms': first_token_ms,
        'output_tokens_per_request': output_lengths,
        'output_tok_per_s': sum(output_lengths) / (wall_ms / 1000),
        **evidence,
    }


def _native_batch_trial(engine, prompts, max_new_tokens):
    from vllm import SamplingParams
    from vllm.sampling_params import RequestOutputKind

    from scd_runtime.batch import observe_batch
    from scd_runtime.resident_runner import _signal_deadline

    def submit(started):
        requests = [
            (uuid.uuid4().hex, {'prompt_token_ids': list(prompt)}, SamplingParams(
                max_tokens=max_new_tokens,
                temperature=0.0,
                ignore_eos=True,
                output_kind=RequestOutputKind.CUMULATIVE,
            ))
            for prompt in prompts
        ]
        return observe_batch(engine, requests, started=started, deadline_factory=_signal_deadline)

    _, wall_ms, observation = _time_batch(submit)
    outputs = list(observation['requests'].values())
    cached_tokens = [output['num_cached_tokens'] for output in outputs]
    if any(type(count) is not int or not 0 <= count <= len(prompt)
           for count, prompt in zip(cached_tokens, prompts)):
        raise RuntimeError(f'vLLM did not report valid prefix-cache hits: {cached_tokens}')
    return _batch_trial(
        wall_ms,
        [output['first_token_ms'] for output in outputs],
        [len(output['token_ids']) for output in outputs],
        max_new_tokens,
        num_cached_tokens=cached_tokens,
        max_outputs_per_step=observation['max_outputs_per_step'],
    )


def _resident_batch_trial(consumer, requests, generation):
    started, wall_ms, batch = _time_batch(lambda _: consumer.run_batch(requests, generation))
    # first_token_ms is consumer-local, so this also accounts for all work
    # before consumer validation, including source prefill, conversion and admission.
    offset_ms = (batch.metrics['consumer_started_at'] - started) * 1000
    return _batch_trial(
        wall_ms,
        [offset_ms + item.metadata['first_token_ms'] for item in batch.generations],
        [len(item.token_ids) for item in batch.generations],
        generation['max_new_tokens'],
        consumer_start_offset_ms=offset_ms,
        imported_tokens=[item.metadata['imported_tokens'] for item in batch.generations],
        memo_hits=batch.metrics['memo_hits'],
        batch_dedup_hits=batch.metrics['batch_dedup_hits'],
        max_outputs_per_step=batch.metrics['max_outputs_per_step'],
    )


def _run_native_batch_trials(engine_cfg, prompts_by_size, prefix_len, max_new_tokens, reps, *, apc):
    from vllm import LLM

    native_config = {
        key: value for key, value in engine_cfg.items()
        if key not in {'model_manifest', 'async_loading', 'memoize_prefix_entries'}
    }
    native_config.update(
        dtype='bfloat16',
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        enable_prefix_caching=apc,
        enable_chunked_prefill=False,
        disable_hybrid_kv_cache_manager=True,
        trust_remote_code=False,
        disable_log_stats=False,
    )
    llm = LLM(**native_config)
    engine = llm.llm_engine
    cold_by_size, warm_by_size = {}, {}
    try:
        if engine.vllm_config.cache_config.enable_prefix_caching is not apc:
            raise RuntimeError('vLLM did not apply the benchmark APC setting')
        for prompts in prompts_by_size.values():
            _native_batch_trial(engine, prompts, 4)

        for size, prompts in prompts_by_size.items():
            print(f'  Native APC={apc}, batch size {size}, {reps} trials', flush=True)
            cold_trials, warm_trials = [], []
            for _ in range(reps):
                if apc and engine.reset_prefix_cache() is not True:
                    raise RuntimeError('unable to reset the native prefix cache between trials')
                cold = _native_batch_trial(engine, prompts, max_new_tokens)
                cold_trials.append(cold)
                if apc:
                    # The first request must miss after reset. Later requests
                    # may reuse blocks allocated by other requests in the cold batch.
                    if cold['num_cached_tokens'][0] != 0:
                        raise RuntimeError('first native APC request hit after cache reset')
                    warm = _native_batch_trial(engine, prompts, max_new_tokens)
                    if any(count < prefix_len for count in warm['num_cached_tokens']):
                        raise RuntimeError(f'native APC hot trial missed shared prefix: {warm["num_cached_tokens"]}')
                    warm_trials.append(warm)
                elif any(cold['num_cached_tokens']):
                    raise RuntimeError('native APC-off trial reported a prefix-cache hit')
            cold_by_size[size] = cold_trials
            warm_by_size[size] = warm_trials
    finally:
        engine.engine_core.shutdown(timeout=5)
        torch.cuda.empty_cache()
    return cold_by_size, warm_by_size


def _summarize_trials(trials):
    return {
        'median_wall_ms': _median([trial['wall_ms'] for trial in trials]),
        'median_mean_first_token_ms': _median([trial['mean_first_token_ms'] for trial in trials]),
        'median_output_tok_per_s': _median([trial['output_tok_per_s'] for trial in trials]),
        'trials': trials,
    }


def run_batch_throughput_suite(
    *,
    profile,
    assets_root: Path,
    target_root: str,
    target_manifest: str,
    base_case: dict,
    prefix_len: int = 2048,
    batch_sizes: tuple[int, ...] = (1, 4, 8, 16),
    max_new_tokens: int = 32,
    reps: int = 3,
) -> list[dict[str, Any]]:
    from scd_runtime.contracts.types import PrefixContract
    from scd_runtime.resident_runner import GPUResidentConsumer

    max_b = max(batch_sizes)
    engine_cfg = {
        'model': target_root,
        'model_manifest': target_manifest,
        'block_size': 16,
        'attention_config': {'backend': 'FLASH_ATTN'},
        'max_model_len': 4096,
        'max_num_batched_tokens': max(4096 * 4, max_b * (prefix_len + 64)),
        'max_num_seqs': max_b,
        'gpu_memory_utilization': 0.50,
        'kv_cache_memory_bytes': 6 * 1024**3,
        'seed': 20261023,
        'enable_prefix_caching': False,
        'enforce_eager': False,
        'async_loading': False,
        'async_scheduling': False,
        'memoize_prefix_entries': 32,
    }
    if reps < 1 or prefix_len <= 0 or prefix_len % engine_cfg['block_size']:
        raise ValueError('batch benchmark needs at least one repetition and a whole-block prefix')
    generation = {'max_new_tokens': max_new_tokens, 'temperature': 0.0, 'ignore_eos': True, 'eos_token_id': None}
    warm_gen = {**generation, 'max_new_tokens': 4}

    base_tokens = list(base_case['prefix_ids'])
    repeats = (prefix_len // len(base_tokens)) + 2
    shared_prefix_ids = tuple((base_tokens * repeats)[:prefix_len])
    shared_prefix = PrefixContract(
        shared_prefix_ids,
        0,
        profile.source.tokenizer_sha256,
        profile.source.chat_template_sha256,
    )
    prompts_by_size = {
        size: [shared_prefix_ids + (100 + i, 200 + i) for i in range(size)]
        for size in batch_sizes
    }

    native, _ = _run_native_batch_trials(
        engine_cfg, prompts_by_size, prefix_len, max_new_tokens, reps, apc=False,
    )
    apc_cold, apc_warm = _run_native_batch_trials(
        engine_cfg, prompts_by_size, prefix_len, max_new_tokens, reps, apc=True,
    )
    consumer = GPUResidentConsumer(
        profile,
        assets_root,
        engine_cfg,
        enable_diagnostics=False,
        enable_output_completion=False,
    )
    rows = []
    try:
        for prompts in prompts_by_size.values():
            consumer.run_batch([(shared_prefix, prompt) for prompt in prompts], warm_gen)
        consumer.synchronize()

        for size, prompts in prompts_by_size.items():
            print(f'  SCD, batch size {size}, {reps} cold/warm trials', flush=True)
            requests = [(shared_prefix, prompt) for prompt in prompts]
            cold_trials, warm_trials = [], []
            for _ in range(reps):
                consumer.clear_memo()
                consumer.synchronize()
                cold = _resident_batch_trial(consumer, requests, generation)
                warm = _resident_batch_trial(consumer, requests, generation)
                if cold['memo_hits'] or cold['batch_dedup_hits'] != size - 1 or not warm['memo_hits']:
                    raise RuntimeError('observed SCD memoization/dedup does not match cold/warm trial')
                if any(count != prefix_len for trial in (cold, warm) for count in trial['imported_tokens']):
                    raise RuntimeError('SCD did not import the full shared prefix')
                cold_trials.append(cold)
                warm_trials.append(warm)

            arms = {
                'native_no_apc': _summarize_trials(native[size]),
                'native_apc_cold': _summarize_trials(apc_cold[size]),
                'native_apc_warm': _summarize_trials(apc_warm[size]),
                'scd_cold': _summarize_trials(cold_trials),
                'scd_warm': _summarize_trials(warm_trials),
            }
            native_summary = arms['native_no_apc']
            cold_summary = arms['scd_cold']
            warm_summary = arms['scd_warm']
            nat_wall = native_summary['median_wall_ms']
            cold_wall = cold_summary['median_wall_ms']
            warm_wall = warm_summary['median_wall_ms']
            rows.append({
                'batch_size': size,
                'prefix_tokens': prefix_len,
                'max_new_tokens': max_new_tokens,
                'ttft_scope': 'outer_batch_call_to_first_observed_output_per_request',
                'trial_count_per_arm': reps,
                'arms': arms,
                'native_batch_wall_ms': nat_wall,
                'scd_cold_shared_batch_wall_ms': cold_wall,
                'scd_memoized_batch_wall_ms': warm_wall,
                'cold_batch_speedup_x': nat_wall / cold_wall,
                'batch_speedup_x': nat_wall / warm_wall,
                'native_req_per_s': size / (nat_wall / 1000),
                'scd_cold_req_per_s': size / (cold_wall / 1000),
                'scd_req_per_s': size / (warm_wall / 1000),
                'native_output_tok_per_s': native_summary['median_output_tok_per_s'],
                'scd_cold_output_tok_per_s': cold_summary['median_output_tok_per_s'],
                'scd_output_tok_per_s': warm_summary['median_output_tok_per_s'],
                'scd_effective_total_tok_per_s': size * (len(prompts[0]) + max_new_tokens) / (warm_wall / 1000),
                'native_mean_first_token_ms': native_summary['median_mean_first_token_ms'],
                'scd_cold_mean_first_token_ms': cold_summary['median_mean_first_token_ms'],
                'scd_mean_first_token_ms': warm_summary['median_mean_first_token_ms'],
                'cold_batch_dedup_hits': cold_trials[0]['batch_dedup_hits'],
                'warm_memo_hits': warm_trials[0]['memo_hits'],
            })
        return rows
    finally:
        consumer.close()
        torch.cuda.empty_cache()

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--profile', required=True, type=Path)
    parser.add_argument('--assets-root', required=True, type=Path)
    parser.add_argument('--target-root', required=True)
    parser.add_argument('--target-manifest', required=True)
    parser.add_argument('--cases-json', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    stages = parser.add_mutually_exclusive_group()
    stages.add_argument('--resume', action='store_true', help='Reuse completed stages in --out if already verified.')
    stages.add_argument('--batch-only', action='store_true', help='Run only the fresh shared-prefix batch comparison.')
    stages.add_argument('--cudagraph-and-batch', action='store_true', help='Run fresh CUDA-Graph single and batch comparisons.')
    parser.add_argument('--batch-reps', type=int, default=3, help='Measured trials per batch size and arm.')
    parser.add_argument('--batch-sizes', type=int, nargs='+', default=(1, 4, 8, 16))
    args = parser.parse_args()
    if args.batch_reps < 1:
        parser.error('--batch-reps must be positive')
    if len(set(args.batch_sizes)) != len(args.batch_sizes) or any(not 1 <= size <= 16 for size in args.batch_sizes):
        parser.error('--batch-sizes must be distinct integers between 1 and 16')

    from scd_vllm.capability import ensure_sampler_jit_safe
    ensure_sampler_jit_safe()
    from scd_runtime.contracts.types import profile_from_dict
    import vllm

    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    device = torch.device('cuda:0')
    props = torch.cuda.get_device_properties(device)
    profile = profile_from_dict(json.loads(args.profile.read_text(encoding='utf-8')))
    cases = json.loads(args.cases_json.read_text(encoding='utf-8'))

    prior: dict[str, Any] = {}
    if args.resume and args.out.is_file():
        prior = json.loads(args.out.read_text(encoding='utf-8'))

    report: dict[str, Any] = {
        'timestamp_utc': prior.get('timestamp_utc') or time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'completed_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'gpu_name': props.name,
        'gpu_total_memory_mib': int(props.total_memory // (1024**2)),
        'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES', ''),
        'torch_version': torch.__version__,
        'vllm_version': getattr(vllm, '__version__', 'unknown'),
        'flashinfer_sampler_env': os.environ.get('VLLM_USE_FLASHINFER_SAMPLER', 'auto'),
    }

    def save_progress():
        report['completed_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        args.out.parent.mkdir(parents=True, exist_ok=True)
        tmp = args.out.with_suffix('.json.tmp')
        tmp.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
        tmp.replace(args.out)

    if not (args.batch_only or args.cudagraph_and_batch):
        if prior.get('convert_parity_and_microbench', {}).get('all_cases_bitwise_equal') is True:
            print('[1/4] Reusing verified Stage 1 bitwise parity & convert microbenchmark from --out...', flush=True)
            report['convert_parity_and_microbench'] = prior['convert_parity_and_microbench']
        else:
            print('[1/4] Running A800 CUDA bitwise parity & standalone convert microbenchmark...', flush=True)
            report['convert_parity_and_microbench'] = run_convert_microbench_and_parity(
                profile,
                args.assets_root,
                cases,
                device,
            )
            save_progress()

        if len(prior.get('vllm_eager', {}).get('cmrc_cases', ())) == len(cases):
            print('[2/4] Reusing verified Stage 2 vLLM V1 eager benchmark from --out...', flush=True)
            report['vllm_eager'] = prior['vllm_eager']
        else:
            print('[2/4] Running paired vLLM V1 suite (enforce_eager=True, apples-to-apples vs dev26)...', flush=True)
            report['vllm_eager'] = run_engine_suite(
                label='vllm_eager_optimized',
                profile=profile,
                assets_root=args.assets_root,
                target_root=args.target_root,
                target_manifest=args.target_manifest,
                cases=cases,
                scaling_lengths=[512, 1024, 2048, 3584],
                enforce_eager=True,
                memoize_entries=0,
            )
            save_progress()

    if not args.batch_only:
        if len(prior.get('vllm_cudagraph', {}).get('cmrc_cases', ())) == len(cases):
            print('[3/4] Reusing verified Stage 3 vLLM V1 CUDA-Graph benchmark from --out...', flush=True)
            report['vllm_cudagraph'] = prior['vllm_cudagraph']
        else:
            print('[3/4] Running paired vLLM V1 suite (enforce_eager=False, CUDA Graphs enabled)...', flush=True)
            report['vllm_cudagraph'] = run_engine_suite(
                label='vllm_cudagraph_optimized',
                profile=profile,
                assets_root=args.assets_root,
                target_root=args.target_root,
                target_manifest=args.target_manifest,
                cases=cases,
                scaling_lengths=[512, 1024, 2048, 3584],
                enforce_eager=False,
                memoize_entries=0,
            )
            save_progress()

    print('Running shared-prefix batch throughput benchmark with native APC controls...', flush=True)
    report['vllm_batch_throughput'] = run_batch_throughput_suite(
        profile=profile,
        assets_root=args.assets_root,
        target_root=args.target_root,
        target_manifest=args.target_manifest,
        base_case=cases[0],
        prefix_len=2048,
        batch_sizes=tuple(args.batch_sizes),
        max_new_tokens=32,
        reps=args.batch_reps,
    )
    save_progress()
    print('Benchmark completed ->', args.out, flush=True)


if __name__ == '__main__':
    main()
