"""Offline-friendly CLI for SCD-ICML runtime. Heavy tensor libraries load lazily."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path

from .contracts.identity import canonical_bytes, profile_fingerprint, protocol_fingerprint
from .contracts.types import GenerationResult, PrefixContract, profile_from_dict, to_data
from .errors import SCDRuntimeError, ValidationIncomplete


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_new(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(canonical_bytes(data) + b'\n')


def parser():
    p = argparse.ArgumentParser(description='SCD-ICML vLLM runtime CLI')
    commands = p.add_subparsers(dest='command', required=True)

    doctor = commands.add_parser('doctor', help='Report profile/asset readiness without loading models')
    doctor.add_argument('--profile', required=True)
    doctor.add_argument('--assets')

    imp = commands.add_parser('import-assets', help='Convert trusted legacy SCD-ICML banks to safetensors')
    imp.add_argument('--profile', required=True)
    imp.add_argument('--input', action='append', default=[], metavar='ROLE=PATH')
    imp.add_argument('--out', required=True)
    imp.add_argument('--trusted-legacy', action='store_true')

    for name in ('prepare', 'infer'):
        q = commands.add_parser(name)
        for option in ('profile', 'assets', 'prefix-ids', 'store', 'out'):
            q.add_argument('--' + option, required=True)
        q.add_argument('--capacity-mib', type=int, default=1024)
        if name == 'infer':
            q.add_argument('--sample-id', help='ID from the frozen validation membership manifest')
            q.add_argument('--prompt-ids', required=True)
            q.add_argument('--engine-config', required=True)
            q.add_argument('--max-new-tokens', type=int, default=64)

    val = commands.add_parser('validate', help='Evaluate recorded evidence; incomplete results return exit 2')
    val.add_argument('--profile', required=True)
    val.add_argument('--receipts', required=True)
    val.add_argument('--out', required=True)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    profile = None
    try:
        if args.command == 'doctor':
            issues = []
            try:
                profile = profile_from_dict(read_json(args.profile))
            except (ValueError, TypeError, KeyError, OSError) as exc:
                issues.append(f'profile binding: {exc}')
            if not args.assets:
                issues.append('asset root not provided')
            elif not (Path(args.assets) / 'manifest.json').is_file():
                issues.append('asset manifest missing')
            versions = {}
            for package in ('torch', 'transformers', 'vllm', 'safetensors'):
                try:
                    versions[package] = importlib.metadata.version(package)
                except importlib.metadata.PackageNotFoundError:
                    versions[package] = None
            issues.append('real-model and vLLM GPU qualification must be supplied separately')
            print(
                json.dumps(
                    {
                        'ready': False,
                        'profile_valid': profile is not None,
                        'missing_or_invalid': issues,
                        'observed_packages': versions,
                        'loads_models': False,
                    },
                    ensure_ascii=False,
                )
            )
            return 2

        profile = profile_from_dict(read_json(args.profile))
        if args.command == 'import-assets':
            from .assets.legacy_import import import_legacy

            inputs = {'profile': Path(args.profile)}
            for item in args.input:
                role, value = item.split('=', 1)
                if role in inputs:
                    raise ValueError('duplicate input role')
                inputs[role] = Path(value)
            path = import_legacy(profile.profile_id, inputs, Path(args.out), trusted_legacy=args.trusted_legacy)
            print(json.dumps({'output': str(path), 'runtime_verified': False}))
            return 0

        if args.command == 'validate':
            from .benchmarks.validate import validate_profile
            from .runner import RunReceipt

            if profile.validation_protocol is None:
                raise ValidationIncomplete('profile has no frozen validation protocol')
            receipts = []
            for path in sorted(Path(args.receipts).rglob('metrics.json')):
                data = read_json(path)
                gen = data['generation']
                receipts.append(
                    RunReceipt(
                        data['profile_fingerprint'],
                        data['artifact_id'],
                        data['request_digest'],
                        data['mode'],
                        GenerationResult(tuple(gen['token_ids']), None, gen.get('metadata', {})),
                        data['metrics'],
                        data['status'],
                    )
                )
            report = validate_profile(profile, receipts, profile.validation_protocol)
            write_new(Path(args.out) / 'summary.json', to_data(report))
            print(json.dumps(to_data(report), ensure_ascii=False))
            return 0 if report.complete else 2

        from .assets.manifest import load_assets
        from .runner import VLLMConsumer, prepare, run_request
        from .transport.store import ArtifactStore

        assets = load_assets(args.assets, profile)
        store = ArtifactStore(args.store, args.capacity_mib * 1024 * 1024)
        prefix = PrefixContract(
            tuple(read_json(args.prefix_ids)),
            0,
            profile.source.tokenizer_sha256,
            profile.source.chat_template_sha256,
        )
        if args.command == 'prepare':
            prepared = prepare(profile, prefix, assets, store)
            write_new(Path(args.out) / 'prepared.json', to_data(prepared))
            print(json.dumps(to_data(prepared)))
            return 0

        consumer = VLLMConsumer(store, read_json(args.engine_config), profile)
        try:
            receipt = run_request(
                profile,
                prefix,
                read_json(args.prompt_ids),
                assets,
                store,
                consumer,
                {'max_new_tokens': args.max_new_tokens, 'temperature': 0.0},
                mode='E',
                sample_id=args.sample_id,
            )
            write_new(Path(args.out) / 'metrics.json', to_data(receipt))
            print(json.dumps(to_data(receipt)))
            return 0
        finally:
            consumer.close()
    except (SCDRuntimeError, ValueError, KeyError, OSError, ImportError, RuntimeError) as exc:
        failure = {'status': 'failed', 'error_type': type(exc).__name__, 'error': str(exc), 'quality_established': False}
        if getattr(args, 'out', None):
            try:
                if args.command == 'infer' and profile is not None:
                    failure['sample_id'] = args.sample_id
                    if profile.validation_protocol is not None:
                        failure['validation_protocol_fingerprint'] = protocol_fingerprint(profile.validation_protocol)
                    failure = {
                        'profile_fingerprint': profile_fingerprint(profile),
                        'artifact_id': '',
                        'request_digest': '',
                        'mode': 'E',
                        'generation': {'token_ids': [], 'metadata': {}},
                        'metrics': failure,
                        'status': 'failed',
                    }
                    write_new(Path(args.out) / 'metrics.json', failure)
                else:
                    write_new(Path(args.out) / 'failure.json', failure)
            except FileExistsError:
                pass
        print(json.dumps(failure, ensure_ascii=False))
        return 2
