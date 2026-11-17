"""Explicit offline conversion of trusted legacy checkpoints; never called by serving."""
from dataclasses import replace
from pathlib import Path
import json
from .manifest import asset_binding, file_sha256
from ..contracts.types import profile_from_dict, to_data
from ..contracts.identity import canonical_bytes
from ..errors import AssetMismatch, MissingAsset


def _flatten(value, prefix, out, torch):
    if isinstance(value, torch.Tensor):
        if prefix in out:
            raise AssetMismatch('duplicate legacy tensor')
        out[prefix] = value.detach().cpu().contiguous().clone()
    elif isinstance(value, dict):
        for k, v in value.items():
            _flatten(v, f'{prefix}.{k}' if prefix else str(k), out, torch)
    elif isinstance(value, (list, tuple)):
        for k, v in enumerate(value):
            _flatten(v, f'{prefix}.{k}', out, torch)


def import_legacy(profile_id, inputs, destination, *, trusted_legacy):
    if not trusted_legacy:
        raise AssetMismatch('legacy import requires explicit trusted_legacy=True')
    if 'profile' not in inputs:
        raise MissingAsset('profile input required')
    destination = Path(destination)
    if destination.exists():
        raise AssetMismatch('destination already exists')
    profile = profile_from_dict(json.loads(Path(inputs['profile']).read_text(encoding='utf-8')))
    if profile.profile_id != profile_id:
        raise AssetMismatch('profile ID differs')
    metadata = json.loads(Path(inputs['metadata']).read_text(encoding='utf-8')) if 'metadata' in inputs else {}

    import torch
    from safetensors.torch import load_file, save_file

    tensors = {}
    raw = {}
    hashes = {}
    for name, filename in inputs.items():
        if name in ('profile', 'metadata'):
            continue
        path = Path(filename)
        if not path.is_file():
            raise MissingAsset(str(path))
        hashes[name] = file_sha256(path)
        expected = metadata.get('expected_input_sha256', {}).get(name)
        if metadata.get('legacy_format') and expected is None:
            raise MissingAsset(f'expected input checksum required: {name}')
        if expected is not None and expected != hashes[name]:
            raise AssetMismatch(f'legacy input checksum differs: {name}')
        if name in ('reuse_meta', 'patch_meta'):
            value = json.loads(path.read_text(encoding='utf-8'))
        else:
            value = load_file(str(path)) if path.suffix == '.safetensors' else torch.load(path, map_location='cpu', weights_only=False)
        raw[name] = value
        if not metadata.get('legacy_format'):
            _flatten(value, name, tensors, torch)

    metadata['input_sha256'] = hashes
    if metadata.get('legacy_format'):
        from .formats import normalize_icml
        if metadata['legacy_format'] != 'icml_banks':
            raise AssetMismatch('unknown legacy format; expected icml_banks')
        tensors, metadata = normalize_icml(raw, profile, metadata)
        from ..providers.hf.loading import verify_model
        from ..providers.hf.projections import extract_projection_tensors
        target = metadata.get('models', {}).get('target')
        if not target:
            raise MissingAsset('target model metadata needed for native Patch projections')
        root = verify_model(target['root'], target['manifest'], profile.target)
        tensors.update(extract_projection_tensors(root, profile.route['patch_layers']))
        metadata['projection_model_sha256'] = profile.target.manifest_sha256

    if not tensors:
        raise MissingAsset('no tensors imported')
    keymap = metadata.pop('tensor_key_map', None)
    if keymap is not None:
        if len(set(keymap.values())) != len(keymap):
            raise AssetMismatch('tensor mapping collision')
        try:
            tensors = {new: tensors[old] for old, new in keymap.items()}
        except KeyError as exc:
            raise MissingAsset(f'legacy tensor not found: {exc}') from exc

    destination.mkdir(parents=True)
    path = destination / 'weights.safetensors'
    save_file(tensors, str(path))
    manifest = {
        'schema_version': 1,
        'binding': asset_binding(profile),
        'metadata': metadata,
        'files': [{'path': path.name, 'size_bytes': path.stat().st_size, 'sha256': file_sha256(path)}],
    }
    (destination / 'manifest.json').write_bytes(canonical_bytes(manifest))
    bound = replace(profile, asset_manifest_sha256=file_sha256(destination / 'manifest.json'))
    (destination / 'profile.json').write_bytes(canonical_bytes(to_data(bound)))
    return destination
