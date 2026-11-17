"""Read local model manifests before letting Transformers/vLLM open weights."""
from pathlib import Path
import json
from ...assets.manifest import file_sha256, safe_path
from ...errors import MissingAsset, AssetMismatch
from .capture import HFSourceProvider
from .config_identity import check_config_identity


def verify_model(root, manifest, identity):
    root = Path(root).resolve()
    manifest = Path(manifest).resolve()
    if not manifest.is_file():
        raise MissingAsset(str(manifest))
    if file_sha256(manifest) != identity.manifest_sha256:
        raise AssetMismatch('model manifest hash differs')
    data = json.loads(manifest.read_text(encoding='utf-8'))
    if 'models' in data:
        entries = [e for e in data['models'] if Path(e['path']).resolve() == root]
        if len(entries) != 1:
            raise AssetMismatch('model root not uniquely bound')
        data = entries[0]
    files = data.get('files', [])
    if not files:
        raise MissingAsset('empty model manifest')
    recorded = set()
    for item in files:
        path = safe_path(root, item['path'])
        recorded.add(path)
        if not path.is_file():
            raise MissingAsset(str(path))
        size = item.get('size', item.get('size_bytes'))
        if path.stat().st_size != size or file_sha256(path) != item['sha256']:
            raise AssetMismatch(f'model file differs: {item["path"]}')
    config = root / 'config.json'
    if config not in recorded or file_sha256(config) != identity.config_sha256:
        raise AssetMismatch('model config not bound')
    weights = set(root.glob('*.safetensors')) | set(root.glob('*.bin'))
    if not weights.issubset(recorded):
        raise AssetMismatch('unregistered model weights')
    indices = set(root.glob('*.index.json'))
    for index in indices:
        if index.resolve() not in recorded:
            raise AssetMismatch('unregistered model weight index')
        mapping = json.loads(index.read_text(encoding='utf-8')).get('weight_map')
        if not isinstance(mapping, dict) or not mapping:
            raise AssetMismatch('invalid model weight index')
        for name in mapping.values():
            if not isinstance(name, str):
                raise AssetMismatch('invalid model shard path')
            shard = safe_path(root, name)
            if shard not in recorded or shard.suffix not in ('.safetensors', '.bin'):
                raise AssetMismatch('unregistered indexed model shard')
    if not any((root / name).is_file() for name in ('model.safetensors', 'pytorch_model.bin', 'model.safetensors.index.json', 'pytorch_model.bin.index.json')):
        raise MissingAsset('supported model weights or index required')
    if (root / 'adapter_config.json').exists():
        raise AssetMismatch('implicit model adapter is not bound')
    for name in ('generation_config.json',):
        path = root / name
        if path.exists() and path.resolve() not in recorded:
            raise AssetMismatch(f'unregistered model config: {name}')
    check_config_identity(json.loads(config.read_text(encoding='utf-8')), identity)
    return root


def load_source_provider(assets, *, device=None, attn_implementation: str | None = None, fast_causal_mask: bool = False):
    model = assets.metadata.get('models', {}).get('source')
    if not model:
        raise MissingAsset('models.source root and manifest must be configured')
    root = verify_model(model['root'], model['manifest'], assets.profile.source)
    import torch
    from transformers import AutoModelForCausalLM
    kwargs = {'local_files_only': True, 'trust_remote_code': False, 'torch_dtype': torch.bfloat16}
    if attn_implementation is not None:
        kwargs['attn_implementation'] = attn_implementation
    target_device = assets.metadata.get('device', 'cpu') if device is None else device
    loaded = AutoModelForCausalLM.from_pretrained(str(root), **kwargs).to(target_device).eval()
    return HFSourceProvider(loaded, assets.profile.source, fast_causal_mask=fast_causal_mask)
