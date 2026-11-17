from pathlib import Path
import hashlib
import json
from ..contracts.types import MethodAssets,to_data
from ..contracts.identity import sha256
from ..errors import MissingAsset,AssetMismatch

def file_sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()

def safe_path(root,path):
    base=Path(root).resolve()
    raw=Path(path)
    if raw.is_absolute() or '..' in raw.parts: raise AssetMismatch('path must be root-relative')
    target=(base/raw).resolve()
    if not target.is_relative_to(base) or target==base: raise AssetMismatch('asset path escapes root')
    return target

def asset_binding(profile):
    data=to_data(profile)
    data.pop('validation_protocol',None);data.pop('asset_manifest_sha256',None)
    return sha256(data)

def load_assets(root,profile):
    root=Path(root).resolve(); manifest=root/'manifest.json'
    if not manifest.is_file(): raise MissingAsset(f'missing asset manifest: {manifest}')
    if file_sha256(manifest)!=profile.asset_manifest_sha256: raise AssetMismatch('asset manifest checksum mismatch')
    try: data=json.loads(manifest.read_text(encoding='utf-8'))
    except (ValueError,OSError) as exc: raise AssetMismatch('invalid asset manifest') from exc
    if set(data)!={'schema_version','binding','files','metadata'} or data['schema_version']!=1: raise AssetMismatch('unknown asset manifest schema')
    if data['binding']!=asset_binding(profile): raise AssetMismatch('profile, reader or statistics binding differs')
    if not isinstance(data['files'],list) or not data['files']: raise MissingAsset('no weight files registered')
    verified=[];seen=set()
    for item in data['files']:
        if set(item)!={'path','size_bytes','sha256'}: raise AssetMismatch('invalid asset file record')
        path=safe_path(root,item['path'])
        if path in seen: raise AssetMismatch('duplicate asset file')
        seen.add(path)
        if path.suffix!='.safetensors': raise AssetMismatch('runtime accepts safetensors weights only')
        if not path.is_file(): raise MissingAsset(f'missing weight: {item["path"]}')
        if type(item['size_bytes']) is not int or path.stat().st_size!=item['size_bytes'] or file_sha256(path)!=item['sha256']: raise AssetMismatch(f'corrupt weight: {item["path"]}')
        verified.append(path)
    from safetensors.torch import load_file
    tensors={}
    for path in verified:
        loaded=load_file(str(path),device='cpu')
        if tensors.keys() & loaded.keys(): raise AssetMismatch('duplicate tensor name')
        tensors.update(loaded)
    return MethodAssets(profile,root,tensors,data['metadata'])
