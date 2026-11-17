import hashlib
import json
from pathlib import Path
from .manifest import safe_path,file_sha256
from ..errors import AssetMismatch,MissingAsset

def verify_snapshot(root,manifest_path):
    entries=json.loads(Path(manifest_path).read_text(encoding='utf-8'))['files']
    for item in entries:
        path=safe_path(root,item['path'])
        if not path.is_file(): raise MissingAsset(item['path'])
        if 'sha256' in item and file_sha256(path)!=item['sha256']: raise AssetMismatch(f'source changed: {item["path"]}')
        if 'git_blob_sha1' in item:
            body=path.read_bytes()
            actual=hashlib.sha1(f'blob {len(body)}\0'.encode()+body).hexdigest()
            if actual!=item['git_blob_sha1']: raise AssetMismatch(f'Git source mismatch: {item["path"]}')
