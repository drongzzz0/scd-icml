from ..contracts.types import ModelIdentity,ReaderIdentity,PrefixContract,TargetKV,strict_construct,to_data
from ..errors import ArtifactCorrupt

def describe(kv):
    import torch
    if not kv.rope_applied or kv.reader.base_manifest_sha256!=kv.target.manifest_sha256:raise ArtifactCorrupt('cache target/reader or RoPE contract invalid')
    shape=(1,kv.target.num_kv_heads,len(kv.prefix.token_ids),kv.target.head_dim)
    if len(kv.layers)!=kv.target.num_layers:raise ArtifactCorrupt('cache layer count differs')
    tensors={}
    for i,pair in enumerate(kv.layers):
        if len(pair)!=2:raise ArtifactCorrupt('cache pair incomplete')
        for kind,t in zip(('k','v'),pair):
            if tuple(t.shape)!=shape or t.dtype!=torch.bfloat16 or not bool(torch.isfinite(t).all()):raise ArtifactCorrupt('cache shape/dtype/finite check failed')
            # One owning, contiguous snapshot even for CPU inputs. A GPU->CPU
            # copy already owns storage, so it needs no second full-size clone.
            tensors[f'{i}.{kind}']=t.detach().to(device='cpu',copy=True,memory_format=torch.contiguous_format)
    metadata={'schema_version':1,'kind':'reconstructed_target_kv','target':to_data(kv.target),'reader':to_data(kv.reader),'prefix':to_data(kv.prefix),'method_fingerprint':kv.method_fingerprint,'rope_applied':True}
    return metadata,tensors

def restore(data,tensors):
    expected={'schema_version','kind','target','reader','prefix','method_fingerprint','rope_applied','payload_sha256','payload_bytes'}
    if set(data)!=expected or data['schema_version']!=1 or data['kind']!='reconstructed_target_kv':raise ArtifactCorrupt('unknown transport manifest')
    try:
        target=strict_construct(ModelIdentity,data['target']);reader=strict_construct(ReaderIdentity,data['reader']);prefix=strict_construct(PrefixContract,data['prefix'])
        if set(tensors)!={f'{i}.{k}' for i in range(target.num_layers) for k in ('k','v')}:raise ValueError('unexpected tensor keys')
        kv=TargetKV(target,reader,prefix,tuple((tensors[f'{i}.k'],tensors[f'{i}.v']) for i in range(target.num_layers)),data['method_fingerprint'],data['rope_applied'])
        # Validation does not rewrite the loaded payload.
        import torch
        shape=(1,target.num_kv_heads,len(prefix.token_ids),target.head_dim)
        if any(tuple(t.shape)!=shape or t.dtype!=torch.bfloat16 for pair in kv.layers for t in pair):raise ValueError('cache geometry differs')
        if not kv.rope_applied or reader.base_manifest_sha256!=target.manifest_sha256:raise ValueError('reader/RoPE mismatch')
        return kv
    except (KeyError,ValueError,TypeError) as exc:raise ArtifactCorrupt('invalid cache metadata') from exc
