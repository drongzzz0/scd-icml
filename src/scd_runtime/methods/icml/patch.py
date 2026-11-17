import torch
from ..common import tensor
from ...errors import MissingAsset,UnsupportedProfile

def patch_layer(hidden,assets,layer,projections):
    mode=assets.profile.route['fit_mode'];p=f'patch.{layer}.';x=hidden.reshape(-1,hidden.shape[-1]).float()
    if mode=='direct_kv_v1':
        aug=torch.cat((x,torch.ones((x.shape[0],1),device=x.device)),1)
        k=aug@tensor(assets,p+'wk',x.device,x.dtype);v=aug@tensor(assets,p+'wv',x.device,x.dtype)
        h,d=assets.profile.target.num_kv_heads,assets.profile.target.head_dim
        return k.reshape(1,-1,h,d).transpose(1,2),v.reshape(1,-1,h,d).transpose(1,2)
    if mode=='xnorm_semantic_v1':
        if projections is None:raise MissingAsset('semantic Patch requires frozen target projections')
        if p+'enc_h' in assets.tensors:x=x@tensor(assets,p+'enc_h',x.device,x.dtype)
        elif not assets.metadata.get('enc_h_is_identity',False):raise MissingAsset('Patch encoder missing')
        y=torch.nn.functional.linear(x,tensor(assets,p+'fc1.weight',x.device,x.dtype),tensor(assets,p+'fc1.bias',x.device,x.dtype))
        y=torch.nn.functional.gelu(y)
        y=torch.nn.functional.linear(y,tensor(assets,p+'fc2.weight',x.device,x.dtype),tensor(assets,p+'fc2.bias',x.device,x.dtype))
        k,v=projections.project(layer,y[None,:]);return k.float(),v.float()
    raise UnsupportedProfile(f'unsupported Patch fit mode: {mode}')
