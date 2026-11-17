"""Frozen target attention parameters, without a target prefill interface."""
import json
from pathlib import Path
import torch
from ...errors import MissingAsset, IdentityMismatch


class TargetProjectionProvider:
    def __init__(self, identity, tensors):
        self.identity = identity
        self.tensors = tensors
        self._compiled_device = None
        self._compiled_layers = {}

    @classmethod
    def from_assets(cls, assets):
        if assets.metadata.get('projection_model_sha256') != assets.profile.target.manifest_sha256:
            raise IdentityMismatch('projection model binding missing or different')
        return cls(assets.profile.target, assets.tensors)

    def prepare_device(self, device, layers):
        """Pre-bind BF16 projection tensors on the target device once at setup."""
        dev = torch.device(device)
        if self._compiled_device == dev and set(layers).issubset(self._compiled_layers):
            return
        cache = {}
        for layer in layers:
            prefix = f'projection.{layer}.'
            try:
                wk = self.tensors[prefix + 'k_weight'].to(device=dev, dtype=torch.bfloat16)
                wv = self.tensors[prefix + 'v_weight'].to(device=dev, dtype=torch.bfloat16)
            except KeyError as exc:
                raise MissingAsset(f'projection weights missing for layer {layer}') from exc
            bk = self.tensors.get(prefix + 'k_bias')
            bv = self.tensors.get(prefix + 'v_bias')
            bk = None if bk is None else bk.to(device=dev, dtype=torch.bfloat16)
            bv = None if bv is None else bv.to(device=dev, dtype=torch.bfloat16)
            norm = self.tensors.get(prefix + 'k_norm')
            eps = None
            if norm is not None:
                raw_eps = self.tensors.get(prefix + 'norm_eps')
                if raw_eps is None:
                    raise MissingAsset('K norm epsilon missing')
                norm = norm.to(device=dev, dtype=torch.float32)
                eps = float(raw_eps)
            cache[layer] = (wk, bk, wv, bv, norm, eps)
        self._compiled_device = dev
        self._compiled_layers = cache

    def project(self, layer, hidden):
        h, d = self.identity.num_kv_heads, self.identity.head_dim
        x = hidden if hidden.dtype == torch.bfloat16 else hidden.to(dtype=torch.bfloat16)
        if self._compiled_device == x.device and layer in self._compiled_layers:
            wk, bk, wv, bv, norm, eps = self._compiled_layers[layer]
            k = torch.nn.functional.linear(x, wk, bk)
            v = torch.nn.functional.linear(x, wv, bv)
            k = k.reshape(1, -1, h, d).transpose(1, 2)
            v = v.reshape(1, -1, h, d).transpose(1, 2)
            if norm is not None:
                kn = k.float()
                kn = kn * torch.rsqrt(kn.square().mean(-1, keepdim=True) + eps)
                k = (kn * norm).to(x.dtype)
            return k, v

        prefix = f'projection.{layer}.'
        try:
            wk = self.tensors[prefix + 'k_weight']
            wv = self.tensors[prefix + 'v_weight']
        except KeyError as exc:
            raise MissingAsset(f'projection weights missing for layer {layer}') from exc
        bk = self.tensors.get(prefix + 'k_bias')
        bv = self.tensors.get(prefix + 'v_bias')
        k = torch.nn.functional.linear(x, wk.to(device=x.device, dtype=x.dtype), None if bk is None else bk.to(device=x.device, dtype=x.dtype))
        v = torch.nn.functional.linear(x, wv.to(device=x.device, dtype=x.dtype), None if bv is None else bv.to(device=x.device, dtype=x.dtype))
        k = k.reshape(1, -1, h, d).transpose(1, 2)
        v = v.reshape(1, -1, h, d).transpose(1, 2)
        norm = self.tensors.get(prefix + 'k_norm')
        if norm is not None:
            eps = self.tensors.get(prefix + 'norm_eps')
            if eps is None:
                raise MissingAsset('K norm epsilon missing')
            kn = k.float()
            kn = kn * torch.rsqrt(kn.square().mean(-1, keepdim=True) + float(eps))
            k = (kn * norm.to(kn.device)).to(x.dtype)
        return k, v


def extract_projection_tensors(root, layers):
    """Called only after verify_model; reads selected tensors, not a target forward."""
    from safetensors import safe_open
    from ...assets.manifest import safe_path
    root = Path(root)
    config = json.loads((root / 'config.json').read_text(encoding='utf-8'))
    index = root / 'model.safetensors.index.json'
    if not index.is_file():
        index = root / 'pytorch_model.bin.index.json'
    bin_cache = {}

    def load_bin(path):
        if str(path) not in bin_cache:
            bin_cache.clear()
            bin_cache[str(path)] = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
        return bin_cache[str(path)]

    if index.is_file():
        mapping = json.loads(index.read_text(encoding='utf-8'))['weight_map']
    else:
        single = root / 'model.safetensors'
        if single.is_file():
            with safe_open(str(single), framework='pt', device='cpu') as f:
                mapping = {k: single.name for k in f.keys()}
        else:
            single = root / 'pytorch_model.bin'
            if not single.is_file():
                raise MissingAsset('target model weights/index required')
            mapping = {k: single.name for k in load_bin(single)}

    result = {}
    for layer in layers:
        prefix = f'model.layers.{layer}.self_attn.'
        keys = [
            ('k_proj.weight', 'k_weight', True),
            ('v_proj.weight', 'v_weight', True),
            ('k_proj.bias', 'k_bias', False),
            ('v_proj.bias', 'v_bias', False),
            ('k_norm.weight', 'k_norm', False),
        ]
        for old, new, required in keys:
            key = prefix + old
            if key not in mapping:
                if required:
                    raise MissingAsset(key)
                continue
            shard = safe_path(root, mapping[key])
            if shard.suffix == '.safetensors':
                with safe_open(str(shard), framework='pt', device='cpu') as f:
                    result[f'projection.{layer}.{new}'] = f.get_tensor(key).clone()
            else:
                result[f'projection.{layer}.{new}'] = load_bin(shard)[key].clone()
            if new == 'k_norm':
                if 'rms_norm_eps' not in config:
                    raise MissingAsset('norm epsilon unavailable')
                result[f'projection.{layer}.norm_eps'] = torch.tensor(config['rms_norm_eps'])
    return result
