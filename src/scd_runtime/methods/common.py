import torch
from ..errors import IdentityMismatch, MissingAsset, UnsupportedProfile


def tensor(assets, key, device, dtype=None):
    if key not in assets.tensors:
        raise MissingAsset(f'missing tensor {key}')
    value = assets.tensors[key]
    target_dtype = dtype or value.dtype
    if value.device == device and value.dtype == target_dtype:
        return value
    return value.to(device=device, dtype=target_dtype)


def validate_source(state, profile):
    if state.source != profile.source:
        raise IdentityMismatch('source model differs')
    if (state.prefix.tokenizer_sha256 != profile.source.tokenizer_sha256
            or state.prefix.chat_template_sha256 != profile.source.chat_template_sha256):
        raise IdentityMismatch('source prefix differs')
    if len(state.kv) != profile.source.num_layers:
        raise IdentityMismatch('incomplete source cache')
    shape = (1, profile.source.num_kv_heads, len(state.prefix.token_ids), profile.source.head_dim)
    device = state.kv[0][0].device
    for pair in state.kv:
        if len(pair) != 2 or any(tuple(t.shape) != shape or t.dtype != torch.bfloat16 or t.device != device for t in pair):
            raise IdentityMismatch('invalid BF16 source geometry/device')
    if (profile.source.num_kv_heads, profile.source.head_dim) != (profile.target.num_kv_heads, profile.target.head_dim):
        raise UnsupportedProfile('KV geometry conversion not implemented')
    return device


def rope_tables(identity, n, device, dtype, *, native_cpu_init=False):
    cfg = identity.rope
    if cfg.get('rope_type', 'default') != 'default' or cfg.get('factor', 1) != 1 or cfg.get('attention_factor', 1) != 1:
        raise UnsupportedProfile('only static unit-scaled RoPE is supported')
    d = identity.head_dim
    if d % 2:
        raise UnsupportedProfile('split-half rotary requires even head dim')
    theta = cfg.get('theta', cfg.get('rope_theta'))
    if theta is None or theta <= 0:
        raise UnsupportedProfile('RoPE theta missing')
    if type(native_cpu_init) is not bool:
        raise UnsupportedProfile('explicit native rotary initialization mode required')
    inv = 1.0 / (float(theta) ** (torch.arange(0, d, 2, device='cpu' if native_cpu_init else device, dtype=torch.float32) / d))
    if native_cpu_init:
        inv = inv.to(device=device)
    angles = torch.arange(n, device=device, dtype=torch.float32)[:, None] * inv[None, :]
    emb = torch.cat((angles, angles), -1)[None, :]
    return emb.cos().to(dtype), emb.sin().to(dtype)


def rotate(x, tables, inverse=False):
    cos, sin = tables
    n = x.shape[-2]
    if cos.shape[-2] > n:
        cos = cos[:, :n, :]
        sin = sin[:, :n, :]
    cos = cos.to(x.dtype).unsqueeze(1)
    sin = sin.to(x.dtype).unsqueeze(1)
    half = x.shape[-1] // 2
    y = torch.cat((-x[..., half:], x[..., :half]), -1)
    return x * cos + y * (-sin if inverse else sin)
