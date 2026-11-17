"""Shared residual equations from pinned ICML direct_kv_residual_patch.py.

Preserves RMS normalization, SwiGLU blocks and layer-conditioned bottleneck,
with optional pre-bound GPU tensor plan for zero per-request dict/cast overhead.
"""
from dataclasses import dataclass
import torch
from ..common import tensor
from ...errors import MissingAsset, UnsupportedProfile


@dataclass(frozen=True)
class CompiledSharedPatch:
    num_blocks: int
    active_layers: tuple[int, ...]
    active_layer_ids: torch.Tensor
    w_input: torch.Tensor
    b_input: torch.Tensor
    embed_active: torch.Tensor
    blocks: tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], ...]
    w_bottleneck: torch.Tensor
    w_delta_active: torch.Tensor
    b_delta_active: torch.Tensor


def compile_shared_patch(assets, active_layers: tuple[int, ...], device: torch.device) -> CompiledSharedPatch | None:
    if not active_layers:
        return None
    blocks = assets.metadata.get('shared_num_blocks')
    if type(blocks) is not int or blocks < 1:
        raise MissingAsset('shared_num_blocks required')

    def w(name):
        return tensor(assets, 'shared.' + name, device, torch.float32).contiguous()

    layer_ids = torch.tensor(active_layers, device=device, dtype=torch.long)
    block_weights = tuple(
        (w(f'blocks.{i}.gate_proj.weight'), w(f'blocks.{i}.up_proj.weight'), w(f'blocks.{i}.down_proj.weight'))
        for i in range(blocks)
    )
    return CompiledSharedPatch(
        num_blocks=blocks,
        active_layers=active_layers,
        active_layer_ids=layer_ids,
        w_input=w('input_proj.weight'),
        b_input=w('input_proj.bias'),
        embed_active=w('layer_embed.weight').index_select(0, layer_ids).unsqueeze(1).contiguous(),
        blocks=block_weights,
        w_bottleneck=w('bottleneck_proj.weight'),
        w_delta_active=w('delta_h_weight').index_select(0, layer_ids).contiguous(),
        b_delta_active=w('delta_h_bias').index_select(0, layer_ids).unsqueeze(1).contiguous(),
    )


def shared_hidden_compiled(x: torch.Tensor, plan: CompiledSharedPatch) -> torch.Tensor:
    L, n, H = x.shape

    def norm(value):
        return value * value.pow(2).mean(-1, keepdim=True).add(1e-6).rsqrt()

    flat = x.reshape(-1, H).float()
    embed = plan.embed_active.expand(-1, n, -1)
    h = torch.nn.functional.linear(norm(flat), plan.w_input, plan.b_input) + embed.reshape(L * n, -1)
    for w_gate, w_up, w_down in plan.blocks:
        z = norm(h)
        gated = torch.nn.functional.silu(torch.nn.functional.linear(z, w_gate)) * torch.nn.functional.linear(z, w_up)
        h = h + torch.nn.functional.linear(gated, w_down)
    bottleneck = torch.nn.functional.linear(norm(h), plan.w_bottleneck).reshape(L, n, -1)
    delta = torch.bmm(bottleneck, plan.w_delta_active) + plan.b_delta_active
    return x + delta


def shared_hidden(x, layer_ids, assets):
    device = x.device
    L, n, H = x.shape

    def weight(name):
        return tensor(assets, 'shared.' + name, device, torch.float32)

    def norm(value):
        return value * value.pow(2).mean(-1, keepdim=True).add(1e-6).rsqrt()

    def linear(value, name, bias=True):
        return torch.nn.functional.linear(value, weight(name + '.weight'), weight(name + '.bias') if bias else None)

    blocks = assets.metadata.get('shared_num_blocks')
    if type(blocks) is not int or blocks < 1:
        raise MissingAsset('shared_num_blocks required')
    flat = x.reshape(-1, H).float()
    embed = weight('layer_embed.weight')[layer_ids].unsqueeze(1).expand(-1, n, -1)
    h = linear(norm(flat), 'input_proj') + embed.reshape(L * n, -1)
    for i in range(blocks):
        z = norm(h)
        p = f'blocks.{i}.'
        h = h + linear(torch.nn.functional.silu(linear(z, p + 'gate_proj', False)) * linear(z, p + 'up_proj', False), p + 'down_proj', False)
    bottleneck = linear(norm(h), 'bottleneck_proj', False).reshape(L, n, -1)
    delta = torch.bmm(bottleneck, weight('delta_h_weight').index_select(0, layer_ids)) + weight('delta_h_bias').index_select(0, layer_ids).unsqueeze(1)
    return x + delta


def build_paired_rope_tables(identity, max_len: int, device: torch.device, dtype: torch.dtype = torch.float32):
    cfg = identity.rope
    if cfg.get('rope_type', 'default') != 'default' or cfg.get('factor', 1) != 1:
        raise UnsupportedProfile('legacy Reuse scaled rotary not qualified')
    d = identity.head_dim
    theta = cfg.get('theta', cfg.get('rope_theta'))
    if not theta or d % 2:
        raise UnsupportedProfile('invalid rotary configuration')
    inv = 1.0 / (float(theta) ** (torch.arange(0, d, 2, device=device, dtype=torch.float32) / d))
    angles = torch.outer(torch.arange(max_len, device=device, dtype=torch.float32), inv)
    cos = torch.repeat_interleave(angles.cos(), 2, dim=-1).to(dtype)[None, None, :].contiguous()
    sin = torch.repeat_interleave(angles.sin(), 2, dim=-1).to(dtype)[None, None, :].contiguous()
    return cos, sin


def apply_paired_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    n = x.shape[2]
    c = cos[:, :, :n, :]
    s = sin[:, :, :n, :]
    half = torch.empty_like(x)
    half[..., ::2] = -x[..., 1::2]
    half[..., 1::2] = x[..., ::2]
    return x * c + half * (-s if inverse else s)


def paired_rope(x, identity, inverse=False):
    """Legacy ICML Reuse uses adjacent pairs, unlike its native Patch path."""
    cfg = identity.rope
    if cfg.get('rope_type', 'default') != 'default' or cfg.get('factor', 1) != 1:
        raise UnsupportedProfile('legacy Reuse scaled rotary not qualified')
    d = x.shape[-1]
    theta = cfg.get('theta', cfg.get('rope_theta'))
    if not theta or d % 2:
        raise UnsupportedProfile('invalid rotary configuration')
    inv = 1 / (float(theta) ** (torch.arange(0, d, 2, device=x.device, dtype=torch.float32) / d))
    angles = torch.outer(torch.arange(x.shape[2], device=x.device, dtype=torch.float32), inv)
    cos = torch.repeat_interleave(angles.cos(), 2, dim=-1).to(x.dtype)[None, None, :]
    sin = torch.repeat_interleave(angles.sin(), 2, dim=-1).to(x.dtype)[None, None, :]
    half = torch.empty_like(x)
    half[..., ::2] = -x[..., 1::2]
    half[..., 1::2] = x[..., ::2]
    return x * cos + half * (-sin if inverse else sin)
