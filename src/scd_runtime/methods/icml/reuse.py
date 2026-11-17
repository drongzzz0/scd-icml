from dataclasses import dataclass
import torch
from ..common import tensor

BATCHED_NUMERIC_VERSION = 'icml-pinned-v2-batched'
NATIVE_ROPE_NUMERIC_VERSION = 'icml-pinned-v3-hf-native-rope'
SUPPORTED_NUMERIC_VERSIONS = frozenset(('icml-v1', 'icml-pinned-v1', BATCHED_NUMERIC_VERSION, NATIVE_ROPE_NUMERIC_VERSION))


def check_numeric_version(version):
    from ...errors import UnsupportedProfile
    if version not in SUPPORTED_NUMERIC_VERSIONS:
        raise UnsupportedProfile('unknown ICML numeric version')


@dataclass(frozen=True)
class CompiledReusePlan:
    indices: tuple[int, ...]
    source_layers: tuple[int, ...]
    enc_k: torch.Tensor
    dec_k: torch.Tensor
    mean_a_k: torch.Tensor | None
    mean_b_k: torch.Tensor | None
    enc_v: torch.Tensor
    dec_v: torch.Tensor
    mean_a_v: torch.Tensor | None
    mean_b_v: torch.Tensor | None
    src_cos: torch.Tensor | None
    src_sin: torch.Tensor | None
    tgt_cos: torch.Tensor | None
    tgt_sin: torch.Tensor | None


def compile_reuse_plan(assets, layers, device: torch.device, max_len: int) -> CompiledReusePlan | None:
    from .shared_patch import build_paired_rope_tables
    from ...errors import UnsupportedProfile

    if not layers:
        return None
    p = assets.profile
    indices = tuple(sorted(layers))
    src_layers = tuple(p.layer_map[i] for i in indices)

    def stack_weight(name, kind):
        entries = [tensor(assets, f'reuse.{i}.{name}_{kind}', device, torch.float32) for i in indices]
        if len({tuple(t.shape) for t in entries}) != 1:
            raise UnsupportedProfile('batched Reuse requires uniform layer matrix shapes')
        return torch.stack(entries).contiguous()

    def stack_mean(name, kind):
        keys = [f'reuse.{i}.{name}_{kind}' for i in indices]
        if not all(key in assets.tensors for key in keys):
            return None
        return torch.stack([tensor(assets, key, device, torch.float32) for key in keys]).contiguous().unsqueeze(1)

    src_cos = src_sin = tgt_cos = tgt_sin = None
    if p.route.get('reuse_key_space') == 'post_rope':
        if p.source.rope != p.target.rope:
            raise UnsupportedProfile('post-RoPE Reuse requires matching rotary parameters')
    else:
        src_cos, src_sin = build_paired_rope_tables(p.source, max_len, device, torch.float32)
        tgt_cos, tgt_sin = build_paired_rope_tables(p.target, max_len, device, torch.float32)

    return CompiledReusePlan(
        indices=indices,
        source_layers=src_layers,
        enc_k=stack_weight('enc', 'k'),
        dec_k=stack_weight('dec', 'k'),
        mean_a_k=stack_mean('mean_a', 'k'),
        mean_b_k=stack_mean('mean_b', 'k'),
        enc_v=stack_weight('enc', 'v'),
        dec_v=stack_weight('dec', 'v'),
        mean_a_v=stack_mean('mean_a', 'v'),
        mean_b_v=stack_mean('mean_b', 'v'),
        src_cos=src_cos,
        src_sin=src_sin,
        tgt_cos=tgt_cos,
        tgt_sin=tgt_sin,
    )


def reconstruct_layerwise_compiled(state, plan: CompiledReusePlan) -> tuple[torch.Tensor, torch.Tensor]:
    """Execute batched Reuse using pre-staged GPU weights and cached RoPE tables.

    Returns (k_all, v_all) each of shape [count, b, h, n, d] in float32,
    bitwise identical to reconstruct_layerwise.
    """
    from .shared_patch import apply_paired_rope

    b, h, n, d = state.kv[0][0].shape
    count = len(plan.indices)

    vals_k = torch.stack([state.kv[s][0].float() for s in plan.source_layers]).contiguous()
    if plan.src_cos is not None:
        vals_k = apply_paired_rope(vals_k.reshape(count * b, h, n, d), plan.src_cos, plan.src_sin, inverse=True).reshape(count, b, h, n, d).contiguous()
    xk = vals_k.reshape(count, b * h * n, d)
    zk = torch.bmm(xk if plan.mean_a_k is None else xk - plan.mean_a_k, plan.enc_k).to(torch.bfloat16).float()
    yk = torch.bmm(zk, plan.dec_k)
    if plan.mean_b_k is not None:
        yk = yk + plan.mean_b_k
    vals_k = yk.reshape(count, b, h, n, d).contiguous()
    if plan.tgt_cos is not None:
        vals_k = apply_paired_rope(vals_k.reshape(count * b, h, n, d), plan.tgt_cos, plan.tgt_sin, inverse=False).reshape(count, b, h, n, d)

    vals_v = torch.stack([state.kv[s][1].float() for s in plan.source_layers]).contiguous()
    xv = vals_v.reshape(count, b * h * n, d)
    zv = torch.bmm(xv if plan.mean_a_v is None else xv - plan.mean_a_v, plan.enc_v).to(torch.bfloat16).float()
    yv = torch.bmm(zv, plan.dec_v)
    if plan.mean_b_v is not None:
        yv = yv + plan.mean_b_v
    vals_v = yv.reshape(count, b, h, n, d)

    return vals_k, vals_v


def reconstruct_layerwise(state, assets, layers):
    """Preserve the frozen release's batch geometry and BF16 wire rounding."""
    from .shared_patch import paired_rope
    from ...errors import UnsupportedProfile

    if not layers:
        return {}
    p = assets.profile
    indices = sorted(layers)
    device = state.kv[0][0].device
    b, h, n, d = state.kv[0][0].shape
    count = len(indices)
    rotate_keys = p.route.get('reuse_key_space') != 'post_rope'
    if not rotate_keys and p.source.rope != p.target.rope:
        raise UnsupportedProfile('post-RoPE Reuse requires matching rotary parameters')
    outputs = {}
    for kind, index in (('k', 0), ('v', 1)):
        values = torch.stack([state.kv[p.layer_map[i]][index].float() for i in indices]).contiguous()
        if kind == 'k' and rotate_keys:
            values = paired_rope(values.reshape(count * b, h, n, d), p.source, True).reshape(count, b, h, n, d).contiguous()
        x = values.reshape(count, b * h * n, d)

        def weight(name):
            entries = [tensor(assets, f'reuse.{i}.{name}_{kind}', device, torch.float32) for i in indices]
            if len({tuple(t.shape) for t in entries}) != 1:
                raise UnsupportedProfile('batched Reuse requires uniform layer matrix shapes')
            return torch.stack(entries).contiguous()

        def mean(name):
            keys = [f'reuse.{i}.{name}_{kind}' for i in indices]
            if not all(key in assets.tensors for key in keys):
                return None
            return torch.stack([tensor(assets, key, device, torch.float32) for key in keys]).contiguous().unsqueeze(1)

        center = mean('mean_a')
        z = torch.bmm(x if center is None else x - center, weight('enc')).to(torch.bfloat16).float()
        y = torch.bmm(z, weight('dec'))
        center = mean('mean_b')
        if center is not None:
            y = y + center
        values = y.reshape(count, b, h, n, d).contiguous()
        if kind == 'k' and rotate_keys:
            values = paired_rope(values.reshape(count * b, h, n, d), p.target).reshape(count, b, h, n, d).contiguous()
        outputs[kind] = values
    return {layer: (outputs['k'][j], outputs['v'][j]) for j, layer in enumerate(indices)}


def reconstruct(value, assets, layer, kind):
    p = f'reuse.{layer}.'
    x = value.reshape(-1, value.shape[-1]).float()
    if p + 'mean_a_' + kind in assets.tensors:
        x = x - tensor(assets, p + 'mean_a_' + kind, x.device, x.dtype)
    z = x @ tensor(assets, p + 'enc_' + kind, x.device, x.dtype)
    z = z.to(torch.bfloat16).float()
    y = z @ tensor(assets, p + 'dec_' + kind, x.device, x.dtype)
    if p + 'mean_b_' + kind in assets.tensors:
        y = y + tensor(assets, p + 'mean_b_' + kind, x.device, x.dtype)
    b, h, n, d = value.shape
    return y.reshape(b, h, n, d).contiguous()
