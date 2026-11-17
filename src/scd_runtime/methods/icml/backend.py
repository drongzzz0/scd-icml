from dataclasses import dataclass
import torch
from ...contracts.types import FeatureRequest, TargetKV
from ...contracts.identity import fingerprint
from ...errors import AssetMismatch, MissingAsset, UnsupportedProfile, IdentityMismatch
from ..common import validate_source, rope_tables, rotate
from .reuse import (
    reconstruct,
    reconstruct_layerwise,
    CompiledReusePlan,
    compile_reuse_plan,
    reconstruct_layerwise_compiled,
    BATCHED_NUMERIC_VERSION,
    NATIVE_ROPE_NUMERIC_VERSION,
    check_numeric_version,
)
from .patch import patch_layer
from .shared_patch import (
    shared_hidden,
    paired_rope,
    CompiledSharedPatch,
    compile_shared_patch,
    shared_hidden_compiled,
)


@dataclass
class CompiledICMLPlan:
    device: torch.device
    max_seq_len: int
    asset_signature: tuple | None
    reuse_layers: tuple[int, ...]
    patch_layers: tuple[int, ...]
    reuse_idx_tensor: torch.Tensor | None
    patch_idx_tensor: torch.Tensor | None
    identity_layers: frozenset[int]
    active_patch_layers: tuple[int, ...]
    reuse_plan: CompiledReusePlan | None
    shared_patch_plan: CompiledSharedPatch | None
    patch_rope_tables: tuple[torch.Tensor, torch.Tensor] | None


def _tensor_signature(tensors_dict) -> tuple | None:
    items = []
    for k in sorted(tensors_dict):
        t = tensors_dict[k]
        if not isinstance(t, torch.Tensor):
            return None
        try:
            ver = t._version
        except RuntimeError:
            return None
        items.append((k, id(t), ver, tuple(t.shape), t.dtype, t.device))
    return tuple(items)


class ICMLBackend:
    def __init__(self):
        self._shared_identity_cache = None
        self._compiled_plan: CompiledICMLPlan | None = None
        self._compiled_assets_ref = None

    def shared_identity_layers(self, assets):
        """Cache the original exact-zero residual predicate, never caller hints."""
        p = assets.profile
        if p.route.get('fit_mode') != 'xnorm_nativekv_shared_residual_mlp_v1':
            return ()
        blocks = assets.metadata.get('shared_num_blocks')
        if type(blocks) is not int or blocks < 1:
            raise MissingAsset('shared_num_blocks required')
        names = [
            'input_proj.weight', 'input_proj.bias', 'layer_embed.weight',
            'bottleneck_proj.weight', 'delta_h_weight', 'delta_h_bias',
        ]
        for i in range(blocks):
            names.extend(f'blocks.{i}.{part}.weight' for part in ('gate_proj', 'up_proj', 'down_proj'))
        try:
            values = tuple(assets.tensors['shared.' + name] for name in names)
        except KeyError as exc:
            raise MissingAsset(str(exc)) from exc
        if any(not isinstance(t, torch.Tensor) or not t.is_floating_point() for t in values):
            raise AssetMismatch('shared Patch requires floating tensor state')
        parameters = dict(zip(names, values))
        w = parameters['delta_h_weight']
        b = parameters['delta_h_bias']
        h = b.shape[1] if b.ndim == 2 else 0
        L = p.target.num_layers
        if h < 1 or w.ndim != 3 or w.shape[0] != L or w.shape[1] < 1 or w.shape[2] != h or b.shape != (L, h):
            raise AssetMismatch('shared residual head geometry differs')
        inp = parameters['input_proj.weight']
        bottleneck = parameters['bottleneck_proj.weight']
        if inp.ndim != 2 or inp.shape[1] != h or inp.shape[0] < 1:
            raise AssetMismatch('shared input projection geometry differs')
        trunk = inp.shape[0]
        if (parameters['input_proj.bias'].shape != (trunk,)
                or parameters['layer_embed.weight'].shape != (L, trunk)
                or bottleneck.shape != (w.shape[1], trunk)):
            raise AssetMismatch('shared Patch trunk geometry differs')
        for i in range(blocks):
            gate = parameters[f'blocks.{i}.gate_proj.weight']
            up = parameters[f'blocks.{i}.up_proj.weight']
            down = parameters[f'blocks.{i}.down_proj.weight']
            if gate.ndim != 2 or gate.shape[1] != trunk or up.shape != gate.shape or down.shape != (trunk, gate.shape[0]):
                raise AssetMismatch('shared SwiGLU geometry differs')
        try:
            signature = tuple((id(t), t._version, tuple(t.shape), t.dtype, t.device) for t in values)
        except RuntimeError:
            signature = None
        cached = self._shared_identity_cache
        if (signature is not None and cached is not None and cached[0] is assets
                and cached[1] == signature and all(a is c for a, c in zip(values, cached[2]))):
            return cached[3]
        self._shared_identity_cache = None
        if any(not bool(torch.isfinite(t).all()) for t in values):
            raise AssetMismatch('nonfinite shared Patch state')
        wz = (w == 0).flatten(1).all(1).cpu().tolist()
        bz = (b == 0).all(1).cpu().tolist()
        identity = tuple(i for i, (a, c) in enumerate(zip(wz, bz)) if a and c)
        if signature is not None:
            self._shared_identity_cache = (assets, signature, values, identity)
        return identity

    def required_features(self, profile):
        cached = getattr(profile, '_cached_required_features', None)
        if cached is not None:
            return cached
        req = FeatureRequest(
            tuple(range(profile.source.num_layers)),
            tuple((profile.layer_map[i], 'attention_input') for i in profile.route.get('patch_layers', ())),
        )
        try:
            object.__setattr__(profile, '_cached_required_features', req)
        except (AttributeError, TypeError):
            pass
        return req

    def compile_plan(self, assets, projections=None, *, device: torch.device, max_seq_len: int = 4096) -> CompiledICMLPlan | None:
        """Pre-stage Reuse matrices, Patch weights, target projections and RoPE tables on device."""
        p = assets.profile
        route = p.route
        if p.numeric_version not in (BATCHED_NUMERIC_VERSION, NATIVE_ROPE_NUMERIC_VERSION):
            return None
        if route.get('reuse_key_space') != 'pre_rope':
            return None
        if route.get('fit_mode') != 'xnorm_nativekv_shared_residual_mlp_v1':
            return None
        if projections is None:
            return None

        sig = _tensor_signature(assets.tensors)
        if (sig is not None and self._compiled_plan is not None
                and self._compiled_assets_ref is assets
                and self._compiled_plan.asset_signature == sig
                and self._compiled_plan.device == device
                and self._compiled_plan.max_seq_len >= max_seq_len):
            return self._compiled_plan

        reuse = tuple(sorted(route.get('reuse_layers', ())))
        patch = tuple(sorted(route.get('patch_layers', ())))
        identity_set = frozenset(self.shared_identity_layers(assets))
        active_patch = tuple(i for i in patch if i not in identity_set)

        if hasattr(projections, 'prepare_device'):
            projections.prepare_device(device, patch)
        reuse_plan = compile_reuse_plan(assets, reuse, device, max_seq_len)
        shared_plan = compile_shared_patch(assets, active_patch, device)
        native_cpu = (p.numeric_version == NATIVE_ROPE_NUMERIC_VERSION)
        patch_tables = rope_tables(p.target, max_seq_len, device, torch.bfloat16, native_cpu_init=native_cpu) if patch else None

        plan = CompiledICMLPlan(
            device=device,
            max_seq_len=max_seq_len,
            asset_signature=sig,
            reuse_layers=reuse,
            patch_layers=patch,
            reuse_idx_tensor=torch.tensor(reuse, device=device, dtype=torch.long) if reuse else None,
            patch_idx_tensor=torch.tensor(patch, device=device, dtype=torch.long) if patch else None,
            identity_layers=identity_set,
            active_patch_layers=active_patch,
            reuse_plan=reuse_plan,
            shared_patch_plan=shared_plan,
            patch_rope_tables=patch_tables,
        )
        if sig is not None:
            self._compiled_assets_ref = assets
            self._compiled_plan = plan
        return plan

    def _validate_convert_contract(self, state, assets, projections):
        p = assets.profile
        device = validate_source(state, p)
        route = p.route
        if p.method_id != 'icml_reuse_patch':
            raise AssetMismatch('wrong method assets')
        check_numeric_version(p.numeric_version)
        reuse = tuple(route.get('reuse_layers', ()))
        patch = tuple(route.get('patch_layers', ()))
        if len(set(reuse + patch)) != len(reuse + patch) or set(reuse + patch) != set(range(p.target.num_layers)):
            raise AssetMismatch('route must cover each target layer exactly once')
        if route.get('fallback_layers') or route.get('selected_recompute_layers'):
            raise UnsupportedProfile('fallback/recompute not permitted in conversion')
        if route.get('fit_mode') not in ('direct_kv_v1', 'xnorm_semantic_v1', 'xnorm_nativekv_shared_residual_mlp_v1'):
            raise UnsupportedProfile('unsupported Patch fit mode')
        if route.get('reuse_key_space') not in ('pre_rope', 'post_rope'):
            raise AssetMismatch('declare reuse key space')
        if projections is not None and projections.identity != p.target:
            raise IdentityMismatch('projection target differs')
        return p, device, route, reuse, patch

    @torch.inference_mode()
    def convert(self, state, assets, projections=None, *, return_packed: bool = False):
        p, device, route, reuse, patch = self._validate_convert_contract(state, assets, projections)
        n = len(state.prefix.token_ids)

        # Fast compiled path when explicitly pre-compiled or executing on CUDA
        if (
            (self._compiled_plan is not None or device.type == 'cuda')
            and p.numeric_version in (BATCHED_NUMERIC_VERSION, NATIVE_ROPE_NUMERIC_VERSION)
            and route['reuse_key_space'] == 'pre_rope'
            and route['fit_mode'] == 'xnorm_nativekv_shared_residual_mlp_v1'
            and projections is not None
        ):
            max_len = max(4096, ((n + 255) // 256) * 256)
            plan = self.compile_plan(assets, projections, device=device, max_seq_len=max_len)
            if plan is not None:
                return self._convert_compiled(state, p, projections, plan, n, return_packed=return_packed)

        if p.numeric_version in (BATCHED_NUMERIC_VERSION, NATIVE_ROPE_NUMERIC_VERSION) and route['reuse_key_space'] != 'pre_rope':
            raise UnsupportedProfile('batched release Reuse requires pre-RoPE space')
        if route['fit_mode'] == 'xnorm_nativekv_shared_residual_mlp_v1' and projections is None:
            raise MissingAsset('shared native Patch requires frozen target projections')

        layers = []
        shared = {}
        batched = {}
        native_tables = (
            rope_tables(p.target, n, device, torch.bfloat16, native_cpu_init=True)
            if p.numeric_version == NATIVE_ROPE_NUMERIC_VERSION and patch
            else None
        )
        if p.numeric_version in (BATCHED_NUMERIC_VERSION, NATIVE_ROPE_NUMERIC_VERSION):
            batched = reconstruct_layerwise(state, assets, reuse)
        if route['fit_mode'] == 'xnorm_nativekv_shared_residual_mlp_v1':
            indices = sorted(patch)
            identity = set(self.shared_identity_layers(assets)) if p.numeric_version in (BATCHED_NUMERIC_VERSION, NATIVE_ROPE_NUMERIC_VERSION) else set()
            active = [i for i in indices if i not in identity]
            try:
                shared = {i: state.hidden[(p.layer_map[i], 'attention_input')].float() for i in indices if i in identity}
                inputs = torch.stack([state.hidden[(p.layer_map[i], 'attention_input')][0].float() for i in active]) if active else None
            except KeyError as exc:
                raise MissingAsset('shared Patch source hidden missing') from exc
            if active:
                predicted = shared_hidden(inputs, torch.tensor(active, device=device, dtype=torch.long), assets)
                shared.update({i: predicted[j:j + 1] for j, i in enumerate(active)})

        for i, s in enumerate(p.layer_map):
            if i in reuse:
                if i in batched:
                    k, v = batched[i]
                else:
                    k, v = state.kv[s]
                    if route['reuse_key_space'] == 'pre_rope':
                        k = paired_rope(k.float(), p.source, True)
                    k = reconstruct(k, assets, i, 'k')
                    v = reconstruct(v, assets, i, 'v')
                    if route['reuse_key_space'] == 'pre_rope':
                        k = paired_rope(k, p.target)
                    elif p.source.rope != p.target.rope:
                        raise UnsupportedProfile('post-RoPE mapping requires equal rotary config')
            else:
                hidden = state.hidden.get((s, 'attention_input'))
                if hidden is None:
                    raise MissingAsset(f'source attention input missing: {s}')
                if i in shared:
                    k, v = projections.project(i, shared[i])
                    k = rotate(k.to(torch.bfloat16), native_tables if native_tables is not None else rope_tables(p.target, n, device, torch.bfloat16))
                else:
                    k, v = patch_layer(hidden, assets, i, projections)
                    k = rotate(k, native_tables if native_tables is not None else rope_tables(p.target, n, device, k.dtype))
            layers.append((k.to(torch.bfloat16).contiguous(), v.to(torch.bfloat16).contiguous()))

        packed_kv = None
        if return_packed:
            L = p.target.num_layers
            H = p.target.num_kv_heads
            D = p.target.head_dim
            with torch.inference_mode(False), torch.no_grad():
                packed_kv = torch.empty((L, n, H, 2 * D), device=device, dtype=torch.bfloat16)
                for idx, (kt, vt) in enumerate(layers):
                    packed_kv[idx, :, :, :D].copy_(kt[0].transpose(0, 1))
                    packed_kv[idx, :, :, D:].copy_(vt[0].transpose(0, 1))
        return TargetKV(p.target, p.reader, state.prefix, tuple(layers), fingerprint(p, state.prefix), packed_kv=packed_kv)

    def _convert_compiled(self, state, p, projections, plan: CompiledICMLPlan, n: int, *, return_packed: bool) -> TargetKV:
        L = p.target.num_layers
        H = p.target.num_kv_heads
        D = p.target.head_dim
        device = plan.device

        # 1. Batched Reuse reconstruction across all Reuse layers
        reuse_kv = {}
        rk_bf16 = rv_bf16 = None
        if plan.reuse_plan is not None:
            rk, rv = reconstruct_layerwise_compiled(state, plan.reuse_plan)
            rk_bf16 = rk.to(torch.bfloat16)
            rv_bf16 = rv.to(torch.bfloat16)
            if not return_packed:
                for j, layer_idx in enumerate(plan.reuse_plan.indices):
                    reuse_kv[layer_idx] = (rk_bf16[j], rv_bf16[j])

        # 2. Shared residual Patch + batched RoPE across all Patch layers
        patch_kv = {}
        rotated_k = stacked_v = None
        if plan.patch_layers:
            shared = {}
            try:
                for i in plan.patch_layers:
                    if i in plan.identity_layers:
                        shared[i] = state.hidden[(p.layer_map[i], 'attention_input')]
                inputs = (
                    torch.stack([state.hidden[(p.layer_map[i], 'attention_input')][0].float() for i in plan.active_patch_layers])
                    if plan.active_patch_layers else None
                )
            except KeyError as exc:
                raise MissingAsset('shared Patch source hidden missing') from exc

            if plan.active_patch_layers and plan.shared_patch_plan is not None:
                predicted = shared_hidden_compiled(inputs, plan.shared_patch_plan)
                for j, i in enumerate(plan.active_patch_layers):
                    shared[i] = predicted[j:j + 1]

            raw_k = []
            raw_v = []
            for i in plan.patch_layers:
                k_i, v_i = projections.project(i, shared[i])
                raw_k.append(k_i.to(torch.bfloat16))
                raw_v.append(v_i.to(torch.bfloat16))

            # Single vectorized RoPE rotation across all P patch layers simultaneously
            stacked_k = torch.cat(raw_k, dim=0)  # [P, H, n, D]
            cos, sin = plan.patch_rope_tables
            cos_n = cos[:, :n, :].unsqueeze(1)  # [1, 1, n, D]
            sin_n = sin[:, :n, :].unsqueeze(1)
            half = D // 2
            rot_y = torch.cat((-stacked_k[..., half:], stacked_k[..., :half]), dim=-1)
            rotated_k = stacked_k * cos_n + rot_y * sin_n  # [P, H, n, D]

            if return_packed:
                stacked_v = torch.cat(raw_v, dim=0)  # [P, H, n, D]
            else:
                for j, i in enumerate(plan.patch_layers):
                    patch_kv[i] = (rotated_k[j:j + 1], raw_v[j])

        if return_packed:
            # Allocate one versioned non-inference [L, n, H, 2*D] buffer ready for direct paged-KV scatter
            with torch.inference_mode(False), torch.no_grad():
                packed_kv = torch.empty((L, n, H, 2 * D), device=device, dtype=torch.bfloat16)
                if rk_bf16 is not None and plan.reuse_idx_tensor is not None:
                    reuse_packed = torch.cat(
                        (rk_bf16.squeeze(1).transpose(1, 2), rv_bf16.squeeze(1).transpose(1, 2)),
                        dim=-1,
                    )
                    packed_kv.index_copy_(0, plan.reuse_idx_tensor, reuse_packed.detach())
                if rotated_k is not None and stacked_v is not None and plan.patch_idx_tensor is not None:
                    patch_packed = torch.cat(
                        (rotated_k.transpose(1, 2), stacked_v.transpose(1, 2)),
                        dim=-1,
                    )
                    packed_kv.index_copy_(0, plan.patch_idx_tensor, patch_packed.detach())
                layers = tuple(
                    (
                        packed_kv[i, :, :, :D].transpose(0, 1).unsqueeze(0),
                        packed_kv[i, :, :, D:].transpose(0, 1).unsqueeze(0),
                    )
                    for i in range(L)
                )
            return TargetKV(p.target, p.reader, state.prefix, layers, fingerprint(p, state.prefix), packed_kv=packed_kv)

        layers = []
        for i in range(L):
            kt, vt = reuse_kv[i] if i in reuse_kv else patch_kv[i]
            layers.append((kt.contiguous(), vt.contiguous()))
        return TargetKV(p.target, p.reader, state.prefix, tuple(layers), fingerprint(p, state.prefix), packed_kv=None)
