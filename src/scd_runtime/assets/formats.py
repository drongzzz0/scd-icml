"""Checked translation of the pinned SCD-ICML Reuse/Patch checkpoint format."""
import torch
from ..errors import AssetMismatch, MissingAsset, UnsupportedProfile


def _bank_metadata(raw, role):
    embedded = raw[role].get('meta') or {}
    sidecar = raw.get(role + '_meta')
    if sidecar is not None:
        if not isinstance(sidecar, dict) or not sidecar:
            raise MissingAsset(f'{role} metadata is empty')
        if embedded and embedded != sidecar:
            raise AssetMismatch(f'{role} embedded and sidecar metadata differ')
        return sidecar
    if not embedded:
        raise MissingAsset(f'{role}_meta sidecar required for this TensorBundle')
    return embedded


def normalize_icml(raw, profile, metadata):
    from ..methods.icml.reuse import check_numeric_version
    check_numeric_version(profile.numeric_version)
    try:
        reuse = raw['reuse']
        patch = raw['patch']
        rmeta = _bank_metadata(raw, 'reuse')
        pmeta = _bank_metadata(raw, 'patch')
        ra = reuse['arrays']
        pa = patch['arrays']
    except KeyError as exc:
        raise MissingAsset('ICML requires reuse/patch TensorBundle files') from exc

    key_space = profile.route.get('reuse_key_space')
    flags = (rmeta.get('k_de_rope'), rmeta.get('k_re_rope'))
    if rmeta.get('group_size') != 1:
        raise UnsupportedProfile('released importer requires layerwise Reuse')
    if ((key_space == 'pre_rope' and flags != (True, True))
            or (key_space == 'post_rope' and (flags != (False, False) or profile.source.rope != profile.target.rope))
            or key_space not in ('pre_rope', 'post_rope')):
        raise UnsupportedProfile('Reuse archive and model RoPE contracts differ')
    if profile.numeric_version in ('icml-pinned-v2-batched', 'icml-pinned-v3-hf-native-rope'):
        groups = rmeta['group_layers']
        if len(groups) != profile.source.num_layers or any(len(value) != 1 for value in groups.values()):
            raise UnsupportedProfile('batched original Reuse requires one group per source layer')
        shapes = {
            tuple(tuple(ra[f'{name}.{gid}'].shape) for name in ('enc_k', 'enc_v', 'dec_k_b', 'dec_v_b'))
            for gid in groups
        }
        if len(shapes) != 1:
            raise UnsupportedProfile('original Reuse does not select its batched path for these shapes')
    if pmeta.get('fit_mode') != profile.route.get('fit_mode'):
        raise AssetMismatch('Patch fit mode differs from profile')

    result = {}
    for layer in profile.route['reuse_layers']:
        matches = [key for key, value in rmeta['group_layers'].items() if value == [layer]]
        if len(matches) != 1:
            raise AssetMismatch('layer-to-group binding differs')
        gid = matches[0]
        for old, new in (('enc_k', 'enc_k'), ('dec_k_b', 'dec_k'), ('enc_v', 'enc_v'), ('dec_v_b', 'dec_v')):
            try:
                result[f'reuse.{layer}.{new}'] = ra[f'{old}.{gid}'].detach().cpu().clone()
            except KeyError as exc:
                raise MissingAsset(str(exc)) from exc
        for kind in ('mean_a_k', 'mean_b_k', 'mean_a_v', 'mean_b_v'):
            if f'{kind}.{gid}' in ra:
                result[f'reuse.{layer}.{kind}'] = ra[f'{kind}.{gid}'].detach().cpu().clone()
            elif rmeta.get('center_before_svd'):
                raise MissingAsset('centered Reuse statistics missing')

    prefix = 'xnorm_nativekv_shared_residual_patch.'
    if pmeta['fit_mode'] == 'xnorm_nativekv_shared_residual_mlp_v1':
        result.update({
            'shared.' + key[len(prefix):]: value.detach().cpu().clone()
            for key, value in pa.items()
            if key.startswith(prefix)
        })
        if not any(key.startswith('shared.') for key in result):
            raise MissingAsset('shared residual state missing')
        metadata = {**metadata, 'shared_num_blocks': pmeta['residual_num_blocks']}
    else:
        raise UnsupportedProfile('automatic ICML import supports shared-residual Patch banks')
    return result, metadata
