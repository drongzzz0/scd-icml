"""Compare declared cache semantics with the configuration actually loaded."""
from ...errors import IdentityMismatch, UnsupportedProfile


def check_config_identity(config, identity):
    cfg = config if isinstance(config, dict) else config.to_dict()
    try:
        heads = cfg.get('num_key_value_heads', cfg['num_attention_heads'])
        dim = cfg.get('head_dim') or cfg['hidden_size'] // cfg['num_attention_heads']
        geometry = (cfg['num_hidden_layers'], heads, dim)
        if geometry != (identity.num_layers, identity.num_kv_heads, identity.head_dim):
            raise IdentityMismatch('model KV geometry differs')
        rope = cfg.get('rope_parameters') or cfg.get('rope_scaling') or {}
        if (set(rope) - {'rope_type', 'type', 'rope_theta'}
                or rope.get('rope_type', rope.get('type', 'default')) != 'default'
                or cfg.get('partial_rotary_factor', 1.) != 1.):
            raise UnsupportedProfile('only full-dimension default RoPE is implemented')
        theta = rope.get('rope_theta', cfg.get('rope_theta'))
        if (theta is None or set(identity.rope) - {'theta', 'rope_type'}
                or identity.rope.get('rope_type', 'default') != 'default'
                or theta != identity.rope.get('theta')):
            raise IdentityMismatch('model RoPE identity differs')
        window = cfg.get('sliding_window')
        if cfg.get('use_sliding_window') is False:
            window = None
        layers = cfg.get('layer_types')
        if layers is None:
            if window is not None and cfg.get('max_window_layers', geometry[0]) != geometry[0]:
                raise UnsupportedProfile('mixed window attention needs explicit layer_types')
            layers = ['sliding' if window is not None else 'full'] * geometry[0]
        aliases = {'full_attention': 'full', 'sliding_attention': 'sliding'}
        attention = {'layer_types': tuple(aliases.get(x, x) for x in layers),
                     'sliding_window': window}
        if attention != dict(identity.attention):
            raise IdentityMismatch('model attention identity differs')
    except (KeyError, TypeError, ZeroDivisionError) as exc:
        raise IdentityMismatch(f'incomplete model cache configuration: {exc}') from exc
