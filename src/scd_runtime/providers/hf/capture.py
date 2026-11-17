import inspect
import torch
from ...contracts.types import SourceState, tokens
from ...errors import IdentityMismatch, UnsupportedProfile
from .config_identity import check_config_identity


def pairs_from_cache(cache):
    if hasattr(cache, 'to_legacy_cache'):
        return tuple(tuple(x) for x in cache.to_legacy_cache())
    if hasattr(cache, 'layers'):
        return tuple((x.keys, x.values) for x in cache.layers)
    return tuple(tuple(x) for x in cache)


def model_layers(model):
    layers = getattr(getattr(model, 'model', None), 'layers', None)
    if layers is None:
        raise UnsupportedProfile('model must expose decoder model.layers')
    return layers


class HFSourceProvider:
    def __init__(self, model, identity, *, fast_causal_mask: bool = False, persistent_hooks: bool = False):
        self.model = model
        self.identity = identity
        self.fast_causal_mask = bool(fast_causal_mask)
        self.persistent_hooks = bool(persistent_hooks)
        check_config_identity(model.config, identity)
        if model.dtype != torch.bfloat16:
            raise IdentityMismatch('source model must be BF16')
        self._layers = model_layers(model)
        self._supports_logits_to_keep = 'logits_to_keep' in inspect.signature(model.forward).parameters
        self._active_capture = None
        self._installed_hook_layers = {}

    def enable_persistent_hooks(self, hidden_hooks):
        self.persistent_hooks = True
        self._ensure_hooks(hidden_hooks)

    def _ensure_hooks(self, hidden_hooks):
        for index, name in hidden_hooks:
            if name != 'attention_input' or not 0 <= index < len(self._layers):
                raise UnsupportedProfile('unknown feature hook')
            key = (index, name)
            if key in self._installed_hook_layers:
                continue

            def make_hook(hook_key):
                def hook(module, args, kwargs):
                    ctx = self._active_capture
                    if ctx is None or hook_key not in ctx['wanted']:
                        return
                    value = kwargs.get('hidden_states', args[0] if args else None)
                    if value is None:
                        raise UnsupportedProfile('attention input not exposed')
                    slice_n = ctx['slice_n']
                    if slice_n is not None:
                        ctx['hidden'][hook_key] = value[:, :slice_n, :].detach().contiguous().clone()
                    else:
                        ctx['hidden'][hook_key] = value.detach().clone()
                return hook

            handle = self._layers[index].self_attn.register_forward_pre_hook(make_hook(key), with_kwargs=True)
            self._installed_hook_layers[key] = handle

    def close(self):
        for handle in self._installed_hook_layers.values():
            handle.remove()
        self._installed_hook_layers.clear()

    @torch.inference_mode()
    def capture(self, prefix, features, *, full_prompt_ids=None):
        if (prefix.tokenizer_sha256 != self.identity.tokenizer_sha256
                or prefix.chat_template_sha256 != self.identity.chat_template_sha256):
            raise IdentityMismatch('source tokenizer identity differs')
        capture_ids = prefix.token_ids
        slice_n = None
        if full_prompt_ids is not None:
            capture_ids = tokens(full_prompt_ids)
            if len(capture_ids) != len(prefix.token_ids) + 1 or capture_ids[:-1] != prefix.token_ids:
                raise IdentityMismatch('full source prompt must be the exact prefix and one final token')
            if max(capture_ids) >= self.model.config.vocab_size:
                raise IdentityMismatch('full source prompt exceeds model vocabulary')
            slice_n = len(prefix.token_ids)
        if any(i < 0 or i >= len(self._layers) for i in features.kv_layers):
            raise UnsupportedProfile('source layer out of range')

        self._ensure_hooks(features.hidden_hooks)
        hidden = {}
        self._active_capture = {'wanted': set(features.hidden_hooks), 'hidden': hidden, 'slice_n': slice_n}
        try:
            dev = self.model.device
            ids = torch.tensor([capture_ids], device=dev, dtype=torch.long)
            extras = {'logits_to_keep': 1} if self._supports_logits_to_keep else {}
            pos = torch.arange(ids.shape[1], device=dev)[None, :]
            if self.fast_causal_mask:
                out = self.model(input_ids=ids, position_ids=pos, use_cache=True, **extras)
            else:
                out = self.model(input_ids=ids, attention_mask=torch.ones_like(ids), position_ids=pos, use_cache=True, **extras)
            kv = pairs_from_cache(out.past_key_values)
            if slice_n is not None:
                kv = tuple(tuple(t[..., :slice_n, :].contiguous() for t in pair) for pair in kv)
            return SourceState(self.identity, prefix, kv, hidden)
        finally:
            self._active_capture = None
            if not self.persistent_hooks:
                self.close()
