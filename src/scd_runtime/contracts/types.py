from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, TYPE_CHECKING
import math
import re

if TYPE_CHECKING:
    from torch import Tensor

JsonValue = Any
TensorPair = tuple['Tensor', 'Tensor']


def freeze(value):
    if isinstance(value, Mapping):
        if any(not isinstance(k, str) for k in value):
            raise ValueError('JSON keys must be strings')
        return MappingProxyType({k: freeze(v) for k, v in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(freeze(v) for v in value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise ValueError('not a finite JSON value')


def to_data(value):
    if is_dataclass(value):
        cached = getattr(value, '_cached_to_data', None)
        if cached is not None:
            return dict(cached)
        res = {f.name: to_data(getattr(value, f.name)) for f in fields(value)}
        try:
            object.__setattr__(value, '_cached_to_data', res)
        except (AttributeError, TypeError):
            pass
        return dict(res)
    if isinstance(value, Mapping):
        return {k: to_data(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        if value and type(value[0]) is int and all(type(x) is int for x in value):
            return list(value)
        return [to_data(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def digest(value: str, name='sha256'):
    if not isinstance(value, str) or re.fullmatch(r'[0-9a-f]{64}', value) is None:
        raise ValueError(f'invalid {name}')


def integer(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f'invalid {name}')


class _TokenTuple(tuple):
    """Marker subclass so repeated tokens(...) calls on validated tuples are O(1)."""
    __slots__ = ()


def tokens(value) -> tuple[int, ...]:
    if isinstance(value, _TokenTuple):
        return value
    result = _TokenTuple(value)
    if not result or any(type(x) is not int or x < 0 for x in result):
        raise ValueError('token IDs must be nonempty nonnegative integers')
    return result


@dataclass(frozen=True)
class ModelIdentity:
    manifest_sha256: str
    config_sha256: str
    tokenizer_sha256: str
    chat_template_sha256: str
    num_layers: int
    num_kv_heads: int
    head_dim: int
    dtype: str
    rope: Mapping
    attention: Mapping

    def __post_init__(self):
        for name in ('manifest_sha256', 'config_sha256', 'tokenizer_sha256', 'chat_template_sha256'):
            digest(getattr(self, name), name)
        for name in ('num_layers', 'num_kv_heads', 'head_dim'):
            integer(getattr(self, name), name)
        if self.dtype != 'bfloat16':
            raise ValueError('initial schema requires bfloat16')
        if not isinstance(self.rope, Mapping) or not self.rope:
            raise ValueError('RoPE config required')
        if not isinstance(self.attention, Mapping) or len(self.attention.get('layer_types', ())) != self.num_layers:
            raise ValueError('attention definition required per layer')
        object.__setattr__(self, 'rope', freeze(self.rope))
        object.__setattr__(self, 'attention', freeze(self.attention))


@dataclass(frozen=True)
class ReaderIdentity:
    kind: str
    base_manifest_sha256: str
    weights_sha256: str | None = None
    rank: int = 0
    alpha: float = 0.0
    targets: tuple[str, ...] = ()
    numeric_mode: str = 'none'

    def __post_init__(self):
        digest(self.base_manifest_sha256)
        object.__setattr__(self, 'targets', tuple(self.targets))
        if self.kind != 'none':
            raise ValueError('SCD-ICML uses target-native Reuse/Patch KV with kind="none"')
        if self.weights_sha256 is not None or self.rank != 0 or self.alpha != 0 or self.targets or self.numeric_mode != 'none':
            raise ValueError('none reader cannot contain adapter settings')


@dataclass(frozen=True)
class PrefixContract:
    token_ids: tuple[int, ...]
    position_start: int
    tokenizer_sha256: str
    chat_template_sha256: str
    mask_mode: str = 'causal_unpadded'

    def __post_init__(self):
        object.__setattr__(self, 'token_ids', tokens(self.token_ids))
        if type(self.position_start) is not int or self.position_start != 0:
            raise ValueError('only position zero supported')
        if self.mask_mode != 'causal_unpadded':
            raise ValueError('unsupported mask')
        digest(self.tokenizer_sha256)
        digest(self.chat_template_sha256)


@dataclass(frozen=True)
class ValidationProtocol:
    profile_fingerprint: str
    dataset_manifest_sha256: str
    reference_run_sha256: str
    metric_name: str
    quality_rule: str
    threshold: float
    tensor_atol: float
    tensor_rtol: float
    generation_config: Mapping
    dataset_manifest_path: str | None = None

    def __post_init__(self):
        for name in ('profile_fingerprint', 'dataset_manifest_sha256', 'reference_run_sha256'):
            digest(getattr(self, name))
        if self.quality_rule not in ('minimum', 'relative_retention', 'max_drop'):
            raise ValueError('unknown quality rule')
        for name in ('threshold', 'tensor_atol', 'tensor_rtol'):
            value = getattr(self, name)
            if type(value) not in (float, int) or not math.isfinite(value) or value < 0:
                raise ValueError(f'invalid {name}')
        if not self.metric_name:
            raise ValueError('metric required')
        if self.dataset_manifest_path is not None and (not isinstance(self.dataset_manifest_path, str) or not self.dataset_manifest_path):
            raise ValueError('invalid dataset manifest path')
        object.__setattr__(self, 'generation_config', freeze(self.generation_config))


@dataclass(frozen=True)
class MethodProfile:
    profile_id: str
    method_id: str
    source: ModelIdentity
    target: ModelIdentity
    reader: ReaderIdentity
    route: Mapping
    layer_map: tuple[int, ...]
    numeric_version: str
    asset_manifest_sha256: str
    validation_protocol: ValidationProtocol | None = None

    def __post_init__(self):
        if self.method_id != 'icml_reuse_patch':
            raise ValueError('unknown method; expected icml_reuse_patch')
        if not isinstance(self.profile_id, str) or not self.profile_id or not isinstance(self.numeric_version, str) or not self.numeric_version:
            raise ValueError('profile identity required')
        digest(self.asset_manifest_sha256)
        if self.reader.base_manifest_sha256 != self.target.manifest_sha256:
            raise ValueError('reader base mismatch')
        mapping = tuple(self.layer_map)
        if len(mapping) != self.target.num_layers or any(type(x) is not int or not 0 <= x < self.source.num_layers for x in mapping):
            raise ValueError('invalid layer map')
        object.__setattr__(self, 'layer_map', mapping)
        object.__setattr__(self, 'route', freeze(self.route))


@dataclass(frozen=True)
class FeatureRequest:
    kv_layers: tuple[int, ...]
    hidden_hooks: tuple[tuple[int, str], ...] = ()


@dataclass(frozen=True)
class SourceState:
    source: ModelIdentity
    prefix: PrefixContract
    kv: tuple[TensorPair, ...]
    hidden: Mapping[tuple[int, str], 'Tensor'] = field(default_factory=dict)


@dataclass(frozen=True)
class TargetKV:
    target: ModelIdentity
    reader: ReaderIdentity
    prefix: PrefixContract
    layers: tuple[TensorPair, ...]
    method_fingerprint: str
    rope_applied: bool = True
    packed_kv: 'Tensor | None' = None  # Optional contiguous [L, N, H, 2*D] BF16 view owner


@dataclass(frozen=True)
class MethodAssets:
    profile: MethodProfile
    root: Path
    tensors: Mapping[str, 'Tensor']
    metadata: Mapping


@dataclass(frozen=True)
class ImportPlan:
    original_prefix_tokens: int
    imported_tokens: int
    tail_recomputed_tokens: int
    suffix_ids: tuple[int, ...]
    request_digest: str


@dataclass(frozen=True)
class FullPrefixImportPlan(ImportPlan):
    """Opt-in full-prefix import plan; legacy ImportPlan wire and digest stay unchanged."""
    import_policy: str = 'fullprefix'

    def __post_init__(self):
        if self.import_policy != 'fullprefix':
            raise ValueError('unknown full-prefix import policy')
        integer(self.original_prefix_tokens, 'original prefix tokens')
        if (self.imported_tokens != self.original_prefix_tokens
                or type(self.imported_tokens) is not int
                or type(self.tail_recomputed_tokens) is not int
                or self.tail_recomputed_tokens != 0):
            raise ValueError('full-prefix plan must import exactly the entire nonempty prefix')
        object.__setattr__(self, 'suffix_ids', tokens(self.suffix_ids))
        digest(self.request_digest, 'request digest')


@dataclass(frozen=True)
class EngineCapability:
    vllm_commit: str
    backend: str
    dtype: str
    tp: int
    pp: int
    block_size: int
    kv_groups: int
    attention: Mapping
    hma: bool
    chunked_prefill: bool
    prefix_cache: bool
    max_model_len: int | None = None
    request_prefix_cache_policy: bool = False


@dataclass(frozen=True)
class GenerationResult:
    token_ids: tuple[int, ...]
    first_logits: 'Tensor | None' = None
    metadata: Mapping = field(default_factory=dict)


def strict_construct(cls, data):
    if not isinstance(data, dict):
        raise ValueError(f'{cls.__name__} must be object')
    known = {x.name for x in fields(cls)}
    if data.keys() - known:
        raise ValueError(f'unknown {cls.__name__} fields: {sorted(data.keys() - known)}')
    try:
        return cls(**data)
    except TypeError as exc:
        raise ValueError(f'invalid {cls.__name__}') from exc


def profile_from_dict(data: dict) -> MethodProfile:
    value = dict(data)
    for name in ('source', 'target'):
        value[name] = strict_construct(ModelIdentity, value[name])
    value['reader'] = strict_construct(ReaderIdentity, value['reader'])
    if value.get('validation_protocol') is not None:
        value['validation_protocol'] = strict_construct(ValidationProtocol, value['validation_protocol'])
    return strict_construct(MethodProfile, value)
