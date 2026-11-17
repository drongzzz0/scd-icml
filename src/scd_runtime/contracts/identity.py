import hashlib
import json
from dataclasses import is_dataclass
from .types import to_data


def canonical_bytes(value) -> bytes:
    return json.dumps(
        to_data(value),
        sort_keys=True,
        separators=(',', ':'),
        ensure_ascii=False,
        allow_nan=False,
    ).encode('utf-8')


def sha256(value) -> str:
    if is_dataclass(value):
        cached = getattr(value, '_cached_sha256', None)
        if cached is not None:
            return cached
        res = hashlib.sha256(canonical_bytes(value)).hexdigest()
        try:
            object.__setattr__(value, '_cached_sha256', res)
        except (AttributeError, TypeError):
            pass
        return res
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def profile_fingerprint(profile) -> str:
    cached = getattr(profile, '_cached_profile_fingerprint', None)
    if cached is not None:
        return cached
    # Evaluation protocol binds this identity, so is not recursively hashed into it.
    data = to_data(profile)
    data.pop('validation_protocol', None)
    res = sha256(data)
    try:
        object.__setattr__(profile, '_cached_profile_fingerprint', res)
    except (AttributeError, TypeError):
        pass
    return res


def fingerprint(profile, prefix) -> str:
    pf = profile_fingerprint(profile)
    cached_pair = getattr(prefix, '_cached_method_fp', None)
    if cached_pair is not None and cached_pair[0] == pf:
        return cached_pair[1]
    res = sha256({'schema': 1, 'profile': pf, 'prefix': to_data(prefix)})
    try:
        object.__setattr__(prefix, '_cached_method_fp', (pf, res))
    except (AttributeError, TypeError):
        pass
    return res


def protocol_fingerprint(protocol) -> str:
    data = to_data(protocol)
    data.pop('dataset_manifest_path', None)
    return sha256(data)
