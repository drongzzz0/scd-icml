"""Validate trusted records without promoting smoke tests to release evidence."""
from collections import Counter
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re

from ..assets.manifest import file_sha256, safe_path
from ..contracts.capability import VLLM_COMMIT
from ..contracts.identity import profile_fingerprint, protocol_fingerprint
from ..contracts.types import to_data
from ..errors import IdentityMismatch


@dataclass(frozen=True)
class ValidationSummary:
    implementation: str
    assets: str
    reference_equivalence: str
    native_injection: str
    reader_equivalence: str
    quality: str
    performance: str
    complete: bool
    reasons: tuple[str, ...]
    profile_fingerprint: str = ''
    protocol_fingerprint: str = ''
    cohort_size: int = 0
    received_samples: int = 0
    failed_samples: int = 0
    missing_samples: tuple[str, ...] = ()


@dataclass(frozen=True)
class ReleaseStatus:
    profiles: dict
    complete: bool
    reasons: tuple[str, ...]


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _model_binding(runtime, profile):
    return (isinstance(runtime, dict)
            and runtime.get('source_manifest_sha256') == profile.source.manifest_sha256
            and runtime.get('target_manifest_sha256') == profile.target.manifest_sha256)


def _real_runtime(evidence, profile, *, needs_vllm):
    runtime = evidence.get('runtime', {})
    return (evidence.get('scope') == 'real_model' and evidence.get('synthetic') is False
            and _model_binding(runtime, profile)
            and isinstance(runtime.get('device'), str)
            and re.fullmatch(r'cuda(?::[0-9]+)?', runtime['device']) is not None
            and (not needs_vllm or runtime.get('vllm_commit') == VLLM_COMMIT))


def _measure_check(evidence, name, profile, protocol):
    m = evidence.get('measurements')
    if not isinstance(m, dict):
        return 'not_run'
    if name == 'implementation':
        if evidence.get('scope') != 'engineering' or type(m.get('passed')) is not int or m['passed'] <= 0 or type(m.get('failed')) is not int or m['failed'] < 0:
            return 'not_run'
        return 'pass' if m['failed'] == 0 else 'fail'
    if name == 'assets':
        if (evidence.get('scope') != 'real_assets' or evidence.get('synthetic') is not False
                or not _model_binding(evidence.get('runtime'), profile)
                or type(m.get('verified_files')) is not int or m['verified_files'] <= 0
                or any(type(m.get(k)) is not int or m[k] < 0 for k in ('missing_files', 'mismatched_files'))):
            return 'not_run'
        return 'pass' if m['missing_files'] == m['mismatched_files'] == 0 else 'fail'
    if not _real_runtime(evidence, profile, needs_vllm=name != 'reference_equivalence'):
        return 'not_run'
    required = ('key', 'value', 'first_logits') if name == 'reference_equivalence' else ('first_logits',)
    comparisons = m.get('comparisons', {})
    if not isinstance(comparisons, dict) or type(m.get('output_tokens_equal')) is not bool:
        return 'not_run'
    states = []
    for key in required:
        item = comparisons.get(key)
        if (not isinstance(item, dict) or type(item.get('compared_elements')) is not int
                or item['compared_elements'] <= 0 or type(item.get('all_finite')) is not bool
                or item.get('atol') != protocol.tensor_atol or item.get('rtol') != protocol.tensor_rtol):
            return 'not_run'
        if item['all_finite'] is False:
            states.append('fail')
            continue
        if (not _finite(item.get('max_abs')) or item['max_abs'] < 0
                or not _finite(item.get('max_tolerance_excess'))):
            return 'not_run'
        # The pointwise allclose condition is max_i(abs(error_i) - rtol*abs(ref_i)) <= atol.
        states.append('pass' if item['max_tolerance_excess'] <= protocol.tensor_atol else 'fail')
    return 'pass' if m['output_tokens_equal'] and all(s == 'pass' for s in states) else 'fail'


def _check(records, name, profile, protocol):
    values = []
    for receipt in records:
        check = receipt.metrics.get('checks', {}).get(name)
        if not isinstance(check, dict) or check.get('executed') is not True or len(check.get('artifact_sha256', '')) != 64:
            continue
        root = receipt.metrics.get('run_root'); relative = check.get('artifact_path')
        if not root or not relative:
            continue
        try:
            path = safe_path(root, relative)
            if not path.is_file() or file_sha256(path) != check['artifact_sha256']:
                continue
            evidence = json.loads(path.read_text(encoding='utf-8'))
            if (evidence.get('schema_version') != 1 or evidence.get('check_type') != name
                    or evidence.get('profile_fingerprint') != receipt.profile_fingerprint
                    or evidence.get('validation_protocol_fingerprint') != protocol_fingerprint(protocol)
                    or evidence.get('executed') is not True or evidence.get('status') not in ('pass', 'fail')
                    or evidence['status'] != check.get('status')):
                continue
            state = _measure_check(evidence, name, profile, protocol)
            if state != 'not_run':
                values.append('fail' if evidence['status'] == 'fail' else state)
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            continue
    return 'fail' if 'fail' in values else ('pass' if values else 'not_run')


def _members(protocol):
    if not protocol.dataset_manifest_path:
        return None
    path = Path(protocol.dataset_manifest_path)
    if not path.is_file():
        return None
    if file_sha256(path) != protocol.dataset_manifest_sha256:
        raise IdentityMismatch('dataset membership manifest checksum differs')
    data = json.loads(path.read_text(encoding='utf-8'))
    ids = data.get('sample_ids')
    if (data.get('schema_version') != 1 or not isinstance(ids, list) or not ids
            or any(not isinstance(x, str) or not x for x in ids) or len(set(ids)) != len(ids)):
        raise IdentityMismatch('dataset manifest requires unique nonempty sample_ids')
    return set(ids)


def validate_profile(profile, receipts, protocol):
    receipts = tuple(receipts)
    identity = profile_fingerprint(profile); protocol_id = protocol_fingerprint(protocol)
    if protocol.profile_fingerprint != identity or any(r.profile_fingerprint != identity for r in receipts):
        raise IdentityMismatch('validation profile mismatch')
    statuses = {name: _check(receipts, name, profile, protocol) for name in
                ('implementation', 'assets', 'reference_equivalence', 'native_injection', 'reader_equivalence')}
    members = _members(protocol); reasons = []
    sample_ids = [r.metrics.get('sample_id') for r in receipts]
    received = {x for x in sample_ids if isinstance(x, str) and x}
    missing = members - received if members is not None else set()
    digests = [r.request_digest for r in receipts if r.status == 'success']
    unique = (all(isinstance(x, str) and x for x in sample_ids)
              and len(received) == len(receipts)
              and all(isinstance(x, str) and re.fullmatch(r'[a-f0-9]{64}', x) for x in digests)
              and all(n == 1 for n in Counter(digests).values()))
    cohort_ok = members is not None and received == members and unique and len(receipts) == len(members)
    failed = sum(r.status != 'success' for r in receipts)
    quality = performance = 'not_run'
    if members is None:
        reasons.append('cohort: verified membership manifest unavailable')
    elif not cohort_ok:
        reasons.append('cohort: missing, foreign, duplicate or unidentified requests')
        quality = performance = 'fail'
    elif failed:
        quality = performance = 'fail'
    scores = []; evaluated = []
    for receipt in receipts:
        m = receipt.metrics; evaluation = m.get('evaluation')
        if not isinstance(evaluation, dict):
            continue
        if (evaluation.get('dataset_manifest_sha256') != protocol.dataset_manifest_sha256
                or evaluation.get('reference_run_sha256') != protocol.reference_run_sha256
                or evaluation.get('generation_config') != to_data(protocol.generation_config)
                or m.get('validation_protocol_fingerprint') != protocol_id):
            raise IdentityMismatch('quality protocol mismatch')
        if evaluation.get('schema_version') != 1 or not _real_runtime(evaluation, profile, needs_vllm=True):
            continue
        score = evaluation.get(protocol.metric_name); native = evaluation.get('native_'+protocol.metric_name)
        if _finite(score) and _finite(native):
            scores.append((score, native)); evaluated.append(receipt)
    if cohort_ok and not failed and len(scores) == len(receipts):
        score = sum(x for x, _ in scores)/len(members); native = sum(x for _, x in scores)/len(members)
        good = (score >= protocol.threshold if protocol.quality_rule == 'minimum' else
                score >= native*protocol.threshold if protocol.quality_rule == 'relative_retention' else
                native-score <= protocol.threshold)
        quality = 'pass' if good else 'fail'
        timing = [r for r in evaluated if r.metrics.get('backend') == 'vllm'
                  and r.metrics.get('vllm_commit') == VLLM_COMMIT and r.metrics.get('native_fallback') is False
                  and type(r.metrics.get('reused_tokens')) is int and r.metrics['reused_tokens'] > 0
                  and all(_finite(r.metrics.get(k)) and r.metrics[k] > 0 for k in
                          ('pipeline_first_token_ms', 'native_first_token_ms'))]
        if len(timing) == len(receipts) and len({r.mode for r in receipts}) == 1 and receipts[0].mode in ('E', 'H'):
            performance = 'pass'
    statuses.update(quality=quality, performance=performance)
    reasons.extend(f'{name}: {state}' for name, state in statuses.items() if state != 'pass')
    if failed:
        reasons.append('failed requests remain in denominator')
    return ValidationSummary(**statuses, complete=not reasons, reasons=tuple(reasons),
        profile_fingerprint=identity, protocol_fingerprint=protocol_id, cohort_size=len(members or ()),
        received_samples=len(received), failed_samples=failed, missing_samples=tuple(sorted(missing)))


def summarize_release(required_profiles, results):
    reasons = []
    for profile in required_profiles:
        report = results.get(profile.profile_id)
        if report is None:
            reasons.append(f'{profile.profile_id}: missing validation')
        elif (profile.validation_protocol is None or report.profile_fingerprint != profile_fingerprint(profile)
                or report.protocol_fingerprint != protocol_fingerprint(profile.validation_protocol)):
            reasons.append(f'{profile.profile_id}: validation identity differs')
        elif not report.complete:
            reasons.append(f'{profile.profile_id}: incomplete')
    if len({p.profile_id for p in required_profiles}) != len(required_profiles):
        reasons.append('duplicate required profile IDs')
    return ReleaseStatus(dict(results), bool(required_profiles) and not reasons,
                         tuple(reasons or ([] if required_profiles else ['no required profiles'])))
