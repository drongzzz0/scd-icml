from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence


@dataclass
class DoctorCheck:
    name: str
    status: str
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "message": self.message,
            "details": self.details,
        }


def run_doctor(
    *,
    producer_model: str = "",
    consumer_model: str = "",
    reuse_ckpt: str = "",
    patch_ckpt: str = "",
    route: str = "",
    require_local_models: bool = False,
    require_mixed: bool = True,
    skip_vllm_import: bool = False,
) -> dict[str, Any]:
    checks: list[DoctorCheck] = []
    checks.append(_check_vllm_import(skip=skip_vllm_import))
    for name, value, required in (
        ("producer_model", producer_model, require_local_models),
        ("consumer_model", consumer_model, require_local_models),
        ("reuse_ckpt", reuse_ckpt, True),
        ("patch_ckpt", patch_ckpt, True),
        ("route", route, True),
    ):
        checks.append(_check_path(name, value, required=required))
    if route:
        checks.append(_check_route(route, require_mixed=require_mixed))
    else:
        checks.append(DoctorCheck("route_strictness", "fail", "missing --route"))
    failed = [item for item in checks if item.status == "fail"]
    warnings = [item for item in checks if item.status == "warn"]
    return {
        "mode": "scd_vllm_doctor",
        "pass": not failed,
        "failure_count": len(failed),
        "warning_count": len(warnings),
        "checks": [item.to_dict() for item in checks],
    }


def _check_vllm_import(*, skip: bool) -> DoctorCheck:
    if skip:
        return DoctorCheck("vllm_import", "skip", "vLLM import check skipped")
    try:
        version = importlib.metadata.version("vllm")
    except importlib.metadata.PackageNotFoundError:
        return DoctorCheck("vllm_import", "fail", "vllm package is not installed")
    try:
        importlib.import_module("vllm")
    except Exception as exc:
        return DoctorCheck("vllm_import", "fail", f"vllm import failed: {exc}", {"version": version})
    return DoctorCheck("vllm_import", "pass", "vllm import passed", {"version": version})


def _check_path(name: str, value: str, *, required: bool) -> DoctorCheck:
    if not value:
        return DoctorCheck(name, "fail" if required else "skip", f"missing --{name.replace('_', '-')}")
    path = Path(value)
    looks_local = path.is_absolute() or value.startswith((".", "~"))
    if path.exists():
        return DoctorCheck(name, "pass", "path exists", {"path": str(path), "is_dir": path.is_dir()})
    if required or looks_local:
        return DoctorCheck(name, "fail", "path does not exist", {"path": str(path)})
    return DoctorCheck(
        name,
        "warn",
        "not a local path; assuming model id or remote path",
        {"value": value},
    )


def _check_route(route_path: str, *, require_mixed: bool) -> DoctorCheck:
    path = Path(route_path)
    if not path.is_file():
        return DoctorCheck("route_strictness", "fail", "route file does not exist", {"path": str(path)})
    try:
        route = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception as exc:
        return DoctorCheck("route_strictness", "fail", f"route JSON parse failed: {exc}", {"path": str(path)})

    errors: list[str] = []
    try:
        num_layers = int(route.get("num_layers", 0))
    except Exception:
        num_layers = 0
    patch_layers = _int_list(route, "patch_layers")
    reuse_layers = _int_list(route, "reuse_layers")
    fallback_layers = _int_list(route, "fallback_layers")
    recompute_layers = _int_list(route, "selected_recompute_layers")
    recompute_map = route.get("recompute_segment_length_by_layer", {}) or {}

    if num_layers <= 0:
        errors.append(f"route.num_layers must be positive, got {route.get('num_layers')!r}")
    if fallback_layers:
        errors.append(f"fallback_layers must be empty, got {fallback_layers}")
    if recompute_layers or recompute_map:
        errors.append(f"recompute must be disabled, got layers={recompute_layers} map={recompute_map}")
    if num_layers > 0:
        all_layers = set(range(num_layers))
        patch_set = set(patch_layers)
        reuse_set = set(reuse_layers)
        overlap = sorted(patch_set & reuse_set)
        missing = sorted(all_layers - (patch_set | reuse_set))
        extra = sorted((patch_set | reuse_set) - all_layers)
        if overlap:
            errors.append(f"patch_layers and reuse_layers overlap: {overlap}")
        if missing:
            errors.append(f"patch_layers plus reuse_layers do not cover layers: {missing}")
        if extra:
            errors.append(f"route contains layers outside 0..{num_layers - 1}: {extra}")
        if require_mixed and (not patch_set or not reuse_set):
            errors.append("route must be mixed: both patch_layers and reuse_layers must be non-empty")
    details = {
        "path": str(path),
        "num_layers": num_layers,
        "patch_layers": patch_layers,
        "reuse_layers": reuse_layers,
        "patch_count": len(patch_layers),
        "reuse_count": len(reuse_layers),
        "fallback_count": len(fallback_layers),
        "recompute_count": len(recompute_layers),
    }
    if errors:
        return DoctorCheck("route_strictness", "fail", "; ".join(errors), details)
    return DoctorCheck("route_strictness", "pass", "strict reuse+patch route validated", details)


def _int_list(route: dict[str, Any], key: str) -> list[int]:
    return [int(layer) for layer in route.get(key, [])]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scd-vllm doctor",
        description="Check SCD-vLLM plugin prerequisites without launching GPU work.",
    )
    parser.add_argument("--producer-model", default="")
    parser.add_argument("--consumer-model", default="")
    parser.add_argument("--reuse-ckpt", required=True)
    parser.add_argument("--patch-ckpt", required=True)
    parser.add_argument("--route", required=True)
    parser.add_argument("--require-local-models", action="store_true")
    parser.add_argument("--allow-non-mixed-route", action="store_true")
    parser.add_argument("--skip-vllm-import", action="store_true")
    parser.add_argument("--out", default="", help="Optional JSON output path")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    payload = run_doctor(
        producer_model=args.producer_model,
        consumer_model=args.consumer_model,
        reuse_ckpt=args.reuse_ckpt,
        patch_ckpt=args.patch_ckpt,
        route=args.route,
        require_local_models=args.require_local_models,
        require_mixed=not args.allow_non_mixed_route,
        skip_vllm_import=args.skip_vllm_import,
    )
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["pass"] else 1

