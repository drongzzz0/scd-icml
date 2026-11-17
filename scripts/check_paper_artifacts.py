#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


INCLUDE_RE = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate that a paper source references existing generated figures "
            "and that its paper artifact bundle manifest is complete."
        )
    )
    parser.add_argument("--paper-tex", required=True, help="Paper .tex file to inspect")
    parser.add_argument("--base-dir", default="", help="Base directory for relative figure paths; defaults to paper dir")
    parser.add_argument("--bundle-dir", default="", help="Paper artifact bundle directory containing paper_artifact_manifest.json")
    parser.add_argument("--forbid-image", action="append", default=[], help="Forbidden image path as written in TeX, normalized to '/'; may repeat")
    parser.add_argument("--require-image", action="append", default=[], help="Required image path relative to base dir; may repeat")
    parser.add_argument("--require-bundle-file", action="append", default=[], help="Required file path relative to bundle dir; may repeat")
    parser.add_argument("--require-text", action="append", default=[], help="Literal text that must appear in the paper source; may repeat")
    parser.add_argument("--forbid-text", action="append", default=[], help="Literal text that must not appear in the paper source; may repeat")
    parser.add_argument("--out-json", default="", help="Optional machine-readable report path")
    return parser


def resolve_image(base_dir: Path, value: str) -> Path | None:
    raw = Path(value)
    candidates = [raw] if raw.suffix else [raw.with_suffix(suffix) for suffix in [".pdf", ".png", ".jpg", ".jpeg"]]
    for candidate in candidates:
        path = candidate if candidate.is_absolute() else base_dir / candidate
        if path.is_file():
            return path
    return None


def parse_included_images(tex_path: Path) -> list[str]:
    text = tex_path.read_text(encoding="utf-8")
    return [match.group(1).strip() for match in INCLUDE_RE.finditer(text)]


def check_bundle(bundle_dir: Path, required_files: list[str], errors: list[str]) -> dict[str, Any]:
    manifest_path = bundle_dir / "paper_artifact_manifest.json"
    if not manifest_path.is_file():
        errors.append(f"missing bundle manifest: {manifest_path}")
        return {}
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    except Exception as exc:
        errors.append(f"invalid bundle manifest {manifest_path}: {exc}")
        return {}
    if manifest.get("status") != "ok":
        errors.append(f"bundle manifest status is not ok: {manifest.get('status')!r}")
    required_missing = [row for row in manifest.get("missing", []) if row.get("required") == "true"]
    if required_missing:
        errors.append(f"bundle manifest has required missing entries: {len(required_missing)}")
    for rel in required_files:
        if not (bundle_dir / rel).is_file():
            errors.append(f"required bundle file missing: {rel}")
    return manifest


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    tex_path = Path(args.paper_tex)
    base_dir = Path(args.base_dir) if args.base_dir else tex_path.parent
    errors: list[str] = []

    tex_text = ""
    if not tex_path.is_file():
        errors.append(f"paper tex missing: {tex_path}")
        included: list[str] = []
    else:
        tex_text = tex_path.read_text(encoding="utf-8")
        included = parse_included_images(tex_path)

    resolved: list[dict[str, str]] = []
    missing_images: list[str] = []
    for image in included:
        path = resolve_image(base_dir, image)
        if path is None:
            missing_images.append(image)
        else:
            resolved.append({"source": image, "path": str(path)})
    for image in missing_images:
        errors.append(f"included image missing: {image}")

    included_norms = {image.replace("\\", "/") for image in included}
    for forbidden in args.forbid_image:
        forbidden_norm = forbidden.replace("\\", "/")
        if forbidden_norm in included_norms:
            errors.append(f"forbidden image referenced: {forbidden}")

    for required in args.require_image:
        if resolve_image(base_dir, required) is None:
            errors.append(f"required image missing: {required}")

    for required_text in args.require_text:
        if required_text not in tex_text:
            errors.append(f"required text missing: {required_text}")
    for forbidden_text in args.forbid_text:
        if forbidden_text in tex_text:
            errors.append(f"forbidden text present: {forbidden_text}")

    manifest: dict[str, Any] = {}
    if args.bundle_dir:
        manifest = check_bundle(Path(args.bundle_dir), args.require_bundle_file, errors)

    report = {
        "paper_tex": str(tex_path),
        "base_dir": str(base_dir),
        "included_images": included,
        "resolved_images": resolved,
        "missing_images": missing_images,
        "bundle_status": manifest.get("status", "") if manifest else "",
        "required_text": args.require_text,
        "forbidden_text": args.forbid_text,
        "errors": errors,
    }
    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 2
    print(f"paper_artifacts_ok={tex_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
