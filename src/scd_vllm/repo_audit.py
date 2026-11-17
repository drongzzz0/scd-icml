from __future__ import annotations

import argparse
import json
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class RepoAudit:
    repo_root: Path
    entry_points: dict[str, str]
    source_modules: list[str]
    scripts: list[str]
    configs: list[str]
    route_files: list[str]
    tests: list[str]
    existing_reports: list[str]
    missing_pieces: list[str]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit SCD runtime repo before vLLM integration work")
    parser.add_argument("--repo-root", default=".", help="Runtime repository root")
    parser.add_argument("--out", default="reports/scd_vllm_repo_audit.md", help="Markdown report path")
    parser.add_argument("--json-out", default="", help="Optional machine-readable JSON audit path")
    return parser.parse_args(argv)


def run_audit(repo_root: str | Path) -> RepoAudit:
    root = Path(repo_root).resolve()
    pyproject = _read_pyproject(root / "pyproject.toml")
    entry_points = dict(pyproject.get("project", {}).get("scripts", {}))
    source_modules = _relative_files(root, "src", "*.py")
    scripts = _relative_files(root, "scripts", "*")
    configs = _relative_files(root, "configs", "*")
    tests = _relative_files(root, "tests", "test_*.py")
    route_files = [p for p in configs if "route" in Path(p).name.lower() or "/routes/" in p.replace("\\", "/")]
    existing_reports = _relative_files(root, "reports", "*.md") if (root / "reports").exists() else []
    missing_pieces = _missing_pieces(root)
    return RepoAudit(
        repo_root=root,
        entry_points=entry_points,
        source_modules=source_modules,
        scripts=scripts,
        configs=configs,
        route_files=route_files,
        tests=tests,
        existing_reports=existing_reports,
        missing_pieces=missing_pieces,
    )


def _read_pyproject(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return tomllib.loads(path.read_text(encoding="utf-8"))


def _relative_files(root: Path, subdir: str, pattern: str) -> list[str]:
    base = root / subdir
    if not base.exists():
        return []
    files = []
    for path in base.rglob(pattern):
        if not path.is_file():
            continue
        rel = str(path.relative_to(root)).replace("\\", "/")
        if "__pycache__/" in rel or rel.endswith((".pyc", ".pyo")):
            continue
        files.append(rel)
    return sorted(files)


def _missing_pieces(root: Path) -> list[str]:
    expected = [
        "src/scd_vllm/kv_spec.py",
        "src/scd_vllm/bundle.py",
        "src/scd_vllm/env_probe.py",
        "src/scd_vllm/hf_export.py",
        "src/scd_vllm/vllm_baseline.py",
        "src/scd_vllm/truekv_oracle.py",
        "src/scd_vllm/injection.py",
        "configs/scd_vllm_mvp.yaml",
        "scripts/scd_vllm/run_02_hf_export.sh",
        "scripts/scd_vllm/run_03_vllm_baseline.sh",
        "scripts/scd_vllm/run_04_truekv_oracle.sh",
        "reports/scd_vllm_status.md",
    ]
    return [item for item in expected if not (root / item).exists()]


def render_markdown(audit: RepoAudit) -> str:
    existing_capture = [p for p in audit.source_modules + audit.scripts if "capture" in p.lower()]
    existing_patch = [p for p in audit.source_modules + audit.scripts if "patch" in p.lower()]
    existing_reuse = [p for p in audit.source_modules + audit.scripts if "reuse" in p.lower()]
    existing_eval = [p for p in audit.source_modules + audit.scripts if "eval" in p.lower()]
    lines = [
        "# SCD vLLM Repo Audit",
        "",
        "- repo_root: `.`",
        f"- python: `{sys.version.split()[0]}`",
        "",
        "## Existing entry points",
        "",
        _format_mapping(audit.entry_points),
        "",
        "## Existing model/config files",
        "",
        _format_list(audit.configs[:80]),
        "",
        "## Existing route files",
        "",
        _format_list(audit.route_files[:80]),
        "",
        "## Existing KV capture/export candidates",
        "",
        _format_list(existing_capture[:80]),
        "",
        "## Existing Reuse/Patch candidates",
        "",
        "### Reuse",
        "",
        _format_list(existing_reuse[:80]),
        "",
        "### Patch",
        "",
        _format_list(existing_patch[:80]),
        "",
        "## Existing metrics/eval candidates",
        "",
        _format_list(existing_eval[:80]),
        "",
        "## Existing tests",
        "",
        _format_list(audit.tests[:80]),
        "",
        "## Missing pieces",
        "",
        _format_list(audit.missing_pieces),
        "",
        "## Proposed mapping to scd_vllm modules",
        "",
        "- `src/scd_vllm/kv_spec.py`: model/runtime KV compatibility checks.",
        "- `src/scd_vllm/bundle.py`: standard SCDCacheBundle manifest and hash helpers.",
        "- `src/scd_vllm/env_probe.py`: local/A800 dependency, model, and vLLM API probe.",
        "- `src/scd_vllm/hf_export.py`: HF full-consumer reference bundle exporter.",
        "- `src/scd_vllm/vllm_baseline.py`: plain official vLLM baseline latency runner.",
        "- `src/scd_vllm/truekv_oracle.py`: vLLM TrueKV API inspection and future save/load correctness oracle.",
        "- `src/scd_vllm/injection.py`: SCD reconstructed KV injection smoke path, gated by TrueKV and greedy decode checks.",
        "",
        "## Implementation/release risks",
        "",
        "- vLLM runtime phases require a Linux CUDA host; Windows local work should stay static/unit-test only.",
        "- vLLM KVConnector APIs are version-sensitive; keep Phase 4 inspection/oracle gates version-pinned to the installed vLLM source.",
        "- Current SCD evidence uses dataset-specific routes; the vLLM MVP must not assume a universal route.",
        "- This is not a production connector yet; KV lifecycle, concurrency, multi-tenant isolation, and request-level routing still need explicit engineering before serving claims.",
        "",
    ]
    return "\n".join(lines)


def _format_mapping(mapping: dict[str, str]) -> str:
    if not mapping:
        return "- N/A"
    return "\n".join(f"- `{key}` -> `{value}`" for key, value in sorted(mapping.items()))


def _format_list(items: list[str]) -> str:
    if not items:
        return "- N/A"
    return "\n".join(f"- `{item}`" for item in items)


def audit_to_dict(audit: RepoAudit) -> dict[str, Any]:
    return {
        "repo_root": str(audit.repo_root),
        "entry_points": audit.entry_points,
        "source_modules": audit.source_modules,
        "scripts": audit.scripts,
        "configs": audit.configs,
        "route_files": audit.route_files,
        "tests": audit.tests,
        "existing_reports": audit.existing_reports,
        "missing_pieces": audit.missing_pieces,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    audit = run_audit(args.repo_root)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_markdown(audit), encoding="utf-8", newline="\n")
    if args.json_out:
        json_path = Path(args.json_out)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(
            json.dumps(audit_to_dict(audit), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
