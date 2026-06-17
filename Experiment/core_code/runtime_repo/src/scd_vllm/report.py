from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build SCD-vLLM status report from phase artifacts")
    parser.add_argument("--run-dir", default="runs/scd_vllm")
    parser.add_argument("--repo-audit", default="reports/scd_vllm_repo_audit.md")
    parser.add_argument("--env-probe", default="runs/scd_vllm/env_probe.json")
    parser.add_argument("--out", default="reports/scd_vllm_status.md")
    return parser.parse_args(argv)


def render_report(args: argparse.Namespace) -> str:
    env_probe = _load_json(Path(args.env_probe))
    hf_bundles = _discover_hf_bundles(Path(args.run_dir))
    vllm_baselines = _discover_vllm_baselines(Path(args.run_dir))
    truekv_oracles = _discover_truekv_oracles(Path(args.run_dir))
    injections = _discover_injections(Path(args.run_dir))
    lines = [
        "# SCD vLLM Integration Status",
        "",
        "## Environment",
        "",
    ]
    if env_probe:
        packages = env_probe.get("packages", {})
        lines.extend(
            [
                f"- python: `{str(env_probe.get('python', '')).split()[0]}`",
                f"- platform: `{env_probe.get('platform')}`",
                f"- torch: `{packages.get('torch')}`",
                f"- transformers: `{packages.get('transformers')}`",
                f"- vllm: `{packages.get('vllm')}`",
                f"- cuda_available: `{env_probe.get('cuda', {}).get('cuda_available')}`",
                "",
            ]
        )
    else:
        lines.extend(["- N/A; env probe has not been run.", ""])
    lines.extend(
        [
            "## Phase Results",
            "",
            _phase_status("Phase 0 Repo Audit", Path(args.repo_audit).exists()),
            _phase_status("Phase 1 Env Probe", bool(env_probe)),
            _phase_status("Phase 2 HF Export", bool(hf_bundles), _hf_export_note(hf_bundles)),
            _phase_status("Phase 3 vLLM Baseline", bool(vllm_baselines), _vllm_baseline_note(vllm_baselines)),
            _phase_status("Phase 4 TrueKV Oracle", _truekv_oracle_passed(truekv_oracles), _truekv_oracle_note(truekv_oracles)),
            _phase_status("Phase 5 SCD Injection", _injection_passed(injections), _injection_note(injections)),
            "",
            "## Blockers",
            "",
        ]
    )
    blockers = _blockers(env_probe)
    lines.extend(blockers if blockers else ["- None recorded for completed scaffold phases."])
    lines.extend(
        [
            "",
            "## Reproduction Commands",
            "",
            "```bash",
            "bash scripts/scd_vllm/run_00_repo_audit.sh",
            "bash scripts/scd_vllm/run_01_env_probe.sh",
            "bash scripts/scd_vllm/run_02_hf_export.sh",
            "bash scripts/scd_vllm/run_03_vllm_baseline.sh",
            "bash scripts/scd_vllm/run_04_truekv_oracle.sh",
            "bash scripts/scd_vllm/run_05_reconstructed_export.sh",
            "bash scripts/scd_vllm/run_05_scd_injection.sh",
            "bash scripts/scd_vllm/run_06_report.sh",
            "```",
            "",
            "## Next Steps",
            "",
            *_next_steps(
                bool(hf_bundles),
                bool(vllm_baselines),
                _truekv_oracle_passed(truekv_oracles),
                _injection_passed(injections),
                _adapter_smoke_passed(injections),
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _discover_hf_bundles(run_dir: Path) -> list[dict[str, Any]]:
    if not run_dir.exists():
        return []
    bundles: list[dict[str, Any]] = []
    for manifest_path in sorted(run_dir.glob("**/manifest.json")):
        data = _load_json(manifest_path)
        if not data or data.get("bundle_type") != "full_consumer":
            continue
        metrics = _load_json(manifest_path.parent / "metrics.json") or {}
        bundles.append(
            {
                "path": str(manifest_path.parent),
                "num_tokens": data.get("num_tokens"),
                "num_layers": metrics.get("num_layers"),
                "payload_bytes": metrics.get("payload_bytes"),
            }
        )
    return bundles


def _discover_vllm_baselines(run_dir: Path) -> list[dict[str, Any]]:
    if not run_dir.exists():
        return []
    baselines: list[dict[str, Any]] = []
    for metrics_path in sorted(run_dir.glob("**/metrics.json")):
        data = _load_json(metrics_path)
        if not data or data.get("mode") != "vllm_plain_baseline":
            continue
        summary = data.get("summary", {}) if isinstance(data.get("summary"), dict) else {}
        baselines.append(
            {
                "path": str(metrics_path.parent),
                "prompt_token_count": data.get("prompt_token_count"),
                "max_new_tokens": data.get("max_new_tokens"),
                "generation_seconds_mean": summary.get("generation_seconds_mean"),
                "ttft_seconds_mean": summary.get("ttft_seconds_mean"),
                "ttft_proxy_seconds_mean": summary.get("ttft_proxy_seconds_mean"),
                "tokens_per_second_mean": summary.get("tokens_per_second_mean"),
            }
        )
    return baselines


def _discover_truekv_oracles(run_dir: Path) -> list[dict[str, Any]]:
    if not run_dir.exists():
        return []
    oracles: list[dict[str, Any]] = []
    for metrics_path in sorted(run_dir.glob("**/metrics.json")):
        data = _load_json(metrics_path)
        if not data or not str(data.get("mode", "")).startswith("truekv_oracle"):
            continue
        oracles.append(
            {
                "path": str(metrics_path.parent),
                "mode": data.get("mode"),
                "oracle_pass": data.get("oracle_pass"),
                "vllm_version": data.get("vllm_version"),
                "blocker": data.get("blocker"),
                "save_safetensor_files": data.get("save_safetensor_files"),
                "load_safetensor_files": data.get("load_safetensor_files"),
                "greedy_1tok_match": data.get("greedy_1tok_match"),
                "greedy_16tok_match": data.get("greedy_16tok_match"),
            }
        )
    return oracles


def _discover_injections(run_dir: Path) -> list[dict[str, Any]]:
    if not run_dir.exists():
        return []
    injections: list[dict[str, Any]] = []
    for metrics_path in sorted(run_dir.glob("**/metrics.json")):
        data = _load_json(metrics_path)
        if not data or data.get("mode") != "scd_vllm_injection_smoke":
            continue
        injections.append(
            {
                "path": str(metrics_path.parent),
                "adapter_smoke_pass": data.get("adapter_smoke_pass"),
                "scd_injection_pass": data.get("scd_injection_pass"),
                "reconstructed_scd": data.get("reconstructed_scd"),
                "bundle_type": data.get("bundle_type"),
                "bundle_token_match": data.get("bundle_token_match"),
                "aligned_tokens": data.get("aligned_tokens"),
                "num_layers": data.get("num_layers"),
                "payload_bytes": data.get("payload_bytes"),
                "greedy_1tok_match": data.get("greedy_1tok_match"),
                "greedy_ntok_match": data.get("greedy_ntok_match"),
                "mtime": metrics_path.stat().st_mtime,
            }
        )
    return injections


def _hf_export_note(bundles: list[dict[str, Any]]) -> str:
    if not bundles:
        return "no full-consumer bundle manifest found"
    latest = bundles[-1]
    return (
        f"{latest['path']} tokens={latest.get('num_tokens')} "
        f"layers={latest.get('num_layers')} payload_bytes={latest.get('payload_bytes')}"
    )


def _vllm_baseline_note(baselines: list[dict[str, Any]]) -> str:
    if not baselines:
        return "no vLLM baseline metrics found"
    latest = baselines[-1]
    return (
        f"{latest['path']} prompt_tokens={latest.get('prompt_token_count')} "
        f"max_new_tokens={latest.get('max_new_tokens')} "
        f"mean_generation_seconds={latest.get('generation_seconds_mean')} "
        f"mean_ttft_seconds={latest.get('ttft_seconds_mean')} "
        f"mean_ttft_proxy_seconds={latest.get('ttft_proxy_seconds_mean')}"
    )


def _truekv_oracle_passed(oracles: list[dict[str, Any]]) -> bool:
    return any(item.get("oracle_pass") is True for item in oracles)


def _truekv_oracle_note(oracles: list[dict[str, Any]]) -> str:
    if not oracles:
        return "no TrueKV oracle or API inspection metrics found"
    latest = _select_truekv_oracle(oracles)
    return (
        f"{latest.get('path')} mode={latest.get('mode')} oracle_pass={latest.get('oracle_pass')} "
        f"vllm={latest.get('vllm_version')} save_files={latest.get('save_safetensor_files')} "
        f"load_files={latest.get('load_safetensor_files')} greedy_1tok={latest.get('greedy_1tok_match')} "
        f"greedy_16tok={latest.get('greedy_16tok_match')} blocker={latest.get('blocker')}"
    )


def _select_truekv_oracle(oracles: list[dict[str, Any]]) -> dict[str, Any]:
    def score(item: dict[str, Any]) -> tuple[int, int]:
        mode = str(item.get("mode", ""))
        return (
            1 if item.get("oracle_pass") is True else 0,
            1 if mode == "truekv_oracle" else 0,
        )

    return max(oracles, key=score)


def _injection_passed(injections: list[dict[str, Any]]) -> bool:
    return any(item.get("scd_injection_pass") is True for item in injections)


def _adapter_smoke_passed(injections: list[dict[str, Any]]) -> bool:
    return any(item.get("adapter_smoke_pass") is True for item in injections)


def _injection_note(injections: list[dict[str, Any]]) -> str:
    if not injections:
        return "no SCD/vLLM injection metrics found"
    latest = _select_injection(injections)
    return (
        f"{latest.get('path')} bundle_type={latest.get('bundle_type')} "
        f"adapter_smoke_pass={latest.get('adapter_smoke_pass')} "
        f"scd_injection_pass={latest.get('scd_injection_pass')} "
        f"aligned_tokens={latest.get('aligned_tokens')} layers={latest.get('num_layers')} "
        f"payload_bytes={latest.get('payload_bytes')} greedy_1tok={latest.get('greedy_1tok_match')} "
        f"greedy_ntok={latest.get('greedy_ntok_match')}"
    )


def _select_injection(injections: list[dict[str, Any]]) -> dict[str, Any]:
    def score(item: dict[str, Any]) -> tuple[int, int, int, float]:
        return (
            1 if item.get("scd_injection_pass") is True else 0,
            1 if item.get("adapter_smoke_pass") is True else 0,
            1 if item.get("reconstructed_scd") is True else 0,
            float(item.get("mtime") or 0.0),
        )

    return max(injections, key=score)


def _next_steps(
    hf_export_done: bool,
    vllm_baseline_done: bool,
    truekv_oracle_passed: bool,
    scd_injection_passed: bool,
    adapter_smoke_passed: bool,
) -> list[str]:
    if not hf_export_done:
        return [
            "- Run Phase 2 HF full-consumer export on a Linux CUDA host after the model pair passes KVSpec compatibility.",
            "- Do not implement SCD injection until TrueKV save/load oracle has an explicit pass or a documented blocker.",
        ]
    if not vllm_baseline_done:
        return [
            "- Implement a plain vLLM baseline for the same prompt/model pair and record TTFT plus output-token latency.",
            "- Implement TrueKV save/load oracle before any SCD reconstructed KV injection.",
            "- Keep SCD injection skipped until the TrueKV oracle has an explicit pass or a documented blocker.",
        ]
    if not truekv_oracle_passed:
        return [
            "- Implement TrueKV save/load oracle using the inspected vLLM KVConnector API.",
            "- Keep SCD injection skipped until the TrueKV oracle has an explicit pass or a documented blocker.",
        ]
    if scd_injection_passed:
        return [
            "- Measure latency and quality for the passing reconstructed-SCD vLLM injection path.",
        ]
    if adapter_smoke_passed:
        return [
            "- Replace the full-consumer adapter smoke bundle with a reconstructed SCD bundle and rerun Phase 5.",
            "- Keep the full-consumer adapter smoke as a bridge regression test, not as a paper speedup result.",
        ]
    return [
        "- Run Phase 5 adapter smoke with a long-prefix full-consumer bundle, then replace it with a reconstructed SCD bundle.",
    ]


def _phase_status(name: str, passed: bool, note: str = "") -> str:
    status = "PASS" if passed else "PENDING"
    suffix = f" - {note}" if note else ""
    return f"- {name}: {status}{suffix}"


def _blockers(env_probe: dict[str, Any] | None) -> list[str]:
    out: list[str] = []
    if env_probe:
        if env_probe.get("packages", {}).get("vllm") is None:
            out.append("- vLLM is not installed in the current environment; runtime phases require a Linux CUDA environment with vLLM installed.")
        errors = env_probe.get("compatibility", {}).get("errors") or []
        out.extend(f"- {err}" for err in errors)
    return out


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_report(args), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
