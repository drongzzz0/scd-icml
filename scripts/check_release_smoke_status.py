from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect a SCD-ICML release smoke or reproduction run root")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--pid", type=int, default=0, help="optional wrapper PID to check")
    parser.add_argument("--tail-lines", type=int, default=40)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_tail(path: Path, lines: int) -> list[str]:
    if not path.is_file():
        return []
    text = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return text[-max(1, int(lines)) :]


def pid_running(pid: int) -> bool | None:
    if pid <= 0:
        return None
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def collect_statuses(run_root: Path) -> dict[str, dict[str, Any]]:
    statuses: dict[str, dict[str, Any]] = {}
    logs = run_root / "logs"
    if not logs.exists():
        return statuses
    for path in sorted(logs.glob("*/status.json")):
        try:
            payload = read_json(path)
        except Exception as exc:
            payload = {"stage": path.parent.name, "success": False, "error": f"invalid status json: {exc}"}
        stage = str(payload.get("stage") or path.parent.name)
        statuses[stage] = payload
    return statuses


def expected_stages(run_config: dict[str, Any]) -> list[str]:
    directions = [part.strip() for part in str(run_config.get("directions", "")).split(",") if part.strip()]
    datasets = [part.strip() for part in str(run_config.get("datasets", "")).split(",") if part.strip()]
    route_mode = str(run_config.get("route_mode", "patch_meta"))
    if route_mode == "patch_meta":
        route_stage = "route_patch_meta"
    elif route_mode == "paper_profile":
        route_stage = "route_paper_scd"
    elif route_mode == "build":
        route_stage = "build_route"
    else:
        route_stage = "route_template"
    stages: list[str] = []
    for direction in directions:
        stages.extend(
            [
                f"{direction}_capture_pair",
                f"{direction}_train_reuse",
                f"{direction}_profile_layers",
                f"{direction}_train_patch",
                f"{direction}_{route_stage}",
            ]
        )
        stages.extend(f"{direction}_eval_{dataset}" for dataset in datasets)
    return stages


def classify(
    *,
    run_root: Path,
    run_config: dict[str, Any],
    statuses: dict[str, dict[str, Any]],
    tail: list[str],
    is_running: bool | None,
) -> tuple[str, list[str]]:
    notes: list[str] = []
    failed = [stage for stage, payload in statuses.items() if payload.get("success") is False]
    if failed:
        return "failed", [f"failed stages: {', '.join(failed)}"]

    expected = expected_stages(run_config)
    missing = [stage for stage in expected if stage not in statuses]
    summary_json = run_root / "scd_icml_mistral_summary.json"
    summary_md = run_root / "scd_icml_mistral_summary.md"
    if expected and not missing and summary_json.is_file() and summary_md.is_file():
        return "complete", ["all expected stages and aggregate summaries are present"]

    joined_tail = "\n".join(tail)
    waiting_patterns = [
        r"waiting to select two GPUs",
        r"waiting for GPU free memory",
    ]
    if any(re.search(pattern, joined_tail, flags=re.IGNORECASE) for pattern in waiting_patterns):
        notes.append("log tail indicates GPU wait")
        return "waiting_gpu", notes

    if is_running is True:
        if missing:
            notes.append(f"missing stages: {', '.join(missing[:8])}" + (" ..." if len(missing) > 8 else ""))
        return "running", notes

    if missing:
        notes.append(f"missing stages: {', '.join(missing[:8])}" + (" ..." if len(missing) > 8 else ""))
    if is_running is False:
        notes.append("process is not running")
    return "incomplete", notes


def main() -> None:
    args = parse_args()
    run_root = Path(args.run_root)
    run_config_path = run_root / "run_config.json"
    log_path = Path(str(run_root) + ".nohup.log")

    run_config: dict[str, Any] = {}
    if run_config_path.is_file():
        run_config = read_json(run_config_path)
    statuses = collect_statuses(run_root)
    tail = read_tail(log_path, args.tail_lines)
    is_running = pid_running(args.pid)
    state, notes = classify(
        run_root=run_root,
        run_config=run_config,
        statuses=statuses,
        tail=tail,
        is_running=is_running,
    )
    payload = {
        "state": state,
        "run_root": str(run_root),
        "pid": args.pid or None,
        "pid_running": is_running,
        "status_count": len(statuses),
        "statuses": statuses,
        "expected_stages": expected_stages(run_config) if run_config else [],
        "summary_json_exists": (run_root / "scd_icml_mistral_summary.json").is_file(),
        "summary_md_exists": (run_root / "scd_icml_mistral_summary.md").is_file(),
        "nohup_log": str(log_path),
        "tail": tail,
        "notes": notes,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
