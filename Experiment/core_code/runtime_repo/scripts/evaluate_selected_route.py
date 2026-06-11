#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from scd_repro.route_selection.config import load_config, write_json
from scd_repro.route_selection.schema import RoutePlan


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate a selected teacher-aligned route and emit strict-eval command metadata.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--route", required=True)
    parser.add_argument("--out", default="")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cfg = load_config(args.config)
    route = RoutePlan.from_dict(__import__("json").loads(Path(args.route).read_text(encoding="utf-8-sig")))
    if not route.patch_layers or not route.reuse_layers:
        raise SystemExit("selected route must be mixed Patch+Reuse before strict SCD evaluation")
    eval_cfg = cfg.get("evaluation", {})
    command = [
        "bash",
        "scripts/run_ser_scd_icml_eval_existing_route.sh",
        "ROUTE_PATH=" + str(Path(args.route)),
    ]
    if eval_cfg.get("run_root"):
        command.append("RUN_ROOT=" + str(eval_cfg["run_root"]))
    if eval_cfg.get("source_run_root"):
        command.append("SOURCE_RUN_ROOT=" + str(eval_cfg["source_run_root"]))
    artifact = {
        "status": "ready_for_strict_eval",
        "route": route.to_dict(),
        "legacy_route": route.to_legacy_route(),
        "strict_eval_command_template": command,
        "note": "This script does not fabricate metrics. Run the command template with the existing strict eval wrapper, then feed completed metrics back to select_patch_reuse_route.py.",
    }
    out = Path(args.out or cfg.get("output", {}).get("dir", "outputs/route_selection") + "/evaluate_selected_route.json")
    write_json(out, artifact)
    print(f"eval_manifest={out}")
    print(" ".join(command))


if __name__ == "__main__":
    main()
