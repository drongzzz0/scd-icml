#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scd_repro.route_selection.config import load_config
from scd_repro.route_selection.profiler import baseline_row_from_eval, write_baseline_outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build baseline route diagnostic tables from completed strict eval outputs.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--routes-manifest", default="", help="JSON list of route eval entries; overrides config baseline.routes_manifest")
    return parser


def _load_manifest(path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if isinstance(payload, dict):
        payload = payload.get("routes", payload.get("candidates", []))
    if not isinstance(payload, list):
        raise ValueError("routes manifest must be a JSON list or object with routes/candidates")
    return [dict(row) for row in payload]


def main() -> None:
    args = build_parser().parse_args()
    cfg = load_config(args.config)
    output_dir = Path(cfg.get("output", {}).get("dir", "outputs/route_selection"))
    baseline_cfg = cfg.get("baseline", {})
    manifest_path = args.routes_manifest or baseline_cfg.get("routes_manifest") or ""
    direction = str(baseline_cfg.get("direction", "lite_to_7b"))
    dataset = str(baseline_cfg.get("dataset", "cmrc2018"))
    rows: list[dict[str, Any]] = []
    if manifest_path:
        for entry in _load_manifest(manifest_path):
            rows.append(
                baseline_row_from_eval(
                    route_name=str(entry.get("route_name") or entry.get("name") or entry.get("method") or "route"),
                    route_path=entry.get("route_path") or entry.get("route"),
                    summary_path=entry.get("summary_path") or entry.get("summary"),
                    method=str(entry.get("method", "scd_full")),
                    direction=str(entry.get("direction", direction)),
                    dataset=str(entry.get("dataset", dataset)),
                )
            )
    else:
        rows.append(
            {
                "route_name": "UNCONFIGURED",
                "status": "no_routes_manifest",
                "patch_layers": [],
                "reuse_layers": [],
                "source_layer_mapping": [],
                "kv_drift_mean": None,
                "hidden_drift_mean": None,
                "logit_kl": None,
                "ppl": None,
                "ppl_delta": None,
                "f1": None,
                "payload_bytes": None,
                "estimated_ttft_ms": None,
                "route_fingerprint": "",
                "route_path": "",
                "summary_path": "",
            }
        )
    out_json = output_dir / "baseline_diagnostic.json"
    out_csv = output_dir / "baseline_diagnostic.csv"
    write_baseline_outputs(rows, out_json, out_csv)
    print(f"baseline_json={out_json}")
    print(f"baseline_csv={out_csv}")
    print("note=missing teacher drift fields remain null unless completed evaluator summaries provide them")


if __name__ == "__main__":
    main()
