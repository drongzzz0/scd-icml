from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .metrics import ppl_delta
from .schema import RoutePlan, route_fingerprint


BASELINE_FIELDS = [
    "route_name",
    "status",
    "patch_layers",
    "reuse_layers",
    "source_layer_mapping",
    "kv_drift_mean",
    "hidden_drift_mean",
    "logit_kl",
    "ppl",
    "ppl_delta",
    "avg_nll",
    "avg_nll_delta",
    "f1",
    "em",
    "payload_bytes",
    "estimated_ttft_ms",
    "route_fingerprint",
    "route_path",
    "summary_path",
]


def _load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _method_rows(summary: Any, *, direction: str, dataset: str) -> dict[str, dict[str, Any]]:
    if isinstance(summary, dict):
        if "rows" in summary:
            summary = summary["rows"]
        elif "results" in summary:
            summary = summary["results"]
    if not isinstance(summary, list):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for row in summary:
        if str(row.get("direction", direction)) == direction and str(row.get("dataset", dataset)) == dataset:
            out[str(row.get("method", row.get("route_name", "")))] = dict(row)
    return out


def baseline_row_from_eval(
    *,
    route_name: str,
    route_path: str | Path | None,
    summary_path: str | Path | None,
    method: str,
    direction: str,
    dataset: str,
) -> dict[str, Any]:
    route_plan: RoutePlan | None = None
    route_payload: dict[str, Any] | None = None
    if route_path:
        route_payload = _load_json(route_path)
        route_plan = RoutePlan.from_dict(route_payload)

    row: dict[str, Any] = {
        "route_name": route_name,
        "status": "missing_summary",
        "patch_layers": [] if route_plan is None else route_plan.patch_layers,
        "reuse_layers": [] if route_plan is None else route_plan.reuse_layers,
        "source_layer_mapping": [] if route_plan is None else route_plan.source_layer_mapping,
        "kv_drift_mean": None,
        "hidden_drift_mean": None,
        "logit_kl": None,
        "ppl": None,
        "ppl_delta": None,
        "avg_nll": None,
        "avg_nll_delta": None,
        "f1": None,
        "em": None,
        "payload_bytes": None,
        "estimated_ttft_ms": None,
        "route_fingerprint": "" if route_plan is None else route_fingerprint(route_plan),
        "route_path": "" if route_path is None else str(route_path),
        "summary_path": "" if summary_path is None else str(summary_path),
    }
    if not summary_path or not Path(summary_path).is_file():
        return row
    methods = _method_rows(_load_json(summary_path), direction=direction, dataset=dataset)
    full = methods.get("full_consumer", {})
    selected = methods.get(method, {})
    if not selected:
        row["status"] = "missing_method"
        return row
    full_ppl = full.get("ppl")
    selected_ppl = selected.get("ppl")
    full_nll = full.get("avg_nll")
    selected_nll = selected.get("avg_nll")
    row.update(
        {
            "status": "ok",
            "kv_drift_mean": selected.get("kv_drift_mean", selected.get("kv_drift")),
            "hidden_drift_mean": selected.get("hidden_drift_mean", selected.get("hidden_drift")),
            "logit_kl": selected.get("logit_kl"),
            "ppl": selected_ppl,
            "ppl_delta": None if full_ppl is None or selected_ppl is None else ppl_delta(float(selected_ppl), float(full_ppl)),
            "avg_nll": selected_nll,
            "avg_nll_delta": None if full_nll is None or selected_nll is None else float(selected_nll) - float(full_nll),
            "f1": selected.get("f1"),
            "em": selected.get("em"),
            "payload_bytes": selected.get("payload_bytes", selected.get("bytes_transferred")),
            "estimated_ttft_ms": selected.get("estimated_ttft_ms", selected.get("ttft_ms")),
        }
    )
    return row


def write_baseline_outputs(rows: list[dict[str, Any]], out_json: str | Path, out_csv: str | Path) -> None:
    json_path = Path(out_json)
    csv_path = Path(out_csv)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps({"rows": rows}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=BASELINE_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in BASELINE_FIELDS})
