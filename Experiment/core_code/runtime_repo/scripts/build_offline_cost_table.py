#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


TIME_FIELDS = {
    "User time (seconds)": "user_seconds",
    "System time (seconds)": "system_seconds",
    "Elapsed (wall clock) time (h:mm:ss or m:ss)": "elapsed_raw",
    "Maximum resident set size (kbytes)": "max_rss_kb",
    "Exit status": "exit_status",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build an offline calibration/training cost table from raw "
            "/usr/bin/time -v logs."
        )
    )
    parser.add_argument(
        "--entry",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="Time log entry; may be repeated. LABEL is used as the table row id.",
    )
    parser.add_argument("--out-dir", required=True)
    return parser


def maybe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def fmt(value: Any, digits: int = 3) -> str:
    number = maybe_float(value)
    if number is None:
        return ""
    return f"{number:.{digits}f}"


def parse_elapsed_seconds(value: str | None) -> float | None:
    if not value:
        return None
    parts = value.strip().split(":")
    try:
        if len(parts) == 2:
            minutes, seconds = parts
            return int(minutes) * 60 + float(seconds)
        if len(parts) == 3:
            hours, minutes, seconds = parts
            return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    except ValueError:
        return None
    return None


def seconds_to_hms(value: float | None) -> str:
    if value is None:
        return ""
    total = int(round(value))
    hours, rem = divmod(total, 3600)
    minutes, seconds = divmod(rem, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    return f"{minutes}:{seconds:02d}"


def parse_time_file(path: Path) -> dict[str, Any]:
    row: dict[str, Any] = {
        "path": str(path),
        "status": "missing",
        "user_seconds": None,
        "system_seconds": None,
        "elapsed_raw": "",
        "elapsed_seconds": None,
        "elapsed_hms": "",
        "max_rss_kb": None,
        "max_rss_gib": None,
        "exit_status": "",
    }
    if not path.exists():
        return row
    row["status"] = "no_time_fields"
    text = path.read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines():
        stripped = line.strip()
        for key, field in TIME_FIELDS.items():
            prefix = f"{key}:"
            if stripped.startswith(prefix):
                row[field] = stripped[len(prefix) :].strip()
                break
    if row["elapsed_raw"]:
        row["status"] = "ok"
        row["elapsed_seconds"] = parse_elapsed_seconds(str(row["elapsed_raw"]))
        row["elapsed_hms"] = seconds_to_hms(maybe_float(row["elapsed_seconds"]))
    for field in ("user_seconds", "system_seconds", "max_rss_kb"):
        row[field] = maybe_float(row[field])
    if row["max_rss_kb"] is not None:
        row["max_rss_gib"] = float(row["max_rss_kb"]) / (1024 * 1024)
    return row


def parse_entry(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"--entry must be LABEL=PATH: {value}")
    label, path = value.split("=", 1)
    label = label.strip()
    if not label:
        raise ValueError(f"--entry label is empty: {value}")
    return label, Path(path)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_md(path: Path, title: str, rows: list[dict[str, Any]], fields: list[str]) -> None:
    lines = [f"# {title}", ""]
    if rows:
        lines.append("| " + " | ".join(fields) + " |")
        lines.append("|" + "|".join("---" for _ in fields) + "|")
        for row in rows:
            lines.append("| " + " | ".join(str(row.get(field, "")) for field in fields) + " |")
    else:
        lines.append("No rows.")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def svg_escape(value: Any) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def build_elapsed_svg(rows: list[dict[str, Any]]) -> str:
    data = [row for row in rows if maybe_float(row.get("elapsed_seconds")) is not None]
    width = 860
    height = max(280, 70 + 36 * len(data))
    left = 260
    right = 45
    top = 45
    row_h = 32
    plot_w = width - left - right
    max_seconds = max((float(row["elapsed_seconds"]) for row in data), default=1.0)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2:.1f}" y="26" text-anchor="middle" font-family="Arial" font-size="18" font-weight="700">Offline Cost Wall Time</text>',
    ]
    for idx, row in enumerate(data):
        y = top + idx * row_h
        seconds = float(row["elapsed_seconds"])
        bar_w = 0.0 if max_seconds <= 0 else seconds / max_seconds * plot_w
        parts.append(
            f'<text x="{left - 12}" y="{y + 18}" text-anchor="end" font-family="Arial" font-size="12">{svg_escape(row["label"])}</text>'
        )
        parts.append(f'<rect x="{left}" y="{y + 5}" width="{bar_w:.1f}" height="18" fill="#1f77b4"/>')
        parts.append(
            f'<text x="{left + bar_w + 6:.1f}" y="{y + 18}" font-family="Arial" font-size="12">{svg_escape(row["elapsed_hms"])}</text>'
        )
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def normalize_rows(entries: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for entry in entries:
        label, path = parse_entry(entry)
        parsed = parse_time_file(path)
        elapsed_seconds = maybe_float(parsed.get("elapsed_seconds"))
        max_rss_gib = maybe_float(parsed.get("max_rss_gib"))
        rows.append(
            {
                "label": label,
                "status": parsed["status"],
                "elapsed_seconds": fmt(elapsed_seconds),
                "elapsed_hms": parsed.get("elapsed_hms", ""),
                "user_seconds": fmt(parsed.get("user_seconds")),
                "system_seconds": fmt(parsed.get("system_seconds")),
                "max_rss_gib": fmt(max_rss_gib),
                "exit_status": parsed.get("exit_status", ""),
                "path": parsed["path"],
            }
        )
    return rows


def main() -> None:
    args = build_parser().parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = normalize_rows(args.entry)
    fields = [
        "label",
        "status",
        "elapsed_seconds",
        "elapsed_hms",
        "user_seconds",
        "system_seconds",
        "max_rss_gib",
        "exit_status",
        "path",
    ]
    write_csv(out_dir / "offline_cost_table.csv", rows, fields)
    write_md(out_dir / "offline_cost_table.md", "Offline Cost Table", rows, fields)
    (out_dir / "offline_cost_wall_time.svg").write_text(build_elapsed_svg(rows), encoding="utf-8")
    (out_dir / "offline_cost_summary.json").write_text(
        json.dumps({"rows": rows}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (out_dir / "README.md").write_text(
        "\n".join(
            [
                "# Offline Cost Evidence",
                "",
                "This directory contains a generated offline calibration/training cost table.",
                "Rows are derived from raw `/usr/bin/time -v` logs supplied through `--entry`.",
                "",
                "Files:",
                "",
                "- `offline_cost_table.csv`: machine-readable cost table.",
                "- `offline_cost_table.md`: Markdown view of the same rows.",
                "- `offline_cost_wall_time.svg`: wall-clock time chart for rows with valid elapsed time.",
                "- `offline_cost_summary.json`: JSON copy of normalized rows.",
                "",
                "`status=no_time_fields` means the source file existed but did not contain the standard `/usr/bin/time -v` fields.",
                "",
            ]
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
