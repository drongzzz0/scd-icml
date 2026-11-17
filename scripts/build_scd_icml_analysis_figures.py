#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build reproducible SCD-ICML analysis figures from a route profile "
            "JSON. The script uses only stored profile metrics."
        )
    )
    parser.add_argument("--profile", required=True, help="profile_cmrc2018_seed42.json")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--title-prefix", default="SCD-ICML")
    return parser


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def maybe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def fmt(value: Any, digits: int = 6) -> str:
    number = maybe_float(value)
    if number is None:
        return ""
    return f"{number:.{digits}f}"


def svg_escape(value: Any) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def normalize_layers(profile: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in profile.get("layers", []):
        if not isinstance(raw, dict):
            continue
        try:
            layer = int(raw.get("layer"))
        except (TypeError, ValueError):
            continue
        reuse_error = maybe_float(raw.get("reuse_error"))
        patch_error = maybe_float(raw.get("patch_error"))
        expected_gain = maybe_float(raw.get("expected_gain"))
        if expected_gain is None and reuse_error is not None and patch_error is not None:
            expected_gain = reuse_error - patch_error
        rows.append(
            {
                "layer": layer,
                "reuse_error": reuse_error,
                "patch_error": patch_error,
                "expected_gain": expected_gain,
                "patch_xnorm_error": maybe_float(raw.get("patch_xnorm_error")),
                "patch_k_error": maybe_float(raw.get("patch_k_error")),
                "patch_v_error": maybe_float(raw.get("patch_v_error")),
                "patch_kv_error": maybe_float(raw.get("patch_kv_error")),
            }
        )
    return sorted(rows, key=lambda row: int(row["layer"]))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") if field == "layer" else fmt(row.get(field)) for field in fields})


def write_md(path: Path, title: str, rows: list[dict[str, Any]], fields: list[str]) -> None:
    lines = [f"# {title}", ""]
    if not rows:
        lines.extend(["No rows.", ""])
        path.write_text("\n".join(lines), encoding="utf-8")
        return
    lines.append("| " + " | ".join(fields) + " |")
    lines.append("|" + "|".join("---" for _ in fields) + "|")
    for row in rows:
        values = [str(row.get(field, "")) if field == "layer" else fmt(row.get(field)) for field in fields]
        lines.append("| " + " | ".join(values) + " |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def scale_bounds(rows: list[dict[str, Any]], fields: list[str]) -> tuple[float, float, int, int]:
    values: list[float] = []
    layers: list[int] = []
    for row in rows:
        layers.append(int(row["layer"]))
        for field in fields:
            value = maybe_float(row.get(field))
            if value is not None:
                values.append(value)
    if not values or not layers:
        return 0.0, 1.0, 0, 1
    y_min = min(0.0, min(values))
    y_max = max(values)
    if y_min == y_max:
        y_max = y_min + 1.0
    pad = (y_max - y_min) * 0.08
    return y_min - pad, y_max + pad, min(layers), max(layers)


def build_line_svg(title: str, rows: list[dict[str, Any]], series: list[tuple[str, str, str]], x_label: str) -> str:
    width = 820
    height = 430
    left = 75
    right = 35
    top = 46
    bottom = 66
    plot_w = width - left - right
    plot_h = height - top - bottom
    y_min, y_max, x_min, x_max = scale_bounds(rows, [field for field, _, _ in series])

    def x_pos(layer: int) -> float:
        if x_min == x_max:
            return left + plot_w / 2
        return left + (layer - x_min) / (x_max - x_min) * plot_w

    def y_pos(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2:.1f}" y="26" text-anchor="middle" font-family="Arial" font-size="18" font-weight="700">{svg_escape(title)}</text>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#333" stroke-width="1"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#333" stroke-width="1"/>',
        f'<text x="{left + plot_w / 2:.1f}" y="{height - 22}" text-anchor="middle" font-family="Arial" font-size="13">{svg_escape(x_label)}</text>',
        f'<text x="18" y="{top + plot_h / 2:.1f}" transform="rotate(-90 18 {top + plot_h / 2:.1f})" text-anchor="middle" font-family="Arial" font-size="13">Profile metric</text>',
        f'<text x="{left}" y="{top + plot_h + 20}" text-anchor="middle" font-family="Arial" font-size="11">{x_min}</text>',
        f'<text x="{left + plot_w}" y="{top + plot_h + 20}" text-anchor="middle" font-family="Arial" font-size="11">{x_max}</text>',
        f'<text x="{left - 8}" y="{top + 4}" text-anchor="end" font-family="Arial" font-size="11">{y_max:.3f}</text>',
        f'<text x="{left - 8}" y="{top + plot_h + 4}" text-anchor="end" font-family="Arial" font-size="11">{y_min:.3f}</text>',
    ]
    for idx, (field, label, color) in enumerate(series):
        points: list[tuple[float, float, dict[str, Any]]] = []
        for row in rows:
            value = maybe_float(row.get(field))
            if value is None:
                continue
            points.append((x_pos(int(row["layer"])), y_pos(value), row))
        if not points:
            continue
        polyline = " ".join(f"{x:.2f},{y:.2f}" for x, y, _ in points)
        parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="2.5" points="{polyline}"/>')
        for x, y, row in points:
            parts.append(
                f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3.4" fill="{color}">'
                f'<title>{x_label.lower()} {row["layer"]}: {label}={fmt(row.get(field))}</title></circle>'
            )
        legend_y = top + 16 + idx * 18
        parts.append(f'<rect x="{left + 10}" y="{legend_y - 9}" width="12" height="12" fill="{color}"/>')
        parts.append(f'<text x="{left + 28}" y="{legend_y + 1}" font-family="Arial" font-size="12">{svg_escape(label)}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def build_bar_svg(title: str, rows: list[dict[str, Any]], selected: set[int]) -> str:
    width = 820
    height = 430
    left = 75
    right = 35
    top = 46
    bottom = 66
    plot_w = width - left - right
    plot_h = height - top - bottom
    y_min, y_max, x_min, x_max = scale_bounds(rows, ["expected_gain"])
    bar_gap = 2
    bar_w = max(4.0, plot_w / max(len(rows), 1) - bar_gap)

    def y_pos(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * plot_h

    zero_y = y_pos(0.0)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2:.1f}" y="26" text-anchor="middle" font-family="Arial" font-size="18" font-weight="700">{svg_escape(title)}</text>',
        f'<line x1="{left}" y1="{zero_y:.2f}" x2="{left + plot_w}" y2="{zero_y:.2f}" stroke="#999" stroke-width="1"/>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#333" stroke-width="1"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#333" stroke-width="1"/>',
        f'<text x="{left + plot_w / 2:.1f}" y="{height - 22}" text-anchor="middle" font-family="Arial" font-size="13">Layer index</text>',
        f'<text x="18" y="{top + plot_h / 2:.1f}" transform="rotate(-90 18 {top + plot_h / 2:.1f})" text-anchor="middle" font-family="Arial" font-size="13">Expected gain</text>',
        f'<text x="{left}" y="{top + plot_h + 20}" text-anchor="middle" font-family="Arial" font-size="11">{x_min}</text>',
        f'<text x="{left + plot_w}" y="{top + plot_h + 20}" text-anchor="middle" font-family="Arial" font-size="11">{x_max}</text>',
        f'<text x="{left - 8}" y="{top + 4}" text-anchor="end" font-family="Arial" font-size="11">{y_max:.3f}</text>',
        f'<text x="{left - 8}" y="{top + plot_h + 4}" text-anchor="end" font-family="Arial" font-size="11">{y_min:.3f}</text>',
    ]
    step = plot_w / max(len(rows), 1)
    for idx, row in enumerate(rows):
        value = maybe_float(row.get("expected_gain"))
        if value is None:
            continue
        y = y_pos(value)
        h = abs(zero_y - y)
        top_y = min(y, zero_y)
        x = left + idx * step + bar_gap / 2
        color = "#2f6fbb" if int(row["layer"]) in selected else "#b8c7d9"
        parts.append(
            f'<rect x="{x:.2f}" y="{top_y:.2f}" width="{bar_w:.2f}" height="{h:.2f}" fill="{color}">'
            f'<title>layer {row["layer"]}: gain={fmt(value)}</title></rect>'
        )
    parts.append(f'<rect x="{left + 10}" y="{top + 12}" width="12" height="12" fill="#2f6fbb"/>')
    parts.append(f'<text x="{left + 28}" y="{top + 23}" font-family="Arial" font-size="12">recommended patch layers</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def configure_matplotlib() -> None:
    matplotlib.rcParams["font.family"] = "DejaVu Sans"
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42


def write_line_plot(
    path_base: Path,
    title: str,
    rows: list[dict[str, Any]],
    series: list[tuple[str, str, str]],
    x_label: str,
    y_label: str = "Profile metric",
) -> None:
    configure_matplotlib()
    fig, ax = plt.subplots(figsize=(7.6, 4.0), dpi=160)
    xs = [int(row["layer"]) for row in rows]
    for field, label, color in series:
        ys = [maybe_float(row.get(field)) for row in rows]
        plot_x = [x for x, y in zip(xs, ys) if y is not None]
        plot_y = [float(y) for y in ys if y is not None]
        if plot_x and plot_y:
            ax.plot(plot_x, plot_y, marker="o", linewidth=1.8, markersize=3.5, color=color, label=label)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True, color="#dddddd", linewidth=0.7, alpha=0.7)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path_base.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)


def write_bar_plot(path_base: Path, title: str, rows: list[dict[str, Any]], selected: set[int]) -> None:
    configure_matplotlib()
    fig, ax = plt.subplots(figsize=(7.6, 4.0), dpi=160)
    xs = [int(row["layer"]) for row in rows]
    ys = [maybe_float(row.get("expected_gain")) or 0.0 for row in rows]
    colors = ["#2f6fbb" if layer in selected else "#b8c7d9" for layer in xs]
    ax.bar(xs, ys, color=colors, width=0.78)
    ax.axhline(0.0, color="#777777", linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Expected gain")
    ax.grid(True, axis="y", color="#dddddd", linewidth=0.7, alpha=0.7)
    fig.tight_layout()
    fig.savefig(path_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path_base.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)


def normalize_gain_curve(profile: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, value in enumerate(profile.get("patch_gain_curve", []), start=1):
        number = maybe_float(value)
        if number is not None:
            rows.append({"layer": idx, "patch_budget": idx, "cumulative_gain": number})
    return rows


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    profile_path = Path(args.profile)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    profile = load_json(profile_path)
    if not isinstance(profile, dict):
        raise ValueError("profile must be a JSON object")
    layer_rows = normalize_layers(profile)
    if not layer_rows:
        raise ValueError("profile contains no layer rows")
    selected = {int(layer) for layer in profile.get("recommended_patch_layers", [])}
    layer_fields = [
        "layer",
        "reuse_error",
        "patch_error",
        "expected_gain",
        "patch_xnorm_error",
        "patch_k_error",
        "patch_v_error",
        "patch_kv_error",
    ]
    write_csv(out_dir / "layer_profile.csv", layer_rows, layer_fields)
    write_md(out_dir / "layer_profile.md", "Layer Profile", layer_rows, layer_fields)
    (out_dir / "layer_error_profile.svg").write_text(
        build_line_svg(
            f"{args.title_prefix}: Reuse vs Patch error",
            layer_rows,
            [("reuse_error", "Reuse error", "#d95f02"), ("patch_error", "Patch error", "#1b9e77")],
            "Layer index",
        ),
        encoding="utf-8",
    )
    write_line_plot(
        out_dir / "layer_error_profile",
        f"{args.title_prefix}: Reuse vs Patch error",
        layer_rows,
        [("reuse_error", "Reuse error", "#d95f02"), ("patch_error", "Patch error", "#1b9e77")],
        "Layer index",
    )
    (out_dir / "layer_gain_profile.svg").write_text(
        build_bar_svg(f"{args.title_prefix}: expected Patch gain by layer", layer_rows, selected),
        encoding="utf-8",
    )
    write_bar_plot(out_dir / "layer_gain_profile", f"{args.title_prefix}: expected Patch gain by layer", layer_rows, selected)
    gain_rows = normalize_gain_curve(profile)
    gain_fields = ["patch_budget", "cumulative_gain"]
    write_csv(out_dir / "patch_gain_curve.csv", gain_rows, gain_fields)
    write_md(out_dir / "patch_gain_curve.md", "Patch Gain Curve", gain_rows, gain_fields)
    if gain_rows:
        (out_dir / "patch_gain_curve.svg").write_text(
            build_line_svg(
                f"{args.title_prefix}: cumulative gain curve",
                gain_rows,
                [("cumulative_gain", "Cumulative gain", "#1b9e77")],
                "Patch budget",
            ),
            encoding="utf-8",
        )
        write_line_plot(
            out_dir / "patch_gain_curve",
            f"{args.title_prefix}: cumulative gain curve",
            gain_rows,
            [("cumulative_gain", "Cumulative gain", "#1b9e77")],
            "Patch budget",
            y_label="Cumulative gain",
        )
    manifest = {
        "profile": str(profile_path),
        "outputs": sorted(path.name for path in out_dir.iterdir() if path.is_file()),
        "num_layers": len(layer_rows),
        "recommended_patch_layers": sorted(selected),
    }
    (out_dir / "analysis_figures_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
