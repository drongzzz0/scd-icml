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
from matplotlib.lines import Line2D
import numpy as np


BLUE = "#3B6FB6"
ORANGE = "#C95F45"
GREY = "#B8B8B8"
DARK = "#2B2B2B"
MUTED = "#777777"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build the SCD stack/t-SNE visualization from stored KV distance "
            "and 2D embedding CSV artifacts."
        )
    )
    parser.add_argument("--distance-csv", required=True)
    parser.add_argument("--distance-meta", required=True)
    parser.add_argument("--tsne-csv", required=True)
    parser.add_argument("--tsne-meta", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--basename", default="stack_viz")
    return parser


def configure_matplotlib() -> None:
    matplotlib.rcParams["font.family"] = "serif"
    matplotlib.rcParams["font.serif"] = [
        "Times New Roman",
        "Liberation Serif",
        "Nimbus Roman",
        "DejaVu Serif",
    ]
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42


def load_distance_rows(path: Path) -> tuple[np.ndarray, np.ndarray]:
    source = []
    translated = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            source.append(float(row["source_rel_l2"]))
            translated.append(float(row["translated_rel_l2"]))
    if not source:
        raise ValueError(f"empty distance CSV: {path}")
    return np.asarray(source), np.asarray(translated)


def load_tsne_points(path: Path) -> dict[str, np.ndarray]:
    grouped: dict[str, list[list[float]]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            grouped.setdefault(row["series"], []).append([float(row["x"]), float(row["y"])])
    required = {"source", "target", "translated_reuse"}
    missing = sorted(required.difference(grouped))
    if missing:
        raise ValueError(f"missing t-SNE series in {path}: {', '.join(missing)}")
    return {key: np.asarray(value, dtype=np.float32) for key, value in grouped.items()}


def style_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#CFCFCF")
    ax.spines["bottom"].set_color("#CFCFCF")
    ax.tick_params(axis="both", colors="#555555", labelsize=6.3, width=0.5, length=2.5)


def draw_distribution(ax: plt.Axes, source: np.ndarray, translated: np.ndarray, meta: dict[str, Any]) -> None:
    rng = np.random.default_rng(42)
    positions = np.array([0.0, 1.0])
    data = [source, translated]
    colors = [BLUE, ORANGE]

    violins = ax.violinplot(data, positions=positions, widths=0.62, showmeans=False, showmedians=False, showextrema=False)
    for body, color in zip(violins["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor("none")
        body.set_alpha(0.16)

    for x, values, color in zip(positions, data, colors):
        q1, median, q3 = np.percentile(values, [25, 50, 75])
        ax.vlines(x, q1, q3, color=color, lw=2.1, alpha=0.95, zorder=4)
        ax.scatter([x], [median], s=19, c=color, edgecolors="white", linewidths=0.45, zorder=5)
        jitter = rng.normal(0, 0.035, size=values.shape[0])
        ax.scatter(np.full_like(values, x) + jitter, values, s=5.0, c=color, alpha=0.18, linewidths=0, zorder=2)

    subset = rng.choice(np.arange(source.shape[0]), size=min(55, source.shape[0]), replace=False)
    for idx in subset:
        ax.plot(positions, [source[idx], translated[idx]], color="#9A9A9A", alpha=0.13, lw=0.45, zorder=1)

    ax.set_xlim(-0.55, 1.55)
    ymin = max(0.0, min(source.min(), translated.min()) - 0.03)
    ymax = max(source.max(), translated.max()) + 0.035
    ax.set_ylim(ymin, ymax)
    ax.set_xticks(positions)
    ax.set_xticklabels(["Source KV", "Translated KV"], fontsize=7.0)
    ax.set_ylabel("Relative L2 distance to target KV", fontsize=7.0, color=DARK)
    ax.set_title("(a) High-dimensional KV distance", fontsize=8.0, loc="left", pad=3, color=DARK)
    ax.grid(axis="y", color="#E7E7E7", lw=0.5, alpha=0.8)
    style_axis(ax)

    reduction = meta.get("median_rel_l2_reduction")
    if reduction is not None:
        ax.text(
            0.98,
            0.96,
            f"median reduction {100.0 * float(reduction):.1f}%",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=7.2,
            color=ORANGE,
            fontweight="bold",
        )


def draw_tsne(ax: plt.Axes, points: dict[str, np.ndarray], tsne_meta: dict[str, Any]) -> None:
    target = points["target"]
    translated = points["translated_reuse"]
    source = points["source"]
    all_points = np.vstack([source, target, translated])
    lo = np.quantile(all_points, 0.01, axis=0)
    hi = np.quantile(all_points, 0.99, axis=0)
    pad = 0.09 * np.maximum(hi - lo, 1e-6)
    ax.set_xlim(float(lo[0] - pad[0]), float(hi[0] + pad[0]))
    ax.set_ylim(float(lo[1] - pad[1]), float(hi[1] + pad[1]))
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.scatter(source[:, 0], source[:, 1], s=5.0, c=BLUE, alpha=0.20, linewidths=0, rasterized=True)
    ax.scatter(target[:, 0], target[:, 1], s=5.2, c=GREY, alpha=0.28, linewidths=0, rasterized=True)
    ax.scatter(translated[:, 0], translated[:, 1], s=5.2, c=ORANGE, alpha=0.52, linewidths=0, rasterized=True)
    ax.set_title("(b) t-SNE view after translation", fontsize=8.0, loc="left", pad=3, color=DARK)
    ax.text(
        0.98,
        0.02,
        f"layer {tsne_meta.get('layer', '?')}, p={tsne_meta.get('perplexity', 50)}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=5.8,
        color=MUTED,
    )


def build_stack_viz(
    distance_csv: Path,
    distance_meta: Path,
    tsne_csv: Path,
    tsne_meta: Path,
    out_dir: Path,
    basename: str,
) -> list[Path]:
    configure_matplotlib()
    for path in (distance_csv, distance_meta, tsne_csv, tsne_meta):
        if not path.exists():
            raise FileNotFoundError(f"missing input: {path}")
    source_dist, translated_dist = load_distance_rows(distance_csv)
    dist_meta = json.loads(distance_meta.read_text(encoding="utf-8-sig"))
    tsne_points = load_tsne_points(tsne_csv)
    tsne_meta_payload = json.loads(tsne_meta.read_text(encoding="utf-8-sig"))

    fig = plt.figure(figsize=(6.65, 2.45))
    grid = fig.add_gridspec(1, 2, width_ratios=[1.12, 1.0], wspace=0.28)
    ax_dist = fig.add_subplot(grid[0, 0])
    ax_tsne = fig.add_subplot(grid[0, 1])

    draw_distribution(ax_dist, source_dist, translated_dist, dist_meta)
    draw_tsne(ax_tsne, tsne_points, tsne_meta_payload)

    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=BLUE, markeredgewidth=0, alpha=0.75, markersize=3.8, label="Source KV"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=GREY, markeredgewidth=0, alpha=0.65, markersize=3.8, label="Target KV"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=ORANGE, markeredgewidth=0, alpha=0.80, markersize=3.8, label="Translated KV"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=6.4,
        bbox_to_anchor=(0.56, -0.03),
        columnspacing=1.1,
        handletextpad=0.35,
    )
    fig.subplots_adjust(left=0.095, right=0.985, top=0.86, bottom=0.20, wspace=0.26)
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = [out_dir / f"{basename}.pdf", out_dir / f"{basename}.png", out_dir / f"{basename}.svg"]
    fig.savefig(outputs[0], bbox_inches="tight")
    fig.savefig(outputs[1], dpi=420, bbox_inches="tight")
    fig.savefig(outputs[2], bbox_inches="tight")
    plt.close(fig)
    return outputs


def main() -> None:
    args = build_parser().parse_args()
    outputs = build_stack_viz(
        distance_csv=Path(args.distance_csv),
        distance_meta=Path(args.distance_meta),
        tsne_csv=Path(args.tsne_csv),
        tsne_meta=Path(args.tsne_meta),
        out_dir=Path(args.out_dir),
        basename=args.basename,
    )
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
