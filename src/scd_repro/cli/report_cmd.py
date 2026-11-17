from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt

from ..io_utils import load_json
from .common import resolve_logger


COLUMNS = [
    "method",
    "f1",
    "kl",
    "tv",
    "ppl",
    "ttft_ms",
    "oracle_ttft_ms",
    "speedup_vs_oracle",
    "payload_mb",
]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="汇总多次运行指标并生成报告")
    p.add_argument("--runs", required=True, help="逗号分隔的运行目录列表")
    p.add_argument("--out", required=True)
    p.add_argument("--log", default="")
    return p


def _collect_metrics(runs: list[Path]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for run in runs:
        metric_path = run / "metrics.json"
        if not metric_path.exists():
            continue
        payload = load_json(metric_path)
        metrics = payload["metrics"]
        rows.append({"run": str(run), **metrics})
    return rows


def _write_csv(rows: list[dict[str, object]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["run", *COLUMNS], extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_md(rows: list[dict[str, object]], out_md: Path) -> None:
    out_md.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# SCD 复现实验报告",
        "",
        "| run | method | f1 | kl | tv | ppl | ttft_ms | oracle_ttft_ms | speedup | payload_mb |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['run']} | {row['method']} | {row['f1']:.4f} | {row['kl']:.4f} | {row['tv']:.4f} | {row['ppl']:.4f} | {row['ttft_ms']:.2f} | {row['oracle_ttft_ms']:.2f} | {row['speedup_vs_oracle']:.3f} | {row['payload_mb']:.3f} |"
        )
    out_md.write_text("\n".join(lines), encoding="utf-8")


def _plot_tradeoff(rows: list[dict[str, object]], out_png: Path) -> None:
    if not rows:
        return
    out_png.parent.mkdir(parents=True, exist_ok=True)
    x = [float(r["payload_mb"]) for r in rows]
    y = [float(r["f1"]) for r in rows]
    labels = [str(r["method"]) for r in rows]

    plt.figure(figsize=(6.0, 4.0))
    plt.scatter(x, y, c="#1f77b4", s=40)
    for xi, yi, lb in zip(x, y, labels):
        plt.annotate(lb, (xi, yi), textcoords="offset points", xytext=(4, 3), fontsize=8)
    plt.xlabel("传输负载 (MB/请求)")
    plt.ylabel("F1")
    plt.title("SCD 复现：质量-负载权衡")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def main() -> None:
    args = build_parser().parse_args()
    logger = resolve_logger(args.log)

    runs = [Path(x.strip()) for x in args.runs.split(",") if x.strip()]
    rows = _collect_metrics(runs)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(rows, out_dir / "summary.csv")
    _write_md(rows, out_dir / "summary.md")
    _plot_tradeoff(rows, out_dir / "payload_vs_f1.png")

    logger.info("报告已生成到 %s", out_dir)


if __name__ == "__main__":
    main()
