from __future__ import annotations

"""CPU-only toy demo for the SCD Reuse/Patch/Route workflow.

这个脚本是给代码展示用的最小机制演示，不依赖真实模型、GPU、数据集或
checkpoint。它用合成的 KV-like tensor 模拟生产者/消费者状态，并把真实
工程里的核心流程压缩成四步：

1. 生成成对状态，模拟 `capture-pair` 的输出。
2. 训练线性 Reuse 映射，模拟低成本 KV 重建。
3. 训练非线性 Patch 修正，模拟关键层的误差补偿。
4. 按层选择 Reuse/Patch 路线，并输出指标和路线图。

生成的数字只用于解释机制，不代表论文实验结果。
"""

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class DemoConfig:
    seed: int
    samples: int
    layers: int
    dim: int
    patch_budget: int
    route_threshold: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a small CPU-only Semantic Cache Distillation toy demo.",
    )
    parser.add_argument("--out", default="demo_outputs/toy_scd", help="Output directory.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--layers", type=int, default=12)
    parser.add_argument("--dim", type=int, default=16)
    parser.add_argument("--patch-budget", type=int, default=4)
    parser.add_argument(
        "--route-threshold",
        type=float,
        default=0.25,
        help="Minimum relative layer-MSE reduction needed before Patch can be selected.",
    )
    return parser.parse_args()


def ridge_fit(x: np.ndarray, y: np.ndarray, ridge: float = 1e-3) -> np.ndarray:
    """Fit a small ridge-regression map used by the toy Reuse/Patch modules."""
    xtx = x.T @ x
    reg = ridge * np.eye(xtx.shape[0], dtype=np.float64)
    return np.linalg.solve(xtx + reg, x.T @ y)


def add_bias(x: np.ndarray) -> np.ndarray:
    """Append a constant feature so the linear model can learn layer bias."""
    return np.concatenate([x, np.ones((x.shape[0], 1), dtype=x.dtype)], axis=1)


def patch_features(x: np.ndarray) -> np.ndarray:
    """Build richer features for Patch so it can correct nonlinear layers."""
    clipped = np.clip(x, -3.0, 3.0)
    return add_bias(np.concatenate([x, clipped * clipped, np.tanh(clipped)], axis=1))


def make_synthetic_pairs(cfg: DemoConfig) -> tuple[np.ndarray, np.ndarray]:
    """Create synthetic producer/consumer states for the demo.

    Most layers are close to linear, so Reuse works reasonably well. A small
    set of hard layers contains stronger nonlinear terms, so Patch has a clear
    role and the selected route is easy to explain.
    """
    rng = np.random.default_rng(cfg.seed)
    producer = rng.normal(size=(cfg.samples, cfg.layers, cfg.dim)).astype(np.float64)
    consumer = np.empty_like(producer)

    hard_layers = {1, 4, 7, 10}
    for layer in range(cfg.layers):
        base = np.eye(cfg.dim) + 0.08 * rng.normal(size=(cfg.dim, cfg.dim))
        bias = 0.03 * rng.normal(size=(cfg.dim,))
        linear = producer[:, layer, :] @ base.T + bias

        centered_square = producer[:, layer, :] * producer[:, layer, :] - 1.0
        if layer in hard_layers:
            curve = 0.22 * centered_square + 0.08 * np.tanh(1.5 * producer[:, layer, :])
        else:
            curve = 0.03 * centered_square
        noise = 0.015 * rng.normal(size=(cfg.samples, cfg.dim))
        consumer[:, layer, :] = linear + curve + noise

    return producer, consumer


def train_layer_models(
    producer_train: np.ndarray,
    consumer_train: np.ndarray,
    producer_eval: np.ndarray,
    consumer_eval: np.ndarray,
) -> list[dict[str, object]]:
    """Train one Reuse model and one Patch model for every synthetic layer."""
    layer_rows: list[dict[str, object]] = []
    for layer in range(producer_train.shape[1]):
        x_train = producer_train[:, layer, :]
        y_train = consumer_train[:, layer, :]
        x_eval = producer_eval[:, layer, :]
        y_eval = consumer_eval[:, layer, :]

        reuse_w = ridge_fit(add_bias(x_train), y_train)
        reuse_pred = add_bias(x_eval) @ reuse_w

        patch_w = ridge_fit(patch_features(x_train), y_train)
        patch_pred = patch_features(x_eval) @ patch_w

        reuse_mse = float(np.mean((reuse_pred - y_eval) ** 2))
        patch_mse = float(np.mean((patch_pred - y_eval) ** 2))
        improvement = max(0.0, reuse_mse - patch_mse)
        improvement_ratio = improvement / max(reuse_mse, 1e-12)
        layer_rows.append(
            {
                "layer": layer,
                "reuse_pred": reuse_pred,
                "patch_pred": patch_pred,
                "target": y_eval,
                "reuse_mse": reuse_mse,
                "patch_mse": patch_mse,
                "improvement_ratio": float(improvement_ratio),
            }
        )
    return layer_rows


def select_route(layer_rows: list[dict[str, object]], cfg: DemoConfig) -> tuple[list[int], list[int]]:
    """Select Patch only where it gives enough relative MSE reduction."""
    candidates = [
        row
        for row in layer_rows
        if float(row["improvement_ratio"]) >= cfg.route_threshold
    ]
    candidates.sort(key=lambda row: float(row["improvement_ratio"]), reverse=True)
    patch_layers = sorted(int(row["layer"]) for row in candidates[: cfg.patch_budget])
    reuse_layers = [layer for layer in range(cfg.layers) if layer not in set(patch_layers)]
    return reuse_layers, patch_layers


def stack_predictions(layer_rows: list[dict[str, object]], key: str) -> np.ndarray:
    """Stack per-layer predictions back into [sample, layer, dim] tensors."""
    return np.stack([row[key] for row in layer_rows], axis=1)


def evaluate(layer_rows: list[dict[str, object]], reuse_layers: list[int], patch_layers: list[int]) -> dict[str, dict[str, float]]:
    """Compare full, reuse-only, patch-only, and routed SCD behavior."""
    target = stack_predictions(layer_rows, "target")
    reuse = stack_predictions(layer_rows, "reuse_pred")
    patch = stack_predictions(layer_rows, "patch_pred")
    route = np.empty_like(target)
    for layer in reuse_layers:
        route[:, layer, :] = reuse[:, layer, :]
    for layer in patch_layers:
        route[:, layer, :] = patch[:, layer, :]

    patch_fraction = len(patch_layers) / max(target.shape[1], 1)
    return {
        "full_consumer": {"kv_mse": 0.0, "simulated_ttft_ms": 100.0},
        "reuse_only": {"kv_mse": float(np.mean((reuse - target) ** 2)), "simulated_ttft_ms": 42.0},
        "patch_only": {"kv_mse": float(np.mean((patch - target) ** 2)), "simulated_ttft_ms": 76.0},
        "scd_route": {
            "kv_mse": float(np.mean((route - target) ** 2)),
            "simulated_ttft_ms": float(42.0 + patch_fraction * (76.0 - 42.0)),
        },
    }


def write_layer_csv(path: Path, layer_rows: list[dict[str, object]], patch_layers: list[int]) -> None:
    """Write per-layer errors so readers can inspect why a route was selected."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["layer", "route_action", "reuse_mse", "patch_mse", "improvement_ratio"],
        )
        writer.writeheader()
        patch_set = set(patch_layers)
        for row in layer_rows:
            layer = int(row["layer"])
            writer.writerow(
                {
                    "layer": layer,
                    "route_action": "patch" if layer in patch_set else "reuse",
                    "reuse_mse": f"{float(row['reuse_mse']):.8f}",
                    "patch_mse": f"{float(row['patch_mse']):.8f}",
                    "improvement_ratio": f"{float(row['improvement_ratio']):.6f}",
                }
            )


def write_route_svg(path: Path, layer_rows: list[dict[str, object]], patch_layers: list[int]) -> None:
    """Render a tiny route diagram without requiring extra plotting packages."""
    width = 760
    height = 180
    margin = 32
    cell_gap = 6
    layers = len(layer_rows)
    cell_w = (width - 2 * margin - (layers - 1) * cell_gap) / layers
    patch_set = set(patch_layers)

    rects = []
    labels = []
    for row in layer_rows:
        layer = int(row["layer"])
        x = margin + layer * (cell_w + cell_gap)
        color = "#f6b26b" if layer in patch_set else "#6fa8dc"
        label = "P" if layer in patch_set else "R"
        rects.append(
            f'<rect x="{x:.1f}" y="64" width="{cell_w:.1f}" height="52" rx="4" fill="{color}" stroke="#333" stroke-width="1" />'
        )
        labels.append(
            f'<text x="{x + cell_w / 2:.1f}" y="93" text-anchor="middle" font-family="Arial" font-size="16" fill="#111">{label}</text>'
        )
        labels.append(
            f'<text x="{x + cell_w / 2:.1f}" y="135" text-anchor="middle" font-family="Arial" font-size="10" fill="#333">L{layer}</text>'
        )
    svg = "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="#ffffff" />',
            '<text x="32" y="30" font-family="Arial" font-size="18" font-weight="700" fill="#111">Toy SCD route</text>',
            '<text x="32" y="50" font-family="Arial" font-size="12" fill="#555">R = Reuse layer, P = Patch layer</text>',
            *rects,
            *labels,
            "</svg>",
        ]
    )
    path.write_text(svg, encoding="utf-8")


def print_table(metrics: dict[str, dict[str, float]]) -> None:
    """Print the demo's main trade-off table to the terminal."""
    print("method          kv_mse      simulated_ttft_ms")
    print("--------------  ----------  -----------------")
    for name in ("full_consumer", "reuse_only", "patch_only", "scd_route"):
        row = metrics[name]
        print(f"{name:<14}  {row['kv_mse']:<10.6f}  {row['simulated_ttft_ms']:<17.2f}")


def main() -> int:
    """Run the full toy workflow and write demo artifacts."""
    args = parse_args()
    cfg = DemoConfig(
        seed=args.seed,
        samples=args.samples,
        layers=args.layers,
        dim=args.dim,
        patch_budget=min(args.patch_budget, args.layers),
        route_threshold=args.route_threshold,
    )
    if cfg.samples < 32:
        raise SystemExit("--samples must be at least 32")
    if cfg.layers < 2:
        raise SystemExit("--layers must be at least 2")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    producer, consumer = make_synthetic_pairs(cfg)
    split = int(0.7 * cfg.samples)
    layer_rows = train_layer_models(
        producer[:split],
        consumer[:split],
        producer[split:],
        consumer[split:],
    )
    reuse_layers, patch_layers = select_route(layer_rows, cfg)
    metrics = evaluate(layer_rows, reuse_layers, patch_layers)

    route = {
        "demo": "toy_scd",
        "num_layers": cfg.layers,
        "reuse_layers": reuse_layers,
        "patch_layers": patch_layers,
        "route_semantics": "toy_sparse_patch_reuse_all_other_layers",
        "selection": {
            "patch_budget": cfg.patch_budget,
            "route_threshold": cfg.route_threshold,
        },
    }
    (out_dir / "route.json").write_text(json.dumps(route, indent=2), encoding="utf-8")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    write_layer_csv(out_dir / "layer_errors.csv", layer_rows, patch_layers)
    write_route_svg(out_dir / "route_layers.svg", layer_rows, patch_layers)

    print_table(metrics)
    print()
    print(f"wrote: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
