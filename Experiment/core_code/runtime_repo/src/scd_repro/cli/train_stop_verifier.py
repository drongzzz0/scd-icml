from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from ..cmrc import score_cmrc_prediction


FEATURE_NAMES = [
    "generated_token_count",
    "answer_char_len",
    "raw_char_len",
    "visible_char_len",
    "answer_in_context",
    "answer_is_visible_prefix",
    "has_newline",
    "ends_ascii_punct",
    "ends_digit",
    "ends_ascii_alpha",
    "top1_prob",
    "top2_prob",
    "top_prob_gap",
    "selected_token_prob",
    "stop_token_prob",
]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train a lightweight generation-stop verifier from CMRC trace JSONL.")
    p.add_argument("--train-jsonl", required=True)
    p.add_argument("--val-jsonl", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--method", default="scd_full")
    p.add_argument("--label-tolerance", type=float, default=1e-6)
    p.add_argument("--allowed-f1-loss", type=float, default=0.002)
    p.add_argument("--allowed-em-loss", type=float, default=0.005)
    p.add_argument("--min-positive-f1", type=float, default=0.5)
    p.add_argument("--min-generated-tokens", type=int, default=1)
    p.add_argument("--min-answer-chars", type=int, default=1)
    p.add_argument("--max-answer-chars", type=int, default=40)
    p.add_argument("--epochs", type=int, default=1200)
    p.add_argument("--lr", type=float, default=0.08)
    p.add_argument("--l2", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    return p


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _rows_for_method(rows: list[dict[str, Any]], method: str) -> list[dict[str, Any]]:
    return [row for row in rows if str(row.get("method", "")) == method]


def _feature_vector(features: dict[str, Any]) -> list[float]:
    return [float(features.get(name, 0.0) or 0.0) for name in FEATURE_NAMES]


def _score_answer(answer: str, gold_answers: list[str]) -> tuple[float, float]:
    score = score_cmrc_prediction(answer, gold_answers)
    return float(score["f1"]), float(score["em"])


def _trace_examples(
    rows: list[dict[str, Any]],
    *,
    label_tolerance: float,
    min_positive_f1: float,
    min_generated_tokens: int,
    min_answer_chars: int,
    max_answer_chars: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    xs: list[list[float]] = []
    ys: list[int] = []
    skipped_no_trace = 0
    positive_rows = 0
    candidate_rows = 0

    for row in rows:
        trace = row.get("generation_trace", [])
        if not isinstance(trace, list) or not trace:
            skipped_no_trace += 1
            continue
        gold_answers = [str(x) for x in row.get("gold_answers", []) if str(x).strip()]
        if not gold_answers:
            continue

        prefix_scores: list[tuple[float, float]] = []
        for step in trace:
            answer = str((step or {}).get("answer_text", ""))
            prefix_scores.append(_score_answer(answer, gold_answers))
        final_f1 = float(row.get("f1", 0.0) or 0.0)
        final_em = float(row.get("em", 0.0) or 0.0)
        best_f1 = max([final_f1, *[score[0] for score in prefix_scores]])
        best_em = max([final_em, *[score[1] for score in prefix_scores]])
        row_has_positive = False

        for step, (f1, em) in zip(trace, prefix_scores):
            features = (step or {}).get("features", {})
            if not isinstance(features, dict):
                continue
            token_count = int(float(features.get("generated_token_count", 0.0) or 0.0))
            answer_len = int(float(features.get("answer_char_len", 0.0) or 0.0))
            eligible = token_count >= min_generated_tokens and answer_len >= min_answer_chars
            if max_answer_chars > 0 and answer_len > max_answer_chars:
                eligible = False
            label = int(
                eligible
                and best_f1 >= min_positive_f1
                and f1 >= best_f1 - label_tolerance
                and em >= best_em - label_tolerance
            )
            xs.append(_feature_vector(features))
            ys.append(label)
            candidate_rows += 1
            row_has_positive = row_has_positive or bool(label)
        positive_rows += int(row_has_positive)

    if not xs:
        raise ValueError("no trace examples were found; run eval with generation_stop.trace_enabled=true first")
    y_arr = np.asarray(ys, dtype=np.float32)
    if float(y_arr.sum()) <= 0.0:
        raise ValueError("trace examples contain no positive stop labels; relax min-positive-f1 or inspect traces")
    stats = {
        "rows": len(rows),
        "skipped_no_trace": skipped_no_trace,
        "candidate_steps": candidate_rows,
        "positive_steps": int(y_arr.sum()),
        "positive_source_rows": positive_rows,
    }
    return np.asarray(xs, dtype=np.float32), y_arr, stats


def _sigmoid_np(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-x))


def _train_logistic(
    x: np.ndarray,
    y: np.ndarray,
    *,
    epochs: int,
    lr: float,
    l2: float,
    seed: int,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray, dict[str, float]]:
    rng = np.random.default_rng(seed)
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale = np.where(scale < 1e-6, 1.0, scale)
    xz = (x - mean) / scale
    w = rng.normal(0.0, 0.01, size=xz.shape[1]).astype(np.float32)
    pos = float(y.sum())
    neg = float(len(y) - pos)
    bias = math.log((pos + 1.0) / (neg + 1.0))
    pos_weight = 0.5 * len(y) / max(pos, 1.0)
    neg_weight = 0.5 * len(y) / max(neg, 1.0)
    sample_weight = np.where(y > 0.5, pos_weight, neg_weight).astype(np.float32)
    denom = float(sample_weight.sum())

    for _ in range(max(1, int(epochs))):
        logits = xz @ w + bias
        pred = _sigmoid_np(logits)
        err = (pred - y) * sample_weight
        grad_w = (xz.T @ err) / denom + float(l2) * w
        grad_b = float(err.sum() / denom)
        w -= float(lr) * grad_w
        bias -= float(lr) * grad_b

    pred = _sigmoid_np(xz @ w + bias)
    eps = 1e-8
    loss = -float(np.mean(y * np.log(pred + eps) + (1.0 - y) * np.log(1.0 - pred + eps)))
    return w, float(bias), mean, scale, {"train_loss": loss, "train_pos_rate": float(y.mean())}


def _predict_scores(rows: list[dict[str, Any]], artifact: dict[str, Any]) -> list[list[float]]:
    w = np.asarray(artifact["weights"], dtype=np.float32)
    mean = np.asarray(artifact["feature_mean"], dtype=np.float32)
    scale = np.asarray(artifact["feature_scale"], dtype=np.float32)
    bias = float(artifact["bias"])
    all_scores: list[list[float]] = []
    for row in rows:
        scores: list[float] = []
        for step in row.get("generation_trace", []) or []:
            features = (step or {}).get("features", {})
            x = np.asarray(_feature_vector(features if isinstance(features, dict) else {}), dtype=np.float32)
            z = float(((x - mean) / scale) @ w + bias)
            scores.append(float(1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, z))))))
        all_scores.append(scores)
    return all_scores


def _simulate(rows: list[dict[str, Any]], scores: list[list[float]], threshold: float, artifact: dict[str, Any]) -> dict[str, Any]:
    total_f1 = total_em = total_tokens = 0.0
    base_f1 = base_em = base_tokens = 0.0
    verifier_stops = 0
    max_new_hits = 0
    n = 0
    min_tokens = int(artifact.get("min_generated_tokens", 1) or 0)
    min_chars = int(artifact.get("min_answer_chars", 1) or 0)
    max_chars = int(artifact.get("max_answer_chars", 0) or 0)

    for row, row_scores in zip(rows, scores):
        if not row.get("generation_trace"):
            continue
        gold_answers = [str(x) for x in row.get("gold_answers", []) if str(x).strip()]
        if not gold_answers:
            continue
        n += 1
        base_f1 += float(row.get("f1", 0.0) or 0.0)
        base_em += float(row.get("em", 0.0) or 0.0)
        base_tokens += float(row.get("generated_token_count", 0.0) or 0.0)
        pred_text = str(row.get("pred_text", ""))
        pred_tokens = int(float(row.get("generated_token_count", 0.0) or 0.0))
        stopped = False
        for step, score in zip(row.get("generation_trace", []) or [], row_scores):
            features = (step or {}).get("features", {})
            if not isinstance(features, dict):
                continue
            token_count = int(float(features.get("generated_token_count", 0.0) or 0.0))
            answer_len = int(float(features.get("answer_char_len", 0.0) or 0.0))
            if token_count < min_tokens or answer_len < min_chars:
                continue
            if max_chars > 0 and answer_len > max_chars:
                continue
            if float(score) >= threshold:
                pred_text = str((step or {}).get("answer_text", ""))
                pred_tokens = token_count
                verifier_stops += 1
                stopped = True
                break
        f1, em = _score_answer(pred_text, gold_answers)
        total_f1 += f1
        total_em += em
        total_tokens += float(pred_tokens)
        if (not stopped) and str(row.get("stop_reason", "")) == "max_new_tokens":
            max_new_hits += 1

    if n <= 0:
        raise ValueError("no rows available for verifier simulation")
    return {
        "num_samples": n,
        "f1": total_f1 / n,
        "em": total_em / n,
        "generated_token_count": total_tokens / n,
        "verifier_stop_rate": verifier_stops / n,
        "max_new_tokens_hit_rate": max_new_hits / n,
        "baseline_f1": base_f1 / n,
        "baseline_em": base_em / n,
        "baseline_generated_token_count": base_tokens / n,
        "delta_f1": total_f1 / n - base_f1 / n,
        "delta_em": total_em / n - base_em / n,
        "delta_generated_token_count": total_tokens / n - base_tokens / n,
    }


def _select_threshold(rows: list[dict[str, Any]], artifact: dict[str, Any], allowed_f1_loss: float, allowed_em_loss: float) -> dict[str, Any]:
    scores = _predict_scores(rows, artifact)
    raw_scores = sorted({round(score, 6) for row_scores in scores for score in row_scores})
    grid = [0.05 + i * 0.005 for i in range(181)]
    thresholds = sorted(set([*grid, *raw_scores]))
    candidates: list[dict[str, Any]] = []
    for threshold in thresholds:
        metrics = _simulate(rows, scores, float(threshold), artifact)
        metrics["threshold"] = float(threshold)
        metrics["passes_gate"] = (
            metrics["f1"] >= metrics["baseline_f1"] - float(allowed_f1_loss)
            and metrics["em"] >= metrics["baseline_em"] - float(allowed_em_loss)
        )
        candidates.append(metrics)
    gated = [row for row in candidates if row["passes_gate"]]
    if gated:
        return max(gated, key=lambda row: (-row["generated_token_count"], row["f1"], row["em"]))
    return max(candidates, key=lambda row: (row["f1"], row["em"], -row["generated_token_count"]))


def main() -> None:
    args = build_parser().parse_args()
    train_rows = _rows_for_method(_load_jsonl(args.train_jsonl), args.method)
    val_rows = _rows_for_method(_load_jsonl(args.val_jsonl), args.method)
    if not train_rows or not val_rows:
        raise SystemExit(f"missing method rows for method={args.method}: train={len(train_rows)} val={len(val_rows)}")

    x_train, y_train, trace_stats = _trace_examples(
        train_rows,
        label_tolerance=float(args.label_tolerance),
        min_positive_f1=float(args.min_positive_f1),
        min_generated_tokens=int(args.min_generated_tokens),
        min_answer_chars=int(args.min_answer_chars),
        max_answer_chars=int(args.max_answer_chars),
    )
    weights, bias, mean, scale, train_metrics = _train_logistic(
        x_train,
        y_train,
        epochs=int(args.epochs),
        lr=float(args.lr),
        l2=float(args.l2),
        seed=int(args.seed),
    )
    artifact: dict[str, Any] = {
        "schema": "scd_stop_verifier_logistic_v1",
        "method": args.method,
        "feature_names": FEATURE_NAMES,
        "weights": [float(x) for x in weights.tolist()],
        "bias": float(bias),
        "feature_mean": [float(x) for x in mean.tolist()],
        "feature_scale": [float(x) for x in scale.tolist()],
        "threshold": 0.5,
        "min_generated_tokens": int(args.min_generated_tokens),
        "min_answer_chars": int(args.min_answer_chars),
        "max_answer_chars": int(args.max_answer_chars),
        "training": {
            **trace_stats,
            **train_metrics,
            "label_tolerance": float(args.label_tolerance),
            "min_positive_f1": float(args.min_positive_f1),
        },
    }
    selected = _select_threshold(val_rows, artifact, float(args.allowed_f1_loss), float(args.allowed_em_loss))
    artifact["threshold"] = float(selected["threshold"])
    artifact["validation"] = selected
    artifact["selection_constraints"] = {
        "allowed_f1_loss": float(args.allowed_f1_loss),
        "allowed_em_loss": float(args.allowed_em_loss),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    metrics_path = out.with_name(out.stem + "_metrics.json")
    metrics_path.write_text(
        json.dumps({"artifact": str(out), "validation": selected, "training": artifact["training"]}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("METRIC verifier_threshold=%s" % artifact["threshold"])
    print("METRIC val_f1=%s" % selected["f1"])
    print("METRIC val_em=%s" % selected["em"])
    print("METRIC val_delta_tokens=%s" % selected["delta_generated_token_count"])
    print(f"artifact={out}")
    print(f"metrics={metrics_path}")


if __name__ == "__main__":
    main()
