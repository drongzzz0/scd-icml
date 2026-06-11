from __future__ import annotations

import argparse
import math

import numpy as np

from ..io_utils import load_json, save_json
from ..metrics import kl_div, mean_f1, perplexity_from_nll, tv_distance
from .common import load_cfg, resolve_logger


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="对单次运行结果进行预言机对比评估")
    p.add_argument("--pred", required=True)
    p.add_argument("--oracle", required=True)
    p.add_argument("--eval-config", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--log", default="")
    return p


def main() -> None:
    args = build_parser().parse_args()
    logger = resolve_logger(args.log)

    pred = load_json(args.pred)
    oracle = load_json(args.oracle)
    eval_cfg = load_cfg(args.eval_config)

    p = np.array(pred["probs"], dtype=np.float64)[None, :]
    q = np.array(oracle["probs"], dtype=np.float64)[None, :]

    kl = kl_div(q, p)
    tv = tv_distance(q, p)

    pred_texts = [s["pred_text"] for s in pred["samples"]]
    ref_texts = [s["pred_text"] for s in oracle["samples"]]
    f1 = mean_f1(pred_texts, ref_texts)

    oracle_token = int(oracle["samples"][0]["pred_token"])
    p_oracle = max(float(p[0, oracle_token]), 1e-12)
    nll = -math.log(p_oracle)
    ppl = perplexity_from_nll(nll)

    ttft_pred = float(pred["timing_ms"]["ttft"])
    ttft_oracle = float(oracle["timing_ms"]["ttft"])
    speedup = ttft_oracle / max(1e-9, ttft_pred)

    metrics = {
        "method": pred["method"],
        "f1": f1,
        "kl": kl,
        "tv": tv,
        "ppl": ppl,
        "ttft_ms": ttft_pred,
        "oracle_ttft_ms": ttft_oracle,
        "speedup_vs_oracle": speedup,
        "payload_mb": float(pred["timing_ms"]["payload_mb"]),
    }

    thresholds = eval_cfg["acceptance"]
    acceptance = {
        "f1_abs_gap_le": float(thresholds["f1_abs_gap_le"]),
        "kl_rel_gap_le": float(thresholds["kl_rel_gap_le"]),
        "tv_rel_gap_le": float(thresholds["tv_rel_gap_le"]),
        "ppl_rel_gap_le": float(thresholds["ppl_rel_gap_le"]),
        "ttft_rel_gap_le": float(thresholds["ttft_rel_gap_le"]),
    }

    # 预言机参考值，用于相对误差检查。
    oracle_baseline = {
        "f1": 1.0,
        "kl": 0.0,
        "tv": 0.0,
        "ppl": 1.0,
        "ttft_ms": ttft_oracle,
    }

    eval_result = {
        "metrics": metrics,
        "acceptance_thresholds": acceptance,
        "oracle_reference": oracle_baseline,
    }

    save_json(args.out, eval_result)
    logger.info("评测指标已保存到 %s", args.out)


if __name__ == "__main__":
    main()

