#!/usr/bin/env bash
set -euo pipefail

SEED="${1:-42}"
CAPTURE_BACKEND="${CAPTURE_BACKEND:-synthetic}"

python -m pip install -e .

capture-pair --backend "$CAPTURE_BACKEND" --models-config configs/models.yaml --data-config configs/data.yaml --out artifacts/pairs/run_main --seed "$SEED"
train-reuse --pairs artifacts/pairs/run_main --train-config configs/train.yaml --out artifacts/ckpt_reuse/ckpt_main --seed "$SEED"
profile-layers --pairs artifacts/pairs/run_main --reuse-ckpt artifacts/ckpt_reuse/ckpt_main --out artifacts/route/profile_main.json
train-patch --pairs artifacts/pairs/run_main --profile artifacts/route/profile_main.json --train-config configs/train.yaml --out artifacts/ckpt_patch/patch_main --seed "$SEED"
build-route --profile artifacts/route/profile_main.json --eval-config configs/eval.yaml --out artifacts/route/route_main.json

METHODS=(raw_kv quantized_4bit selective_recompute reuse_only scd_full)
RUNS=()
for METHOD in "${METHODS[@]}"; do
  OUT="artifacts/reports/${METHOD}"
  infer --pairs artifacts/pairs/run_main --reuse-ckpt artifacts/ckpt_reuse/ckpt_main --patch-ckpt artifacts/ckpt_patch/patch_main --route artifacts/route/route_main.json --eval-config configs/eval.yaml --method "$METHOD" --out "$OUT" --seed "$SEED"
  python -m scd_repro.cli.eval_cmd --pred "$OUT/predictions.json" --oracle "$OUT/oracle.json" --eval-config configs/eval.yaml --out "$OUT/metrics.json"
  RUNS+=("$OUT")
done

RUNS_CSV=$(IFS=, ; echo "${RUNS[*]}")
report --runs "$RUNS_CSV" --out artifacts/reports/final

echo "done: artifacts/reports/final/summary.md"
