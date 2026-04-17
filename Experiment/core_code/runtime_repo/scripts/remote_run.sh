#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 5 ]]; then
  echo "用法: $0 <ssh_host> <远端项目目录> <会话名> <seed> <method> [infer 额外参数...]"
  exit 1
fi

SSH_HOST="$1"
REMOTE_DIR="$2"
SESSION="$3"
SEED="$4"
METHOD="$5"
CAPTURE_BACKEND="${CAPTURE_BACKEND:-synthetic}"
shift 5

EXTRA_ARGS="$*"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="artifacts/reports/${METHOD}_${STAMP}"

REMOTE_CMD=$(cat <<CMD
set -euo pipefail
cd "${REMOTE_DIR}"
python -m pip install -e .
capture-pair --backend ${CAPTURE_BACKEND} --models-config configs/models.yaml --data-config configs/data.yaml --out artifacts/pairs/run_main --seed ${SEED}
train-reuse --pairs artifacts/pairs/run_main --train-config configs/train.yaml --out artifacts/ckpt_reuse/ckpt_main --seed ${SEED}
profile-layers --pairs artifacts/pairs/run_main --reuse-ckpt artifacts/ckpt_reuse/ckpt_main --out artifacts/route/profile_main.json
train-patch --pairs artifacts/pairs/run_main --profile artifacts/route/profile_main.json --train-config configs/train.yaml --out artifacts/ckpt_patch/patch_main --seed ${SEED}
build-route --profile artifacts/route/profile_main.json --eval-config configs/eval.yaml --out artifacts/route/route_main.json
infer --pairs artifacts/pairs/run_main --reuse-ckpt artifacts/ckpt_reuse/ckpt_main --patch-ckpt artifacts/ckpt_patch/patch_main --route artifacts/route/route_main.json --eval-config configs/eval.yaml --method ${METHOD} --out ${RUN_DIR} --seed ${SEED} ${EXTRA_ARGS}
python -m scd_repro.cli.eval_cmd --pred ${RUN_DIR}/predictions.json --oracle ${RUN_DIR}/oracle.json --eval-config configs/eval.yaml --out ${RUN_DIR}/metrics.json
CMD
)

ssh "$SSH_HOST" "tmux new-session -d -s ${SESSION} '${REMOTE_CMD}'"
echo "已提交 tmux 会话: ${SESSION}"
echo "查看命令: ssh ${SSH_HOST} 'tmux attach -t ${SESSION}'"
