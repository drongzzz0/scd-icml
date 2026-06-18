#!/usr/bin/env bash
set -euo pipefail

REMOTE_WORK_ROOT="/disk_n/c101-3-mql"
REMOTE_VENV="/disk_n/c101-3-mql/venvs/A800/bin/activate"
REMOTE_CACHE_ROOT="/disk_n/c101-3-mql/.modelscope_cache"
TARGET_DIR="/disk_n/c101-3-mql/models/Qwen3.5-4B-Base"
MODEL_ID="${1:-Qwen/Qwen3.5-4B-Base}"

if [[ ! -f "${REMOTE_VENV}" ]]; then
  echo "missing venv activate script: ${REMOTE_VENV}" >&2
  exit 1
fi

source "${REMOTE_VENV}"
export MODELSCOPE_CACHE="${REMOTE_CACHE_ROOT}"

mkdir -p "${REMOTE_CACHE_ROOT}"
mkdir -p "${REMOTE_WORK_ROOT}/models"

python3 - <<'PY'
import importlib.util
import sys

if importlib.util.find_spec("modelscope") is None:
    raise SystemExit(
        "modelscope is missing in the active environment. "
        "Run `python3 -m pip install -U modelscope` and retry."
    )
print("modelscope_import_ok")
PY

python3 /Users/weimingming/dr-claw/scd-new/Experiment/core_code/download_qwen35_4b_base_from_modelscope.py \
  --model-id "${MODEL_ID}" \
  --output-dir "${TARGET_DIR}"

echo "download_finished target_dir=${TARGET_DIR}"
