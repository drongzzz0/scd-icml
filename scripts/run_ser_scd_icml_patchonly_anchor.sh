#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
VENV="${VENV:-/disk_n/c101-3-mql/venvs/A800}"
SNAPSHOT_ROOT="${SNAPSHOT_ROOT:-/disk_n/c101-3-mql/projects/SCD-ICML/kv/snapshots/scd-github-f7d7854}"
DATA_ROOT="${DATA_ROOT:-${SNAPSHOT_ROOT}}"
RUN_ROOT="${RUN_ROOT:-/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_patchonly_anchor_$(date +%Y%m%d_%H%M%S)}"

MODELS_CONFIG="${MODELS_CONFIG:-configs/models_ser_mistrallite_to_mistral7_mql.json}"
ROUTE_PATH="${ROUTE_PATH:-configs/route_reuse_patchonly_all32_20260404.json}"
BASELINE_REUSE_EVAL_CONFIG="${BASELINE_REUSE_EVAL_CONFIG:-configs/eval_baseline_reuse_only_fulldev3219_20260405.json}"
PATCH_ONLY_EVAL_CONFIG="${PATCH_ONLY_EVAL_CONFIG:-configs/eval_patch_only_fulldev3219_20260405.json}"

DATASETS="${DATASETS:-cmrc2018,hotpotqa_compressed,wikitext2}"
SEED="${SEED:-42}"
EVAL_LIMIT="${EVAL_LIMIT:-0}"
MAX_LENGTH="${MAX_LENGTH:-1024}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"
BANDWIDTH_GBPS="${BANDWIDTH_GBPS:-200}"
CMRC_SPLIT="${CMRC_SPLIT:-dev}"
PRODUCER_DEVICE="${PRODUCER_DEVICE:-cuda:0}"
CONSUMER_DEVICE="${CONSUMER_DEVICE:-cuda:1}"
WAIT_GPU_POLL_SEC="${WAIT_GPU_POLL_SEC:-120}"
WAIT_EVAL_FREE_MB="${WAIT_EVAL_FREE_MB:-18000}"
AUTO_GPU_COUNT="${AUTO_GPU_COUNT:-2}"
INSTALL_EDITABLE="${INSTALL_EDITABLE:-0}"
PATCH_CKPT_DEVICE="${PATCH_CKPT_DEVICE:-cpu}"
RUNTIME_PATCH_WINDOW_SIZE="${RUNTIME_PATCH_WINDOW_SIZE:-8}"
RUNTIME_PATCH_PERSIST_SHARDS="${RUNTIME_PATCH_PERSIST_SHARDS:-false}"
RUNTIME_PATCH_PERSIST_MODULES="${RUNTIME_PATCH_PERSIST_MODULES:-false}"
RESUME="${RESUME:-0}"

CMRC_REUSE_DIR="${CMRC_REUSE_DIR:-${SNAPSHOT_ROOT}/runs/mistral_strict_default_gpu1_200_b2_20260421/ckpt_reuse/cmrc2018_seed42}"
CMRC_PATCH_DIR="${CMRC_PATCH_DIR:-${SNAPSHOT_ROOT}/runs/mistral_strict_default_gpu1_200_b2_20260421/ckpt_patch/patch_cmrc2018_seed42}"
HOTPOT_REUSE_DIR="${HOTPOT_REUSE_DIR:-${SNAPSHOT_ROOT}/runs/strict_hotpotqa_200_total_heldout100_promptfix_max512_20260422/ckpt_reuse/hotpotqa_seed42}"
HOTPOT_PATCH_DIR="${HOTPOT_PATCH_DIR:-${SNAPSHOT_ROOT}/runs/strict_hotpotqa_200_total_heldout100_promptfix_max512_20260422/ckpt_patch/patch_hotpotqa_seed42}"
WIKITEXT_REUSE_DIR="${WIKITEXT_REUSE_DIR:-${SNAPSHOT_ROOT}/runs/strict_wikitext2_train200_test200_20260422/ckpt_reuse/wikitext2_seed42}"
WIKITEXT_PATCH_DIR="${WIKITEXT_PATCH_DIR:-${SNAPSHOT_ROOT}/runs/strict_wikitext2_train200_test200_20260422/ckpt_patch/patch_wikitext2_seed42}"

pick_free_gpus() {
  local required_free_mb="$1"
  local count="$2"
  nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits | \
    awk -F',' -v required="${required_free_mb}" '{gsub(/ /, "", $1); gsub(/ /, "", $2); gsub(/ /, "", $3); if ($2 >= required) print $1 "," $2 "," $3}' | \
    sort -t',' -k2,2nr -k3,3n | \
    awk -F',' -v count="${count}" 'NR <= count {print $1}'
}

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  while true; do
    mapfile -t picked_gpus < <(pick_free_gpus "${WAIT_EVAL_FREE_MB}" "${AUTO_GPU_COUNT}")
    if [[ "${#picked_gpus[@]}" -ge "${AUTO_GPU_COUNT}" ]]; then
      export CUDA_VISIBLE_DEVICES="$(IFS=','; echo "${picked_gpus[*]}")"
      echo "selected CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
      break
    fi
    echo "waiting to select ${AUTO_GPU_COUNT} GPU(s) with free memory >= ${WAIT_EVAL_FREE_MB}MB"
    nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits
    sleep "${WAIT_GPU_POLL_SEC}"
  done
fi

ckpt_for_dataset() {
  local dataset="$1"
  case "${dataset}" in
    cmrc2018)
      echo "${CMRC_REUSE_DIR}|${CMRC_PATCH_DIR}"
      ;;
    hotpotqa|hotpotqa_compressed)
      echo "${HOTPOT_REUSE_DIR}|${HOTPOT_PATCH_DIR}"
      ;;
    wikitext2|wikitext-2)
      echo "${WIKITEXT_REUSE_DIR}|${WIKITEXT_PATCH_DIR}"
      ;;
    *)
      echo "unsupported dataset: ${dataset}" >&2
      return 2
      ;;
  esac
}

mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/tmp_cfg" "${RUN_ROOT}/strict_eval"
cd "${ROOT}"
source "${VENV}/bin/activate"
if [[ "${INSTALL_EDITABLE}" == "1" ]]; then
  python -m pip install -e .
fi
export PYTHONPATH="${ROOT}/src"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

for required in "${MODELS_CONFIG}" "${ROUTE_PATH}" "${BASELINE_REUSE_EVAL_CONFIG}" "${PATCH_ONLY_EVAL_CONFIG}"; do
  if [[ ! -f "${required}" ]]; then
    echo "missing required file: ${required}" >&2
    exit 2
  fi
done

make_eval_config() {
  local src="$1"
  local dst="$2"
  python - "${src}" "${dst}" "${EVAL_LIMIT}" "${BANDWIDTH_GBPS}" "${PATCH_CKPT_DEVICE}" \
    "${RUNTIME_PATCH_WINDOW_SIZE}" "${RUNTIME_PATCH_PERSIST_SHARDS}" "${RUNTIME_PATCH_PERSIST_MODULES}" "${CMRC_SPLIT}" <<'PY'
import json
import sys
from pathlib import Path

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
limit = int(sys.argv[3])
bandwidth = float(sys.argv[4])
patch_ckpt_device = sys.argv[5].strip()
runtime_patch_window_size = sys.argv[6].strip()
persist_shards = sys.argv[7].strip().lower() in {"1", "true", "yes", "on"}
persist_modules = sys.argv[8].strip().lower() in {"1", "true", "yes", "on"}
cmrc_split = sys.argv[9].strip() or "dev"

cfg = json.loads(src.read_text(encoding="utf-8-sig"))
runtime_eval = cfg.setdefault("runtime_eval", {})
runtime_eval["max_eval_samples"] = limit
runtime_eval["cmrc_split"] = cmrc_split
runtime_eval["patch_ckpt_device"] = patch_ckpt_device
runtime_eval["runtime_patch_window_size"] = int(runtime_patch_window_size)
runtime_eval["runtime_patch_persist_shards"] = persist_shards
runtime_eval["runtime_patch_persist_modules"] = persist_modules
cfg.setdefault("network", {})["bandwidth_gbps"] = bandwidth
dst.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY
}

BASELINE_REUSE_CFG="${RUN_ROOT}/tmp_cfg/eval_baseline_reuse.json"
PATCH_ONLY_CFG="${RUN_ROOT}/tmp_cfg/eval_patch_only.json"
make_eval_config "${BASELINE_REUSE_EVAL_CONFIG}" "${BASELINE_REUSE_CFG}"
make_eval_config "${PATCH_ONLY_EVAL_CONFIG}" "${PATCH_ONLY_CFG}"

cat > "${RUN_ROOT}/run_config.json" <<EOF
{
  "root": "${ROOT}",
  "snapshot_root": "${SNAPSHOT_ROOT}",
  "data_root": "${DATA_ROOT}",
  "run_root": "${RUN_ROOT}",
  "datasets": "${DATASETS}",
  "eval_limit": ${EVAL_LIMIT},
  "max_length": ${MAX_LENGTH},
  "max_new_tokens": ${MAX_NEW_TOKENS},
  "bandwidth_gbps": ${BANDWIDTH_GBPS},
  "cmrc_split": "${CMRC_SPLIT}",
  "cuda_visible_devices": "${CUDA_VISIBLE_DEVICES:-}",
  "producer_device": "${PRODUCER_DEVICE}",
  "consumer_device": "${CONSUMER_DEVICE}",
  "resume": "${RESUME}",
  "route": "${ROUTE_PATH}",
  "cmrc_reuse_dir": "${CMRC_REUSE_DIR}",
  "cmrc_patch_dir": "${CMRC_PATCH_DIR}",
  "hotpot_reuse_dir": "${HOTPOT_REUSE_DIR}",
  "hotpot_patch_dir": "${HOTPOT_PATCH_DIR}",
  "wikitext_reuse_dir": "${WIKITEXT_REUSE_DIR}",
  "wikitext_patch_dir": "${WIKITEXT_PATCH_DIR}"
}
EOF

run_logged() {
  local name="$1"
  shift
  local stage_dir="${RUN_ROOT}/logs/${name}"
  mkdir -p "${stage_dir}"
  echo "+ $*" | tee "${stage_dir}/command.txt"
  /usr/bin/time -v -o "${stage_dir}/time.txt" "$@" >"${stage_dir}/stdout.log" 2>"${stage_dir}/stderr.log"
}

IFS=',' read -r -a DATASET_LIST <<< "${DATASETS}"
for dataset in "${DATASET_LIST[@]}"; do
  dataset="$(echo "${dataset}" | xargs)"
  [[ -z "${dataset}" ]] && continue
  ckpt_pair="$(ckpt_for_dataset "${dataset}")"
  reuse_dir="${ckpt_pair%%|*}"
  patch_dir="${ckpt_pair##*|}"
  if [[ ! -d "${reuse_dir}" ]]; then
    echo "missing Reuse checkpoint for ${dataset}: ${reuse_dir}" >&2
    exit 2
  fi
  if [[ ! -d "${patch_dir}" ]]; then
    echo "missing Patch checkpoint for ${dataset}: ${patch_dir}" >&2
    exit 2
  fi

  base_out="${RUN_ROOT}/strict_eval/${dataset}/baseline_reuse"
  patch_out="${RUN_ROOT}/strict_eval/${dataset}/patch_only"
  if [[ "${RESUME}" == "1" && -f "${base_out}/per_mode_summary.json" ]]; then
    echo "skip ${dataset}_baseline_reuse (RESUME=1, existing per_mode_summary.json)"
  else
    run_logged "${dataset}_baseline_reuse" \
      python -m scd_repro.cli.eval_text_strict \
        --dataset "${dataset}" \
        --models-config "${MODELS_CONFIG}" \
        --eval-config "${BASELINE_REUSE_CFG}" \
        --data-root "${DATA_ROOT}" \
        --route "${ROUTE_PATH}" \
        --reuse-ckpt "${reuse_dir}" \
        --patch-ckpt "${patch_dir}" \
        --out "${base_out}" \
        --limit "${EVAL_LIMIT}" \
        --max-length "${MAX_LENGTH}" \
        --max-new-tokens "${MAX_NEW_TOKENS}" \
        --bandwidth-gbps "${BANDWIDTH_GBPS}" \
        --producer-device "${PRODUCER_DEVICE}" \
        --consumer-device "${CONSUMER_DEVICE}" \
        --dtype bfloat16 \
        --cmrc-split "${CMRC_SPLIT}" \
        --seed "${SEED}"
  fi

  if [[ "${RESUME}" == "1" && -f "${patch_out}/per_mode_summary.json" ]]; then
    echo "skip ${dataset}_patch_only (RESUME=1, existing per_mode_summary.json)"
  else
    run_logged "${dataset}_patch_only" \
      python -m scd_repro.cli.eval_text_strict \
        --dataset "${dataset}" \
        --models-config "${MODELS_CONFIG}" \
        --eval-config "${PATCH_ONLY_CFG}" \
        --data-root "${DATA_ROOT}" \
        --route "${ROUTE_PATH}" \
        --reuse-ckpt "${reuse_dir}" \
        --patch-ckpt "${patch_dir}" \
        --out "${patch_out}" \
        --limit "${EVAL_LIMIT}" \
        --max-length "${MAX_LENGTH}" \
        --max-new-tokens "${MAX_NEW_TOKENS}" \
        --bandwidth-gbps "${BANDWIDTH_GBPS}" \
        --producer-device "${PRODUCER_DEVICE}" \
        --consumer-device "${CONSUMER_DEVICE}" \
        --dtype bfloat16 \
        --cmrc-split "${CMRC_SPLIT}" \
        --seed "${SEED}"
  fi
done

python - "${RUN_ROOT}" "${DATASETS}" <<'PY'
import json
import sys
from pathlib import Path

run_root = Path(sys.argv[1])
datasets = [x.strip() for x in sys.argv[2].split(",") if x.strip()]
rows = []
for dataset in datasets:
    for group in ["baseline_reuse", "patch_only"]:
        path = run_root / "strict_eval" / dataset / group / "per_mode_summary.json"
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        for method, row in payload.items():
            item = {"dataset": dataset, "method": method}
            item.update(row)
            rows.append(item)

(run_root / "patchonly_anchor_summary.json").write_text(
    json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
lines = [
    "# SCD-ICML Patch-only Anchor Summary",
    "",
    "| dataset | method | F1 | EM | avg NLL | PPL | TTFT ms | payload MB |",
    "|---|---|---:|---:|---:|---:|---:|---:|",
]
for row in rows:
    ppl = row.get("ppl", row.get("avg_ppl", ""))
    lines.append(
        f"| {row.get('dataset')} | {row.get('method')} | "
        f"{float(row.get('f1', 0.0)):.6f} | "
        f"{float(row.get('em', 0.0)):.6f} | "
        f"{float(row.get('avg_nll', 0.0)):.6f} | "
        f"{float(ppl) if ppl not in {'', None} else 0.0:.6f} | "
        f"{float(row.get('ttft_ms', 0.0)):.6f} | "
        f"{float(row.get('payload_mb', 0.0)):.6f} |"
    )
(run_root / "patchonly_anchor_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
PY

echo "patch-only anchor finished."
echo "RUN_ROOT=${RUN_ROOT}"
