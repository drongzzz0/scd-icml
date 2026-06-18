#!/usr/bin/env bash
set -euo pipefail
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate cache
REPO=/home/c101-800-mql/maqianli/SCD-ICML/runtime_repo_codex_38dcdca_tp_20260618_1128
QUEUE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_PARENT=/home/c101-800-mql/nas/SCD-ICML/runs
MODELS_CONFIG="$QUEUE_ROOT/models_qwen25_32b_base_to_instruct_actual.json"
DATA_ROOT=/home/c101-800-mql/maqianli/no4/data
MODEL_CACHE_ROOT=/home/c101-800-mql/nas/SCD-ICML/model_cache
NAS_QWEN32_INSTRUCT=/home/c101-800-mql/nas/models/Qwen2.5-32B-Instruct
REQUIRED_GPUS=2
REQUIRED_FREE_MB=70000
MONITOR_SECONDS=60
SAMPLE_INTERVAL=10
MAX_WAIT_SECONDS=${MAX_WAIT_SECONDS:-43200}
PREFERRED=${PREFERRED_GPUS:-3,4,6,7}
select_gpus() {
  local tmp
  tmp="$(mktemp)"
  local samples=$(( MONITOR_SECONDS / SAMPLE_INTERVAL + 1 ))
  for i in $(seq 1 "$samples"); do
    nvidia-smi --query-gpu=index,memory.used,memory.free,memory.total,utilization.gpu --format=csv,noheader,nounits | awk -v sample="$i" '{print sample","$0}' >> "$tmp"
    if [ "$i" -lt "$samples" ]; then sleep "$SAMPLE_INTERVAL"; fi
  done
  python - "$tmp" "$REQUIRED_GPUS" "$REQUIRED_FREE_MB" "$PREFERRED" <<'PY'
import csv, sys, collections
path, required, min_free, preferred = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), [int(x) for x in sys.argv[4].split(',') if x]
rows=[]
with open(path) as f:
    for r in csv.reader(f):
        if len(r) >= 6:
            rows.append(dict(sample=int(r[0]), index=int(r[1]), used=int(r[2]), free=int(r[3]), total=int(r[4]), util=int(r[5])))
by=collections.defaultdict(list)
for r in rows: by[r['index']].append(r)
sample_count=len({r['sample'] for r in rows})
cands=[]
for idx, rs in by.items():
    if len(rs) != sample_count:
        continue
    if all(r['util'] == 0 for r in rs):
        mf=min(r['free'] for r in rs); mu=max(r['used'] for r in rs)
        if mf >= min_free:
            cands.append((idx,mf,mu))
def order(items):
    return sorted(items, key=lambda x: (0 if x[0] in preferred else 1, -x[1], x[2], x[0]))
selected=order(cands)[:required]
print(','.join(str(x[0]) for x in selected))
PY
  rm -f "$tmp"
}
start=$(date +%s)
while true; do
  now=$(date +%s)
  if [ $((now-start)) -gt "$MAX_WAIT_SECONDS" ]; then
    echo "timeout waiting for ${REQUIRED_GPUS} GPUs with util=0 and free>=${REQUIRED_FREE_MB}MB" >&2
    exit 3
  fi
  selected="$(select_gpus | tail -n 1)"
  if [ -n "$selected" ] && [ "$(echo "$selected" | awk -F, '{print NF}')" -ge "$REQUIRED_GPUS" ]; then
    echo "$(date -Is) selected=$selected"
    break
  fi
  echo "$(date -Is) waiting; selected='${selected}'"
  sleep 30
done
until [ -f "$NAS_QWEN32_INSTRUCT/config.json" ] && [ "$(find "$NAS_QWEN32_INSTRUCT" -maxdepth 1 -name '*.safetensors' 2>/dev/null | wc -l)" -ge 17 ]; do
  echo "$(date -Is) waiting for NAS Qwen2.5-32B-Instruct copy at $NAS_QWEN32_INSTRUCT"
  sleep 60
done
cd "$REPO"
export RUN_PARENT="$RUN_PARENT"
export MODELS_CONFIG="$MODELS_CONFIG"
export DATA_ROOT="$DATA_ROOT"
export MODEL_CACHE_ROOT="$MODEL_CACHE_ROOT"
export HF_HOME="$MODEL_CACHE_ROOT/hf_home"
export HF_HUB_CACHE="$MODEL_CACHE_ROOT/hf_hub"
export TRANSFORMERS_CACHE="$MODEL_CACHE_ROOT/transformers"
export MODELSCOPE_CACHE="$MODEL_CACHE_ROOT/modelscope"
export TORCH_HOME="$MODEL_CACHE_ROOT/torch"
export REQUIRED_GPUS=2
export REQUIRED_FREE_MB=70000
export GPU_MONITOR_SECONDS=60
export GPU_SAMPLE_INTERVAL=10
export TRAIN_LIMIT=200
export EVAL_LIMIT=200
export TRAIN_CMRC_SPLIT=train
export EVAL_CMRC_SPLIT=dev
export DATASETS=cmrc2018
export MAX_LENGTH=0
export MAX_NEW_TOKENS=32
export CONDA_ENV=cache
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
bash scripts/run_a800_scd_icml_qwen32_repro.sh
