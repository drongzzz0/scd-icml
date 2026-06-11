#!/usr/bin/env bash
set -euo pipefail

RUNTIME_ROOT="${RUNTIME_ROOT:-$(pwd)}"
SOURCE_RUN_ROOT="${SOURCE_RUN_ROOT:-}"
DIRECTION="${DIRECTION:-lite_to_7b}"
SEED="${SEED:-42}"
TRAIN_LIMIT="${TRAIN_LIMIT:-200}"
RUN_ROOT="${RUN_ROOT:-${RUNTIME_ROOT}/runs/scd_icml_parallel_nonshared_patch31_$(date +%Y%m%d_%H%M%S)}"
TRAIN_CONFIG="${TRAIN_CONFIG:-configs/mistral_self_patch_xnorm_nativekv_residual_mlp_parallel_e1_h2048_b2_20260404.json}"
MODELS_CONFIG="${MODELS_CONFIG:-configs/models_ser_mistrallite_to_mistral7_mql.json}"
EVAL_CONFIG="${EVAL_CONFIG:-configs/eval_scd_icml_no_fallback_no_recompute.json}"
PATCH_LAYERS="${PATCH_LAYERS:-1-31}"
SHARDS="${SHARDS:-0:1-8:shard_01_08;1:9-16:shard_09_16;2:17-24:shard_17_24;3:25-31:shard_25_31}"
PAIRS="${PAIRS:-}"
PROFILE="${PROFILE:-}"
WAIT_FOR_SHARDS="${WAIT_FOR_SHARDS:-0}"
MERGE_AFTER="${MERGE_AFTER:-0}"
FINAL_PATCH_DIR="${FINAL_PATCH_DIR:-${RUN_ROOT}/${DIRECTION}/ckpt_patch/patch_cmrc2018_seed${SEED}}"
FINAL_ROUTE_PATH="${FINAL_ROUTE_PATH:-${RUN_ROOT}/${DIRECTION}/route/route_cmrc2018_seed${SEED}.json}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"

if [[ -z "${SOURCE_RUN_ROOT}" && ( -z "${PAIRS}" || -z "${PROFILE}" ) ]]; then
  echo "Set SOURCE_RUN_ROOT, or set both PAIRS and PROFILE." >&2
  exit 2
fi

if [[ -z "${PAIRS}" ]]; then
  PAIRS="${SOURCE_RUN_ROOT}/${DIRECTION}/pairs/cmrc2018_train_seed${SEED}_n${TRAIN_LIMIT}"
fi
if [[ -z "${PROFILE}" ]]; then
  PROFILE="${SOURCE_RUN_ROOT}/${DIRECTION}/route/profile_cmrc2018_seed${SEED}.json"
fi

if [[ ! -d "${PAIRS}" ]]; then
  echo "missing pairs directory: ${PAIRS}" >&2
  exit 2
fi
if [[ ! -f "${PROFILE}" ]]; then
  echo "missing profile file: ${PROFILE}" >&2
  exit 2
fi

cd "${RUNTIME_ROOT}"
mkdir -p "${RUN_ROOT}/${DIRECTION}/route" "${RUN_ROOT}/${DIRECTION}/ckpt_patch_shards" "${RUN_ROOT}/logs"

python - "${RUN_ROOT}" "${DIRECTION}" "${SHARDS}" "${PAIRS}" "${PROFILE}" "${TRAIN_CONFIG}" "${MODELS_CONFIG}" "${PATCH_LAYERS}" <<'PY'
import json
import sys
from pathlib import Path


def parse_layers(spec: str) -> list[int]:
    layers: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            layers.extend(range(int(start), int(end) + 1))
        else:
            layers.append(int(part))
    return layers


run = Path(sys.argv[1])
direction, shard_spec, pairs, profile, train_config, models_config, patch_layers_spec = sys.argv[2:9]
route_dir = run / direction / "route"
route_dir.mkdir(parents=True, exist_ok=True)

shards: dict[str, dict[str, object]] = {}
for row in shard_spec.split(";"):
    row = row.strip()
    if not row:
        continue
    gpu, layers_spec, name = row.split(":", 2)
    layers = parse_layers(layers_spec)
    shards[name] = {"gpu": int(gpu), "layers": layers}
    (route_dir / f"train_route_{name}.json").write_text(
        json.dumps(
            {
                "num_layers": 32,
                "patch_layers": layers,
                "reuse_layers": [layer for layer in range(32) if layer not in layers],
                "fallback_layers": [],
                "selected_recompute_layers": [],
                "recompute_segment_length_by_layer": {},
                "route_semantics": "train_patch_shard_layer_local_only",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

manifest = {
    "run_root": str(run),
    "direction": direction,
    "source_pairs": pairs,
    "source_profile": profile,
    "train_config": train_config,
    "models_config": models_config,
    "target_final_patch_layers": parse_layers(patch_layers_spec),
    "target_final_reuse_layers": [0],
    "fallback_layers": [],
    "recompute_layers": [],
    "shards": shards,
}
(run / "parallel_shard_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
PY

declare -a shard_out_dirs=()
declare -a shard_pids=()

IFS=";" read -r -a shard_rows <<< "${SHARDS}"
for row in "${shard_rows[@]}"; do
  [[ -z "${row}" ]] && continue
  IFS=":" read -r gpu _layers name <<< "${row}"
  route_path="${RUN_ROOT}/${DIRECTION}/route/train_route_${name}.json"
  out_dir="${RUN_ROOT}/${DIRECTION}/ckpt_patch_shards/${name}"
  log_dir="${RUN_ROOT}/logs/${name}"
  mkdir -p "${out_dir}" "${log_dir}"
  CUDA_VISIBLE_DEVICES="${gpu}" nohup bash -lc \
    "cd '${RUNTIME_ROOT}' && export PYTHONPATH=src:\${PYTHONPATH:-} OMP_NUM_THREADS='${OMP_NUM_THREADS}' MKL_NUM_THREADS='${MKL_NUM_THREADS}' OPENBLAS_NUM_THREADS='${OPENBLAS_NUM_THREADS}' OPENBLAS_MAIN_FREE=1 && /usr/bin/time -v -o '${log_dir}/time.txt' python -m scd_repro.cli.train_patch --pairs '${PAIRS}' --profile '${PROFILE}' --route '${route_path}' --train-config '${TRAIN_CONFIG}' --models-config '${MODELS_CONFIG}' --out '${out_dir}' --seed '${SEED}' --train-device cuda:0" \
    > "${log_dir}/stdout.log" 2> "${log_dir}/stderr.log" &
  pid="$!"
  echo "${pid}" > "${log_dir}/pid.txt"
  shard_pids+=("${pid}")
  shard_out_dirs+=("${out_dir}")
  echo "launched ${name} gpu=${gpu} pid=${pid} route=${route_path} out=${out_dir}"
done

echo "run_root=${RUN_ROOT}"
echo "manifest=${RUN_ROOT}/parallel_shard_manifest.json"

if [[ "${WAIT_FOR_SHARDS}" == "1" ]]; then
  for pid in "${shard_pids[@]}"; do
    wait "${pid}"
  done
fi

if [[ "${MERGE_AFTER}" == "1" ]]; then
  if [[ "${WAIT_FOR_SHARDS}" != "1" ]]; then
    echo "MERGE_AFTER=1 requires WAIT_FOR_SHARDS=1" >&2
    exit 2
  fi
  python scripts/merge_patch_ckpts.py --out "${FINAL_PATCH_DIR}" "${shard_out_dirs[@]}"
  python scripts/make_paper_scd_route.py \
    --profile "${PROFILE}" \
    --patch-dir "${FINAL_PATCH_DIR}" \
    --models-config "${MODELS_CONFIG}" \
    --eval-config "${EVAL_CONFIG}" \
    --patch-layers "${PATCH_LAYERS}" \
    --out "${FINAL_ROUTE_PATH}"
  python scripts/validate_scd_route.py --route "${FINAL_ROUTE_PATH}" --require-mixed
  echo "merged_patch_dir=${FINAL_PATCH_DIR}"
  echo "final_route=${FINAL_ROUTE_PATH}"
fi
