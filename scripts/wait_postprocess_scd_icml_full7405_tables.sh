#!/usr/bin/env bash
set -euo pipefail

: "${POST_ROOT:?set POST_ROOT to the full7405 postprocess directory}"
: "${EVAL_ROOTS:?set EVAL_ROOTS to colon-separated full7405 strict eval roots}"
: "${PATCH_ANCHOR_JSONS:?set PATCH_ANCHOR_JSONS to colon-separated patchonly_anchor_summary.json files}"
: "${BASE_ROUTE_METRICS_JSONS:?set BASE_ROUTE_METRICS_JSONS to colon-separated pre-full7405 route_metrics.json files}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNTIME="${RUNTIME:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
PYTHON="${PYTHON:-python}"
export PYTHONPATH="${RUNTIME}/src${PYTHONPATH:+:${PYTHONPATH}}"
DIRECTION="${DIRECTION:-lite_to_7b}"
DATASET="${DATASET:-hotpotqa_compressed}"
METHOD="${METHOD:-scd_full}"
METRIC="${METRIC:-f1}"
MAX_METRIC_DROP="${MAX_METRIC_DROP:-0.03}"
RESULT_TABLES_DIR="${RESULT_TABLES_DIR:-${POST_ROOT}/result_tables_with_full7405}"
SLEEP_SECONDS="${SLEEP_SECONDS:-300}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-0}"
START_SECONDS="$(date +%s)"
LIST_SEPARATOR="${LIST_SEPARATOR:-:}"

IFS="${LIST_SEPARATOR}" read -r -a eval_roots <<< "${EVAL_ROOTS}"
IFS="${LIST_SEPARATOR}" read -r -a patch_anchor_jsons <<< "${PATCH_ANCHOR_JSONS}"
IFS="${LIST_SEPARATOR}" read -r -a base_route_metrics_jsons <<< "${BASE_ROUTE_METRICS_JSONS}"

if [[ "${#eval_roots[@]}" -eq 0 ]]; then
  echo "EVAL_ROOTS did not contain any roots" >&2
  exit 2
fi
if [[ "${#patch_anchor_jsons[@]}" -eq 0 ]]; then
  echo "PATCH_ANCHOR_JSONS did not contain any files" >&2
  exit 2
fi
if [[ "${#base_route_metrics_jsons[@]}" -eq 0 ]]; then
  echo "BASE_ROUTE_METRICS_JSONS did not contain any files" >&2
  exit 2
fi

mkdir -p "${POST_ROOT}"

echo "wait_postprocess_started=$(date -Is)"
echo "post_root=${POST_ROOT}"
echo "result_tables_dir=${RESULT_TABLES_DIR}"

while true; do
  ready=0
  for root in "${eval_roots[@]}"; do
    summary="${root}/scd_icml_mistral_summary.json"
    if [[ -s "${summary}" ]]; then
      ready=$((ready + 1))
    fi
  done
  if [[ "${ready}" -eq "${#eval_roots[@]}" ]]; then
    echo "full7405_summaries_ready=$(date -Is) ready=${ready}/${#eval_roots[@]}"
    break
  fi
  if [[ "${TIMEOUT_SECONDS}" != "0" ]]; then
    now="$(date +%s)"
    elapsed=$((now - START_SECONDS))
    if (( elapsed >= TIMEOUT_SECONDS )); then
      echo "timeout_waiting_for_full7405_summaries elapsed=${elapsed} ready=${ready}/${#eval_roots[@]}" >&2
      exit 124
    fi
  fi
  echo "waiting $(date -Is) summaries_ready=${ready}/${#eval_roots[@]}"
  sleep "${SLEEP_SECONDS}"
done

collect_cmd=(
  "${PYTHON}"
  "${RUNTIME}/scripts/collect_route_eval_metrics.py"
  "--direction" "${DIRECTION}"
  "--dataset" "${DATASET}"
  "--method" "${METHOD}"
  "--out-json" "${POST_ROOT}/full7405_route_metrics.json"
  "--out-jsonl" "${POST_ROOT}/full7405_route_metrics.jsonl"
)
for root in "${eval_roots[@]}"; do
  collect_cmd+=("--eval-root" "${root}")
done

echo "running_collect=$(date -Is)"
"${collect_cmd[@]}"

select_cmd=(
  "${PYTHON}"
  "${RUNTIME}/scripts/select_quality_gated_scd_route.py"
  "--dataset" "${DATASET}"
  "--metric" "${METRIC}"
  "--max-metric-drop" "${MAX_METRIC_DROP}"
  "--out-report" "${POST_ROOT}/hotpotqa_full7405_f1_drop003_selection.json"
)
for root in "${eval_roots[@]}"; do
  select_cmd+=("--eval-root" "${root}")
done

echo "running_select=$(date -Is)"
"${select_cmd[@]}" \
  > "${POST_ROOT}/hotpotqa_full7405_f1_drop003_selection.stdout" \
  2> "${POST_ROOT}/hotpotqa_full7405_f1_drop003_selection.stderr"

tables_cmd=(
  "${PYTHON}"
  "${RUNTIME}/scripts/build_scd_icml_result_tables.py"
  "--out-dir" "${RESULT_TABLES_DIR}"
)
for path in "${patch_anchor_jsons[@]}"; do
  tables_cmd+=("--patch-anchor-json" "${path}")
done
for path in "${base_route_metrics_jsons[@]}"; do
  tables_cmd+=("--route-metrics-json" "${path}")
done
tables_cmd+=("--route-metrics-json" "${POST_ROOT}/full7405_route_metrics.json")

echo "running_tables=$(date -Is)"
"${tables_cmd[@]}"
echo "wait_postprocess_done=$(date -Is)"
