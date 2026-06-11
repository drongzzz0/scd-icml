#!/usr/bin/env bash
set -euo pipefail

: "${POST_ROOT:?set POST_ROOT to the full7405 postprocess directory}"
: "${PROFILE_FIGURES_DIR:?set PROFILE_FIGURES_DIR to generated profile figures}"
: "${STACK_VIZ_DIR:?set STACK_VIZ_DIR to generated stack/t-SNE artifacts}"
: "${OFFLINE_COST_DIR:?set OFFLINE_COST_DIR to generated offline-cost artifacts}"
: "${NEIGHBORING_BASELINES_DIR:?set NEIGHBORING_BASELINES_DIR to generated neighboring-baseline table artifacts}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNTIME="${RUNTIME:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
PYTHON="${PYTHON:-python}"
export PYTHONPATH="${RUNTIME}/src${PYTHONPATH:+:${PYTHONPATH}}"
RESULT_TABLES_DIR="${RESULT_TABLES_DIR:-${POST_ROOT}/result_tables_with_full7405}"
OUT_DIR="${OUT_DIR:-${POST_ROOT}/paper_artifact_bundle_with_full7405}"
SLEEP_SECONDS="${SLEEP_SECONDS:-300}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-0}"
if [[ -z "${SKIP_PAPER_CHECK:-}" ]]; then
  if [[ -n "${PAPER_TEX:-}" ]]; then
    SKIP_PAPER_CHECK="0"
  else
    SKIP_PAPER_CHECK="1"
  fi
fi
START_SECONDS="$(date +%s)"

echo "wait_finalize_started=$(date -Is)"
echo "result_tables_dir=${RESULT_TABLES_DIR}"
echo "out_dir=${OUT_DIR}"
echo "neighboring_baselines_dir=${NEIGHBORING_BASELINES_DIR}"

while true; do
  table="${RESULT_TABLES_DIR}/mixed_route_boundary_table.csv"
  if [[ -s "${table}" ]] && grep -q "full7405" "${table}"; then
    echo "full7405_table_ready=$(date -Is)"
    break
  fi
  if [[ "${TIMEOUT_SECONDS}" != "0" ]]; then
    now="$(date +%s)"
    elapsed=$((now - START_SECONDS))
    if (( elapsed >= TIMEOUT_SECONDS )); then
      echo "timeout_waiting_for_full7405_table elapsed=${elapsed}" >&2
      exit 124
    fi
  fi
  echo "waiting $(date -Is) table_ready=no"
  sleep "${SLEEP_SECONDS}"
done

cmd=(
  "${PYTHON}"
  "${RUNTIME}/scripts/finalize_scd_icml_full7405_bundle.py"
  "--result-tables-dir" "${RESULT_TABLES_DIR}"
  "--profile-figures-dir" "${PROFILE_FIGURES_DIR}"
  "--stack-viz-dir" "${STACK_VIZ_DIR}"
  "--offline-cost-dir" "${OFFLINE_COST_DIR}"
  "--neighboring-baselines-dir" "${NEIGHBORING_BASELINES_DIR}"
  "--out-dir" "${OUT_DIR}"
)

if [[ "${SKIP_PAPER_CHECK}" == "1" ]]; then
  cmd+=("--skip-paper-check")
else
  : "${PAPER_TEX:?set PAPER_TEX or use SKIP_PAPER_CHECK=1}"
  cmd+=("--paper-tex" "${PAPER_TEX}")
  if [[ -n "${PAPER_BASE_DIR:-}" ]]; then
    cmd+=("--paper-base-dir" "${PAPER_BASE_DIR}")
  fi
fi

echo "running_finalizer=$(date -Is)"
"${cmd[@]}"
echo "wait_finalize_done=$(date -Is)"
