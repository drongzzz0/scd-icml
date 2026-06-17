#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/disk_n/c101-3-mql/projects/SCD-ICML/repro_current_from_codex_20260607-154030/runtime_repo}"
WAIT_PID="${WAIT_PID:-}"
CHECK_INTERVAL_SEC="${CHECK_INTERVAL_SEC:-300}"

ROUTE_DIR="${ROUTE_DIR:-/disk_n/c101-3-mql/projects/SCD-ICML/runs/route_ablation_cmrcbank_20260617_1253/routes}"
RUN_PARENT="${RUN_PARENT:-/disk_n/c101-3-mql/projects/SCD-ICML/runs/route_ablation_cmrcbank_20260617_1253}"
VARIANTS="${VARIANTS:-patch_all32}"

HOTPOT_PRODUCER_GPU="${HOTPOT_PRODUCER_GPU:-0}"
HOTPOT_CONSUMER_GPU="${HOTPOT_CONSUMER_GPU:-1}"
WIKITEXT_PRODUCER_GPU="${WIKITEXT_PRODUCER_GPU:-2}"
WIKITEXT_CONSUMER_GPU="${WIKITEXT_CONSUMER_GPU:-3}"

cd "${ROOT}"

if [ -n "${WAIT_PID}" ]; then
  while kill -0 "${WAIT_PID}" 2>/dev/null; do
    echo "$(date +%F_%T) waiting_for_pid=${WAIT_PID}"
    sleep "${CHECK_INTERVAL_SEC}"
  done
fi

for variant in ${VARIANTS}; do
  route_path="${ROUTE_DIR}/${variant}.json"
  if [ ! -f "${route_path}" ]; then
    echo "missing route: ${route_path}" >&2
    exit 1
  fi
  echo "$(date +%F_%T) start_variant=${variant} route=${route_path}"
  RUN_BASE="${RUN_PARENT}/${variant}" \
    HOTPOT_LIMIT=0 \
    WIKITEXT_LIMIT=0 \
    RUN_HOTPOT="${RUN_HOTPOT:-1}" \
    RUN_WIKITEXT="${RUN_WIKITEXT:-1}" \
    HOTPOT_PRODUCER_GPU="${HOTPOT_PRODUCER_GPU}" \
    HOTPOT_CONSUMER_GPU="${HOTPOT_CONSUMER_GPU}" \
    WIKITEXT_PRODUCER_GPU="${WIKITEXT_PRODUCER_GPU}" \
    WIKITEXT_CONSUMER_GPU="${WIKITEXT_CONSUMER_GPU}" \
    ROUTE_PATH="${route_path}" \
    WAIT=1 \
    bash scripts/run_ser_extra_dataset_eval.sh
  echo "$(date +%F_%T) done_variant=${variant}"
done
