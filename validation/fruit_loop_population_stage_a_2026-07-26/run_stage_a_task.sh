#!/bin/bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/work/toltec/commissioning2025-test/2026-ENG-hero-multiyear-pointings-v1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/diagnostics/fruitloop_population_v1/stage_a}"
SETUP_DIR="${SETUP_DIR:-${OUTPUT_ROOT}/setup}"
JOBS="${SETUP_DIR}/stage_a_jobs.tsv"
task_id="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"

source "${SETUP_DIR}/binary.env"
line="$(sed -n "$((task_id + 1))p" "${JOBS}")"
test -n "${line}"
IFS=$'\t' read -r task obsnum config output apt rank stratum source <<<"${line}"
test "${task}" = "${task_id}"
test -f "${SETUP_DIR}/${config}"
test -f "${apt}"
echo "${CITLALI_SHA256}  ${CITLALI_SNAPSHOT}" | sha256sum -c -
if find "${output}" -mindepth 1 -print -quit 2>/dev/null | grep -q .; then
    echo "Refusing nonempty output for obs ${obsnum}: ${output}" >&2
    exit 1
fi
echo "Starting obs=${obsnum} rank=${rank} stratum=${stratum} binary=${CITLALI_SHA256}"
exec "${CITLALI_SNAPSHOT}" -l info "${SETUP_DIR}/${config}"
