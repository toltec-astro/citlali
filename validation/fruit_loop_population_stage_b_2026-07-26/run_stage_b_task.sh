#!/bin/bash
set -euo pipefail
umask 0022

PROJECT_ROOT="${PROJECT_ROOT:-/work/toltec/commissioning2025-test/2026-ENG-hero-multiyear-pointings-v1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/diagnostics/fruitloop_population_v1/stage_b}"
SETUP_DIR="${SETUP_DIR:-${OUTPUT_ROOT}/setup}"
JOBS="${SETUP_DIR}/stage_b_jobs.tsv"
task_id="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"

source "${SETUP_DIR}/binary.env"
line="$(sed -n "$((task_id + 1))p" "${JOBS}")"
test -n "${line}"
IFS=$'\t' read -r task obsnum config output apt rank stratum source <<<"${line}"
test "${task}" = "${task_id}"
test -f "${SETUP_DIR}/${config}"
test -f "${apt}"
echo "${CITLALI_SHA256}  ${CITLALI_SNAPSHOT}" | sha256sum -c -
if test -n "0f7685ad2b89cc2fc2cbe330c9e5ed75fc8972dc1bf60ab37e3a4b9209965330"; then
    test "${CITLALI_SHA256}" = "0f7685ad2b89cc2fc2cbe330c9e5ed75fc8972dc1bf60ab37e3a4b9209965330"
fi
if find "${output}" -mindepth 1 -print -quit 2>/dev/null | grep -q .; then
    echo "Refusing nonempty output for obs ${obsnum}: ${output}" >&2
    exit 1
fi
echo "Starting obs=${obsnum} rank=${rank} stratum=${stratum} binary=${CITLALI_SHA256}"
"${CITLALI_SNAPSHOT}" -l info "${SETUP_DIR}/${config}"

for copied_config in "${output}"/redu??/"${config}"; do
    test -f "${copied_config}"
    chmod --reference="${SETUP_DIR}/${config}" "${copied_config}"
    test -r "${copied_config}"
    expected="$(awk -v file="${config}" '$2 == file {print $1}' "${SETUP_DIR}/config_checksums.sha256")"
    test -n "${expected}"
    echo "${expected}  ${copied_config}" | sha256sum -c -
done
