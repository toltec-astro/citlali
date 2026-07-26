#!/bin/bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/work/toltec/commissioning2025-test/2026-ENG-hero-multiyear-pointings-v1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/diagnostics/fruitloop_population_v1/stage_a}"
SETUP_DIR="${SETUP_DIR:-${OUTPUT_ROOT}/setup}"
JOBS="${SETUP_DIR}/stage_a_jobs.tsv"
MIN_FREE_KIB="${MIN_FREE_KIB:-31457280}"

test -f "${SETUP_DIR}/binary.env"
source "${SETUP_DIR}/binary.env"
test -x "${CITLALI_SNAPSHOT}"
echo "${CITLALI_SHA256}  ${CITLALI_SNAPSHOT}" | sha256sum -c -
(cd "${SETUP_DIR}" && sha256sum -c config_checksums.sha256)
test -d "${PROJECT_ROOT}/apts/hero_rc1"
test -d "${PROJECT_ROOT}/data"
free_kib="$(df -Pk "${OUTPUT_ROOT}" | awk 'NR == 2 {print $4}')"
if test -z "${free_kib}" || test "${free_kib}" -lt "${MIN_FREE_KIB}"; then
    echo "Refusing launch with less than ${MIN_FREE_KIB} KiB free at ${OUTPUT_ROOT}" >&2
    exit 1
fi

n_jobs=0
while IFS=$'\t' read -r task obsnum config output apt rank stratum source; do
    test "${task}" = "$((n_jobs + 1))"
    test -f "${SETUP_DIR}/${config}"
    test -f "${apt}"
    while read -r input_path; do
        test -f "${input_path}"
    done < <(awk '/^[[:space:]]*- filepath: / {print $3}' "${SETUP_DIR}/${config}")
    if find "${output}" -mindepth 1 -print -quit 2>/dev/null | grep -q .; then
        echo "Refusing nonempty output for obs ${obsnum}: ${output}" >&2
        exit 1
    fi
    n_jobs=$((n_jobs + 1))
done < <(tail -n +2 "${JOBS}")

test "${n_jobs}" -eq 16
mkdir -p "${OUTPUT_ROOT}/logs"
echo "Stage A preflight passed for ${n_jobs} jobs; ${free_kib} KiB free."
