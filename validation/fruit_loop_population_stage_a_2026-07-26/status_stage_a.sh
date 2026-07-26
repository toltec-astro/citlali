#!/bin/bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/work/toltec/commissioning2025-test/2026-ENG-hero-multiyear-pointings-v1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/diagnostics/fruitloop_population_v1/stage_a}"
SETUP_DIR="${SETUP_DIR:-${OUTPUT_ROOT}/setup}"
JOBS="${SETUP_DIR}/stage_a_jobs.tsv"

squeue -u "${USER}" -n flpop-a || true
printf 'obsnum\trank\tstratum\titerations\tstate\n'
while IFS=$'\t' read -r task obsnum config output apt rank stratum source; do
    iterations=0
    if test -d "${output}"; then
        iterations="$(find "${output}" -mindepth 1 -maxdepth 1 -type d -name 'redu??' | wc -l | tr -d ' ')"
    fi
    state=not_started
    if test "${iterations}" -eq 10; then
        state=products_present
    elif test "${iterations}" -gt 0; then
        state=partial
    fi
    printf '%s\t%s\t%s\t%s\t%s\n'         "${obsnum}" "${rank}" "${stratum}" "${iterations}" "${state}"
done < <(tail -n +2 "${JOBS}")

echo
echo "Potential error-level log lines:"
grep -EHi '(^|[^[:alpha:]])(error|fatal)([^[:alpha:]]|$)'     "${OUTPUT_ROOT}"/logs/flpop-a-*.out 2>/dev/null || true
