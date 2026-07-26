#!/bin/bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/work/toltec/commissioning2025-test/2026-ENG-hero-multiyear-pointings-v1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/diagnostics/fruitloop_population_v1/stage_a}"
SETUP_DIR="${SETUP_DIR:-${OUTPUT_ROOT}/setup}"
ARRAY_CONCURRENCY="${ARRAY_CONCURRENCY:-4}"

"${SETUP_DIR}/preflight_stage_a.sh"
sbatch     --job-name=flpop-a     --array="1-16%${ARRAY_CONCURRENCY}"     --output="${OUTPUT_ROOT}/logs/flpop-a-%A_%a.out"     --time=24:00:00     --mem=64G     --cpus-per-task=6     --partition=toltec-cpu     --chdir="${PROJECT_ROOT}"     "${SETUP_DIR}/run_stage_a_task.sh"
