#!/bin/bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/work/toltec/commissioning2025-test/2026-ENG-hero-multiyear-pointings-v1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/diagnostics/fruitloop_population_v1/stage_a}"
SETUP_DIR="${SETUP_DIR:-${OUTPUT_ROOT}/setup}"
CITLALI_SOURCE="${CITLALI_SOURCE:-/work/toltec/citlali_dev/citlali_refactor/build/bin/citlali}"
BIN_DIR="${SETUP_DIR}/bin"

test -x "${CITLALI_SOURCE}"
mkdir -p "${BIN_DIR}"
source_sha="$(sha256sum "${CITLALI_SOURCE}" | awk '{print $1}')"
snapshot="${BIN_DIR}/citlali-${source_sha}"
if test ! -e "${snapshot}"; then
    install -m 0755 "${CITLALI_SOURCE}" "${snapshot}"
fi
echo "${source_sha}  ${snapshot}" | sha256sum -c -
"${snapshot}" --version >"${BIN_DIR}/citlali-${source_sha}.version.txt" 2>&1 || true
printf 'CITLALI_SNAPSHOT=%s\nCITLALI_SHA256=%s\n'     "${snapshot}" "${source_sha}" >"${SETUP_DIR}/binary.env"
echo "Frozen Citlali ${source_sha} at ${snapshot}"
