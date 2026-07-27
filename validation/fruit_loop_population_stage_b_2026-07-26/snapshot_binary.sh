#!/bin/bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/work/toltec/commissioning2025-test/2026-ENG-hero-multiyear-pointings-v1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/diagnostics/fruitloop_population_v1/stage_b}"
SETUP_DIR="${SETUP_DIR:-${OUTPUT_ROOT}/setup}"
CITLALI_SOURCE="${CITLALI_SOURCE:-/work/toltec/commissioning2025-test/2026-ENG-hero-multiyear-pointings-v1/diagnostics/fruitloop_population_v1/stage_a/setup/bin/citlali-0f7685ad2b89cc2fc2cbe330c9e5ed75fc8972dc1bf60ab37e3a4b9209965330}"
EXPECTED_CITLALI_SHA256="${EXPECTED_CITLALI_SHA256:-0f7685ad2b89cc2fc2cbe330c9e5ed75fc8972dc1bf60ab37e3a4b9209965330}"
BIN_DIR="${SETUP_DIR}/bin"

test -x "${CITLALI_SOURCE}"
mkdir -p "${BIN_DIR}"
source_sha="$(sha256sum "${CITLALI_SOURCE}" | awk '{print $1}')"
if test -n "${EXPECTED_CITLALI_SHA256}" &&
   test "${source_sha}" != "${EXPECTED_CITLALI_SHA256}"; then
    echo "Citlali SHA256 mismatch: expected ${EXPECTED_CITLALI_SHA256}, got ${source_sha}" >&2
    exit 1
fi
snapshot="${BIN_DIR}/citlali-${source_sha}"
if test ! -e "${snapshot}"; then
    install -m 0755 "${CITLALI_SOURCE}" "${snapshot}"
fi
echo "${source_sha}  ${snapshot}" | sha256sum -c -
"${snapshot}" --version >"${BIN_DIR}/citlali-${source_sha}.version.txt" 2>&1 || true
printf 'CITLALI_SNAPSHOT=%s\nCITLALI_SHA256=%s\n'     "${snapshot}" "${source_sha}" >"${SETUP_DIR}/binary.env"
echo "Frozen Citlali ${source_sha} at ${snapshot}"
