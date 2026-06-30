#!/usr/bin/env bash
# Compare two Citlali manifests with the deterministic refactor validation policy.
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  tools/baseline/compare_deterministic_manifests.sh BASELINE.json CANDIDATE.json [compare_manifests.py args...]

Defaults:
  CITLALI_BASELINE_ATOL=2e-8
  CITLALI_BASELINE_RTOL=1e-10
  PYTHON=$HOME/tolteca/bin/python

This wrapper is intended for deterministic seq/one-thread refactor comparisons.
It ignores file hashes, paths, mtimes, aggregate byte totals, per-file byte
sizes, run case labels, and log line-count drift while preserving structured
FITS, netCDF, CSV, and ECSV comparisons.
USAGE
}

if [[ $# -lt 2 ]]; then
  usage >&2
  exit 2
fi

baseline_manifest=$1
candidate_manifest=$2
shift 2

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${PYTHON:-${HOME}/tolteca/bin/python}"
if [[ ! -x "${python_bin}" ]]; then
  python_bin="$(command -v python3 || true)"
fi
if [[ -z "${python_bin}" ]]; then
  echo "No Python executable found. Set PYTHON or install python3." >&2
  exit 2
fi

atol="${CITLALI_BASELINE_ATOL:-2e-8}"
rtol="${CITLALI_BASELINE_RTOL:-1e-10}"

exec "${python_bin}" "${script_dir}/compare_manifests.py" \
  "${baseline_manifest}" \
  "${candidate_manifest}" \
  --ignore-sha256 \
  --atol "${atol}" \
  --rtol "${rtol}" \
  --ignore run.case \
  --ignore 'files.*.path' \
  --ignore 'aggregate.total_size_bytes' \
  --ignore 'files.*.size_bytes' \
  --ignore 'files.*log*.summary.line_count' \
  "$@"
