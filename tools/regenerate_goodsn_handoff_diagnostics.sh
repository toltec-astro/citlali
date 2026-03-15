#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

GOODSN_ROOT="${GOODSN_ROOT:-/Users/wilson/work_toltec/local_data/2025-C1-COM-04/GOODS-N}"
UTILS_ROOT="${UTILS_ROOT:-}"
PYTHON_BIN="${PYTHON_BIN:-}"

usage() {
    cat <<'EOF'
Usage:
  tools/regenerate_goodsn_handoff_diagnostics.sh [--dry-run]

Environment overrides:
  GOODSN_ROOT  Local GOODS-N workdir root
  UTILS_ROOT   Path to toltec-data-product-utilities
  PYTHON_BIN   Python executable with numpy/netCDF4/scipy/matplotlib

What this regenerates:
  - redu04 a1100 RTC/PTC residual report
  - redu04 a1100 blank-sky audit smoketest
  - redu09 a1100 blank-sky audit
  - redu10 a1100 blank-sky audit
  - redu09 a1100 MP low-band estimate
  - redu10 a1100 MP low-band estimate
  - redu09 a1100 MP full-band estimate
  - redu10 a1100 MP full-band estimate

Outputs are written into the same directory names referenced by the March 13
handoff note.
EOF
}

dry_run=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        -n|--dry-run)
            dry_run=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

pick_utils_root() {
    local candidates=(
        "$UTILS_ROOT"
        "$HOME/Documents/GitHub/toltec-data-product-utilities"
        "$HOME/GitHub/toltec-data-product-utilities"
    )
    local p
    for p in "${candidates[@]}"; do
        if [[ -n "$p" && -d "$p" ]]; then
            printf '%s\n' "$p"
            return 0
        fi
    done
    return 1
}

python_has_deps() {
    local py="$1"
    "$py" - <<'PY' >/dev/null 2>&1
import importlib
for name in ("numpy", "netCDF4", "scipy", "matplotlib"):
    importlib.import_module(name)
PY
}

pick_python() {
    local candidates=(
        "$PYTHON_BIN"
        "$HOME/toltec/bin/python"
        "$HOME/tolteca/bin/python"
        "$HOME/miniforge3/envs/tolteca/bin/python"
        "$HOME/miniforge3/bin/python"
        "$HOME/mambaforge/envs/tolteca/bin/python"
        "$HOME/mambaforge/bin/python"
        "/opt/homebrew/bin/python3"
        "/usr/local/bin/python3"
    )
    local py
    for py in "${candidates[@]}"; do
        if [[ -n "$py" && -x "$py" ]] && python_has_deps "$py"; then
            printf '%s\n' "$py"
            return 0
        fi
    done
    return 1
}

run_cmd() {
    printf '+'
    printf ' %q' "$@"
    printf '\n'
    if [[ "$dry_run" == false ]]; then
        "$@"
    fi
}

require_file() {
    local p="$1"
    if [[ ! -e "$p" ]]; then
        echo "Required file is missing: $p" >&2
        exit 1
    fi
}

if [[ ! -d "$GOODSN_ROOT" ]]; then
    echo "GOODSN_ROOT does not exist: $GOODSN_ROOT" >&2
    exit 1
fi

if ! UTILS_ROOT="$(pick_utils_root)"; then
    echo "Could not find toltec-data-product-utilities. Set UTILS_ROOT." >&2
    exit 1
fi

if ! PYTHON_BIN="$(pick_python)"; then
    cat >&2 <<'EOF'
Could not find a Python with the required analysis dependencies.
Set PYTHON_BIN to an interpreter that has:
  numpy, netCDF4, scipy, matplotlib
EOF
    exit 1
fi

REDU04_DIR="$GOODSN_ROOT/reduced/redu04"
REDU09_DIR="$GOODSN_ROOT/reduced/redu09"
REDU10_DIR="$GOODSN_ROOT/reduced/redu10"

REDU04_PTC="$REDU04_DIR/151930/raw/toltec_commissioning_science_151930_ptc_timestream.nc"
REDU04_RTC="$REDU04_DIR/151930/raw/toltec_commissioning_science_151930_rtc_timestream.nc"
REDU09_PTC="$REDU09_DIR/151930/raw/toltec_commissioning_science_151930_ptc_timestream.nc"
REDU09_RTC="$REDU09_DIR/151930/raw/toltec_commissioning_science_151930_rtc_timestream.nc"
REDU10_PTC="$REDU10_DIR/152524/raw/toltec_commissioning_science_152524_ptc_timestream.nc"
REDU10_RTC="$REDU10_DIR/152524/raw/toltec_commissioning_science_152524_rtc_timestream.nc"

for p in \
    "$REDU04_PTC" "$REDU04_RTC" \
    "$REDU09_PTC" "$REDU09_RTC" \
    "$REDU10_PTC" "$REDU10_RTC"
do
    require_file "$p"
done

echo "Python:      $PYTHON_BIN"
echo "Utils root:  $UTILS_ROOT"
echo "GOODS-N:     $GOODSN_ROOT"
if [[ "$dry_run" == true ]]; then
    echo "Mode:        dry-run"
else
    echo "Mode:        live"
fi

run_cmd "$PYTHON_BIN" \
    "$GOODSN_ROOT/tools/analyze_rtc_ptc_residuals.py" \
    --redu-dir "$REDU04_DIR" \
    --utils-root "$UTILS_ROOT" \
    --array a1100 \
    --outdir "$REDU04_DIR/analysis_timestream_residuals_a1100"

run_cmd "$PYTHON_BIN" \
    "$REPO_ROOT/tools/blank_sky/blank_sky_null_audit.py" \
    --nc-file "$REDU04_PTC" \
    --array a1100 \
    --networks 0,1,2,3,4,5 \
    --scans all \
    --utils-root "$UTILS_ROOT" \
    --outdir "$REDU04_DIR/151930/raw/blank_sky_null_audit_smoketest"

run_cmd "$PYTHON_BIN" \
    "$REPO_ROOT/tools/blank_sky/blank_sky_null_audit.py" \
    --nc-file "$REDU09_PTC" \
    --array a1100 \
    --networks 0,1,2,3,4,5 \
    --scans all \
    --utils-root "$UTILS_ROOT" \
    --outdir "$REDU09_DIR/151930/raw/blank_sky_null_audit_a1100"

run_cmd "$PYTHON_BIN" \
    "$REPO_ROOT/tools/blank_sky/blank_sky_null_audit.py" \
    --nc-file "$REDU10_PTC" \
    --array a1100 \
    --networks 0,1,2,3,4,5 \
    --scans all \
    --utils-root "$UTILS_ROOT" \
    --outdir "$REDU10_DIR/152524/raw/blank_sky_null_audit_a1100"

run_cmd "$PYTHON_BIN" \
    "$REPO_ROOT/tools/blank_sky/mp_mode_estimator.py" \
    --nc-file "$REDU09_RTC" \
    --array a1100 \
    --networks 0,1,2,3,4,5 \
    --scans all \
    --band-low-hz 0.05 \
    --band-high-hz 0.5 \
    --configured-k 18 \
    --outdir "$REDU09_DIR/151930/raw/mp_mode_estimate_lowband"

run_cmd "$PYTHON_BIN" \
    "$REPO_ROOT/tools/blank_sky/mp_mode_estimator.py" \
    --nc-file "$REDU10_RTC" \
    --array a1100 \
    --networks 0,1,2,3,4,5 \
    --scans all \
    --band-low-hz 0.05 \
    --band-high-hz 0.5 \
    --configured-k 18 \
    --outdir "$REDU10_DIR/152524/raw/mp_mode_estimate_lowband"

run_cmd "$PYTHON_BIN" \
    "$REPO_ROOT/tools/blank_sky/mp_mode_estimator.py" \
    --nc-file "$REDU09_RTC" \
    --array a1100 \
    --networks 0,1,2,3,4,5 \
    --scans all \
    --configured-k 18 \
    --outdir "$REDU09_DIR/151930/raw/mp_mode_estimate_fullband"

run_cmd "$PYTHON_BIN" \
    "$REPO_ROOT/tools/blank_sky/mp_mode_estimator.py" \
    --nc-file "$REDU10_RTC" \
    --array a1100 \
    --networks 0,1,2,3,4,5 \
    --scans all \
    --configured-k 18 \
    --outdir "$REDU10_DIR/152524/raw/mp_mode_estimate_fullband"
