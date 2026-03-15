#!/usr/bin/env bash

set -euo pipefail

UNITY="${UNITY:-unity_toltec}"
SRC="${SRC:-/work/toltec/commissioning2025-test/2025-C1-COM-04/wilson/GOODS-N}"
DST="${DST:-/Users/wilson/work_toltec/local_data/2025-C1-COM-04/GOODS-N}"

usage() {
    cat <<'EOF'
Usage:
  tools/sync_goodsn_handoff_from_unity.sh [--dry-run] [--essential-timestreams] [--historical-ptc]

Environment overrides:
  UNITY   SSH host alias for Unity
  SRC     GOODS-N source directory on Unity
  DST     Local destination directory

Examples:
  tools/sync_goodsn_handoff_from_unity.sh --dry-run
  tools/sync_goodsn_handoff_from_unity.sh
  tools/sync_goodsn_handoff_from_unity.sh --essential-timestreams
  tools/sync_goodsn_handoff_from_unity.sh --essential-timestreams --historical-ptc

Notes:
  --essential-timestreams pulls the minimum TOD set to rebuild the current
  restart diagnostics:
    redu04 151930 rtc+ptc
    redu09 151930 rtc+ptc
    redu10 152524 rtc+ptc
  This is about 8.6G on Unity.

  --historical-ptc pulls older PTC files used for the earlier null-audit
  progression:
    redu03 151930 ptc
    redu05 151930 ptc
    redu06 151930 ptc
    redu07 151930 ptc
    redu08 151930 ptc
  This is about 8.9G on Unity.
EOF
}

dry_run=false
include_essential_timestreams=false
include_historical_ptc=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        -n|--dry-run)
            dry_run=true
            shift
            ;;
        --essential-timestreams)
            include_essential_timestreams=true
            shift
            ;;
        --historical-ptc)
            include_historical_ptc=true
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

if ! command -v rsync >/dev/null 2>&1; then
    echo "rsync not found on PATH" >&2
    exit 1
fi

if [[ ! -d "$DST" ]]; then
    echo "Destination directory does not exist: $DST" >&2
    exit 1
fi

rsync_args=(
    -avz
    --info=progress2
    --prune-empty-dirs
    "--exclude=*.fits"
)

if [[ "$dry_run" == true ]]; then
    rsync_args+=(-n)
fi

include() {
    rsync_args+=("--include=$1")
}

include "/DEEP_DIVE_2026-03-11.md"

include "/reduced/"

include "/reduced/redu03/"
include "/reduced/redu03/*.yaml"
include "/reduced/redu03/151930/"
include "/reduced/redu03/151930/raw/"
include "/reduced/redu03/151930/raw/blank_sky_null_audit/***"
include "/reduced/redu03/151930/raw/localize_scan046_nw02/***"
include "/reduced/redu03/151930/raw/localize_scan098_nw04/***"
include "/reduced/redu03/151930/raw/corr_analysis_a1100_quick/***"

include "/reduced/redu04/"
include "/reduced/redu04/*.yaml"
include "/reduced/redu04/analysis_timestream_residuals_a1100/***"
include "/reduced/redu04/151930/"
include "/reduced/redu04/151930/raw/"
include "/reduced/redu04/151930/raw/blank_sky_null_audit_smoketest/***"
include "/reduced/redu04/151930/raw/corr_analysis_a1100/***"

include "/reduced/redu05/"
include "/reduced/redu05/*.yaml"
include "/reduced/redu05/151930/"
include "/reduced/redu05/151930/raw/"
include "/reduced/redu05/151930/raw/blank_sky_null_audit_a1100/***"

include "/reduced/redu06/"
include "/reduced/redu06/*.yaml"
include "/reduced/redu06/151930/"
include "/reduced/redu06/151930/raw/"
include "/reduced/redu06/151930/raw/blank_sky_null_audit_a1100/***"

include "/reduced/redu07/"
include "/reduced/redu07/*.yaml"
include "/reduced/redu07/151930/"
include "/reduced/redu07/151930/raw/"
include "/reduced/redu07/151930/raw/blank_sky_null_audit_a1100/***"

include "/reduced/redu08/"
include "/reduced/redu08/*.yaml"
include "/reduced/redu08/151930/"
include "/reduced/redu08/151930/raw/"
include "/reduced/redu08/151930/raw/blank_sky_null_audit_a1100/***"

include "/reduced/redu09/"
include "/reduced/redu09/*.yaml"
include "/reduced/redu09/151930/"
include "/reduced/redu09/151930/raw/"
include "/reduced/redu09/151930/raw/blank_sky_null_audit_a1100/***"
include "/reduced/redu09/151930/raw/mp_mode_estimate_lowband/***"
include "/reduced/redu09/151930/raw/mp_mode_estimate_fullband/***"

include "/reduced/redu10/"
include "/reduced/redu10/*.yaml"
include "/reduced/redu10/152524/"
include "/reduced/redu10/152524/raw/"
include "/reduced/redu10/152524/raw/blank_sky_null_audit_a1100/***"
include "/reduced/redu10/152524/raw/mp_mode_estimate_lowband/***"
include "/reduced/redu10/152524/raw/mp_mode_estimate_fullband/***"

if [[ "$include_essential_timestreams" == true ]]; then
    include "/reduced/redu04/151930/raw/toltec_commissioning_science_151930_ptc_timestream.nc"
    include "/reduced/redu04/151930/raw/toltec_commissioning_science_151930_rtc_timestream.nc"
    include "/reduced/redu09/151930/raw/toltec_commissioning_science_151930_ptc_timestream.nc"
    include "/reduced/redu09/151930/raw/toltec_commissioning_science_151930_rtc_timestream.nc"
    include "/reduced/redu10/152524/raw/toltec_commissioning_science_152524_ptc_timestream.nc"
    include "/reduced/redu10/152524/raw/toltec_commissioning_science_152524_rtc_timestream.nc"
fi

if [[ "$include_historical_ptc" == true ]]; then
    include "/reduced/redu03/151930/raw/toltec_commissioning_science_151930_ptc_timestream.nc"
    include "/reduced/redu05/151930/raw/toltec_commissioning_science_151930_ptc_timestream.nc"
    include "/reduced/redu06/151930/raw/toltec_commissioning_science_151930_ptc_timestream.nc"
    include "/reduced/redu07/151930/raw/toltec_commissioning_science_151930_ptc_timestream.nc"
    include "/reduced/redu08/151930/raw/toltec_commissioning_science_151930_ptc_timestream.nc"
fi

rsync_args+=("--exclude=*")

echo "Unity source: ${UNITY}:${SRC}/"
echo "Local dest:   ${DST}/"
if [[ "$dry_run" == true ]]; then
    echo "Mode:         dry-run"
else
    echo "Mode:         live"
fi
echo "Essential TODs:  ${include_essential_timestreams}"
echo "Historical PTC:  ${include_historical_ptc}"

rsync "${rsync_args[@]}" "${UNITY}:${SRC}/" "${DST}/"
