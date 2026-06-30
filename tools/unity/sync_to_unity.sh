#!/usr/bin/env bash

set -euo pipefail

UNITY_HOST="${UNITY_HOST:-unity_toltec}"
UNITY_BASELINE_REPO="${UNITY_BASELINE_REPO:-/home/toltec_umass_edu/work_toltec/citlali_dev/citlali}"
UNITY_REPO="${UNITY_REPO:-/home/toltec_umass_edu/work_toltec/citlali_dev/citlali_refactor}"
LOCAL_REPO="${LOCAL_REPO:-}"
ALLOW_BASELINE_REPO="${ALLOW_BASELINE_REPO:-false}"

live=false
delete=false
checksum=false

usage() {
    cat <<'EOF'
Usage:
  tools/unity/sync_to_unity.sh [options]

Options:
  --dry-run              preview transfer; default
  --live                 perform the transfer
  --delete               delete remote source files absent locally, after filters
  --checksum             compare file checksums instead of size/mtime
  --host HOST            SSH host alias (default: $UNITY_HOST or unity_toltec)
  --repo PATH            remote refactor repo path
  --local PATH           local repo path; default is git top-level
  -h, --help             show this help

Environment:
  UNITY_HOST             default SSH host alias
  UNITY_REPO             default remote refactor repo path
  UNITY_BASELINE_REPO    protected gw_dev comparison repo path
  LOCAL_REPO             default local repo path
  ALLOW_BASELINE_REPO    set true to allow targeting UNITY_BASELINE_REPO

Notes:
  This script excludes .git, build directories, reduction outputs, coadds, and
  Python caches. It is meant to copy source/docs/tools into a separate Unity
  refactor tree, not to copy build products or overwrite the gw_dev comparison
  checkout.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run|-n)
            live=false
            shift
            ;;
        --live)
            live=true
            shift
            ;;
        --delete)
            delete=true
            shift
            ;;
        --checksum)
            checksum=true
            shift
            ;;
        --host)
            UNITY_HOST="$2"
            shift 2
            ;;
        --repo)
            UNITY_REPO="$2"
            shift 2
            ;;
        --local)
            LOCAL_REPO="$2"
            shift 2
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

if [[ -z "$LOCAL_REPO" ]]; then
    LOCAL_REPO="$(git rev-parse --show-toplevel)"
fi

if [[ "${UNITY_REPO%/}" == "${UNITY_BASELINE_REPO%/}" && "$ALLOW_BASELINE_REPO" != true ]]; then
    echo "Refusing to sync into protected baseline repo: ${UNITY_BASELINE_REPO}" >&2
    echo "Use the refactor target, for example: /home/toltec_umass_edu/work_toltec/citlali_dev/citlali_refactor" >&2
    echo "Set ALLOW_BASELINE_REPO=true only if you intentionally want to override this guard." >&2
    exit 2
fi

filter_file="${LOCAL_REPO}/tools/unity/rsync_filter.txt"
if [[ ! -f "$filter_file" ]]; then
    echo "Missing rsync filter file: $filter_file" >&2
    exit 1
fi

rsync_args=(
    -avz
    --human-readable
    --itemize-changes
    "--filter=merge ${filter_file}"
)

if [[ "$live" != true ]]; then
    rsync_args+=(-n)
fi
if [[ "$delete" == true ]]; then
    rsync_args+=(--delete)
fi
if [[ "$checksum" == true ]]; then
    rsync_args+=(--checksum)
fi

echo "Local repo:   ${LOCAL_REPO}/"
echo "Unity target: ${UNITY_HOST}:${UNITY_REPO}/"
echo "Protected:    ${UNITY_BASELINE_REPO}/"
if [[ "$live" == true ]]; then
    echo "Mode:         live"
else
    echo "Mode:         dry-run"
fi
echo "Delete:       ${delete}"
echo "Checksum:     ${checksum}"

rsync "${rsync_args[@]}" "${LOCAL_REPO}/" "${UNITY_HOST}:${UNITY_REPO}/"
