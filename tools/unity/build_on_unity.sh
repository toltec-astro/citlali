#!/usr/bin/env bash

set -euo pipefail

UNITY_HOST="${UNITY_HOST:-unity_toltec}"
UNITY_BASELINE_REPO="${UNITY_BASELINE_REPO:-/home/toltec_umass_edu/work_toltec/citlali_dev/citlali}"
UNITY_REPO="${UNITY_REPO:-/home/toltec_umass_edu/work_toltec/citlali_dev/citlali_refactor}"
BUILD_DIR="${BUILD_DIR:-build}"
PRESET="${PRESET:-unity_release}"
TARGET="${TARGET:-citlali_cli}"
JOBS="${JOBS:-8}"
ALLOW_BASELINE_REPO="${ALLOW_BASELINE_REPO:-false}"

live=false
configure=false

usage() {
    cat <<'EOF'
Usage:
  tools/unity/build_on_unity.sh [options]

Options:
  --live                  run the remote ssh command; default is dry-run print
  --configure             run cmake configure before building
  --host HOST             SSH host alias (default: $UNITY_HOST or unity_toltec)
  --repo PATH             remote refactor repo path
  --build-dir DIR         remote build dir relative to repo
  --preset NAME           configure preset used when configuring
  --target NAME           CMake build target
  -j, --jobs N            build parallelism
  -h, --help              show this help

Environment:
  UNITY_HOST              default SSH host alias
  UNITY_REPO              default remote refactor repo path
  UNITY_BASELINE_REPO     protected gw_dev comparison repo path
  BUILD_DIR               default build directory
  PRESET                  default configure preset
  TARGET                  default build target
  JOBS                    default build parallelism
  ALLOW_BASELINE_REPO     set true to allow targeting UNITY_BASELINE_REPO
EOF
}

shell_quote() {
    printf "'%s'" "$(printf "%s" "$1" | sed "s/'/'\\\\''/g")"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --live)
            live=true
            shift
            ;;
        --configure)
            configure=true
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
        --build-dir)
            BUILD_DIR="$2"
            shift 2
            ;;
        --preset)
            PRESET="$2"
            shift 2
            ;;
        --target)
            TARGET="$2"
            shift 2
            ;;
        -j|--jobs)
            JOBS="$2"
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

if [[ "${UNITY_REPO%/}" == "${UNITY_BASELINE_REPO%/}" && "$ALLOW_BASELINE_REPO" != true ]]; then
    echo "Refusing to build in protected baseline repo: ${UNITY_BASELINE_REPO}" >&2
    echo "Use the refactor target, for example: /home/toltec_umass_edu/work_toltec/citlali_dev/citlali_refactor" >&2
    echo "Set ALLOW_BASELINE_REPO=true only if you intentionally want to override this guard." >&2
    exit 2
fi

repo_q="$(shell_quote "$UNITY_REPO")"
build_q="$(shell_quote "$BUILD_DIR")"
preset_q="$(shell_quote "$PRESET")"
target_q="$(shell_quote "$TARGET")"
jobs_q="$(shell_quote "$JOBS")"

remote_cmd="set -euo pipefail; cd ${repo_q}; "
remote_cmd+="echo \"repo: \$(pwd)\"; "
remote_cmd+="echo \"HEAD: \$(git rev-parse --short HEAD 2>/dev/null || echo unknown)\"; "
remote_cmd+="echo \"branch: \$(git branch --show-current 2>/dev/null || echo unknown)\"; "
if [[ "$configure" == true ]]; then
    remote_cmd+="cmake -S . -B ${build_q} --preset ${preset_q}; "
else
    remote_cmd+="if [ ! -f ${build_q}/CMakeCache.txt ]; then cmake -S . -B ${build_q} --preset ${preset_q}; fi; "
fi
remote_cmd+="cmake --build ${build_q} --target ${target_q} -j ${jobs_q}; "
remote_cmd+="if [ -x ${build_q}/bin/citlali ]; then ${build_q}/bin/citlali --version; "
remote_cmd+="elif [ -x ${build_q}/citlali ]; then ${build_q}/citlali --version; fi"

echo "Unity host:   ${UNITY_HOST}"
echo "Remote repo:  ${UNITY_REPO}"
echo "Protected:    ${UNITY_BASELINE_REPO}"
echo "Build dir:    ${BUILD_DIR}"
echo "Preset:       ${PRESET}"
echo "Target:       ${TARGET}"
echo "Jobs:         ${JOBS}"
if [[ "$live" == true ]]; then
    echo "Mode:         live"
    ssh -- "$UNITY_HOST" "$remote_cmd"
else
    echo "Mode:         dry-run"
    echo
    echo "ssh -- $(shell_quote "$UNITY_HOST") $(shell_quote "$remote_cmd")"
fi
