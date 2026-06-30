#!/usr/bin/env bash

set -euo pipefail

OUT="${OUT:-/tmp/citlali-refactor-source.tar.gz}"
REPO="${REPO:-}"

usage() {
    cat <<'EOF'
Usage:
  tools/unity/make_source_bundle.sh [options]

Options:
  --out PATH       output tar.gz path
  --repo PATH      source repo path; default is git top-level
  -h, --help       show this help

The bundle is intended for:
  scp /tmp/citlali-refactor-source.tar.gz unity_toltec:/tmp/
  ssh unity_toltec 'cd /path/to/citlali && tar -xzf /tmp/citlali-refactor-source.tar.gz'
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --out)
            OUT="$2"
            shift 2
            ;;
        --repo)
            REPO="$2"
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

if [[ -z "$REPO" ]]; then
    REPO="$(git rev-parse --show-toplevel)"
fi

mkdir -p "$(dirname "$OUT")"

tar -czf "$OUT" \
    -C "$REPO" \
    --exclude="./.git" \
    --exclude="./build" \
    --exclude="./build_*" \
    --exclude="./cmake-build-*" \
    --exclude="./redu[0-9][0-9]" \
    --exclude="./coadded" \
    --exclude="./**/__pycache__" \
    --exclude="./**/*.pyc" \
    --exclude="./.DS_Store" \
    .

echo "Wrote source bundle: $OUT"
echo "Inspect with: tar -tzf $OUT | sed -n '1,40p'"
