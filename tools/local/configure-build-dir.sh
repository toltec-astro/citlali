#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export BUILD_TYPE="${BUILD_TYPE:-Release}"
export BUILD_DIR="${BUILD_DIR:-$repo_root/build}"
export CMAKE_GENERATOR="${CMAKE_GENERATOR:-Unix Makefiles}"

exec "$repo_root/tools/local/configure-local-build.sh"
