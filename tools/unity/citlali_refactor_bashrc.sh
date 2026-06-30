# Source this from Unity's ~/.bashrc after cloning citlali_refactor.
#
# Example:
#   source "${HOME}/work_toltec/citlali_dev/citlali_refactor/tools/unity/citlali_refactor_bashrc.sh"

citlali_refactor_update() {
  local repo="${CITLALI_REFACTOR_REPO:-${HOME}/work_toltec/citlali_dev/citlali_refactor}"
  local branch="${CITLALI_REFACTOR_BRANCH:-codex/structural-refactor}"
  local remote="${CITLALI_REFACTOR_REMOTE:-origin}"
  local build_dir="${CITLALI_REFACTOR_BUILD_DIR:-${repo}/build}"
  local preset="${CITLALI_REFACTOR_PRESET:-}"
  local target="${CITLALI_REFACTOR_TARGET:-citlali_cli}"
  local jobs="${CITLALI_REFACTOR_JOBS:-${CITLALI_BUILD_JOBS:-15}}"
  local build_type="${CITLALI_REFACTOR_BUILD_TYPE:-Release}"
  local wiener_omp="${CITLALI_USE_WIENER_FILTER_OMP:-ON}"
  local tula_dir="${CITLALI_TULA_DIR:-}"
  local baseline_repo="${CITLALI_BASELINE_REPO:-${HOME}/work_toltec/citlali_dev/citlali}"

  if [[ "${build_dir}" != /* ]]; then
    build_dir="${repo}/${build_dir}"
  fi

  if ! git -C "${repo}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "Missing citlali refactor checkout: ${repo}" >&2
    return 1
  fi

  if git -C "${baseline_repo}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    local repo_real
    local baseline_real
    repo_real=$(readlink -f "${repo}")
    baseline_real=$(readlink -f "${baseline_repo}")
    if [[ "${repo_real}" == "${baseline_real}" && "${CITLALI_REFACTOR_ALLOW_BASELINE:-false}" != true ]]; then
      echo "Refusing to build in protected gw_dev comparison checkout: ${baseline_repo}" >&2
      echo "Set CITLALI_REFACTOR_REPO to the separate citlali_refactor checkout." >&2
      return 2
    fi
  fi

  if [[ "${build_dir}" != "${repo}"/* ]]; then
    echo "Refusing to build outside the refactor checkout: ${build_dir}" >&2
    echo "Set CITLALI_REFACTOR_BUILD_DIR to a path under ${repo}." >&2
    return 2
  fi

  (
    set -e

    cd "${repo}"

    git fetch "${remote}" "${branch}"
    if git show-ref --verify --quiet "refs/heads/${branch}"; then
      git switch "${branch}"
    else
      git switch --track "${remote}/${branch}"
    fi
    git pull --ff-only "${remote}" "${branch}"

    git log -1 --oneline

    local cmake_args=(
      -S .
      -B "${build_dir}"
      -DCMAKE_BUILD_TYPE="${build_type}"
      -DCITLALI_USE_WIENER_FILTER_OMP="${wiener_omp}"
    )

    if [[ -n "${preset}" ]]; then
      cmake_args+=(--preset "${preset}")
    fi

    if [[ -n "${tula_dir}" ]]; then
      if [[ ! -d "${tula_dir}" ]]; then
        echo "Missing tula checkout: ${tula_dir}" >&2
        echo "Either create it or unset CITLALI_TULA_DIR to let CMake fetch tula into build/_deps." >&2
        return 1
      fi
      cmake_args+=(-DFETCHCONTENT_SOURCE_DIR_TULA="${tula_dir}")
    else
      # Match the existing citlali/build behavior: do not force a sibling tula
      # checkout; let FetchContent populate build/_deps/tula-src.
      cmake_args+=(-U FETCHCONTENT_SOURCE_DIR_TULA)
    fi

    cmake "${cmake_args[@]}"
    cmake --build "${build_dir}" --target "${target}" -j "${jobs}"

    if [[ -x "${build_dir}/bin/citlali" ]]; then
      "${build_dir}/bin/citlali" --version
    elif [[ -x "${build_dir}/citlali" ]]; then
      "${build_dir}/citlali" --version
    fi
  )
}

alias citlali-refactor-update='citlali_refactor_update'
alias citref-update='citlali_refactor_update'
