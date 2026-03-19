#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

build_type="${BUILD_TYPE:-Debug}"
build_dir="${BUILD_DIR:-$repo_root/build_local_patched}"
generator="${CMAKE_GENERATOR:-Ninja}"
brew_prefix="${BREW_PREFIX:-$(brew --prefix)}"
sdk_root="${CMAKE_OSX_SYSROOT:-$(xcrun --show-sdk-path)}"
c_compiler="${CMAKE_C_COMPILER:-$(xcrun --find clang)}"
cxx_compiler="${CMAKE_CXX_COMPILER:-$(xcrun --find clang++)}"

required_formulae=(
  boost
  ccfits
  cfitsio
  cmake
  fftw
  hdf5
  libomp
  netcdf
  eigen@3
)

missing_formulae=()
for formula in "${required_formulae[@]}"; do
  if ! brew --prefix "$formula" >/dev/null 2>&1; then
    missing_formulae+=("$formula")
  fi
done

if ((${#missing_formulae[@]} > 0)); then
  printf 'Missing Homebrew packages: %s\n' "${missing_formulae[*]}" >&2
  printf 'Install them with: brew install %s\n' "${missing_formulae[*]}" >&2
  exit 1
fi

prefix_entries=(
  "$brew_prefix"
  "$(brew --prefix eigen@3)"
  "$(brew --prefix libomp)"
  "$(brew --prefix netcdf)"
  "$(brew --prefix hdf5)"
  "$(brew --prefix ccfits)"
  "$(brew --prefix cfitsio)"
  "$(brew --prefix fftw)"
  "$(brew --prefix boost)"
)

cmake_prefix_path="$(IFS=';'; echo "${prefix_entries[*]}")"

cmake_args=(
  -S "$repo_root"
  -B "$build_dir"
  -G "$generator"
  -DCMAKE_BUILD_TYPE="$build_type"
  -DCMAKE_C_COMPILER="$c_compiler"
  -DCMAKE_CXX_COMPILER="$cxx_compiler"
  -DCMAKE_OSX_SYSROOT="$sdk_root"
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5
  -DCMAKE_PREFIX_PATH="$cmake_prefix_path"
  -DUSE_INSTALLED_BOOST=ON
  -DUSE_INSTALLED_NETCDF=ON
  -DUSE_INSTALLED_CCFITS=ON
  -DUSE_INSTALLED_FFTW=ON
  -DUSE_INSTALLED_EIGEN3=ON
  -DFETCH_EIGEN3=OFF
  -DCONAN_INSTALL_LOGGING_LIBS=OFF
  -DFETCH_LOGGING_LIBS=ON
  -DCONAN_INSTALL_YAML=OFF
  -DFETCH_YAML=ON
  -DCONAN_INSTALL_CERES=OFF
  -DFETCH_CERES=ON
  -DCONAN_INSTALL_RE2=OFF
  -DFETCH_RE2=ON
  -DCONAN_INSTALL_SPECTRA=OFF
  -DFETCH_SPECTRA=ON
  -DKIDS_BUILD_CLI=OFF
  -DCITLALI_BUILD_TESTS=OFF
)

apply_git_patch() {
  local repo_dir="$1"
  local patch_file="$2"

  if git -C "$repo_dir" apply --check "$patch_file" >/dev/null 2>&1; then
    git -C "$repo_dir" apply "$patch_file"
    return 0
  fi

  if git -C "$repo_dir" apply -R --check "$patch_file" >/dev/null 2>&1; then
    return 0
  fi

  printf 'Failed to apply patch %s in %s\n' "$patch_file" "$repo_dir" >&2
  return 1
}

printf 'Configuring %s (%s)\n' "$build_dir" "$build_type"
cmake "${cmake_args[@]}"

apply_git_patch "$build_dir/_deps/tula-src" "$repo_root/patches/local/tula-local-build.patch"
apply_git_patch "$build_dir/_deps/kidscpp-src" "$repo_root/patches/local/kidscpp-local-build.patch"

printf 'Reconfiguring %s after local fetched-dependency patches\n' "$build_dir"
cmake "${cmake_args[@]}"

printf 'Configured build tree: %s\n' "$build_dir"
