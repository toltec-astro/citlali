# Apple Silicon macOS Citlali Setup Handoff

## Goal

A separate Codex agent on an Apple Silicon Mac should set up a reliable local `citlali` build without relying on the Unity machine.

This note summarizes what worked on the Intel Mac and what is likely to need adaptation for Apple Silicon.

## Executive Summary

The Intel macOS work established that:

- a local macOS build is feasible
- the main blockers are Apple toolchain setup, package-path handling, and fetched dependency compatibility
- the current local tooling / patches should be treated as the starting point, not rebuilt from scratch

The Apple Silicon agent should begin by porting the existing local tooling path, especially the scripts and patch files under `tools/local/` and `patches/local/`.

## Files To Start From

- `tools/local/configure-local-build.sh`
- `tools/local/measure-incremental-build.sh`
- `tools/local/run-sensitivity-smoke.sh`
- `tools/local/README.md`
- `patches/local/tula-local-build.patch`
- `patches/local/kidscpp-local-build.patch`
- `CMakeLists.txt`
- `include/citlali/core/engine/engine.h`
- `include/citlali/core/engine/lali.h`
- `include/citlali/core/timestream/rtc/rtcproc.h`

## What Was Learned On The Intel Mac

### Toolchain requirements

Required before CMake was trustworthy:

- full Xcode installed
- Xcode license accepted
- `xcode-select` pointed at `/Applications/Xcode.app/Contents/Developer`
- `xcrun --show-sdk-path` returned a usable SDK

### Brew packages that mattered

The working local path assumed:

- `boost`
- `ccfits`
- `cfitsio`
- `cmake`
- `fftw`
- `git`
- `hdf5`
- `libomp`
- `netcdf`

### Fetched dependency fixes that were necessary

The patched fetched-dep path was required because upstream `kidscpp` / `tula` did not cleanly build on the Mac as-is.

The local patch files currently cover:

- `Eigen3` package discovery bootstrap fixes
- `Ceres` bootstrap / versioning fixes
- Apple libc++ formatting fixes
- `ceres` 2.0 API compatibility in fetched `kidscpp`
- AppleClang OpenMP handling for fetched deps

### Repo-local fixes that were necessary

These repo-local source fixes were needed on the Intel Mac:

- `include/citlali/core/engine/engine.h`
- `include/citlali/core/engine/lali.h`
- `include/citlali/core/timestream/rtc/rtcproc.h`

The Apple Silicon agent should assume those same fixes are still relevant unless the primary repo has since absorbed them.

## Apple Silicon-Specific Risks

### 1. Hardcoded Homebrew prefix

The current local configure script assumes `/usr/local/...`.

On Apple Silicon, Homebrew is usually under `/opt/homebrew/...`.

The first likely script task is to replace hardcoded `/usr/local` references with a detected prefix, e.g.:

```bash
brew_prefix="$(brew --prefix)"
```

Then derive package paths from that.

Current hardcoded areas to inspect:

- `CCFITS_PREFIX`
- `netcdf_pkg_src`
- `hdf5_pkg_src`
- FFTW include / lib dirs
- `CMAKE_PREFIX_PATH`

### 2. Compiler preset paths

`CMakePresets.json` contains compiler paths under `/usr/local/opt/...`.

Those are likely wrong on Apple Silicon.

The Apple Silicon agent should not assume the checked-in brew compiler presets are usable as-is.

### 3. OpenMP / AppleClang behavior

The Intel Mac needed Apple-specific OpenMP handling:

- disable GrPPI OpenMP codegen in fetched deps
- keep `libomp` on the link line explicitly
- avoid propagating problematic OpenMP compile flags broadly

That same class of issue is likely to appear on Apple Silicon too.

The Apple-specific logic already added to `CMakeLists.txt` and encoded in `patches/local/tula-local-build.patch` should be reused, not rediscovered.

### 4. Package export path rewriting

The local script copies installed `netCDF` / `HDF5` CMake package exports into the bootstrap tree and rewrites stale SDK references to the active Xcode SDK.

That rewrite step is likely still necessary on Apple Silicon.

## Recommended Apple Silicon Agent Plan

1. Verify Xcode / SDK state.
2. Install required brew packages.
3. Detect and record `brew --prefix`.
4. Patch `tools/local/configure-local-build.sh` so all package paths derive from the brew prefix instead of `/usr/local`.
5. Reuse the existing local patch files under `patches/local/`.
6. Configure a fresh `Debug` build first.
7. Build `citlali_cli` and verify `--version`.
8. Only after `Debug` works, configure a fresh `Release` tree.
9. Add timing measurements with `tools/local/measure-incremental-build.sh`.

## Suggested Commands For The Apple Silicon Agent

The agent should aim to make a flow like this work:

```bash
tools/local/configure-local-build.sh
cmake --build "$PWD/build_local_patched" --target citlali_cli -j4
./build_local_patched/bin/citlali --version

BUILD_TYPE=Release \
BOOTSTRAP_DIR="$PWD/build_local_bootstrap_release" \
BUILD_DIR="$PWD/build_local_release" \
tools/local/configure-local-build.sh

cmake --build "$PWD/build_local_release" --target citlali_cli -j4
./build_local_release/bin/citlali --version
```

## Recommended Checks If The Apple Silicon Agent Gets Stuck

### If configure fails early

Check:

- Xcode license / `xcode-select`
- active SDK path
- brew package locations
- hardcoded `/usr/local` assumptions

### If fetched deps fail

Check:

- whether `patches/local/tula-local-build.patch` actually applied
- whether `patches/local/kidscpp-local-build.patch` actually applied
- whether the `Eigen3` package redirect was generated in the build tree

### If OpenMP link errors appear

Check:

- local Apple-specific logic in `CMakeLists.txt`
- OpenMP handling in `patches/local/tula-local-build.patch`
- actual `libomp` path under the Apple Silicon Homebrew prefix

### If CLI compile is extremely slow

That is expected to some degree. The heaviest compile path is still `src/citlali/cli/main.cpp`.

The Apple Silicon agent should focus first on getting a reliable build, not on speeding that path up yet.

## Bottom Line

The Apple Silicon agent should not start from scratch. It should port the existing local macOS solution, replace Intel/Homebrew path assumptions, verify `Debug`, verify `Release`, and only then move on to performance work.
