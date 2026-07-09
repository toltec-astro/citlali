# macOS Homebrew Build

This directory contains the macOS build path for Apple Silicon/Homebrew
development machines.

## Prerequisites

- working `xcrun` toolchain selection
- Homebrew packages:
  - `boost`
  - `ccfits`
  - `cfitsio`
  - `cmake`
  - `fftw`
  - `hdf5`
  - `libomp`
  - `netcdf`
  - `eigen@3`

As of 2026-03-19, this machine can build with Apple Command Line Tools alone; full Xcode was not required for the verified build.

## Normal Local Workflow

Bootstrap the standard `build/` directory once:

```bash
make local-bootstrap
```

If an old `build/` tree was configured against a different source path or was left incomplete,
the bootstrap script will now discard it and regenerate a clean `build/` tree automatically.

After that, the usual loop is:

```bash
cd build
git pull
make -j6
```

Override `BUILD_TYPE=Debug` before running the bootstrap if you want a debug `build/` tree instead of the default release one.
Alternatively, use:

```bash
make local-bootstrap-debug
```

## Debug Build

```bash
tools/macos/configure-homebrew-build.sh
cmake --build "$PWD/build_local_patched" --target citlali_cli -j4
./build_local_patched/bin/citlali --version
```

## Release Build

```bash
BUILD_TYPE=Release \
BUILD_DIR="$PWD/build_local_release" \
tools/macos/configure-homebrew-build.sh

cmake --build "$PWD/build_local_release" --target citlali_cli -j4
./build_local_release/bin/citlali --version
```

## Notes

- `tools/macos/configure-homebrew-build.sh` detects the active Homebrew prefix and active SDK path.
- `tools/macos/configure-build-dir.sh` is the wrapper for the persistent `build/` + `make` workflow.
- The script configures once to populate fetched dependencies, applies the local `tula` and `kidscpp` patches under `patches/local/`, then reconfigures the same build tree.
- It also resets a stale or incomplete build tree before configuring, so `cd build && make -j6` stays reliable after repo moves or interrupted configures.
- Repo-local macOS fixes live in the main source tree and are no longer part of the fetched-dependency patches.
