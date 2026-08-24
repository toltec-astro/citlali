# Unity Git Pull Workflow - 2026-06-30

This is the preferred workflow for validating the refactor branch on Unity.
Push from this machine to GitHub, then pull and build on Unity. Do not overlay
the refactor source into the existing `gw_dev` checkout.

## Local Machine

From `/Users/gwilson/GitHub/citlali-refactor`:

```bash
git status --short
git add doc handoff tools
git commit -m "Add structural refactor planning and Unity workflow tools"
git push -u origin codex/structural-refactor
```

Use a narrower `git add` if there are files you do not want in the first
refactor commit. Unity can only pull files that have been committed and pushed.

## One-Time Unity Setup

On Unity:

```bash
cd "${HOME}/work_toltec/citlali_dev"

# Keep this existing checkout untouched for gw_dev comparisons:
test -d citlali/.git

# Clone a separate refactor checkout.
git clone git@github.com:toltec-astro/citlali.git citlali_refactor
cd citlali_refactor
git fetch origin codex/structural-refactor
git switch --track origin/codex/structural-refactor
```

The refactor build must live under:

```text
${HOME}/work_toltec/citlali_dev/citlali_refactor/build
```

The existing comparison build stays under:

```text
${HOME}/work_toltec/citlali_dev/citlali/build_unity_release_native_lto
```

The existing `citlali/build` directory does not use a sibling `tula` checkout.
Its cache has `FETCHCONTENT_SOURCE_DIR_TULA` empty and uses:

```text
${HOME}/work_toltec/citlali_dev/citlali/build/_deps/tula-src
```

The refactor helper mirrors that behavior by default. It does not pass a
configure preset and it removes any stale `FETCHCONTENT_SOURCE_DIR_TULA` cache
entry, so CMake can populate `citlali_refactor/build/_deps/tula-src`.

If Unity already has a maintained `tula` checkout and you want to force it, set:

```bash
export CITLALI_TULA_DIR=/path/to/tula
```

before running `citlali-refactor-update`. If you also want the tracked
`unity_release` preset, set `CITLALI_REFACTOR_PRESET=unity_release` and provide
`CITLALI_TULA_DIR`, because that preset expects a manual tula source directory.

## Bashrc Integration

After the first clone, add this to Unity's `~/.bashrc`:

```bash
source "${HOME}/work_toltec/citlali_dev/citlali_refactor/tools/unity/citlali_refactor_bashrc.sh"
```

Then reload the shell:

```bash
source ~/.bashrc
```

The snippet defines:

```bash
citlali-refactor-update
citref-update
```

Both aliases pull `origin/codex/structural-refactor`, configure
`citlali_refactor/build`, build `citlali_cli`, and print the resulting Citlali
version when available.

## Normal Unity Update

After pushing new commits from this machine, run on Unity:

```bash
citlali-refactor-update
```

Equivalent explicit command:

```bash
cd "${HOME}/work_toltec/citlali_dev/citlali_refactor"
git fetch origin codex/structural-refactor
git switch codex/structural-refactor
git pull --ff-only origin codex/structural-refactor
rm -f build/FindHDF5.cmake build/Findhdf5.cmake build/FindnetCDF.cmake build/FindNetCDF.cmake build/FindCURL.cmake
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCITLALI_USE_WIENER_FILTER_OMP=ON \
  -DCONAN_CMD=/work/toltec/toltec_shared/toltec_astro/extern/pyenv/versions/conan1/bin/conan \
  -DUSE_INSTALLED_NETCDF=ON \
  -DCONAN_INSTALL_NETCDF=OFF \
  -DFETCH_NETCDF=OFF \
  -DUSE_INSTALLED_NETCDFCXX4=OFF \
  -DCONAN_INSTALL_NETCDFCXX4=OFF \
  -DFETCH_NETCDFCXX4=ON \
  -U FETCHCONTENT_SOURCE_DIR_TULA
cmake --build build --target citlali_cli -j "$(nproc)"
git rev-parse --short HEAD
./build/bin/citlali --version
```

The build refreshes Citlali's generated Git-version header before compilation,
including after a checkout or fast-forward that does not otherwise require
CMake reconfiguration. The short revision printed by `citlali --version` must
match `git rev-parse --short HEAD`. A mismatch is a failed provenance build;
do not submit a reduction with that executable.

## Useful Overrides

Set these in the Unity shell before calling `citlali-refactor-update` if needed:

```bash
export CITLALI_REFACTOR_BRANCH=codex/structural-refactor
export CITLALI_REFACTOR_BUILD_TYPE=Release
export CITLALI_REFACTOR_TARGET=citlali_cli
export CITLALI_CONAN_CMD=/work/toltec/toltec_shared/toltec_astro/extern/pyenv/versions/conan1/bin/conan
export CITLALI_USE_INSTALLED_NETCDF=ON
# Optional: cap or override auto-detected build parallelism.
# export CITLALI_REFACTOR_JOBS=8
```

When `CITLALI_REFACTOR_JOBS` and `CITLALI_BUILD_JOBS` are unset,
`citlali-refactor-update` detects build parallelism with `nproc`, then
`getconf _NPROCESSORS_ONLN`, and falls back to 15 only if neither detector is
available.

The helper refuses to use the protected baseline checkout
`${HOME}/work_toltec/citlali_dev/citlali` unless explicitly overridden. That
guard exists to preserve the `gw_dev` executable for output and performance
comparisons.

## Relationship To Rsync Tools

`tools/unity/sync_to_unity.sh` remains available as a fallback for uncommitted
local experiments, but the normal path should be:

```text
local commit -> git push -> Unity git pull -> Unity build
```

This gives every Unity build a real Git commit SHA and avoids ambiguous source
overlays.
