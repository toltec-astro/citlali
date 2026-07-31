# Native Spack Build Lane

This directory owns Citlali's successor native build entry. It does not
replace the existing CMake/FetchContent build until the acceptance gates in
`doc/TOLTECA_SPACK_BUILD_INTEGRATION_REVIEW_2026-07-31.md` pass.

## Supported Development Sequence

1. Keep `tula_cmake`, `tula`, `kidscpp`, and this Citlali checkout as sibling
   directories.
2. Use exact Homebrew LLVM 20 on Apple Silicon. AppleClang and unversioned
   Homebrew `llvm` are not accepted substitutes.
3. Use Spack 1.2.2 as the dependency and environment authority.
4. Build and run fast gates natively on macOS.
5. Push the accepted commit, then build that exact commit in user-owned space
   on Unity and run the required reduction validation.

Containers are optional CI or troubleshooting tools. They are not part of the
required local workflow.

## Prerequisite Check

Run:

```console
$HOME/tolteca/bin/python tools/build/check_macos_spack_prerequisites.py \
  --spack "$SPACK_ROOT/bin/spack"
```

The check rejects AppleClang, the wrong Spack release, missing sibling package
repositories, and shell flags that force the independently versioned Homebrew
`libomp` into an LLVM 20 build. The Spack environment must supply a compatible
OpenMP runtime instead.

## Foundation Environment

The first environment deliberately stops below Kidscpp and Citlali:

```console
export SPACK_ROOT="$HOME/GitHub/spack"
export SPACK_PYTHON="$HOME/tolteca/bin/python"
. "$SPACK_ROOT/share/spack/setup-env.sh"

spack -e spack/environments/foundation-macos-llvm20 concretize --force
spack -e spack/environments/foundation-macos-llvm20 \
  install --show-log-on-error
spack -e spack/environments/foundation-macos-llvm20 clean --stage tula
spack -e spack/environments/foundation-macos-llvm20 \
  install -y --test=root --overwrite --show-log-on-error tula
$HOME/tolteca/bin/python tools/build/test_spack_foundation.py
```

It builds the Tula component closure needed by Citlali without OpenMP. This
separates native compiler, package-repository, and installed-consumer failures
from the Kidscpp API and full Citlali source port. It does not claim production
equivalence.

The environment constrains NetCDF-C to a shared build with explicit Spack
Szip and Zstandard dependencies and a matching shared HDF5 1.14 ABI. The HDF5
core has its optional Szip filter disabled; NetCDF owns the explicit Szip
plugin edge used by this stack. NetCDF's
CMake otherwise auto-detects unrelated Homebrew compression libraries on this
host, placing the generic Homebrew include directory ahead of the declared
HDF5 dependency. That can compile against Homebrew HDF5 2 headers while
linking Spack HDF5 1.14. The explicit graph prevents that undeclared mixed-ABI
build.

The environment also uses a bounded local compatibility adapter for NetCDF
C++ 4.3.1. That release installs neither the `netcdf-cxx4.pc` metadata expected
by the upstream Tula adapter nor a complete CMake imported target. The local
adapter preserves `tula_deps::netcdf_cxx4` and locates only the already
concretized package. Its removal condition is documented with the adapter.

The two installation steps are intentional. Dependencies are installed
normally first; then the first-party Tula root is rebuilt with package tests.
The stale root build stage is removed before that rebuild so Spack re-runs
configuration with `TULA_BUILD_TESTS=ON`; otherwise its incremental-stage
optimization can reuse the initial no-test configuration.
Asking Spack to test the root during the initial graph install also enables
build-time test configuration in some dependencies. The resulting fmt 9.1 and
Eigen 3.4 private test failures against macOS 26 libc++ are not failures of the
libraries or Citlali contracts. Third-party private test compatibility is not
a Citlali gate.

The production environment and full Citlali recipe will land with the parallel
CMake package-consumer path. OpenMP remains a separate required gate because
Homebrew LLVM 20 does not bundle `libomp`, and the current Tula perflibs recipe
does not declare the separate LLVM OpenMP runtime as a package dependency on
macOS.
