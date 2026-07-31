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

The versioned environment and full Citlali recipe will land with the parallel
CMake package-consumer path. Until then this check establishes the host and
workspace contract without claiming that the refactored application already
installs through Spack.
