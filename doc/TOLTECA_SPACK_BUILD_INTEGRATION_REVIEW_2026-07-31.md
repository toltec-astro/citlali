# TolTECA Spack Build Integration Review

Date: 2026-07-31; updated 2026-08-10

Status: architecture review complete; **Adapt with Spack** selected as the
successor-build direction. The existing build remains the operational
fallback. The upstream Spack application is substantial evidence, but it is
not a drop-in replacement for the full refactored Citlali tree.

## Reviewed Revisions

The review used isolated checkouts at the exact pushed revisions:

| Repository | Branch | Commit |
| --- | --- | --- |
| `toltec-astro/tula_cmake` | `v3.x_spack` | `6433cdabe7010d0af2d0ba69da7af27510391b80` |
| `toltec-astro/tula` | `v3.x_spack` | `aa16c853c6b596c04ccdc90dc3acc4ce2006d947` |
| `toltec-astro/kidscpp` | `v3.x_spack` | `498ece1113001ae2d42d96d9fc29152aea3eaaef` |
| `toltec-astro/citlali` | `v3.x_spack` | `ceb4335c3f00b52af58ce2c09093b863040434b6` |
| `toltec-astro/tolteca_deploy` | `main` | `0a6b896` (`v0.1.1`) |

The reconciled design authority is the `design/` directory in the reviewed
Tula CMake branch. The earlier Conan 2 review remains historical evidence but
no longer governs the package-manager choice.

The 2026-08-10 update clarifies the separation between immutable artifact
identity and runtime deployment identity and advances the release/deployment
design. It does not change the installed `tula-netcdf-cxx4` target's incomplete
NetCDF-C include interface; the bounded Citlali direct dependency remains
necessary until that target is corrected upstream. The reviewed
`tolteca_deploy` release profiles are design evidence only: their committed
Ubuntu ARM64 GCC 14/LLVM 20 locks do not replace the measured Unity x86_64
GCC 13.3 acceptance profile owned by this lane.
Tula's installed version header still reports `cxx_standard=0`, whereas
Kidscpp and Citlali correctly report C++23. This is retained upstream metadata
debt rather than a reason to falsify the effective compiler standard.

## Build-Owner Intent Clarification

The build owner supplied the following intended operating model after the
initial review:

- the developer workspace is four sibling checkouts, with workflow recipes in
  the Tula CMake `Justfile`; the private development container is not an
  exported project deliverable;
- native macOS is intended to be supported, and Homebrew LLVM 20 is a tested
  compiler;
- Spack may consume documented system packages or build dependencies from
  source;
- Unity releases will be installed in user-owned space, matching current
  operational practice;
- the large real-data fixture, release composition, and release locking are
  not yet settled; and
- existing provenance may be retained initially, with improvements reviewed
  where they materially strengthen reproducibility.

Project policy further requires native macOS development. Containers may be
used for CI or troubleshooting, but are not required for local development.
For the bounded Citlali adaptation, those upstream sources are materialized at
the checked revisions under Citlali's ignored `build/spack-sources/` directory.
This avoids modifying unrelated developer checkouts. `tolteca_deploy` remains
an external, read-only deployment project and is not an input to this lane.

## Decision

Use Spack for dependency resolution, variants, compiler identity,
environments, concretization, source acquisition, binary caches, and the
complete package graph. Use Tula CMake only for target-scoped CMake
conventions. Keep Tula, Kidscpp, and Citlali as independent first-party
packages whose source repositories own their Spack recipes.

Carry the full refactored Citlali application into this model through the
bounded Adapt lane. Do not merge the upstream Citlali branch wholesale and do
not replace the refactored source, CLI, tests, configuration, provenance, or
validation history with the smaller upstream tree.

The active build worktree branch is `codex/build-adaptation`. The package
manager is an implementation decision inside that lane rather than part of
the branch identity.

## Why Spack Is A Good Fit

The new architecture removes the duplicated provider model that complicated
the Conan 2 implementation:

- Spack alone owns the dependency graph and version/variant constraints.
- CMake consumes already installed package targets and does not download or
  select providers.
- Each production repository owns a package API v2 recipe.
- Dependency adapters use explicit `tula_deps::*` targets, while higher-level
  Tula behavior uses explicit `tula::*` components.
- Optional Tula components have measured minimal dependency closures.
- Package tests exercise installed prefixes and independent consumers.
- NetCDF C++ is now a first-class relocatable adapter package, resolving the
  exported-header defect found in the Conan review.
- GCC 14 and LLVM 20 use C++23 in the reported matrix.
- The same environment model is compatible in principle with HPC deployment,
  module generation, and binary caches.

This separation is simpler and more defensible than maintaining a second
dependency/provider language inside Tula CMake.

## Evidence

### Reported Upstream Evidence

The reviewed design records the following Ubuntu 24.04 arm64 results:

- Tula: 16 package tests under GCC 14.2 and LLVM 20.1.2.
- Kidscpp: seven package tests in both lanes, including real TolTEC NetCDF
  metadata and slice ingestion with no required-data skips.
- Citlali: six package tests, installed library consumer, installed CLI, and
  version command in both lanes.
- Focused Tula component matrices for ECSV, perflibs/OpenMP, enum, CLI,
  NetCDF, GrPPI, and fitting.
- A GCC 14 installed Citlali run on observation 149101 that processed all 123
  scans and wrote raw and filtered maps for all three arrays.

These are stronger application and package-boundary results than the earlier
Conan milestone provided.

### Independently Reproduced Evidence

On the review Mac:

- all four checked-out revisions match their recorded commits;
- every project-owned Spack recipe passes Python syntax compilation;
- Spack 1.2.2 loads all four package API v2 repositories and the Citlali
  recipe with its declared dependency graph;
- Tula CMake configures and its installed producer/consumer fixture passes;
- the fixture and its downstream consumer compile and run with exact Homebrew
  LLVM 20.1.8 rather than AppleClang.
- the native prerequisite checker passes with Spack 1.2.2, CMake 4.3, Ninja
  1.13, declared Homebrew FFTW and GCC 15 Fortran externals, all sibling
  package repositories, and no globally forced Homebrew `libomp`; its eight
  focused tests pass;
- a concrete native macOS environment source-builds the Tula component closure
  with explicit NetCDF-C, NetCDF C++, HDF5, Szip, and Zstandard identities;
- all 14 enabled Tula root tests pass under LLVM 20; and
- an independent installed Tula consumer configures, builds, and passes CTest
  against the concrete dependency graph.
- a concrete Kidscpp environment source-builds with an explicit
  `llvm-openmp@20.1.8` dependency;
- the independent installed Kidscpp consumer configures, links, and passes
  CTest against that installed graph; and
- a second independent reader consumer opens a current raw TolTEC pointing
  file, reads a two-sample I/Q slice, and records fixture SHA-256
  `cc44075693ab19161eaac390a84b8bc82ab3cf18bdbff7b76ff8d4c02e531edc`;
- the full refactored application builds all eight active compiled sources,
  generated configuration/version headers, production CLI, and complete test
  targets through the native Spack graph under C++23 and exact LLVM 20;
- all 539 enabled CTests pass, with the one intentionally disabled lifecycle
  test reported explicitly;
- the full 123-test/four-mode config preflight passes;
- the installed CLI preserves the complete operational help surface and an
  independent installed `find_package(citlali)` consumer passes; and
- the CLI reports source dirty state, Kidscpp version, build type, compiler
  family, Wiener variant, and concrete Spack root DAG hash. Managed activation
  supplies the profile, lock digest, and environment path; Citlali refuses to
  run when that lock's root DAG differs from the compiled DAG and records the
  accepted deployment identity in FITS, NetCDF, and product-index metadata. A
  persistent Ninja tree provides a measured 0.82-second no-op build without
  restaging the development package.
- the 2026-08-03 upstream revisions replace the local NetCDF C++ and perflibs
  adapters with Tula CMake-owned packages, add the normalized CCfits target,
  and distinguish general pipeline OpenMP from Wiener OpenMP; the resulting
  full Citlali package concretizes and installs under native LLVM 20.

The native reproduction exposed two portability defects that the reported
Ubuntu external-package lane did not exercise. NetCDF-C's CMake build can
auto-detect undeclared Homebrew compression libraries, mixing Homebrew HDF5 2
headers with the declared Spack HDF5 1.14 library. The environment now uses
explicit shared NetCDF, HDF5, Szip, and Zstandard constraints, eliminating
that mixed ABI. NetCDF C++ 4.3.1 also installs neither the
`netcdf-cxx4.pc` file used by an earlier adapter nor a complete CMake imported
target. The accepted Tula CMake package now resolves the concrete C++ and C
libraries directly and exports `tula_deps::netcdf_cxx4`; the local Citlali
adapter is no longer required.

After the original review, all four upstream branches advanced. The current
revisions recorded above were re-reviewed in isolated clean checkouts. Only
Tula CMake, Tula, and Kidscpp are consumed as build inputs; upstream Citlali is
a reference implementation and is not merged into the refactored application.
The newest dependency tips build the full refactored CLI and pass all 533
pre-existing enabled CTests in an isolated graph; with the six deployment
identity tests added by this adaptation, the accepted tree passes 539 enabled
tests. TolTECA Deploy's 20 focused Spack tests pass from an isolated package
install. Its complete Python suite is not counted because the shared local
TolTECA/Tollan/Pydantic environment is incompatible with that checkout and
does not provide controlled deployment evidence.

The complete production reduction matrix was not independently reproduced
locally. The pinned-source workflow now reaches the full refactored Citlali
application. The repositories do not identify an accessible immutable revision of
`tolteca_test_data`. A native Kidscpp package-test rebuild therefore ran six
of seven discovered tests successfully but failed the historical real-file
test at its missing path. Because an empty CMake-provided
`TOLTECA_TEST_DATA_ROOT` is treated as present, that test does not skip and its
invalid-stride companion can pass for the wrong reason. This limitation is
recorded rather than counted as a green package suite. Citlali instead
publishes a path-independent content manifest for a current pointing file;
the installed-reader gate verifies its basename, byte size, and SHA-256 before
opening the external payload and reading a two-sample I/Q slice. That gate
passes under the exact native LLVM 20 graph and replaces the inaccessible
historical fixture as acceptance evidence. Docker is not a prerequisite for
acceptance.

The local Citlali recipe uses source-buildable `cfitsio@4.3.0`, the version
available in the current builtin Spack repository, rather than relying on the
upstream Ubuntu `/usr` external for unavailable `4.3.1`. The macOS environment
declares Homebrew FFTW and GCC 15 Fortran as checked host externals. All
Citlali, Kidscpp, Tula, and other C/C++ application nodes remain exact LLVM 20.

## Requirement Assessment

| Area | Disposition | Evidence or remaining gap |
| --- | --- | --- |
| Dependency ownership | Pass | Spack owns one concrete graph; Tula CMake is CMake-only. |
| First-party package identity | Pass in development | Tula CMake, Tula, Kidscpp, and Citlali are explicit decentralized packages. |
| Installed package consumers | Pass | Native Tula, Kidscpp, and full refactored Citlali installed consumers pass; the Citlali consumer also passes on Unity GCC 13.3. |
| NetCDF C++ propagation | Pass upstream | Tula CMake owns a normalized target that does not depend on missing NetCDF C++ pkg-config metadata. |
| Compiler matrix | Pass in development | GCC 14 and LLVM 20 pass in Ubuntu, native macOS LLVM 20 passes through the full application, and Unity GCC 13.3 passes the complete build and test gate. Release support policy remains separate. |
| Native developer bootstrap | Pass | Exact LLVM 20/Spack host gate, source-built full graph, persistent Ninja workflow, and installed-artifact gate pass locally; the corresponding Unity GCC 13.3 workflow also passes. |
| Real-data fixtures | Pass | The published manifest binds pointing observation 152389, network 0, scan 2 by basename, byte size, and SHA-256; the installed Kidscpp reader verifies it and passes without making the external payload part of Git. |
| Release source identity | Partial | Exact development revisions are machine-readable and checked; immutable release archives/checksums remain open. |
| Portable lock | Planned | Local locks are intentionally ignored; no release environment lock exists. |
| Full refactored application | Pass | All eight active compiled sources, full header surface, generated inputs, library, CLI, and tests build through Spack on native macOS and Unity. |
| Full CLI and config | Pass locally | Full operational CLI/help and complete 123-test/four-mode config preflight pass. |
| Source/dependency provenance | Partial | Source/dirty state, compiler, build type, OpenMP/Wiener variants, semantic package versions, concrete DAG hash, exact first-party development revisions, and managed runtime lock-root binding are checked; a portable release lock and complete release manifest remain open. |
| Direct dependencies | Pass | HDF5 and Zlib are explicit Citlali recipe and CMake target edges. |
| Kidscpp compatibility | Pass pending product validation | A bounded V3 raw-timestream adapter compiles and is tested; legacy config remains accepted and the unused sweep fitter is omitted only in V3. Unity product validation remains. |
| Full tests | Pass in current environments | All 539 enabled CTests pass on native macOS LLVM 20 and Unity GCC 13.3; complete config preflight passes locally. Baseline/ledger/exit gates remain part of final acceptance. |
| Unity operation | Partial | Exact-source job `62890572` at `7ee2c4f7` installs the GCC 13.3 graph, passes source hygiene, both executions of all 539 enabled CTests, the independent installed consumer, and complete managed provenance. A later point smoke exposed concurrent NetCDF-C/HDF5 output access and segfaulted during file close. Repair commit `b8e80fb15` passes the full local gate and interactive Unity acceptance; two subsequent point runs reproduce all 1,930 scientific comparison records exactly without recurrence. Direct Citlali invocation verifies the managed profile, lock, and DAG binding, but TolTECA-launched products remain marked unmanaged. The external deployment owner now owns that propagation fix and one final managed point rerun. |
| Build timing | Partial | Persistent local no-op build is 0.82 seconds. Unity job `62890572` spent 6m51s in package installation and completed the full acceptance workflow in 13m24s; representative incremental timing still needs a formal campaign. |

## Required Adaptation Work

### A. Make The Environment Reconstructible

1. Maintain the checked first-party source manifest and Citlali-owned
   build-source preparation command; keep deployment tooling and containers
   optional and external.
2. Maintain the content-addressed real-data fixture manifest and keep the
   external payload available to acceptance environments.
3. Add and verify the required native macOS environment using exact Homebrew
   LLVM 20 and a compatible OpenMP runtime.
4. Add a user-owned Unity environment using the installed
   Spack/compiler/module facts retrieved by the user.
5. Add immutable release sources/checksums and a portable release lock.
6. Ensure dependencies such as CFITSIO can build from the declared graph or
   are explicitly documented as required environment externals.

### B. Port The Full Refactored Application

1. Preserve the current build unchanged while introducing the parallel Spack
   package path.
2. Carry all active refactor implementation sources, generated config inputs,
   CLI behavior, and required headers into the installed target.
3. Port `KidsDataProc` from legacy Kidscpp APIs to V3's
   `kids::toltec::RawTimeStream` boundary while preserving accepted data
   interpretation and configuration compatibility.
4. Remove the constructed but unused `SweepFitter`; continue accepting its
   legacy configuration leaves until an explicit compatibility retirement.
5. Add direct HDF5 and Zlib package/CMake edges.
6. Preserve the OpenMP Wiener option as package/build identity.

### C. Restore Provenance And Gates

1. Generate Citlali source commit and dirty-state identity.
2. Record Kidscpp/Tula package identity, concrete Spack DAG or lock identity,
   compiler, build type, and performance variants.
3. Preserve embedded default-config dependency tracking.
4. Register the complete CTest surface and run config, baseline, ledger,
   intended-science-change, and session-exit gates without required-data
   skips.
5. Exercise installed library and CLI consumers against the full refactor.

### D. Operational Acceptance

1. Measure clean, no-op, and representative incremental builds.
2. Build the exact commit in a user-writable Unity Spack environment without
   source edits.
3. Snapshot the resulting executable through the existing TolProj contract.
4. Run a Unity point smoke reduction.
5. Freeze and validate the same-SHA point, OOF, science, and Beammap matrix.

## Stop Conditions

Do not remove or weaken the existing build until:

1. the full refactor builds through the native macOS Spack path with exact
   Homebrew LLVM 20;
2. all required local gates pass;
3. installed package consumers pass;
4. provenance is truthful and sufficient to reproduce the graph;
5. the exact source builds and runs on Unity without edits;
6. the same-SHA four-mode matrix passes; and
7. rollback remains straightforward.

Do not combine this adaptation with RTC, PTC, mapmaking, Wiener, JINC,
fruit-loop, fitting, or calibration algorithm changes. Kidscpp compatibility
changes are isolated and require product validation.

## Remaining Open Evidence

1. Define exact first-party dependency source revisions in addition to the
   concrete DAG identity already exposed by Citlali.
2. Define release repository composition, immutable sources, lock, and
   buildcache trust policy.
3. Measure clean and representative incremental builds in addition to the
   accepted no-op timing.
4. Demonstrate the user-owned Unity environment and reduction workflow.

These are implementation and acceptance gaps, not unresolved policy questions
and not reasons to require a local container.
