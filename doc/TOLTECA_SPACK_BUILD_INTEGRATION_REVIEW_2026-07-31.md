# TolTECA Spack Build Integration Review

Date: 2026-07-31

Status: architecture review complete; **Adapt with Spack** selected as the
successor-build direction. The existing build remains the operational
fallback. The upstream Spack application is substantial evidence, but it is
not a drop-in replacement for the full refactored Citlali tree.

## Reviewed Revisions

The review used isolated checkouts at the exact pushed revisions:

| Repository | Branch | Commit |
| --- | --- | --- |
| `toltec-astro/tula_cmake` | `v3.x_spack` | `1ea93f600055e14248b2dbfcf1c16c5487a7b757` |
| `toltec-astro/tula` | `v3.x_spack` | `61f862c9cc08f335e946a4f55c5aa5cf35401bb0` |
| `toltec-astro/kidscpp` | `v3.x_spack` | `e3c05ebc75da42151a450bbc8c1b27f1e2e5e61b` |
| `toltec-astro/citlali` | `v3.x_spack` | `8a1be68354d78110c0c3e0f1d4ee5fd3cea20864` |

The reconciled design authority is the `design/` directory in the reviewed
Tula CMake branch. The earlier Conan 2 review remains historical evidence but
no longer governs the package-manager choice.

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
  1.13, all sibling package repositories, and no globally forced Homebrew
  `libomp`; its six focused tests pass;
- a concrete native macOS environment source-builds the Tula component closure
  with explicit NetCDF-C, NetCDF C++, HDF5, Szip, and Zstandard identities;
- all 14 enabled Tula root tests pass under LLVM 20; and
- an independent installed Tula consumer configures, builds, and passes CTest
  against the concrete dependency graph.

The native reproduction exposed two portability defects that the reported
Ubuntu external-package lane did not exercise. NetCDF-C's CMake build can
auto-detect undeclared Homebrew compression libraries, mixing Homebrew HDF5 2
headers with the declared Spack HDF5 1.14 library. The environment now uses
explicit shared NetCDF, HDF5, Szip, and Zstandard constraints, eliminating
that mixed ABI. NetCDF C++ 4.3.1 also installs neither the
`netcdf-cxx4.pc` file expected by the upstream Tula adapter nor a complete
CMake imported target. A bounded local compatibility adapter preserves
`tula_deps::netcdf_cxx4` until the upstream package boundary is corrected.

After the original review, Tula CMake advanced only to add the accepted Tlaloc
ECSV integration matrix and Tula advanced to fix ECSV table-view lifetimes
with a focused regression test. Those bounded changes were reviewed and are
the revisions now recorded above. Kidscpp and upstream Citlali did not move.

The complete production matrix was not independently reproduced locally. The
four-sibling foundation workflow is now reproduced from a native macOS
environment, but it deliberately stops below Kidscpp and Citlali and does not
yet exercise OpenMP. The repositories do not identify an accessible immutable
revision of `tolteca_test_data`. Docker is not a prerequisite for acceptance.

A fresh native Spack concretization also fails because the Citlali recipe pins
`cfitsio@4.3.1`, while the current builtin repository does not provide that
version. The reported container satisfies it only through a hardcoded Ubuntu
`/usr` external. This does not invalidate the measured container build, but it
means the build owner's intended system-or-source policy is not yet expressed
by a portable concrete graph.

## Requirement Assessment

| Area | Disposition | Evidence or remaining gap |
| --- | --- | --- |
| Dependency ownership | Pass | Spack owns one concrete graph; Tula CMake is CMake-only. |
| First-party package identity | Pass in development | Tula CMake, Tula, Kidscpp, and Citlali are explicit decentralized packages. |
| Installed package consumers | Pass for upstream slice | Tula, Kidscpp, and Citlali installed consumers are reported in both compiler lanes. |
| NetCDF C++ propagation | Pass with bounded adapter | Source-built 4.3.1 lacks the pkg-config metadata required by upstream Tula; the local target adapter and installed consumer pass. |
| Compiler matrix | Partial | GCC 14 and LLVM 20 pass in Ubuntu; native macOS and Unity profiles remain unmeasured. |
| Native developer bootstrap | Partial, foundation passed | Exact LLVM 20/Spack host gate, source-built Tula closure, 14 root tests, and installed consumer pass; OpenMP, Kidscpp, and full Citlali remain. |
| Real-data fixtures | Partial | Upstream reports real-data tests, but the large fixture has no accessible immutable manifest for collaborators. |
| Release source identity | Fail | First-party recipes provide versions without immutable source URLs/checksums and rely on local `develop` paths. |
| Portable lock | Planned | Local locks are intentionally ignored; no release environment lock exists. |
| Full refactored application | Fail pending adaptation | Upstream Citlali has 42 headers and five compiled library sources, not the full refactor graph. |
| Full CLI and config | Pass only for upstream slice | Upstream CLI runs, but refactor compiled sources, config gates, and operational behavior are not ported. |
| Source/dependency provenance | Fail pending adaptation | Semantic versions exist; exact source, dirty state, package DAG, and lock identity are not published by Citlali products. |
| Direct dependencies | Partial | Full refactor uses HDF5 and Zlib directly; they are not explicit Citlali recipe/CMake edges. |
| Kidscpp compatibility | Partial | V3 owns the required raw reader, but its API replaces legacy `kids/toltec/toltec.h` and removes `SweepFitter`. |
| Full tests | Fail pending adaptation | The upstream 16/7/6 matrix does not include the refactor's complete CTest, config, baseline, and ledger gates. |
| Unity operation | Not demonstrated | Existing Spack/module availability is promising but no Citlali environment or reduction has been tested there. |
| Build timing | Not demonstrated | No clean, incremental, or no-op timing evidence was supplied. |

## Required Adaptation Work

### A. Make The Environment Reconstructible

1. Document and reproduce the four-sibling-checkout workflow through the Tula
   CMake `Justfile`; keep any container workflow optional.
2. Identify an accessible real-data fixture and record an immutable manifest.
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

1. Extend the proven native Mac foundation through OpenMP, Kidscpp, and full
   Citlali.
2. Identify the real-data fixture and publish an immutable manifest without
   requiring the large payload to live in Git.
3. Demonstrate that exact dependencies such as `cfitsio@4.3.1` are either
   source-buildable or explicitly selected as platform externals.
4. Define the exact source, dirty-state, and concrete-DAG provenance exposed
   to Citlali.
5. Define release repository composition, immutable sources, lock, and
   buildcache trust policy.
6. Demonstrate the user-owned Unity environment and reduction workflow.

These are implementation and acceptance gaps, not unresolved policy questions
and not reasons to require a local container.
