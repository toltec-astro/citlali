# ADR 0009: Spack Build Foundation

- **Status:** Accepted; supersedes ADR 0008 for build-tool and branch identity
- **Recorded:** 2026-07-31
- **Decision owners:** Citlali project owner and engineering

## Context

ADR 0008 established separate application and build-adaptation lanes while the
available successor used Conan 2. The build owner subsequently replaced that
implementation with a Spack architecture after the Conan package-consumer
failure exposed a deeper provider/export problem.

The Spack implementation now has explicit project-owned recipes, component
packages, installed consumers, GCC 14 and LLVM 20 Ubuntu matrices, real
Kidscpp data tests, and a complete upstream Citlali observation run. It remains
a smaller application tree than the refactor and does not yet provide the
refactor's provenance, test surface, native Mac environment, Unity
environment, or release lock.

## Decision

Retain the two-lane integration model and select Spack as the dependency and
environment foundation for the successor build.

`codex/refactor-mainline` remains the sole application authority.
`codex/build-adaptation` is the isolated successor-build lane. The branch name
is package-manager-neutral so another infrastructure correction would not
create a new application authority or require another workflow branch.

Spack owns dependency resolution, versions, variants, compilers, source
acquisition, externals, binary caches, environments, and concrete DAGs. Tula
CMake owns target-scoped CMake mechanics only. Production repositories own
their decentralized recipes.

The full refactored application is adapted into this model. The upstream
Citlali Spack branch is not merged wholesale. Existing numerical behavior,
CLI contracts, configuration, tests, and provenance remain authoritative.

The existing build remains available until the Spack successor passes all
entry, local, package-consumer, Unity, provenance, and same-SHA mode gates.

Native macOS development with exact Homebrew LLVM 20 is required. The normal
acceptance sequence is native Mac build and fast gates, followed by GitHub
transfer and a user-owned Unity build and reduction. Containers may support CI
or troubleshooting but are not a required developer interface. Citlali owns a
checked manifest of the exact Tula CMake, Tula, and Kidscpp revisions consumed
by this adaptation and materializes them in an ignored build-only source area.
This prevents unrelated developer checkouts from becoming implicit build
inputs. External deployment tooling remains outside Citlali and is not modified
as part of the adaptation.

## Consequences

- Native Spack concepts remain visible; Citlali does not add a wrapper that
  becomes a second dependency language.
- The small source-preparation command implements the checked revision
  manifest; it does not replace Spack's package or dependency model.
- A container-only success cannot satisfy the native Mac gate.
- Unity deployment remains in project-owned user space.
- Unity's existing use of Spack is useful infrastructure evidence but is not
  accepted deployment evidence until a user-writable environment is tested.
- Kidscpp V3 compatibility is a bounded source port, not a license to alter
  timestream mathematics.
- Release sources, locks, buildcache trust, Mac support, and complete
  provenance remain required work rather than assumed properties of Spack.
- The historical Conan branch and review remain available as evidence but are
  no longer active build authorities.

## Rejected Alternatives

- **Continue adapting Conan 2:** upstream ownership has ended and the Spack
  graph directly resolves the provider/export problem that blocked it.
- **Merge upstream Citlali wholesale:** would discard the validated refactor
  surface and obscure compatibility changes.
- **Use Spack only as a Unity module loader:** would retain separate local and
  cluster dependency systems and lose graph identity.
- **Remove the old build immediately:** leaves no operational fallback before
  the full application and four-mode matrix pass.

## Supersession

Review this decision after the Spack build is operationally accepted or if the
build owner abandons the published Spack architecture. Any successor must
preserve one dependency authority, installed consumer tests, exact source and
graph provenance, and the application/build lane separation.

## Evidence

- [`../TOLTECA_SPACK_BUILD_INTEGRATION_REVIEW_2026-07-31.md`](../TOLTECA_SPACK_BUILD_INTEGRATION_REVIEW_2026-07-31.md)
- [`../TOLTECA_BUILD_INTEGRATION_REQUIREMENTS_2026-07-23.md`](../TOLTECA_BUILD_INTEGRATION_REQUIREMENTS_2026-07-23.md)
- [`0008-application-mainline-and-build-adaptation-lanes.md`](0008-application-mainline-and-build-adaptation-lanes.md)
