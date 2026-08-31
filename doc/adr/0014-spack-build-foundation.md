# ADR 0014: Spack Build Foundation

- **Status:** Accepted and operationally demonstrated by the owner-accepted
  Spack-backed `citlali-validation/v2` campaign; supersedes ADR 0008 for
  build-tool and branch identity
- **Recorded:** 2026-07-31
- **Decision owners:** Citlali project owner and engineering

## Context

ADR 0008 established separate application and build-adaptation lanes while the
available successor used Conan 2. The build owner subsequently replaced that
implementation with a Spack architecture after the Conan package-consumer
failure exposed a deeper provider/export problem.

At the time of the original decision, the upstream Spack implementation was a
smaller application tree than the refactor and did not yet provide the full
refactor provenance, test surface, native Mac environment, Unity environment,
or release lock. The bounded Adapt work subsequently brought the complete
refactored application and its test surface into native macOS LLVM 20 and
Unity GCC 13 Spack environments.

The 2026-08-31 owner correction records the operational result: the most
recent accepted end-to-end application validation, `citlali-validation/v2`,
was built and run under the Unity `unity-gcc13` Spack realization. That fact is
bound to exact source, executable, DAG, lock digest, jobs, and audit results in
`validation/citlali_v2_spack_validation_authority.json`.

## Decision

Retain the two-lane integration model and select Spack as the current
dependency and environment authority for this application generation.

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

The established non-Spack build remains available as fallback compatibility
and supplemental compilation/regression infrastructure until release and
rollback policy explicitly retires it. Passing that build does not reproduce
the accepted Spack-backed V2 campaign.

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
- The project-owned Unity Spack environment is accepted build/deployment
  evidence for the V2 application generation. This does not itself promote a
  release or authorize production expansion.
- Kidscpp V3 compatibility is a bounded source port, not a license to alter
  timestream mathematics.
- Portable release sources and locks, buildcache trust, and final release
  composition remain required work rather than assumed properties of Spack.
- The historical Conan branch and review remain available for older-generation
  reproduction and explicitly bounded compatibility only. They are not active
  build authorities. New Conan dependencies, recipe extensions, opportunistic
  infrastructure repair, or Conan-centered successor architecture require an
  explicit owner-authorized compatibility work order.
- Build-environment work retains separate ownership and WIP from
  scientific/application implementation. That separation does not make Spack
  experimental, and neither lane may silently change the other's contracts.

## Rejected Alternatives

- **Continue adapting Conan 2:** upstream ownership has ended and the Spack
  graph directly resolves the provider/export problem that blocked it.
- **Merge upstream Citlali wholesale:** would discard the validated refactor
  surface and obscure compatibility changes.
- **Use Spack only as a Unity module loader:** would retain separate local and
  cluster dependency systems and lose graph identity.
- **Remove the old build immediately:** would remove bounded compatibility and
  rollback machinery before its explicit release disposition.

## Supersession

Review this decision if the owner selects a different current build authority
or explicitly retires the fallback compatibility build. Any successor must
preserve one dependency authority, installed consumer tests, exact source and
graph provenance, and the application/build lane separation.

## Evidence

- [`../TOLTECA_SPACK_BUILD_INTEGRATION_REVIEW_2026-07-31.md`](../TOLTECA_SPACK_BUILD_INTEGRATION_REVIEW_2026-07-31.md)
- [`../TOLTECA_BUILD_INTEGRATION_REQUIREMENTS_2026-07-23.md`](../TOLTECA_BUILD_INTEGRATION_REQUIREMENTS_2026-07-23.md)
- [`0008-application-mainline-and-build-adaptation-lanes.md`](0008-application-mainline-and-build-adaptation-lanes.md)
- [`../../validation/citlali_v2_spack_validation_authority.json`](../../validation/citlali_v2_spack_validation_authority.json)
