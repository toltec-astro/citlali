# ADR 0010: Spack Release Bundle Contract

- **Status:** Accepted
- **Recorded:** 2026-08-14
- **Decision owners:** Citlali project owner and engineering

## Context

The Spack adaptation now builds and tests the full application on native macOS
LLVM 20 and Unity GCC 13. Development environments concretize exact graphs, but
they bind first-party packages to absolute local checkout paths. Their ignored
`spack.lock` files reproduce an observed workspace, not a portable release.

Citlali also needs a stable contract that deployment tooling can consume
without making that external repository authoritative for application source,
scientific acceptance, or provenance requirements.

## Decision

Citlali owns a versioned release-manifest schema and an executable validator.
One manifest binds the exact Citlali, Tula CMake, Tula, and Kidscpp commits to
immutable HTTPS archives and SHA-256 values. It also binds the Spack revision,
accepted evidence, build-cache trust policy, and one source-based environment
plus concrete lock for each supported platform/compiler profile.

A release lock contains no `develop` or `dev_path` source binding. Portability
means that the bundle can be reconstructed on another compatible host within
the declared profile; it does not mean one concrete lock spans macOS LLVM and
Unity GCC.

Source builds remain mandatory. Signed build caches may accelerate deployment,
but their mirrors and trusted key fingerprints are explicit release inputs.
Unsigned build caches are not accepted.

The release manifest is published beside the source archives and profile
artifacts it describes. It is not required to live inside a source archive,
which avoids a circular archive-checksum dependency.

External deployment tooling may realize and activate an accepted bundle. It
does not redefine this acceptance contract, and Citlali does not vendor or
modify that tooling in this lane.

## Consequences

- Current development locks remain valid build evidence but cannot be promoted
  directly to release locks.
- A development-candidate manifest may be valid while failing the stricter
  release-ready gate.
- Each supported profile receives its own reviewed lock and compiler policy.
- Branch names remain review context; full commits and archive checksums are
  authoritative.
- A release is blocked until every manifest remainder is closed and its
  acceptance status is `accepted`.

## Rejected Alternatives

- **Commit the current development locks:** they contain machine-specific
  checkout paths and cannot reconstruct the source graph elsewhere.
- **Use one lock for all platforms:** concrete dependencies and compilers are
  profile-specific.
- **Let deployment tooling define the release:** this would move source and
  scientific acceptance authority outside the application repository.
- **Require a binary cache:** it would make independent source reconstruction
  contingent on cache availability and trust configuration.

## Supersession

Revisit this decision if Spack gains a demonstrably equivalent cross-profile
release abstraction or if project ownership of release composition changes.
Any successor must retain immutable source identity, concrete dependency
identity, source reconstruction, explicit trust, and profile-specific
acceptance evidence.

## Evidence

- [`../../spack/release/README.md`](../../spack/release/README.md)
- [`../../spack/release/development-candidate.json`](../../spack/release/development-candidate.json)
- [`../TOLTECA_SPACK_BUILD_INTEGRATION_REVIEW_2026-07-31.md`](../TOLTECA_SPACK_BUILD_INTEGRATION_REVIEW_2026-07-31.md)
- [`../BUILD_TIMING_BASELINE_2026-08-14.md`](../BUILD_TIMING_BASELINE_2026-08-14.md)
