# Timestream Successor Unity GCC13 ABI Conformance 001

Status: bounded build-adaptation implementation in progress; no candidate,
owner acceptance, integration, or activation is established by this record

Work-order identity:
`TIMESTREAM-SUCCESSOR-UNITY-GCC13-ABI-CONFORMANCE-001`

Owner: Citlali project owner

## Authorization and preflight

The owner authorized option 1 from the representative-acceptance diagnosis:
repair Citlali's repository-owned `unity-gcc13` development profile so that
Citlali and its concrete Spack dependencies use one target/ABI realization.
This is a separately governed build-adaptation lane. It does not occupy the
Timestream Successor spine or scientific-module-probe WIP slots.

Risk tier: Tier 2. The change is build-environment and executable-evidence
conformance. It changes no scientific operation, typed product, pipeline
route, or application configuration.

Applicable governance:

- `doc/governance/ENGINEERING_GOVERNANCE.md`;
- `doc/governance/TIMESTREAM_SUCCESSOR_GOVERNANCE.md`;
- `doc/governance/REVIEW_AND_CONFORMANCE.md`;
- ADR 0014, `doc/adr/0014-spack-build-foundation.md`; and
- `doc/TOLTECA_BUILD_INTEGRATION_REVIEW_2026-07-26.md`.

Branch and worktree:

- exact base: `655160ba59a8abcddb5984d3241857c4ae969589`;
- base tree: `3236d5ec3e19fe2bc30a33386fb788344c4c4ca8`;
- branch: `codex/timestream-successor-unity-gcc13-abi-conformance`;
- worktree:
  `/private/tmp/citlali-timestream-successor-unity-gcc13-abi-conformance`.

The base is the pushed, independently reviewed identity-route evidence
checkpoint. The owner performs every GitHub push. Codex must provide a normal,
non-force `git -C` command containing the full absolute repository path.

## Trigger and evidence

The exact-source Unity `unity-gcc13` build configured Citlali with direct
`/usr/bin/gcc` and `/usr/bin/g++`. Those explicit CMake compiler paths bypassed
the compiler wrappers established by `spack build-env`. The concrete Spack
root and installed Kidscpp were built for `cascadelake`; the direct Citlali
compile did not receive that concrete target's compiler arguments.

The resulting mismatch produced a repeatable invalid free in Eigen storage
destruction at Kidscpp `loadfitreport`, reached through
`kids::TimeStreamSolver`. The behavior reproduced on multiple Unity compute
nodes. An O1 non-native Citlali/Kidscpp pair passed, an O3 concrete-target
Kidscpp with non-native Citlali failed, and a diagnostic Citlali build with
`-march=cascadelake -mtune=cascadelake` passed the complete representative
witness. This establishes a build-realization mismatch rather than a repaired
Kidscpp source defect.

The retained diagnostic witness remains valid evidence under its truthful
matched-native label. It is not evidence that the unmodified repository
profile passed.

## Bounded implementation

The repair must use the concrete Spack root as target authority rather than
hard-code today's `cascadelake` flags in Citlali. Spack 1.2.2's compiler-wrapper
package exports:

- `CC` and `CXX` as compiler-wrapper paths;
- `SPACK_CC` and `SPACK_CXX` as the concrete underlying compilers;
- `SPACK_COMPILER_WRAPPER_PATH` as wrapper ownership; and
- `SPACK_TARGET_ARGS_CC` and `SPACK_TARGET_ARGS_CXX` from the dependent
  concrete spec's microarchitecture.

Included changes are limited to repository-owned Spack build tooling:

1. inspect and validate those compiler controls before build-profile work;
2. require the underlying compilers to match the named profile;
3. require C and C++ target arguments to be present and equal;
4. stop passing direct compiler paths to CMake, allowing fresh configuration
   to honor the wrapper-valued `CC` and `CXX` environment;
5. reject a persistent CMake cache that retains a direct compiler and instruct
   the operator to reconfigure with `--fresh`;
6. keep installed-consumer compilation inside the same `spack build-env`; and
7. record wrapper and target controls in future build-timing manifests.

The profile retains exact expected underlying compiler paths as preflight
identity checks. The repair does not parse the lock to reconstruct compiler
flags, infer flags from the hostname, or name a fixed Unity microarchitecture.

Excluded:

- changes to Kidscpp, Eigen, or their package recipes;
- changes to Identity Route 001 implementation or acceptance runner;
- filtering, factor selection, downsampling, RTC/PTC/CAL/AST/MAP behavior, or
  any scientific policy;
- route activation, canonical integration, production use, or representative
  science acceptance;
- dependency reconcretization or replacement of the retained Unity lock; and
- a generic build-system redesign outside the Citlali Spack profile lane.

## Owners, significance, and gates

The active owner is build/profile tooling. The direct consumers are the
persistent Citlali development build, installed-package consumer acceptance,
and build-timing campaign. No scientific identity, unit, frame, shape,
validity, cause, or lineage changes.

Expected performance significance is limited to using the concrete root's
already selected target arguments. Memory pressure from parallel compilation
is separate from the ABI defect; Unity validation should use conservative
parallelism where necessary and record it truthfully.

Required gates:

- focused unit tests for wrapper ownership, concrete compiler identity,
  matched target arguments, direct-compiler rejection, and CMake-cache
  rejection;
- the complete build-tool suite and Python syntax compilation;
- configuration and baseline-tool regression gates because the durable status
  and work record change;
- local supplemental build/CTest validation where the retained local Spack
  realization permits it;
- independent fresh-context, read-only review of one exact full candidate SHA;
- after owner acceptance and user-performed push, a fresh unmodified
  `unity-gcc13` configure/build/test and complete representative identity-route
  acceptance with retained evidence.

Stop and reassess if the concrete environment does not expose wrapper-owned C
and C++ compilers, does not expose one matching nonempty C/C++ target argument
set, identifies underlying compilers inconsistent with the profile, requires a
dependency-source change, or reveals a scientific/application defect.

No push, canonical integration, route activation, or production authority is
included. The implementation candidate and Unity execution each require their
own reported owner disposition.

## Candidate validation before commit

The bounded implementation changes only the developer-profile runner, shared
Spack build checks, installed-consumer acceptance, build-timing evidence,
their focused tests, and this work-order record. It changes no CMake project,
package recipe, dependency pin, application source, public header, scientific
contract, route, or acceptance-runner source.

Focused and broad non-Unity results:

- Ruff check and format check passed for all seven affected Python paths;
- Python syntax compilation passed for `tools/build`;
- build-tool tests passed 73/73, including new failure cases for a direct
  compiler, wrong underlying compiler, mismatched target arguments, stale
  direct-compiler CMake cache, missing cache, and an installed-consumer build
  that remains inside `spack build-env`;
- configuration preflight passed 130/130, all four mode-kit hashes, 8/8
  compatibility cases, 100% compact surface coverage, and every authority and
  boundary audit;
- baseline tools passed 207/207 after the focused CLI was available;
- historical WP-7 tools passed 26/26;
- the validation ledger reported 60 valid records; and
- the science-change ledger reported 3 changes and 5 valid integration
  commits.

The first baseline invocation ran before the focused CLI existed in this new
worktree. Its 11 CLI-protocol cases failed only their explicit executable
precondition; the other tests completed. After the focused executable linked,
the complete suite was rerun and passed 207/207. The precondition failure is
not a product or protocol finding.

A local Spack 1.2.2 mechanism check used installed root
`u2qcns6o5rtnzzzblqvqsnbdr7uldxgf`. Its build environment reported wrapper
paths for `CC` and `CXX`, the expected underlying Clang 20.1.8 paths in
`SPACK_CC` and `SPACK_CXX`, and identical concrete target arguments
`-mcpu=apple-m1`. A fresh configure with no CMake compiler override stored the
two wrapper paths in `CMakeCache.txt`. The wrapper then invoked the real
compiler with `-mcpu=apple-m1`, and the focused `citlali_cli` target compiled
and linked successfully. The CLI reported Kidscpp 3.1.0 and the exact installed
root DAG. One existing C++23 deprecation warning in GrPPI and duplicate-rpath
linker warnings were present; there were no error-level build messages.

That local check is supplemental mechanism and regression evidence. It used a
pre-existing macOS installed dependency root and a dirty candidate worktree;
it is not an installed-artifact acceptance, a reproduction of Unity, or
representative science evidence. The authoritative remaining gate is a clean
exact-candidate Unity run of the repository's unmodified `unity-gcc13`
profile. It must show wrapper-valued cached compilers, concrete
`cascadelake` target arguments supplied by Spack rather than a manual CMake
flag, a complete build and CTest result, and the representative identity-route
witness without the invalid free.
