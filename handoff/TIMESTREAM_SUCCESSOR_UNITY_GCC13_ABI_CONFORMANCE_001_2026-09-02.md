# Timestream Successor Unity GCC13 ABI Conformance 001

Status: owner-accepted build-adaptation candidate and authoritative Unity gate
complete; closure-record candidate pending independent exact-SHA review and
owner disposition; no canonical integration or activation

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
- ADR 0014, `doc/adr/0014-spack-build-foundation.md`;
- `doc/TOLTECA_BUILD_INTEGRATION_REQUIREMENTS_2026-07-23.md`; and
- `doc/TOLTECA_SPACK_BUILD_INTEGRATION_REVIEW_2026-07-31.md`.

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

- Ruff 0.13.2 lint (`ruff check --no-cache`) passed for all seven affected
  Python paths; Ruff formatter conformance was not selected as a gate because
  four of those files retain the repository's existing non-Ruff formatting;
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

## Independent review repair

Independent fresh-context review of exact first candidate
`aa3b31510b41d750c3475a5071e8f6d51f5f98ca`, tree
`28421540b0fb307ae7e7268e80f6ae179ae6e42a`, passed the
scientific/behavioral and architecture/ownership/anti-bloat categories with no
findings. It failed repository/evidence conformance on two minor record defects:

1. the applicable-governance list cited the superseded 2026-07-26 Conan
   review instead of the active 2026-07-31 Spack review and its requirements;
   and
2. the validation summary claimed a Ruff formatter check that was not
   reproducible after pre-existing file formatting was intentionally retained.

This follow-up changes only this work-order record: it replaces the stale
authority reference and reports the selected Ruff lint gate accurately. It
does not alter the reviewed implementation, tests, scope, exclusions, or
remaining Unity gate. The repaired exact commit requires a new independent
exact-SHA review; the earlier verdict does not transfer.

## Repaired-candidate review and owner acceptance

Independent fresh-context review of repaired exact candidate
`b3f03ae20059022ecb709dd411dfbdde3723fef3`, tree
`1712f13d16506895fb94b72b845f2cbb821bccb7`, sole parent
`aa3b31510b41d750c3475a5071e8f6d51f5f98ca`, passed with no
scientific/behavioral, architecture/ownership/anti-bloat, or
repository/evidence findings. The review reproduced 73/73 build-tool tests,
Python syntax compilation, Ruff 0.13.2 lint, 130/130 configuration tests,
26/26 historical WP-7 tests, and both validation ledgers. Its local installed
Spack probe confirmed wrapper ownership and matched target arguments. It
correctly retained the fresh unmodified Unity build/CTest and representative
identity-route witness as owner-run gates.

The project owner accepted exact candidate `b3f03ae2...` and performed the
normal non-force push of its literal ancestry:

`655160ba59a8abcddb5984d3241857c4ae969589`
-> `aa3b31510b41d750c3475a5071e8f6d51f5f98ca`
-> `b3f03ae20059022ecb709dd411dfbdde3723fef3`.

The local branch and cached remote feature ref were both verified at
`b3f03ae2...` before the Unity gate. That acceptance authorized only the
feature-branch build/profile candidate and its bounded owner-run validation.

## Authoritative Unity execution

The owner ran the unmodified repository `unity-gcc13` profile at exact source
`b3f03ae20059022ecb709dd411dfbdde3723fef3` on Unity. The source checkout was
clean and detached at that SHA. A fresh configure/build/test used Spack root
DAG `ryu6zhjvp5sh7fsbwrzyykbkgkglzkkx`. The CMake cache identified Spack
compiler-wrapper paths for both C and C++ rather than direct `/usr/bin`
compilers, and the retained preflight verified matching concrete target
arguments `-march=cascadelake -mtune=cascadelake` for C and C++.

All 884 runnable CTests passed. The sole non-run test was the established
disabled `citlali::MapFitterLifecycle.ExactProductSequence`; there were no
failed tests. The opt-in acceptance executable was then built explicitly
because its intentional `EXCLUDE_FROM_ALL` declaration keeps it out of the
ordinary build. The complete bounded real-data Identity Route witness ran,
wrote its v2 record, and passed the repository validator. The prior invalid
free in Eigen storage destruction through Kidscpp `loadfitreport` did not
recur.

The authoritative timestamped evidence directory is:

`/work/toltec/citlali_dev/citlali_acceptance_evidence/TIMESTREAM-SUCCESSOR-UNITY-GCC13-ABI-CONFORMANCE-001/b3f03ae20059022ecb709dd411dfbdde3723fef3/authoritative-20260903T120212Z`

Exact retained evidence identities are:

- acceptance JSON SHA-256
  `0c4189970739f98ae3d63bb5fe06a140dc86268f1b9b99c791140cf239b11b4d`;
- runner-log SHA-256
  `27d69c67eb75b8fb404500a90fa877e52134ecea508fb61ed45b8f6ada817ea4`;
- validator-log SHA-256
  `4f19dec0a09efb7af84d286fe53d475986ff024e92a2b6fabf2c8729a9841b98`;
  and
- evidence-manifest `SHA256SUMS` SHA-256
  `4abeecc828817cf890e53bce52b039981fe6bb1b1a2fa8b1b5852a69dc4687e5`.

Two preceding operator attempts are not acceptance evidence. A long
interactive-shell paste was corrupted before the runner executed. The first
checksum-verified script invocation then stopped at preflight because the
deliberately non-default acceptance target had not yet been built. Neither
attempt created an acceptance record or ran the representative witness. The
timestamped directory above is the sole authoritative execution reported by
this closure.

## Completion and conformance

Disposition: evidence/closure-record candidate. The exact implementation
subject remains `b3f03ae2...`; this later documentation-only record does not
change its source or tree.

Scientific and behavioral conformance: pass. The accepted wrapper-based build
repair changes no scientific operation, product, identity, unit, frame,
validity, cause, lineage, configuration, or route. The representative tool
proved the complete typed identity witness with no comparison mismatch and no
unexpected error or critical record. It explicitly makes no representative
science claim.

Architectural conformity and ownership: pass. The repair remains wholly owned
by build/profile tooling and adds no `Engine` state, cross-stage dependency,
generic framework, application behavior, or scientific policy.

Repository, branch, and evidence hygiene: pass subject to independent review
of the closure commit. The implementation candidate has exact linear ancestry,
was independently reviewed and owner-pushed, and was clean at local and Unity
checkpoints. The evidence bytes remain retained on Unity and are represented
here by their exact path and digests; they have not been copied into Git.

Affected-mode result: no ordinary application route was activated or changed,
so no affected-mode reduction was triggered. This was an acceptance-tool
execution, not a Citlali reduction or MAP action.

Intentional scientific changes: none. Canonical integration, route activation,
legacy retirement, production use, MAP action, release, and representative
science acceptance remain unauthorized and unperformed.

The next owner decision, after independent fresh-context exact-SHA review of
the closure commit, is whether to accept this durable evidence record and
separately authorize a canonical-integration proposal. No integration, push,
activation, production, cleanup, or subsequent Timestream Successor increment
is implied.
