# Timestream Successor D2 Admission Repair 001

Status: owner-authorized repair/reconciliation candidate; local gates pass;
independent exact-SHA review is pending at record preparation, with fresh
Unity evidence and owner admission still required

Work order: `TIMESTREAM-SUCCESSOR-D2-ADMISSION-REPAIR-001`

Owner: Citlali project owner. Risk tier: Tier 2.

## Owner authorization and preflight

On 2026-09-04 the owner approved the narrow repair of `D2-ADM-F01` and
canonical reconciliation after independent review of exact candidate
`218988dc21cfc279cdfe7a4318b574f070cd45f9` returned **repair required**.
The approved follow-up rejects non-finite line-frequency metadata before
sorting, adds regression coverage, preserves both histories, validates the
new candidate, and obtains a fresh exact-SHA review. This supersedes the
previous work order's byte-preservation restriction only for the two repair
source/test paths and the bounded new status/evidence records.

No canonical ref movement, GitHub push, route activation, production,
cleanup, later implementation, scientific-policy change, or governance
amendment is authorized. The owner performs Unity operations and all pushes.

The toltec-context skill and repository `AGENTS.md` govern this task.
Applicable governance read: `doc/governance/ENGINEERING_GOVERNANCE.md`,
`doc/governance/TIMESTREAM_SUCCESSOR_GOVERNANCE.md`, and
`doc/governance/REVIEW_AND_CONFORMANCE.md`. Their unchanged accepted digests
are bound by governance commit `06a3ade51c1b3f38887295433d913811bf25cd14`
and effectiveness record `77507836325eff9f469062d5884481ea37599594` on
canonical ancestry. Current sequencing uses `doc/REFACTOR_STATUS.md` and
`doc/INTEGRATION_LEDGER.md`; architecture and ownership use
`doc/ARCHITECTURE.md`, `doc/SCIENTIFIC_CONVENTIONS.md`, and those effective
governance documents.

Scientific authority remains WP-7.1 source
`170ecea9de1ee810da7d7e45a489a4545ccd623d`, closure
`20ba6ae5dcf0b90a24ac3e778a75eff0a1bbe2aa`, the canonical authority router
`validation/wp7_timestream_successor_authority.json`, ADRs 0017--0023,
and the explicit D2/VAL ownership dispositions preserved in the original
D2 handoff. This repair makes an existing mechanical rejection reliable; it
selects no unresolved scientific authority.

## Exact ancestry and owned worktree

| Role | Exact identity |
| --- | --- |
| Verified local canonical base / first parent | `ae953ed4d87d1f693d2bbf42aebbc25ef730c771` |
| Local canonical tree | `37ee17cf001ceb2c193fbea5e2b5ae3d147ba4a1` |
| Live GitHub canonical at preflight / common ancestor | `9f42d348298d76c5d5145aaf0c3eace1f3e154c1` |
| Preserved reviewed admission candidate / second parent | `218988dc21cfc279cdfe7a4318b574f070cd45f9` |
| Prior candidate tree | `449cd2f8eb15ff5c813ae2d49db16f9773cfd336` |
| Accepted, pushed original D2 closure | `54c3254aaf8a6639d3fe1e56e23709e29cd34b3b` |
| Original D2 implementation | `bb060947175523d6fc6a777ae4ad4606693e9e5f` |
| Literal original D2 base | `4d14d0dce8c80b6bc9d0d39c9a90a8f4b2504538` |

The post-approval read-only GitHub check confirmed canonical `9f42...` and
D2 feature `54c3254a...`; local canonical was clean at `ae953ed4...`.
The owner-approved reconciliation preserves the four scientific-contract
manager-handoff paths added by that local canonical child. It does not infer
new scientific, validation, or production authority from those documents.

Owned branch: `codex/timestream-successor-d2-admission-repair`.
Owned clean starting worktree:
`/private/tmp/citlali-timestream-successor-d2-admission-repair`.
It was created literally at local canonical `ae953ed4...`, then the preserved
`218988dc...` candidate was merged without conflicts. The repair and records
are incorporated into this merge candidate. Neither parent is rewritten.
The prior admission branch remains frozen at `218988dc...`, the original
feature branch remains at `54c3254a...`, and all original D2 commits stay
reachable. Unrelated dirty worktrees and retained artifacts remain untouched.

This bounded source repair occupies the single spine slot during preparation;
no module probe is active. The prior admission candidate is retained evidence,
not another active implementation lane. Canonical reconciliation is part of
this same bounded candidate and does not create another implementation slot.

## Review finding and smallest repair

Independent fresh-context read-only review of `218988dc...` found one major
behavioral issue, `D2-ADM-F01`, and no architecture/ownership or frozen-
candidate repository/evidence findings. `D2LineOperatorEvidence::admit`
called `std::sort` before rejecting non-finite low/high frequencies. The
comparator treats low frequencies `1` and `NaN`, and `NaN` and `3`, as
equivalent while ordering `1 < 3`; its induced equivalence is not transitive.
This violates sorting's strict-weak-order precondition before guaranteed
rejection can occur. The same defect exists in original `bb060947...`.

The repair moves the existing two finiteness checks into a read-only
pre-sort loop. The exception type and message, frequency comparator,
nonnegative/ordered/non-overlap rules, line identities, effective-before-
decimation requirement, and operator-evidence identity checks are unchanged.
Valid finite records have the same deterministic order and admission outcome.

Only these source/test paths change relative to the preserved admission
candidate:

- `include/citlali/core/pipeline/timestream_d2_native_measurement.h`;
- `tests/test_timestream_d2_native_measurement.cpp`.

The new test covers NaN, positive infinity, and negative infinity in either
field at all 32 input positions (192 invalid-input cases), plus a NaN high
frequency among equal low frequencies. Existing finite ordering and other
structural-rejection cases remain in the focused suite.

This is line-operator evidence metadata, not residual sample validation.
Non-finite residual `x/r` payload remains mechanically present as before.
VAL retains sole semantic sample/coordinate-validation authority. D2 still
owns residual storage and mechanical completeness, uses native axes, and
binds the exact immutable realization-time snapshot explicitly. Source-mask
evidence, route/profile and parent bindings, all lifetimes, and zero-copy
prefilter references are unchanged. No filter is designed or applied.

The added work is one O(number-of-line-records) scan and no allocation in
this cold admission boundary. There is no detector-occurrence inner-loop or
new memory owner, and no performance campaign is triggered.

## Focused regression and local gates

A diagnostic build of the new test against the inherited, unmodified header
used Apple libc++ debug hardening:
`-D_LIBCPP_HARDENING_MODE=_LIBCPP_HARDENING_MODE_DEBUG`. It aborted with
SIGABRT and the explicit strict-weak-order assertion. The same diagnostic
build with the repaired header passes all eight D2 tests. This establishes
a failing/passing regression under the available local standard-library
check; ordinary release-mode exception tests alone need not expose undefined
behavior. Debug hardening is a diagnostic compile option only and is not
added to repository CMake or production flags. No intentional assertion is
reported as a passing gate or hidden as unexpected output.

The diagnostic links unchanged dependencies/library objects from the prior
local build and recompiles the actual test translation unit without its
PCH so the hardening mode reaches `std::sort`. The ordinary candidate build
is configured separately in its own fresh `build/` tree.

Local gates completed before commit:

| Gate | Result |
| --- | --- |
| Fresh supplemental configure | PASS |
| CLI, D2 target, isolated public-header compilation, safety target | PASS |
| Focused D2 CTest selection | 8/8 PASS |
| Full `check` target | 892/892 runnable tests PASS; 893 registered |
| Baseline-tool unit tests | 207/207 PASS |
| Full config preflight `--require-all` | 130 tests PASS; four-mode kits 8/8; all coverage and drift audits PASS |
| Validation ledger | PASS; 60 records |
| Science-change ledger | PASS; 3 changes, 5 integration commits |
| Inherited-source debug-hardening negative experiment | Expected SIGABRT at strict-weak-order assertion |
| Repaired-source debug-hardening diagnostic | 8/8 D2 tests PASS |

The established disabled test remains
`citlali::MapFitterLifecycle.ExactProductSequence`; it is not counted as a
passing runnable test. There is zero unexpected error-level gate output.
The fresh local build emits two existing glog macOS `syscall` deprecation
warnings. The intentional negative experiment is retained separately from
the passing gates. The configure took 45.24 seconds, CLI/D2/safety build
162.06 seconds, and full build/check 329.50 seconds (CTest 122.69 seconds).
These are local gate timings, not a performance claim or Unity measurement.

The canonical manager-handoff verifier was run successfully at its own
exact subject `ae953ed4...`. Its direct-parent/four-path assertions bind that
snapshot and do not transfer to this D2 merge. This candidate instead proves
all four of those canonical paths byte-identical and preserves that commit
on its first-parent ancestry; no verifier semantics are changed.

The local environment remains supplemental AppleClang 21.0.0, CMake 4.3.0,
Homebrew dependencies, and `$HOME/tolteca/bin/python` 3.13.2. It uses the
previously recorded fallback configuration and cached dependency sources;
pre-existing `kids 04088da-dirty` remains a disclosed limitation. No
dependency patch, upgrade, Conan execution, or Spack-policy change occurs.
Python verification uses `MPLBACKEND=Agg` and task-specific writable caches.

Task-local commands, regression logs, gate reports, and the eventual
exact-candidate proof are retained under
`/private/tmp/citlali-d2-admission-repair-evidence-2026-09-04`. The path is a
local evidence locator, not an immutable source identity or Unity archive.
Final candidate SHA/tree/parents, path digests, postcommit CLI provenance,
review verdict, and clean-worktree state are reported externally after commit
to avoid a circular claim about the containing commit.

Writer's three-axis preparation assessment (not independent approval):

- Scientific/behavioral: the reported mechanical ordering defect is repaired
  and local gates pass; no intentional scientific change. Representative
  Unity evidence for this new source remains required.
- Architecture/ownership: unchanged D2 storage/admission ownership and VAL
  semantic-validation boundary; no new interface, lifecycle, or dependency.
- Repository/evidence: explicit two-parent reconciliation and bounded path
  preservation; final exact-SHA proof, postcommit provenance, and independent
  review are required before a later admission decision.

## Fresh Unity gate and admission boundary

The original Unity job `63987273` passed 891 runnable tests at original
implementation `bb060947...`. That evidence remains intact in the original
handoff but cannot validate this changed source. The prior admission
candidate's whole-application input-equivalence reuse is superseded for this
repair. A fresh owner-run `unity-gcc13` configure/build/check at the final
reviewed repair SHA is required before canonical admission. Expected test
count increases by the one new D2 test; the established disabled
`MapFitterLifecycle.ExactProductSequence` remains unchanged.

Prepare the same resource-matched gate pattern: six CPUs/build jobs,
64 GiB, and an empty job-local non-NFS build tree. Use the accepted Spack
compiler wrappers and existing environment; explicitly build the safety
target before full `check`. Bind source cleanliness, CLI version, full/focused
results, configuration/compiler context, final status, and checksums to that
exact SHA. All uploads and the isolated owner-run source checkout go under
`~/work_toltec/wilson/citlali_testing`; no shared checkout is reset or switched.
No Unity connection, upload, job, build, reduction, or cleanup is performed
by Codex. Affected-mode/representative-science reduction is not triggered:
valid data and every executable route remain unchanged and unactivated.

## Preservation, review, and completion conditions

Beyond the two repair paths, newly authored repository content is limited to
this handoff plus bounded additions in `doc/REFACTOR_STATUS.md` and
`doc/INTEGRATION_LEDGER.md`. Both previous D2 handoffs, the isolated-header
test, CMake test registration, and all four reconciled canonical handoff
paths remain byte-identical to their parent authorities. No validation
acceptance ledger or frozen scientific contract is edited.

Fresh independent review must assess the exact final SHA on all three axes,
including `D2-ADM-F01`, actual regression evidence, precise two-parent
ancestry, source/path preservation, D2/VAL ownership, environment limits,
and the fresh Unity hold. The writer does not approve the repair.

Moving canonical refs, conflicting content, unexpected validation results,
or a new semantic/ownership dependency trigger recorded reassessment. Do not
silently change base, add a second repair, or relax an evidence gate. The
final candidate must contain the then-agreed canonical history before any
later owner-authorized fast-forward. If either canonical authority moves
again, preserve this candidate and obtain the necessary exact ancestry
disposition rather than discard work.

Explicit exclusions: filtering, factor selection, downsampling, substantive
RTC/PTC/AST, common-grid projection, MAP, CAL, route activation, ordinary-route
changes, new persistent schemas, production, release, legacy retirement,
cleanup, scientific-contract changes, and subsequent Timestream Successor
work. Completion of this candidate does not authorize any of them.
