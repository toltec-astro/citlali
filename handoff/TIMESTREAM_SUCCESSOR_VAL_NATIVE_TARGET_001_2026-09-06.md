# Timestream Successor VAL Native Target 001

Status: bounded implementation and precommit local gates complete;
full committed-source gate and final exact-SHA review bind externally.

## Work order and owner disposition

Work order: `TIMESTREAM-SUCCESSOR-VAL-NATIVE-TARGET-001`. Owner: Citlali
project owner. Risk tier: 2. One integration/spine slot; no module probe.
The owner accepted the independently reviewed prerequisite
`080df2a1431487e8cabc255beb9ddc59c0721b59` and approved Section 5's source
unit: "I accept and approve this. Let's proceed."

Local canonical was fast-forwarded literally from reverified live GitHub
`5244e04638db5aa92180feab52acd782229cfa32` to that accepted prerequisite,
tree `a619f802a11eca8f9adccbebbf230517b2891945`. The accepted review is
preserved under `/private/tmp/citlali-val-d2-prerequisite-evidence-2026-09-06`;
its exact digest and completed manifest are recorded in the integration ledger.
No prior evidence record is changed. GitHub remains at the pushed D2 closure;
the one-commit documentation difference is expected and owner-authorized.
No push is needed to start from the now-accepted local canonical authority.

Exact implementation base: `080df2a1431487e8cabc255beb9ddc59c0721b59`.
Branch: `codex/timestream-successor-val-native-target-001`.
Worktree: `/private/tmp/citlali-timestream-successor-val-native-target-001`.
Initial staged/unstaged/untracked state: clean. Unrelated worktrees remain
untouched. The original D2 implementation, repair, closure and admission
commits are ancestors and are not rewritten.

Applicable `AGENTS.md`, toltec-context routing, all three effective governance
documents, living status/ledger, architecture/scientific conventions and the
accepted prerequisite were read. Normative governance bytes match accepted
`06a3ade51c1b3f38887295433d913811bf25cd14`; effectiveness record
`77507836325eff9f469062d5884481ea37599594` remains on ancestry. Six frozen
SCI-VAL Core digests match the accepted prerequisite record.

## Authority, ownership and scope

The [accepted prerequisite](TIMESTREAM_SUCCESSOR_VAL_D2_PREREQUISITE_001_2026-09-06.md)
is the full subject-specific recovery/requirement crosswalk. This increment
implements only mechanical targeting under SCI-VAL-REQ-001/002/006/043
(producer/use-owner/VAL separation), 003/008/010 (exact identity),
004/005/011--015/036--038 (no inference from absence), and 027--029/040
(immutable lifecycle/replay). REQ-023/024 and 009/019/020/044--046 remain
constraints on later numerical/use-profile adoption, not policies to implement
here. WP-7.1 authority JSON, ADRs 0017--0023 and the six frozen Core modules
remain unchanged. Original D2 owner corrections 3--5 continue to require exact
realization-time/publication snapshot handles.

VAL owns the fact-target container and immutable snapshot mechanics. The real
producer explicitly supplies product and realization identity; a consumer
supplies an exact key to look up opaque facts. A descriptor binds one exact
Paired-D1 network and its unchanged native occurrence/detector support. It
does not own a numerical payload or snapshot, avoiding a reference cycle.
Qualified targets identify one explicit detector-occurrence and x or r;
unqualified facts retain their original domain with no broadcast or fallback.
Fact-author identity and numerical-subject identity remain distinct.

This unit has no numerical values, units conversion, frame/grid change,
validity projection, cause interpretation, scientific threshold, persistent
schema or route wiring. x/r identity uses the existing native-coordinate type;
its scientific conventions remain those of the producer. Exact handle identity
is scoped to retained in-memory objects and is not a serialization identity.

Allowed source/test paths are the VAL header, its behavioral test, and its
isolated-header test. Existing CMake registration is expected to suffice.
Bounded status/ledger/handoff records accompany them. D2 code/tests, Paired-D1,
identity route, registry, science, configs and offline PSD tooling stay intact.
No producer, full VAL evaluator, mask exporter, new scientific owner, generic
fact framework, filter, factor, downsampling, RTC/PTC/AST/CAL/MAP, common-grid,
activation, production or cleanup enters this unit.

## Gates, risks and stop conditions

Focused: independent VAL header compilation; x/r, raw/derived and exact-instance
separation; foreign parent/network/row/detector rejection; old/new domain
coexistence; duplicate rejection; immutable overlay and branch separation;
ordering; missing-fact noninference; lifetime and compact memory evidence.
Existing D2, Paired-D1 and identity-route tests remain regression gates.

Broader: local CLI/safety build, runnable CTest suite, config preflight with
all required data, baseline-tool tests and both ledgers. The Mac
AppleClang/Homebrew/cached dependency realization is supplemental and its
actual dependency dirt is disclosed. A clean owner-run Unity GCC13/Spack gate
at the final changed source is required before source admission. Job 64018093
does not cover new VAL code. No scientific reduction is triggered by this
unactivated identity-only interface change.

Measure representation size, per-finding overhead and retained references;
avoid heavy identity strings, dense validity planes, payload copies, ownership
cycles, process-global counters and allocations in lookup/comparison. Exact
target ordering must be a strict total order for retained identities; no
cross-process byte serialization or persistent pointer ordering is claimed.

Stop/reassess if implementation needs D2 changes, a new policy, altered snapshot
equality, cross-stage reach-through, persistence, a producer, scope expansion
or changed canonical/authority overlap. Missing scientific decisions are not
filled from tests or existing tool behavior. The unmodified SCI-VAL verifier's
historical MAP-unbound failure remains a recorded base limitation; it must not
be weakened or relabeled PASS.

Every final candidate receives fresh-context independent read-only exact-SHA
review on all three governance axes. Canonical admission of this changed
source, GitHub push, activation, production and cleanup remain separate owner
dispositions. The user performs all GitHub pushes and Unity operations.

Evidence root: `/private/tmp/citlali-val-native-target-evidence-2026-09-06`.
Final candidate/tree, changed-content digests, gate outcomes and independent
review are bound after implementation and commit, without self-approval.

## Implemented boundary and local verification

`ValNativeRealization::create` checks an explicit parent, named producer,
nonzero realization instance, original-input/derived-residual role and one
existing parent network. The descriptor is noncopyable, has no public mutable
alias, and retains only the parent handle and scalar identity. It establishes
the entire unchanged native network support, not a selected numerical subset.

`ValSnapshot::native_target` binds that descriptor to an existing exact
`ValAddress` and `NativeReadoutCoordinate`; it rejects absent/foreign handles,
foreign network/address, missing detector and invalid coordinate. Address
construction retains the established native-row/detector range checks. An
explicit detector is required to avoid implicit coordinate broadcast.
`ValFindingKey` stores either the existing address or the qualified target;
its author identity is separate from the descriptor's producer identity.
`ValDeltaBuilder` validates either domain before accepting a proposal. Freeze,
duplicate rejection, opaque states/causes, inherited lookup and immutable
snapshot branching retain their existing semantics. A moved-from target is
rejected. All qualified keys retain their exact descriptor handles.

The seven new behavioral tests cover the accepted target cases, 24 proposal
permutations, strict ordering/equality/transitivity, same-description distinct
instances, unknown facts, nonfinite stored input without inference, two equal-
generation snapshot branches, frozen builders and acyclic lifetime release.
No D2/header/test or existing authority bytes change. CMake changes were not
needed; the standalone header target also checks noncopyable realization and
non-default-constructible target properties.

Precommit Mac results: CLI, safety, identity and D2 builds PASS; focused VAL
10/10 and D2 8/8 PASS; config preflight PASS with all required kits/audits;
baseline tools 207/207 PASS; both ledgers PASS. Full `check` and the expanded
Paired-D1/identity regression selection run against the committed candidate,
with their exact results in the external completion record. No precommit run
is falsely labeled execution at that later commit. The final CLI revision
must match the clean committed source before review.

An initial baseline-tool attempt ran before the CLI existed and failed 11
protocol assertions. The original log is retained as
`baseline_tools-before-cli.log`; after the CLI build, all 207 tests passed
without source/test changes. This was local gate sequencing, not a product
regression or waived test. No application error-level failure is accepted.

Measured on the Mac: `ValAddress` remains 56 bytes; finding key grows from
80 to 112 bytes and finding from 88 to 120 bytes. Each shared realization
descriptor is 48 bytes. The additional 32 bytes apply per stored sparse fact,
including existing unqualified findings because the variant reserves its
largest alternative. No dense validity plane is allocated. These are logical
inline sizes, excluding allocator/control-block overhead and RSS. Memory
evidence counts target handle references in each delta, not unique descriptors.
The lifetime test proves no retained snapshot/descriptor/parent cycle; the
numeric payload pointer is unchanged. No performance optimization or new
throughput claim is made.

The local fallback uses AppleClang, Homebrew and the existing cached sources;
Kids remains at `04088da182622c3e879f04314974a7c0d60ee2d6` with the three
pre-existing dirty headers recorded by digest in `ENVIRONMENT.json`. They
were verified unchanged before building. This is supplemental evidence; the
new clean pinned-source Unity GCC13 gate is still required. The historical
SCI-VAL registry-verifier limitation remains unrepaired and fully disclosed.
