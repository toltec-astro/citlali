# Timestream Successor VAL/D2 Prerequisite 001

Status: authority recovery and bounded implementation proposal complete;
documentation candidate pending independent exact-SHA review;
no source implementation, canonical admission or push performed

Work order: `TIMESTREAM-SUCCESSOR-VAL-D2-PREREQUISITE-001`.
Owner: Citlali project owner. Risk tier: 2, interface/authority preflight.

## Authorization, base and limits

The owner approved the proposed next approach: specify VAL coordinate and
residual-realization targeting, establish the immutable binding needed by a
future producer/consumer, identify missing scientific-owner decisions, and
define bounded changes, gates and independent review. The owner asked whether
another push was necessary and directed continuation if not. No prerequisite
push is needed: live GitHub canonical and clean local canonical both resolved
to `5244e04638db5aa92180feab52acd782229cfa32`, tree
`10136ecdb4c503cdd1c4315832855472075cbaa8`.

The owned branch is `codex/timestream-successor-val-d2-prerequisite-001` in
`/private/tmp/citlali-timestream-successor-val-d2-prerequisite-001`, created
clean and literally at that exact base. This is one documentation work order,
not a scientific-module probe or application implementation slot. Its only
changed paths are this handoff, `doc/REFACTOR_STATUS.md` and
`doc/INTEGRATION_LEDGER.md`. Unrelated dirty worktrees and every accepted D2
implementation/closure are preserved. No successor source unit is begun by
this record; Section 5 is the concrete proposed boundary for owner disposition.

Applicable `AGENTS.md`, toltec-context routing, current status/ledger,
architecture/scientific conventions and all three effective engineering,
Timestream Successor and review/conformance governance documents were read.
Their accepted/effective ancestry remains `06a3ade51c1b3f38887295433d913811bf25cd14`
and `77507836325eff9f469062d5884481ea37599594`, with unchanged normative bytes.
The living status supersedes historical program and freeze-time status text.

## 1. Recovered authority and disposition

This is an implementation prerequisite review against existing science, not
new scientific derivation or a revision of a frozen package. Authority is
resolved by subject rather than by selecting the newest file globally.

| Recovered source | Disposition and consequence |
| --- | --- |
| WP-7.1 source `170ecea9de1ee810da7d7e45a489a4545ccd623d`, closure `20ba6ae5dcf0b90a24ac3e778a75eff0a1bbe2aa`, canonical authority JSON and ADRs 0017--0023 | Adopt existing bounded science. Contract closure and retained scope limits are not reopened by an unimplemented stronger interface. |
| SCI-VAL v0.1/r0.3 Core, frozen from `3ad018e97e134a0b0324d3fa2674ef96d5a680d4` under `SCIENTIFIC_OWNER_FREEZE_R0.3.md` | Adopt stable identity, fact/policy ownership, knowledge states, immutable lifecycle and replay requirements. Preserve all six Core modules. |
| SCI-VAL `PROFILE_REGISTRY.md`, `SOURCE_BINDING_REGISTER.md`, immutable JINC Stage A Q002 and NOI Stage A r0.18 successors | Inspect exact registered uses and source generations; do not replace historical bindings with an unspecified current registry. No inspected record binds a D2 native PSD use. |
| Approved `WP7_VAL_SUCCESSOR_BINDING_COMPATIBILITY_PROPOSAL_2026-08-25.md` | Adopt its exact additive compatibility rule for the five named PTC profiles. It does not authorize arbitrary new uses or promote a D2 profile. |
| Accepted D2 native-measurement handoff, implementation `bb060947175523d6fc6a777ae4ad4606693e9e5f`, repair `7d57a5acf893ae0f34c3639499484b3f5976768a`, admission `aeea0eef04ec70d8142c8a20fd7b09dfb64725ed` and closure base `5244e046...` | Adopt exact mechanical storage/publication and explicit owner corrections. Preserve all accepted code and history. |
| Existing `timestream_val_state.h` and its three behavioral tests; D2 header and eight behavioral tests | Inspect implementation evidence, not scientific authority. Identify missing typed targeting without claiming the current fact store is a full SCI-VAL evaluator. |
| Canonical D2 PSD/line tooling and certification-plan D2 gate | Adopt the measurement goal and inspect input expectations. Tool implementation is not authority for a new named-use policy or generic validity projection. |
| Historical producer prototype, legacy rectangular TOD, other package profiles | Exclude as implementation or scientific authority for this unit. No prototype import or legacy-data promotion. |

The freeze record's embedded registry/status snapshots are historical. The
current immutable MAP/JINC/NOI records remain valid in their own exact domains;
their existence does not confer applicability to D2. In particular,
`SCI-NOI:generation_input_admission@1` explicitly concerns
`NOI-GEN/PTC-TO-FROZEN-MAP-CONDITIONAL-SIGN@1`, not native PSD measurement.

## 2. Ownership and exact requirement trace

The owner's instruction that VAL retains semantic validation authority at the
D2 boundary prohibits a D2-local validator or usability plane. It does not
transfer producer facts or a consumer's scientific predicates to VAL.
SCI-VAL retains shared mechanics; producers retain their facts/supports;
the actual named-use owner supplies policy. These statements are compatible.

| Obligation | Exact authority | Consequence for this prerequisite |
| --- | --- | --- |
| Separate producer facts, use-owner policy and VAL mechanics | SCI-VAL-REQ-001, 002, 006, 043 | D2 does not author validity; VAL does not originate facts or PSD predicates. |
| Stable object, role, stage, parent and lifecycle identity | SCI-VAL-REQ-003, 008, 010 | A sample address alone cannot qualify which coordinate of which derived product a fact concerns. |
| Unknown, false, nonfinite, unavailable and inapplicable remain distinct | SCI-VAL-REQ-004, 005, 011--015, 036--038 | Absence of a targeted finding is not permission; no Boolean default or cause inference. |
| Named uses require exact owner-approved binding | SCI-VAL-REQ-009, 019, 020, 044--046 | Independent exposure, PTC output retention, MAP admission and NOI generation are not aliases for PSD eligibility. |
| Invalid inputs excluded before numerics; admitted nonfinite input fails or becomes unavailable at owner-declared scope | SCI-VAL-REQ-023, 024 | A later PSD adapter must establish its predicate and numerical failure scope before adopting the tool's mask behavior. |
| Nonretroactive lifecycle and deterministic replay | SCI-VAL-REQ-027--029, 040 | Preserve old snapshots/actions; qualify exact producer realization; no ambient current-state lookup. |
| Representation-independent science and retained limits | SCI-VAL-REQ-041, 042; WP-7.1 owner closure | A mechanical target type grants no new numerical auxiliary-R use, production state, profile or scientific algorithm. |
| Exact realization/publication snapshot handle | Accepted D2 handoff owner corrections 3--5 and current `D2ResidualNetworkPayload::realize` / `D2NativeMeasurement::publish` | Keep exact handle equality; matching generation numbers, rows or descriptive text are insufficient. |
| Processing controls stay separate from validity | SCI-VAL definitions and accepted D2 correction 6 | Source exclusions and line-operator evidence never become validation facts by translation. |

## 3. Current implementation and adoption gaps

`ValAddress` identifies exact Paired-D1, native sample/occurrence keys and an
optional detector. `ValFindingKey` combines producer-product identity, that
address and an opaque producer fact code. The snapshot stores immutable
generation deltas and resolves inherited findings. It has no typed `x/r`
target or separate typed raw-versus-derived payload subject. Opaque fact codes
must not be assigned new shared coordinate semantics by convention. Existing
unqualified addresses must not silently be reinterpreted as pair-wide facts.

`D2ResidualNetworkPayload` binds one supplied residual realization to its
exact Paired-D1 parent and explicitly supplied immutable `ValSnapshot`.
Publication checks the same handles and realization identity. It deliberately
has no `valid`, `x_usable`, `r_usable` or causes interface, and accepts
mechanically complete nonfinite values. No admitted D2 invariant is defective
merely because a future consumer needs more typed information.

The offline evidence builder accepts a `valid` Boolean array and computes
`valid & ~source_excluded & isfinite(signal)` before centering/PSD evaluation.
That implementation neither binds a named-use profile nor distinguishes an
already-admitted numerical failure from a supplied exclusion. It is retained
as accepted bounded tooling evidence. Feeding a future scientifically
admitted stream through it without an explicit policy/failure disposition
would be an adoption gap; this review does not change or newly certify it.

The inspected base, JINC and NOI registry files contain no D2 PSD profile.
The negative result is bounded to these canonical authority sources and the
linked D2 records. No permission is inferred from `diagnostic_display`,
`estimator_fit`, existing output retention, or a profile in another package.

## 4. Required binding model

The mechanical target must distinguish the exact native sample/detector,
the `x` versus `r` coordinate, the original-versus-derived product role and
the exact owning producer realization. Fact-author identity remains distinct
from the identity of the numerical object being described. Equal axes,
generation numbers, descriptive realization strings or producer-local
integers in different scopes do not establish identity.

For a derived realization, a compact immutable identity descriptor may be
created by its actual producer and shared by its facts and later consumers.
It binds the exact Paired-D1 parent, declared producer product/realization and
native support. It contains no numeric payload, policy or snapshot reference,
so a snapshot can retain the descriptor without a payload/snapshot ownership
cycle. This is a proposed bounded engineering representation, not a scientific
identity based on a pointer or a new durable serialization contract.
VAL must not include or reach into D2, RTC or PTC implementation to invent the
descriptor. Producer identity is supplied explicitly; no process-global
counter, ambient generation, registry or per-cell identity-string copy.

A coordinate target references that descriptor and the existing exact native
address. The descriptor's role does not authorize computation of a residual.
Qualified and existing unqualified findings occupy distinct key domains;
there is no implicit promotion, pair-wide inference, broadcast or fallback.
Same-key successor facts retain the established immutable overlay behavior.
Fact state and cause stay opaque under their actual producer contract.

For future integration, computation input facts, resulting producer facts,
published carrier binding and PSD-use decisions are separate lifecycle steps.
The existing SCI-VAL sequence is `F_k -> V_k -> A_k -> F_(k+1)`; new output
facts or use decisions cannot rewrite the prior snapshot/action. D2's existing
publication must continue to receive exactly the snapshot bound when its
payload wrapper was realized. A later snapshot is never substituted silently.
This prerequisite does not choose the residual algorithm or define which
scientific output facts/profile are produced at those steps. The later
producer work order must demonstrate that full acyclic handoff explicitly.

## 5. Smallest proposed source unit

Proposed work order: `TIMESTREAM-SUCCESSOR-VAL-NATIVE-TARGET-001`.
Purpose: add and test VAL-owned typed native-coordinate fact targeting and
immutable realization identity mechanics, without evaluating a policy.
This source unit is proposed for subsequent owner disposition; the present
approval covers recovery and work-order preparation.

Expected source/test paths are:

- `include/citlali/core/pipeline/timestream_val_state.h`;
- `tests/test_timestream_val_state.cpp`;
- `tests/timestream_val_state_header.cpp`.

Existing CMake registration should suffice; change it only if a concrete
test-registration need is established and recorded. Bounded status/ledger/
handoff records accompany implementation. The accepted D2 header, all D2
tests, Paired-D1, identity route, scientific packages, registry, authority
JSON, Python evidence tools, configs and numerical owners remain unchanged.
Do not implement a generic fact framework, full VAL evaluator, named-use mask,
exporter, native producer, new `ValProducer::d2` scientific owner, persistence
or route wiring in this unit.

Acceptance examples for the proposed mechanical unit:

| Case | Required result |
| --- | --- |
| Same native cell, `x` versus `r` | Distinct typed targets; one cannot find or overwrite the other's fact. |
| Original input versus derived residual on equal axes | Distinct subjects; no promotion by shape or row equality. |
| Two derived realizations with equal descriptive identifiers but different immutable instances | Distinct targets under exact scoped binding. |
| Foreign Paired-D1, network, native row, detector or realization parent | Fail closed before a delta is committed. |
| Existing unqualified finding and a qualified finding at the same address/code | Both retain distinct meanings; no automatic coordinate/pair interpretation. |
| Duplicate exact qualified key in one delta | Reject deterministically; differently targeted facts may coexist. |
| Later snapshot with matching generation number on another branch | No exact-binding substitution; earlier snapshot remains unchanged. |
| Missing targeted fact or nonfinite payload elsewhere | No inferred cause, permission, validity or numeric coercion. |
| Reordered proposals and equivalent lookup | Deterministic key ordering and identical stored facts. |
| Descriptor/snapshot lifetime and memory | No ownership cycle, payload copy, dense validation plane or per-cell identity text. |

Focused gates: isolated VAL header, VAL tests, existing D2/Paired-D1/identity
regressions, strict key-order/foreign-parent cases and compact memory evidence.
Broader gates for the eventual changed source: CLI/safety build, runnable
CTest suite, full config preflight, baseline-tool tests and both ledgers.
Local fallback results remain supplemental. An owner-run clean Unity GCC13/
Spack source gate at the final changed source is required before its canonical
admission; job 64018093 cannot cover a future VAL source change. No scientific
reduction is triggered while no numerical operator/route changes.
Fresh independent exact-SHA review must cover all three governance axes.

Start that source unit from the then-verified live canonical SHA. If canonical
has moved beyond this proposal's base, record path overlap, authority and
ancestry reassessment before creating the implementation worktree. No rebase,
cherry-pick, scope expansion or second active spine is silently implied.

## 6. Deferred owner policy decisions and stop rules

No new scientific decision is needed to preserve exact target identity and
opaque producer facts within Section 5. That statement does not approve the
source implementation or an end-to-end D2 consumer.

Before a native PSD consumer is implemented, its scientific owner must supply
or identify a complete immutable profile for the exact intended measurement:

1. actual policy owner, named use and raw/residual `x/r` domain, including any
   retained auxiliary-R limitation;
2. authoritative producer facts/supports and their declared relevance, source
   and line processing controls, missing/conflict behavior, and exceptions;
3. exact projection to the numerical estimator's accepted sample domain,
   including owner-declared scope for an admitted nonfinite value or failure;
4. immutable producer/fact/profile generations and any distinct aggregation
   proposition needed by the later corpus.

These are future adoption decisions, not invented defaults, new VAL-owned
science or reopened WP-7.1 findings. In particular, no PSD envelope, factor,
filter, line-mitigation policy, cleaning operator or source-mask algorithm is
selected here. The existing tool's finite-value intersection cannot settle
item 3. Existing PTC, MAP and NOI policies retain their own exact domains.

Stop and reassess if the proposed mechanical implementation requires a D2
carrier change, a new scientific predicate, different snapshot equality,
cross-stage reach-through, a persistent identity scheme or an active producer.
Raise an owner question only after identifying the concrete missing authority;
do not ask the owner to reapprove an already frozen rule.

## 7. Verification and completion boundary

This prerequisite changes documentation only. Required checks are exact
three-path scope, unchanged base content outside those paths, prior status
preservation, links, normative-source digests, registry search disposition,
whitespace and clean committed state. Scientific-contract layout and the
unmodified SCI-VAL Core verifier were run as document checks. No new
application build, CTest campaign or Unity job is needed for this record, and
no prior result is relabeled as execution at its containing SHA.

Scientific-contract layout passed. The unmodified SCI-VAL verifier exited 1
with `FAIL: MAP reserved profile must remain explicitly unbound`, identically
on this candidate worktree and the clean exact canonical base. Its lines
114--116 retain the r0.3 reserved-profile assumption; the current registry
explicitly registers the later MAP profiles. This is a pre-existing mismatch
between that historical check and the accepted additive registry, not a PASS
or a candidate regression. The verifier and registry are unchanged. All six
frozen Core digests match, and complete-tree preservation independently binds
all scientific authority and executable inputs to the accepted base. This
limited prerequisite does not repair that verifier or certify all of its
remaining checks. Evidence is preserved under
`/private/tmp/citlali-val-d2-prerequisite-evidence-2026-09-06`.

The final candidate SHA/tree, changed-content digests and fresh independent
review bind externally after commit. This record does not approve itself,
move canonical, publish to GitHub or begin Section 5. All pushes remain
owner-run with the absolute-path, explicit-URL and full-refspec convention.
