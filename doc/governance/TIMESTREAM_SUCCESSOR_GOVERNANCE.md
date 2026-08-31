# Timestream Successor Governance

Status: candidate; not effective until owner-accepted and incorporated into
canonical integration authority

Owner: Citlali project owner

Scope: implementation of the Timestream Successor within the Citlali
application

## Name and authority boundary

**Timestream Successor** is the enduring implementation-program name. The
existing names `WP-7.1 Timestream Successor Program`, `WP7-REPLAY-*`, and
`codex/wp7-*` remain searchable historical, scientific-packet, work-order, or
branch provenance. New durable branches, governance documents, work orders,
and implementation generations MUST NOT use bare `WP-7` or `WP7` as the
program name.

The scientific authority remains precisely the **WP-7.1 Timestream Contract
Baseline**:

- source commit `170ecea9de1ee810da7d7e45a489a4545ccd623d`;
- closure commit `20ba6ae5dcf0b90a24ac3e778a75eff0a1bbe2aa`;
- canonical authority router
  `validation/wp7_timestream_successor_authority.json`; and
- canonical ADRs 0017 through 0023 and their recorded owner dispositions.

The contract name is not erased or generalized. Contract closure does not
establish implementation conformance, executable acceptance, integration,
readiness, release, or production authorization.

## Program shape and WIP budget

The program uses two conceptual queues, not permanent Git branches:

1. one active **integration/spine increment**; and
2. at most one active **scientific-module probe**.

An integration/spine increment changes the accepted typed route, carrier, or
cross-stage seam intended for canonical integration. A module probe evaluates
or repairs one named scientific owner behind an already stated boundary. A
probe becomes active only when implementation begins; conceptual design and
pre-implementation contract authoring do not require a branch.

The budget excludes canonical integration operations, separately governed
Spack/build-adaptation lanes, owner-retained evidence refs, read-only audits,
and scientific-contract authoring that has not entered application
implementation. A second spine or second active module branch requires an
owner-recorded bounded exception or completion/parking of existing work.

Each approved increment owns one bounded branch and clean worktree under the
repository-wide rules. Historical branch existence does not consume WIP and
does not authorize resumed work.

## Enduring architecture

### Orchestration

Session and pipeline orchestration MUST own the direct execution sequence,
lifecycle boundaries, failure propagation, and route selection. It MUST NOT
perform scientific calculations, invent cross-stage facts, or store
stage-specific learned policy in `Engine`.

The first complete route SHOULD remain explicit and readable rather than use a
generic DAG, plugin registry, universal processor, or speculative framework.

### Paired native ingress

The successor starts from one explicit paired x/r product per originating
network. The product and its adapters MUST preserve:

- the exact pairing and producer-interface identity;
- network, detector, parent-readout, and paired-occurrence identity;
- each network's authoritative native occurrence/time axis and physical-run
  partition;
- x-local, r-local, and pair-derived availability, validity, causes, origin,
  support, and lineage;
- coordinate meaning, units/scale, sign, reference, normalization, metric,
  validity domain, and uncertainty state; and
- bounded memory ownership without duplicating heavy identity or provenance
  per detector-occurrence cell.

Network-local storage indices are not persistent identities. Engineering
chunks MUST NOT redefine scientific occurrence, support, timing, or lineage.
A common analysis grid is a separate explicitly requested ALIGN relation; it
MUST NOT replace the native axis merely for rectangular storage or convenience.

### Learn–consider–apply

An operation that learns from data MUST separate:

1. producer observations and diagnostics;
2. a named owner considering those facts under an accepted policy;
3. an immutable plan or decision product; and
4. application of that exact plan with realized evidence.

Learning MUST NOT silently mutate the request, apply an unrecorded threshold,
or collapse into a pass-through shortcut that falsely claims the approved
architecture. Producer causes MUST remain distinct from downstream admission
decisions.

### AST and RTC ownership

AST owns authoritative pointing facts, sample-to-sky association, accepted
sky-motion facts, their validity, and their scientific identity. RTC owns
accepted filter planning, filtering, pre-decimation protection, and
downsampling.

RTC MUST NOT reach into AST internals or `Engine` for sky-motion facts. Any
AST-to-RTC dependency MUST be an explicit typed product or interface with a
named producer, consumer, scope, units/frame, occurrence relation, validity,
causes, and uncertainty. AST MUST NOT select an RTC filter or factor; RTC MUST
NOT manufacture pointing authority.

Detector and pointing transformations that describe the same retained or
resampled occurrence MUST remain synchronized through explicit source/output
relations. A data-only transform or pointing-only transform is not a complete
scientific operation.

### Stage ownership and terminal outcomes

RTC operates in raw detector units. CAL follows RTC and MUST NOT be hidden
inside RTC. PTC and each VAL-owned named-use decision remain explicit.
Scientific-use admission belongs to the named consumer, not to the producer
that records raw facts.

The RTC-only terminal route is legitimate and MUST publish truthful completion,
identity, support, realized-operation, and failure semantics. The ordinary
route MUST eventually reach a real typed MAP-facing boundary; MAP owns its
admission decisions, while the timestream producer owns the facts it emits.

No route may silently fall back between legacy and successor semantics.
Activation, legacy retirement, and production use remain separate owner
decisions after conformance and executable acceptance.

## Increment preflight

Before implementation, the work order MUST state:

- whether it occupies the spine or module-probe slot;
- exact canonical base, branch, and worktree;
- exact scientific requirements, scenarios, ADRs, and owner dispositions;
- the named module, product, interface, and lifecycle owners;
- direct producer/consumer relationships and cross-stage dependencies;
- affected identities, units, frames, shapes, validity, causes, and lineage;
- expected memory and performance significance;
- focused, broad, affected-mode, and independent-review gates;
- actual build environments and whether representative Spack validation is
  required or absent;
- explicit exclusions and review triggers; and
- integration, push, activation, and cleanup authority.

An increment MUST stop when its prerequisite contract is absent, when an
accepted scientific choice is ambiguous, when ownership cannot be maintained
without cross-stage reach-through, or when scope expands beyond the work order.
A prerequisite defect discovered mid-increment may be diagnosed and preserved;
repair requires either a within-scope determination recorded under the same
owner or a separately approved work order.

## Completion and acceptance

Completion MUST use the full report in
`REVIEW_AND_CONFORMANCE.md`. Focused results and broader repository results
MUST be reported separately with their actual build environments. A passing
suite MUST NOT substitute for architectural review.

Every candidate MUST receive independent fresh-context, read-only review at an
exact full SHA. The writer may repair supported findings but MUST NOT approve
the candidate. Integration, push, successor activation, and production
authorization require separate explicit owner dispositions.

## Current historical boundary

The paired-D1 implementation at
`d7d19bc90d7c994fa767ec2a9fd35e4d8599f032` and its closure record at
`2f1d836c1db122d22015853582133abf3611bc30` are candidate evidence under the
historical alias `WP7-REPLAY-002A`. They are not integrated, pushed, or
production-authorized by this document. No D2 seam, RTC/PTC wiring, filter,
factor, downsampling, common-grid projection, AST change, or production route
is implied.
