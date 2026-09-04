# MAP-SPACE-CONTRACT-TO-IMPLEMENTATION-CONFORMANCE-001 Work Order

Status: **candidate source-level conformance study; owner review required**

Date: `2026-09-04`

## Program adherence and prior-work recovery

This is a bounded Tier 2 audit/traceability package, not a new scientific-
contract package and not implementation work.  It begins from the frozen
scientific oracle and stable graph established by
`MAP-SPACE-HORIZONTAL-SCIENTIFIC-CONTRACT-AUDIT-001`, as amended only by the
accepted shared-conventions repair.  The prior-work registry, contract-program
instructions, horizontal audit, repair candidate, and exact owner acceptance
record were reviewed before source inspection.  Recovered work is disposed as
follows:

- **adopt** the exact 17 product IDs, 32 edge IDs, and 16 representative trace
  IDs without renaming or semantic broadening;
- **cite** frozen SCI-MAP, SCI-JINC, SCI-FLT-FIXED,
  SCI-FLT-MATCHED, SCI-NOI, SCI-POINT, and SCI-VAL authority through the
  accepted horizontal-audit manifest and the current repaired shared view;
- **supersede** the horizontal audit's four repair-required shared-source
  findings only to the extent stated by the accepted repair record;
- **defer** every scientific-policy choice that remains owner-owned,
  unregistered, or unavailable; and
- **exclude** active FRUIT implementation/science, the historical ALIGN
  worktree, OOF implementation claims, numerical changes, configuration
  changes, tests, validation changes, Unity, networking, and dependency work.

No implementation-derived material is promoted to scientific authority.

## Exact launch identity

| Field | Exact value |
| --- | --- |
| Study ID | `MAP-SPACE-CONTRACT-TO-IMPLEMENTATION-CONFORMANCE-001` |
| Base commit | `9f42d348298d76c5d5145aaf0c3eace1f3e154c1` |
| Base tree | `e51f22760c64454ce7233c45dd740aa710777bae` |
| Base local integration ref | `refs/heads/codex/refactor-mainline` at the base commit |
| Initial worktree | clean |
| Dedicated task branch | `codex/map-space-contract-to-implementation-conformance-001` |
| Governance tier | Tier 2 |
| Timestream Successor governance | not applicable; this packet makes no WP-7 application change |
| Permitted mutation | this packet directory only |

The exact commit and tree, not the branch label, bind the inspected source.

## Objective

Trace the frozen map-space product and route graph into the implementation
source at the exact base; classify source-level route availability and gaps;
identify dangerous implementation failure modes; and convert the 16 accepted
representative traces into a validation-planning matrix.  The study is meant
to answer what source appears to implement, what remains legacy, missing,
forbidden, contradictory, or indeterminate, and what evidence would be needed
before any runtime-conformance claim.

## Required state vocabulary

Each product and route receives exactly one primary state from this closed
vocabulary:

- `IMPLEMENTED_CONFORMANT_AT_SOURCE_LEVEL`: direct source evidence implements
  the bounded contract property.  This is never a runtime, validation,
  performance, readiness, production, or Unity claim.
- `IMPLEMENTED_LEGACY_SEMANTICS`: an executable analogue exists, but its
  identity, lifecycle, roles, policy binding, or semantics are the predecessor
  behavior rather than the frozen contract.
- `DECLARED_NOT_IMPLEMENTED`: an admitted source explicitly declares the
  capability not implemented; absence is not inferred.
- `UNAVAILABLE_BY_DESIGN`: the frozen graph prohibits or intentionally omits
  the route.  It is not an implementation backlog item.
- `MISSING_AUTHORITY`: implementation cannot conform because an owner choice,
  profile, coefficient family, boundary, or other scientific authority is
  absent or unbound.
- `MISSING_IMPLEMENTATION`: sufficient authority exists for the bounded
  question, but the required source representation or behavior was not found
  in the inspected tree.
- `CONTRADICTORY`: direct source behavior conflicts with a frozen invariant.
- `NOT_APPLICABLE`: the item is outside this work order's admitted domain.
- `INDETERMINATE`: admitted source evidence is insufficient to choose another
  state without inference.

The state is not an ordering.  A route can have a legacy analogue and still
be `CONTRADICTORY`; a conformant source fragment can remain unavailable
because another required edge is `MISSING_AUTHORITY`.

## Evidence grades

| Grade | Meaning |
| --- | --- |
| `A` | Direct implementation dataflow or branch plus exact source citation. |
| `B` | Direct declaration/type/config/lifecycle evidence without a complete executable route. |
| `C` | Focused test or checked-in validation artifact corroborates the source; it does not upgrade source evidence to runtime conformance. |
| `D` | Bounded absence search or negative structural evidence. |
| `E` | Scientific/manager declaration only; no implementation claim. |

Every conformance statement names its grade and source location.  Tests for
legacy contracts are inventories, not evidence that the frozen oracle is
implemented.

## Stop rules

Stop and classify rather than infer when any required coefficient identity,
response/covariance state, profile registration, parent identity, original-
coordinate lineage, or failure state is absent or conflicting.  In
particular:

- never infer a MAP/JINC coefficient from `sens`, scatter, unity, a field
  called `weight`, or numerical agreement;
- never treat a zero as unavailable response, covariance, support, or product;
- never treat processed contribution membership as unique-original exposure;
- never treat a legacy Wiener/convolution path as either frozen filter merely
  because the names resemble each other;
- never infer POINT policy or SCI-VAL decisions from fit success; and
- never enter the active FRUIT branch or historical ALIGN worktree.

## Required deliverables

This directory contains only:

1. `WORK_ORDER.md`;
2. `SOURCE_AUTHORITY_MANIFEST.md`;
3. `PRODUCT_AND_BOUNDARY_IMPLEMENTATION_TRACEABILITY.md`;
4. `ROUTE_AVAILABILITY_CLASSIFICATION.md`;
5. `FAILURE_MODE_SOURCE_AUDIT.md`;
6. `REPRESENTATIVE_TRACE_VALIDATION_PLAN.md`;
7. `PRIORITIZED_REPAIR_BACKLOG.md`;
8. `FRUIT_ATTACHMENT_ENVELOPE.md`;
9. `OOF_ATTACHMENT_ENVELOPE.md`;
10. `OWNER_DECISION_LEDGER.md`;
11. `FINAL_REPORT.md`; and
12. `verify_packet.py`.

The verifier checks base identity, admitted-source digests, exact stable-ID
closure, state vocabulary, artifact set, and packet-only repository mutation.

## Closure boundary

Completion means a verified and committed documentation candidate on the
dedicated task branch.  It does not authorize repairs, source/config/test/
validation edits, scientific-contract changes, registry changes, FRUIT or OOF
attachment, Unity activity, or integration-ref movement.  The candidate stops
for scientific-owner review.
