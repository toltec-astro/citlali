# Owner Decision Ledger

Status: **candidate decision packet; all new decisions remain open**

This ledger separates inherited accepted authority from decisions requested by
the conformance study.  Listing a requested decision does not approve it.

## Inherited decisions applied without reopening

| Authority | Applied disposition |
| --- | --- |
| MAP-space horizontal audit and MSP-OD-001 | Frozen package meanings control; exact product/edge/trace IDs are preserved. |
| Shared-conventions repair acceptance | MAP signal is nonpolarimetric total-intensity-equivalent; coadd uses observation-row `u_op=1`; exposure is unique-original AST-coordinate accounting; JINC has exactly five numerical roles. |
| Frozen package records | Package science is an oracle and was not edited or re-derived. |
| SCI-VAL Registry | A reserved/template/unregistered name is unavailable and cannot be inferred from a predecessor or numeric match. |
| Current program sequencing | FRUIT remains independent; OOF is envelope-only; Unity and application work are outside this study. |

## Scientific-owner decisions and inherited sequencing

CTI-OD-001--CTI-OD-006 remain open.  CTI-OD-007 is an inherited/closed
program-sequencing record, not a new scientific-owner question.  Its authority
is the accepted `2026-09-04` section of
[`DOWNSTREAM_CONTRACT_ROADMAP_2026-08-26.md`](../../DOWNSTREAM_CONTRACT_ROADMAP_2026-08-26.md)
and the original study's envelope-only work order; the
`2026-09-04` manager handoff at
`ae953ed4d87d1f693d2bbf42aebbc25ef730c771` directs this bookkeeping
correction.  It neither pauses independent FRUIT work nor launches OOF or a
FRUIT attachment review.

<!-- BEGIN-OWNER-DECISIONS -->
| Decision ID | Requested decision | Evidence requiring the decision | Options bounded by existing authority | Current state |
| --- | --- | --- | --- | --- |
| CTI-OD-001 | Accept, amend, or reject the product/route source classifications and zero-complete-route conclusion | Traceability, route matrix, and CTI-FM-001--CTI-FM-016 | Accept as bounded source study; return named rows for evidence correction; reject with exact contrary source/authority | `OPEN` |
| CTI-OD-002 | Select the exact PTC-owned coefficient families and QC/profile bindings for MAP, JINC, and NOI design balance | MSP-E001, MSP-E005, MSP-E017; CTI-FM-006 | Publish distinct immutable owner-bound choices; explicitly declare a route unavailable; do not authorize inference from legacy `weights` | `OPEN` |
| CTI-OD-003 | Approve or reorder the P0 repair sequence | CTI-FM-001--CTI-FM-005 | Coefficient nonmutation, original exposure, equal-observation coadd, five-role JINC as separate units; any reordering must preserve prerequisites | `OPEN` |
| CTI-OD-004 | Decide the compatibility lifetime and naming of predecessor MAP/JINC/filter/NOI/POINT outputs | Multiple `IMPLEMENTED_LEGACY_SEMANTICS` rows | Retain under explicit legacy identities; add one-way adapters after frozen product implementation; or retire under a separately reviewed migration | `OPEN` |
| CTI-OD-005 | Supply or explicitly defer the POINT parent-compatibility, formal-error, named-use, and SCI-VAL profiles | MSP-E023--MSP-E029; CTI-FM-010/CTI-FM-015 | Register complete immutable profiles; declare selected roles unavailable; no fit-success or namespace-template fallback | `OPEN` |
| CTI-OD-006 | Authorize one next bounded implementation-mapping unit, if any | Prioritized backlog | Choose exactly one P0/P1 unit with source/config/test/validation scope and gates; this study itself grants no work order | `OPEN` |
| CTI-OD-007 | Inherited program sequencing: preserve FRUIT independently and OOF as envelope-only for this study | Accepted downstream roadmap, original work order, and dated manager handoff cited above | Applied without reopening; any later attachment or OOF work requires its own owner-authorized scope | `INHERITED_CLOSED` |
<!-- END-OWNER-DECISIONS -->

## Manager-only follow-up

If the owner accepts this study, a later manager record may index its exact
commit/tree and the owner's disposition.  This candidate does not update
`doc/REFACTOR_STATUS.md`, the integration ledger, any Registry, or any frozen
record, and it does not move an integration ref.
