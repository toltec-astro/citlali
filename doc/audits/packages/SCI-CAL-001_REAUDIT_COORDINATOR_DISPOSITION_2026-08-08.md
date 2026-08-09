# SCI-CAL-001 exact-repair re-audit coordinator disposition

Date: 2026-08-08

Authority: scientific-audit coordinator; documentation-only disposition

Package: `SCI-CAL-001`

## Exact identities

- Repair candidate: `7894346a91fa78ceb2a8b3d625335f466e5e1756`
  (parent `46ad23888a40f5102cdfd50c06e49a549bdf8a20`, tree
  `991f96c64e4d2d973ed5fc02630bfe29149109d9`).
- Exact-repair re-audit: `4140923de4ae33d36224493b5937e291bd552d30`
  (parent `7894346a91fa78ceb2a8b3d625335f466e5e1756`, tree
  `2f051f6831b25e0b297fb61905f40e5e17c6a925`).
- Frozen coordination authority:
  `dd5894679bf12bf4a5fb551e871b3c6010ef9b9b` (tree
  `e87b507a6dc5246da0f65e563d96b94824e61ba1`).
- Frozen atmosphere authority:
  `7156881bd1a47e8cece97b8c541a013c93ac03e1` (tree
  `316c5c5a0188ead742f55e21ae1bd62a89e02677`).
- Immutable report:
  `doc/audits/SCI-CAL-001_EXACT_REPAIR_REAUDIT_2026-08-08.md`,
  SHA-256
  `7a9eeae603871f3e2c157b123c15970dd2b2e472257479d100b02bea43101d34`.
- Immutable ledger proposal:
  `doc/audits/proposals/SCI-CAL-001_REAUDIT_LEDGER_PROPOSAL_2026-08-08.yaml`,
  SHA-256
  `47a63a5c2a2fcc1000547dd5cdc64d24382818666e299b6629e92afff28e9ee2`.
- Source status object:
  `4140923de4ae33d36224493b5937e291bd552d30:doc/REFACTOR_STATUS.md`,
  SHA-256
  `14d3190c001b33f7ee094a269b99edc58287774f3803b0890788ca4badbfaefb`.
  Its digest is recorded; that source status is not overlaid on the current
  coordination line.

The source commits, parents, trees, branches, artifacts, authority objects,
and proposal cross-bindings were independently recomputed before integration.

## Coordinator disposition

The exact-repair re-audit is accepted as a documentation-only audit package.
The candidate is rejected as complete CAL closure. The canonical axes remain:

- contract: `approved`;
- implementation: `nonconformant`;
- validation: `in_progress`;
- production: `fail_closed`;
- verdict: `amend`.

`SCI-CAL-001-F002` alone receives narrow implementation closure. The candidate
removes the finite-positive low-opacity unity plateau and implements the
approved fixed-DJF25 operator's LOS-optical-depth interpolation from the
analytic zero anchor through the first nonzero anchor, with exact nodes,
endpoints, monotonicity, and seam behavior. This closure does not establish
atmospheric truth, model fidelity, observational calibration, or permission to
use the candidate downstream.

`SCI-CAL-001-F003` and `SCI-CAL-001-F004` remain P0. Calibration factors are
not admitted atomically as one complete product; `CAL.VALID` describes the
tau/extinction decision rather than complete calibration validity; and APT
coefficients still bind to TOD columns positionally without admitted row-order
proof or an explicit keyed acquisition join. `F001` and `F005` through `F010`
remain open as recorded. The owner decision associated with `F006` is settled,
but its implementation remains nonconformant.

The fixed operator and its exact generated artifacts are retained as successor
material. This preserves completed work without promoting the rest of the
candidate or changing any scientific, implementation, validation, or
production axis.

## Bounded successor-repair handoff

A future coordinator-authorized successor repair should preserve the fixed
operator unchanged while addressing only the unresolved CAL closure gates:

1. admit or reject the complete calibration product atomically before any
   observation mutation or publication, and represent complete CAL validity
   with typed causes;
2. prove the validated acquisition-row binding or implement a keyed join from
   APT identity to TOD detector identity, including mismatch and permutation
   tests;
3. fail closed for unsupported units and invalid required factors, and make
   approved factor, uncertainty/weight, nuisance, response, and provenance
   semantics reconstructible without overstating unavailable covariance;
4. complete deterministic production-path fixtures, exact-successor evidence,
   required recipient dispositions, and ALIGN-conditioned gates before a fresh
   re-audit.

This record is readiness and scope control only. It does not select a successor
repair base or authorize repair, re-audit, Unity evidence, FLT, another
downstream audit, merge, operator adoption, or any production change. The
exact successor base remains subject to a separate coordinator/owner launch
decision.

## Successor-2 owner consolidation

The project owner's complete successor-2 dispositions are preserved in
`doc/audits/packages/SCI-CAL-001_SUCCESSOR_2_OWNER_DISPOSITIONS_2026-08-09.md`
(SHA-256
`f0e0500e0ba809c1b51a36f69a97a71ab980d66337f62a9ff6985309b43df1d6`).
That successor authority narrows and supersedes this record's generic repair
readiness language without changing the immutable re-audit artifacts or the
canonical axes.

The smallest bounded repair handoff is
`doc/audits/packages/SCI-CAL-001_SUCCESSOR_2_BOUNDED_REPAIR_HANDOFF_2026-08-09.md`
(SHA-256
`05d2db5c4c099943498f5458ccef44f6acc40553e30d382df60e0c59797bffbf`).
It proposes application base
`7894346a91fa78ceb2a8b3d625335f466e5e1756` (parent
`46ad23888a40f5102cdfd50c06e49a549bdf8a20`, tree
`991f96c64e4d2d973ed5fc02630bfe29149109d9`) and branch
`codex/repair-sci-cal-001-successor-2` for separate owner launch approval.

Closure accounting is now: F002 retains narrow structural closure; F006 is
closed only for the approved `mJy/beam` configuration boundary; F003, F004,
F005, F007, F008, and the local implementation portion of F009 define the
bounded repair; and F001/F010 plus Unity, astronomical-standard, and empirical
response-fidelity evidence remain conditioned external dependencies. Local
implementation conformance may therefore precede those external results, but
production precision/accuracy claims remain `fail_closed`.

The repair remains unlaunched. This successor record does not authorize a
branch or worktree, application/config/test edits, evidence execution, Unity,
re-audit, downstream use, merge, push, or production change.

## Stop

Return to the coordinator and scientific owner. Preserve `fail_closed`; do not
launch any action from this disposition.
