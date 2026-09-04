# MAP-SPACE-SHARED-CONVENTIONS-REPAIR-001 Candidate Report

Status: revised repair candidate; fresh independent exact-SHA review required

Date: 2026-09-04

Recommended candidate disposition: `READY FOR FRESH INDEPENDENT REVIEW`

## Exact inputs

- Canonical repair base: commit
  `5f0fc20042b88fb6cd883c92d1b59b7f22832901`, tree
  `97a4d908061e51418f93afc1d97d27433af441b8`.
- Preserved horizontal-audit commit:
  `34a29a1eac8a2c41a97263bbd775bd36c3d06398`.
- Horizontal-audit source manifest SHA-256:
  `d21d1446ebcdda8597cf08a4568be91906e3cc22e97f9e7f5544a5fa590b2cd5`.
- Owner disposition: `MSP-OD-001`, recorded in the preserved audit packet.
- Work-order SHA-256:
  `dc4d32493554fca78c99af3984bfdc37d05be79e0e9759a975be32d8003b0148`.

## Repair implemented

`doc/SCIENTIFIC_CONVENTIONS.md`, candidate SHA-256
`c29ad515eb84aa2ee2d13b04245ebacf99ba8a790ea2ccb86ff568bcb84284d7`,
now records the four owner-selected meanings:

1. Ordinary SCI-MAP is the frozen nonpolarimetric
   total-intensity-equivalent quantity; legacy component label `I`, a
   STOKES-axis slot, and `mJy/beam` do not establish formal Stokes I.
2. Ordinary SCI-MAP coaddition uses exactly dimensionless `u_op = 1` for each
   admitted observation-output row.  It does not flatten other information,
   acquire precision or empirical-weight meaning, or authorize a JINC coadd.
3. MAP exposure uses the exact
   `upstream_eligible_original_footprint_exposure` and
   `retained_original_footprint_exposure` products, with unique-original
   placement at each original's own AST ALIGN-grid coordinate in the target
   WCS.  Descendant signal membership, filtering, support, influence, and
   statistical weight do not define it.
4. Base SCI-JINC v0.1 has exactly five numerical map-plane roles.  Conditional
   mathematics and the compact generative record do not create response,
   variance, weight, covariance, exposure, standalone-support, diagnostic,
   generalized-provenance, or coadd products.

The shared document cites the exact MAP r0.7.1 and JINC r0.3 package
authorities.  Their precedence is product-scoped and does not widen either
contract.

## Bounded reassessment

The initial audit named the principal contradictory spans.  Whole-document
cross-reference review found two additional generic restatements of the same
resolved MAP quantity identity: the historical WCS/component paragraph and
the deferred-polarimetry paragraph.  Leaving those restatements untouched
would have preserved the formal-Stokes ambiguity.  They were changed only to
repeat `MSP-OD-001`: component index `0` and legacy label `I` do not establish
formal Stokes I.  This required no additional owner choice and introduced no
new product or method.

The historical SCI-MAP-001/002 implementation-evidence paragraphs are now
explicitly separated from the SCI-JINC section. Their meaning is retained as
historical evidence and makes no present implementation-conformance claim.

## First exact-SHA review and bounded repair

An independent xhigh review of commit
`50fd98d7f5b57738ef265080bff223feee6c4e92`, tree
`201e61577c95ff44586c6daf22bb5e0f74cdb84e`, returned `repair required`.
The four owner-selected replacement meanings passed. The review instead found
collateral loss or misclassification around those replacements. The owner
authorized the following bounded repair on 2026-09-04:

1. Restore the frozen MAP order-statistic selector, separate
   numerical-normalization and science-policy support predicates, exact
   `coverage_cut` admission state, one-way lifecycle, raw-parent validity, and
   provenance, explicitly distinguished from observation-row `u_op` and JINC.
2. Correct the JINC geometry distinction between `(r_max)_a` and cache
   half-width `h_a`, and retain certified finite-precision numerical support as
   unavailable under frozen r0.3.
3. Move the historical two-level binary64 policy and the integrated
   SCI-MAP-001/SCI-NOI-002 writer, lifecycle, publication, and coefficient-stage
   material under the explicit historical implementation-evidence boundary.
4. Restore the literal `maps_to_stokes`, `map_stokes`, and `STOKES` transport
   identifiers while stating that none establishes formal Stokes identity.
5. Make the verifier branch-independent, require an explicitly supplied exact
   candidate SHA and tree plus a clean checkout, remove the erroneous
   `coverage_cut / 10` prohibition, and assert preservation of the repaired
   support, lifecycle, JINC, and transport clauses.

No owner-selected scientific meaning, product role, estimator, exposure,
coaddition rule, unavailable route, or frozen package byte was changed by this
follow-up repair.

## Finding disposition proposed for independent review

| Finding | Candidate repository-documentation state | Scientific/package state |
| --- | --- | --- |
| `MSP-F-001` | repaired in shared conventions | owner-resolved under `MSP-OD-001` |
| `MSP-F-002` | repaired in shared conventions | owner-resolved under `MSP-OD-001` |
| `MSP-F-003` | repaired in shared conventions | owner-resolved under `MSP-OD-001` |
| `MSP-F-004` | repaired in shared conventions | owner-resolved under `MSP-OD-001` |
| `MSP-F-005` | open; outside this work order | unchanged |
| `MSP-F-006` | open; outside this work order | unchanged |

The four MAJOR findings are not declared closed by the author.  Closure
requires independent review of the exact repair commit and later owner
integration disposition.  All `MSP-U-*` unavailable, conditional,
not-authorized, and not-applicable route states remain unchanged.

## Checks and exact-SHA gate

- `verify_repair.py` now requires `--expected-candidate <full-sha>` and
  `--expected-tree <full-tree-sha>`. It rejects a mismatched or dirty checkout,
  is independent of branch attachment, preserves the seven audit hashes, and
  checks the four owner-selected meanings plus the bounded review repairs.
  Post-commit execution is reported by exact candidate identity in the task
  completion and fresh independent review records.
- `doc/scientific_contracts/verify_layout.py`: PASS.
- Relative Markdown links in `doc/SCIENTIFIC_CONVENTIONS.md`: 12 checked, 0
  missing.
- `git diff --check 34a29a1eac8a2c41a97263bbd775bd36c3d06398..candidate`:
  PASS for the repair range. The earlier preserved audit work order uses a
  legitimate Markdown setext `=======` heading, which Git's whole-base
  conflict-marker heuristic reports if the audit-preservation commit is
  included; its byte identity is intentionally unchanged.
- Changed-path inspection: only the preserved audit packet,
  `doc/SCIENTIFIC_CONVENTIONS.md`, and this repair directory differ from the
  exact base.

These are documentation identity, scope, reference, and internal-consistency
checks.  They are not application tests or scientific validation.

## Three-axis author disposition

- Scientific and behavioral conformance: candidate conforms to the exact
  owner disposition and frozen package meanings; independent review pending.
- Architectural conformity and ownership: shared summary defers to the exact
  product-specific authorities and changes no producer, consumer, lifecycle,
  or application boundary; independent review pending.
- Repository, branch, and evidence hygiene: bounded clean worktree and branch;
  preserved audit separated from repair; no canonical or remote ref moved;
  independent review and owner integration decision pending.

## Claim and nonmutation boundary

No frozen package, application source, validation product, algorithm, default,
numerical route, FRUIT artifact, ALIGN artifact, canonical ref, or remote ref
was changed.  No implementation, validation, performance, readiness,
production, activation, deployment, or Unity claim is made.  No dependency was
installed and no push, merge, rebase, cleanup, or deletion was performed.
