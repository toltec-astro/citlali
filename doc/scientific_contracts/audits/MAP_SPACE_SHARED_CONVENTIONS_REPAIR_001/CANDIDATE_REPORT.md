# MAP-SPACE-SHARED-CONVENTIONS-REPAIR-001 Candidate Report

Status: repair candidate complete; independent exact-SHA review required

Date: 2026-09-04

Recommended candidate disposition: `READY FOR INDEPENDENT REVIEW`

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
`7b5df9a9f24d48e95510080aeee2242b924596c44b4fc9522a7e3713fc8635ca`,
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

The historical SCI-MAP-001 implementation-evidence paragraphs are now
explicitly separated from the SCI-JINC section.  Their bytes remain historical
evidence and make no present implementation-conformance claim.

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

## Checks performed

- `verify_repair.py`: PASS against the exact base and preserved audit commit;
  seven preserved audit artifacts remain byte-identical; all four repaired
  meanings are present; the superseded clauses are absent; frozen package and
  application paths are unchanged.
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
