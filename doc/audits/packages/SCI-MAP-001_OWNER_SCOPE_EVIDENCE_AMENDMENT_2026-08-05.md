# SCI-MAP-001 project-owner scope and evidence amendment — 2026-08-05

## Authority and preservation

This is an additive, immutable project-owner amendment conveyed through the
post-re-audit coordinator follow-up. It changes the governing scope for a
later repair, re-audit, and coordinator disposition. It does not edit,
withdraw, or soften the independent re-auditor's observations or the decision
that followed the criteria registered at the time.

The historical re-audit package remains intact at
`851035e67f63bdb2bacc122b17566877a9e6db97`:

- `SCI-MAP-001_INDEPENDENT_REAUDIT_2026-08-05.md`, SHA-256
  `77cf9c5dd4ccb3d382a5a365cc6693cdcc0301cc5a36bcb0b2a6c616c36d7f00`;
- `SCI-MAP-001_REAUDIT_DECISION_2026-08-05.md`, SHA-256
  `f4841e89e380e75b3fcc5efb4f65ba0ffbdcaa6f10df3affad5f7bc4c9492ae8`;
  and
- `SCI-MAP-001_REAUDIT_LEDGER_UPDATE_PROPOSAL_2026-08-05.yaml`, SHA-256
  `642d603440a01265cbc0563a969fb72142e20e6df74fc2a96d579453c63dd88d`.

The report remains correct historical evidence under its then-registered
`1e-12 degree` WCS bound, exact FITS-threshold-card rule, and complete external
evidence expectation. This amendment supersedes those criteria only as stated
below for prospective disposition.

## Issue and context

The independent re-audit found that the exact application candidate
`ed28dafb37f9113c0d3c95297148157129a90886` had strong local and seven-case
evidence but remained nonconformant under the registered criteria. The owner
has now decided which discrepancies are scientifically material application
defects, which are bounded persistence defects, and which missing campaign
records remain limitations rather than failed scientific gates.

The four operative clauses are recorded as three decisions: product-
persistence authority and tolerances, S-X observation-realization persistence,
and bounded acceptance of the existing seven-case corpus.

## Approved decision 1: product-persistence authority and tolerances

### WCS persistence

The maximum allowed sky-coordinate separation between the lossless binary64
typed/sidecar WCS authority and the physical FITS WCS is **0.1 arcsec**. This
replaces the prior `1e-12 degree` typed-to-FITS bound.

The following remain exact requirements and are not relaxed:

- axis sign, handedness, and orientation;
- the centered integer observation/coadd placement and its exact shape and
  reference-pixel relations;
- the prohibition on fractional shifts, reprojection, interpolation, or
  implicit recentering; and
- binary64 typed/sidecar identity as the lossless admission and provenance
  authority.

A later repair must add a production-writer-path regression test that computes
the maximum sky separation and enforces `<= 0.1 arcsec` while independently
checking the exact sign/orientation and integer-centering requirements. This
decision does not itself require a WCS writer change if the production path
passes the amended contract.

For context only, the re-audit's representative maximum separation,
`1.8081951134495923e-5 degree`, is `0.06509502408418533 arcsec` and therefore
falls inside the amended bound. That observation does not replace the required
production-path regression test.

### Threshold FITS cards

The binary64 sidecar decimal/hex value remains the exact realized-threshold
authority. FITS threshold cards are non-authoritative convenience metadata.
Each card must:

- be finite;
- identify the correct threshold and carry the correct unit;
- keep the policy value and its alias mutually consistent; and
- agree with the sidecar authority at `rtol=1e-12`.

The prior exact binary64 FITS-card equality requirement is withdrawn. For
context only, the report's example card difference has relative magnitude
`2.745167819465854e-15`, below the amended relative tolerance. The later test
must cover the production card-writing path and every required card relation;
the example is not a blanket pass.

No mapmaking threshold selection, support predicate, order statistic,
normalization, or coadd arithmetic change is authorized by this metadata
decision.

### Rationale

The binary64 typed/sidecar record is already the lossless scientific identity
and provenance authority. The FITS WCS remains usable persisted coordinate
metadata under a bounded angular error, while exact orientation and centered
registration prevent a tolerance from hiding a sign or placement error.
Similarly, a rounded convenience card need not be a second binary64 authority
when the exact threshold and its derivation are preserved losslessly. The
finite, identity, unit, alias, and relative-agreement checks still detect
mislabeling, non-finite state, inconsistent cards, or material drift without
forcing a mapmaking arithmetic change.

## Approved decision 2: S-X observation-realization persistence

Absent same-case S-X observation-noise FITS files are a bounded required-output
persistence defect when observation products are enabled together with
coaddition. They are not evidence that the in-memory observation or coadd
operator failed.

A later repair must serialize the observation realization products in that
mode and add exact local tests for required inventory, observation ownership,
shape/component identity, support/validity, provenance/cardinality, failure
propagation, and use of the same admitted observation operator before coadd.
The existing S-E/S-X product identity remains supporting evidence only; it is
not relabeled as same-case direct evidence.

Unity must not be repeated solely to obtain those files. The exact later
repair SHA is established through the local output-contract and operator tests
specified below. Any future external run requires an independent scientific
reason beyond this bounded serialization defect.

### Rationale

The returned cardinality and sibling products localize the gap to required
output persistence, while the candidate source and local fixtures already
exercise the shared in-memory signal/kernel/realization operator. Repairing
and fail-closed testing the writer boundary directly addresses the defect more
precisely than repeating a multi-case reduction whose missing file inventory
is already known.

## Approved decision 3: bounded seven-case evidence acceptance

The existing seven-case corpus is accepted as bounded external evidence for:

- exact-`ed28dafb` execution and successful completion;
- returned product identity and inventory, subject to the named S-X
  observation-realization persistence defect;
- observation-to-coadd behavior visible in the persisted products; and
- sequential/OpenMP product agreement within the registered regression
  bounds.

The absent independent raw/sample ledgers, wrapper/Slurm records, environment
and retrieval chain, and per-scan pre-normalization planes/traces remain
explicit limitations. They are not failed scientific gates, evidence of a
numerical failure, or instructions to rerun the campaign. Local contract tests
remain authoritative for internal primitive, failure, atomicity, and
determinism behavior.

The full external protocol is prospective: use it only when a future change
has a scientifically necessary external-evidence question that local contract
tests and the accepted corpus cannot answer.

### Rationale

The corpus directly supports the bounded product-level claims for which it
contains immutable candidate binaries, accepted logs, products, sidecars, and
SEQ/OMP pairs. Missing operational-chain and pre-normalization records limit
stronger reconstruction or resource/provenance claims but do not negate the
evidence actually present. Keeping those distinctions explicit avoids both an
unsupported pass and an unnecessary rerun.

## Effect on findings

| Finding | Prospective effect of this amendment |
| --- | --- |
| F004 | **Remains open and is narrowed for the repair.** The absent S-X observation-realization files are a bounded output-persistence defect. Closure requires the local same-operator serialization and failure tests; S-E remains supporting only. No covariance or precision claim is added. |
| F007 | **Remains open pending the production-path tests.** The former `1e-12 degree` WCS failure and exact threshold-card mismatch are no longer failures under the amended contract. Closure requires the `0.1 arcsec` WCS test plus exact orientation/centering/sidecar checks and the finite/unit/identity/alias/`rtol=1e-12` card tests. |
| F010 | **Remains `addressed_pending_reaudit`.** Exact sidecar threshold authority and mapmaking arithmetic are unchanged; exact card equality is no longer a blocker. Aggregate floating/count overflow still conditions the exact F010 claim, and the amended card contract needs local coverage. |
| F011 | **Remains open.** Its required local surface now includes the amended production WCS and threshold-card contracts, coadd-enabled observation-realization serialization, aggregate overflow and finite index-range failures, concurrent realization merge, realization gamma bounds, and complete unchanged-WCS atomicity. Missing external operational records are not substituted for these tests. |
| F012 | **Resolved in bounded scope by owner decision.** The corpus is sufficient for the named external product/execution/SEQ-OMP claims. The coordinator may close F012 while retaining every missing lane as an explicit limitation. No Unity rerun is required, and F012 closure proves no unobserved internal behavior. |

The scoped closure proposals for F001, F002, F003, F006, F008, and F009 are
not reopened by this amendment.

## What remains genuinely open

- F005 remains an application defect: finite floating and integer contributions
  can overflow during aggregate merge, and a finite projected coordinate can
  reach integer rounding outside the representable index domain. Both paths
  require fail-closed rejection before live map or product state mutates.
- F004, F007, F010, and F011 remain open or pending exactly as bounded above
  until an exact-repair-SHA local re-audit accepts the implementation and
  tests.
- No precision, inverse-variance, covariance, uncertainty, significance, GLS,
  or NOI authorization follows from this amendment.
- F013 remains open and conditioned on SCI-ALIGN-001, SCI-CAL-001,
  SCI-AST-001, SCI-PTC-001, and SCI-VAL-001. MAP evidence closes none of those
  dependencies.
- Existing production status remains `existing_use_only`; this amendment does
  not authorize integration or production expansion.

## Exact bounded repair and re-audit route

1. A later separate repair starts from the commit containing this amendment,
   or a coordinator-integrated descendant whose application tree is still
   exactly `ed28dafb37f9113c0d3c95297148157129a90886`, and records its new
   exact SHA. It changes only the F005 fail-closed aggregate/index boundary and
   the required coadd-enabled observation-realization output path. Normal
   finite-domain accumulation order, threshold arithmetic, coadd arithmetic,
   and WCS policy remain unchanged.
2. Add focused production-path tests for:
   - maximum typed/sidecar-to-FITS sky separation `<= 0.1 arcsec`, exact axis
     sign/orientation, and exact centered integer observation/coadd relations;
   - finite, identified, unit-bearing threshold cards, exact policy/alias
     consistency, and sidecar agreement at `rtol=1e-12`;
   - required observation-realization serialization with coaddition enabled,
     including same-operator support/validity, identity, cardinality,
     provenance, and required-write failure propagation;
   - floating and integer aggregate overflow plus finite out-of-range projected
     coordinates, all rejected before mutation; and
   - concurrent realization commits, realization gamma bounds, and complete
     coadd/WCS atomicity.
3. Run the exact-repair-SHA focused truth and TSan suites, the full enabled
   CTest set, baseline-tool tests, config preflight, and product/provenance
   gates with no required-data skip or unexpected serious record.
4. Perform a fresh bounded independent re-audit of F004, F005, F007, F010, and
   F011 against this amendment. Record F012 as accepted bounded external
   evidence with the named limitations; do not request Unity solely for S-X
   observation files or the unavailable historical operational records.
5. Return the repair commit, tests, re-audit disposition, unchanged F013
   dependencies, and this amendment to the coordinator. Only the coordinator
   may update the canonical ledger or coordination line or authorize
   integration/production changes.

This record launches no repair or evidence campaign and modifies no
application source, original audit artifact, canonical ledger, coordination
snapshot, or external corpus.
