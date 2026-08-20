# SCI-PTC — Correlated-Mode Cleaning And Detector Coefficients

Status: Stage B v0.1/r0.3 bounded freeze-candidate revision complete and ready
for scientific-owner review; `PTC-OWNER-Q001` and `PTC-OWNER-Q002` resolved;
scientific authority not frozen

Proposed scientific contract version: `v0.1`

## Program Adherence And Prior-Work Recovery

This package is governed by the
[Citlali Scientific Contract Library Program](../../../README.md). Work began
with [`PRIOR_WORK.md`](PRIOR_WORK.md), not a new derivation. The recovered
implementation-independent PTC core and six approved owner decisions are
reused through a binding supersession cover. Historical implementation,
audit, repair, test, validation, Unity, and production material remains in
[`INTERNAL_DOSSIER.md`](INTERNAL_DOSSIER.md) and is excluded from scientific
authorship.

The proposed v0.1 progression is:

`SCI-RTC conditioned x -> SCI-CAL calibrated detector signal -> SCI-PTC
correlated-mode cleaning and detector coefficients -> optional SCI-MAP`.

SCI-PTC does not repair or complete an unavailable upstream RTC or CAL state.

## Package Contents

- [`PRIOR_WORK.md`](PRIOR_WORK.md): exact recovery snapshots,
  classifications, conflicts, and reuse disposition
- [`INTERNAL_DOSSIER.md`](INTERNAL_DOSSIER.md): implementation-informed scope
  evidence and quarantined audit/validation history
- [`SCOPE_BRIEF.md`](SCOPE_BRIEF.md): proposed sanitized scientific boundary
  for owner approval
- [`DECISION_LOG.md`](DECISION_LOG.md): proposed package-scope decisions and
  preserved historical owner decisions
- [`AUTHOR_SUPERSESSION_COVER.md`](AUTHOR_SUPERSESSION_COVER.md): binding
  corrections and limitations on the reusable independent core
- [`AUTHOR_CONVENTIONS_AND_OWNERSHIP.md`](AUTHOR_CONVENTIONS_AND_OWNERSHIP.md):
  sanitized units, identity, lifecycle, and adjacent-package responsibilities
- [`AUTHOR_PACKET_MANIFEST.md`](AUTHOR_PACKET_MANIFEST.md): exact proposed
  implementation-blind author packet and firewall
- [`AUTHOR_METHOD_REFERENCE_BOUNDARY.md`](AUTHOR_METHOD_REFERENCE_BOUNDARY.md):
  bounded verified context for six primary method references
- [`SCOPE_REVIEW_R0.1.md`](SCOPE_REVIEW_R0.1.md): complete disposition of the
  two `2026-08-19` scope reviews
- [`SCOPE_REVIEW_R0.2.md`](SCOPE_REVIEW_R0.2.md): bounded final-review
  amendments, Q001 resolution, and packet-preflight record
- [`SCIENTIFIC_OWNER_REVIEW_R0.1.md`](SCIENTIFIC_OWNER_REVIEW_R0.1.md): exact
  r0.1 scientific-owner review record, targeted r0.2 dispositions, and the
  owner-approved frozen-subspace projection decision
- [`SCIENTIFIC_OWNER_REVIEW_R0.2.md`](SCIENTIFIC_OWNER_REVIEW_R0.2.md): exact
  r0.2 review hash, bounded r0.3 dispositions, and the recorded high-effort
  scope decision
- [`CROSS_PACKAGE_FOLLOWUP.md`](CROSS_PACKAGE_FOLLOWUP.md): RTC, CAL, raw
  Beammap, MAP, VAL, NOI, and BEAM routing raised by the revision
- [`CROSSWALK.md`](CROSSWALK.md): exact 138-row mapping for 89 requirements and
  49 falsifiable predictions
- [`AUTHOR_DRAFT_DECISIONS.md`](AUTHOR_DRAFT_DECISIONS.md): 24 author choices,
  the decided Q001 and Q002 dispositions, and 12 still-open,
  known-but-not-supplied, or deferred owner entries
- [`SCIENTIFIC_OWNER_DECISION_LEDGER.md`](SCIENTIFIC_OWNER_DECISION_LEDGER.md):
  manager-facing decision summary that refers to the detailed author ledger
  rather than duplicating it
- `src/common/`: the six-file shared canonical notation, definition, equation,
  assumption, requirement, and edge-prediction authority
- `src/scientific-rationale.tex`: the standalone science-team rationale with
  compact traceability and no duplicated normative register
- `src/engineering-conformance.tex`: the complete formal view, importing the
  six shared normative modules exactly once
- `src/generate_crosswalk.py`: deterministic generator and checker for the
  exact requirement/prediction crosswalk
- `src/verify_contract.py`: repeatable approved-packet, identifier, crosswalk,
  audience-view, and PDF coverage checks
- `pdf/`: the canonical 11-page scientific rationale and 20-page engineering
  conformance draft PDFs

## Protected Boundary

SCI-PTC begins with one admitted, calibrated detector timestream from SCI-CAL,
bound to the complete RTC parent, top-of-atmosphere point-source-equivalent mJy
per fixed nominal beam meaning, detector and sample identity, validity/influence state,
conditional uncertainty availability, and upstream response status. It owns
the selected correlated-mode fit, removed subspace, subtraction,
centering/scaling/null-space state, stage-specific support, bounded detector
refinement, within-array grouping, typed coefficient families, sample response,
and the transformed detector-timestream interface. An optional separately
conditioned `r` parent supports diagnostic-only, inert/advisory PCA. It may not
alter calibrated-`x` membership, subtraction, output, or coefficients in base
v0.1; stronger use requires successor owner authority.

It does not own RTC temporal filtering or replacement, CAL factor or
atmosphere science, raw-`Delta f/f` Beammap processing, AST coordinates, VAL
eligibility policy, MAP estimation, NOI empirical uncertainty, FRUIT
recurrence, or FLT map filtering.

## Stage B Result And Next Gate

The owner-approved packet and the r0.1/r0.2 reviews have been consolidated into
one shared normative core containing 41 definitions, 25 numbered equations,
28 assumptions, 89 sequential requirements, and 49 sequential falsifiable
predictions. Every r0.2 normative ID is preserved; r0.3 appends only the
fit-excluded application-availability definition, assumption, requirement,
and prediction plus the complete-upstream-response definition. The engineering
view imports that authority exactly once; the standalone rationale explains
the science without reproducing the full register. Packet-hash, identifier,
crosswalk, audience-separation, PDF coverage, and content checks pass at
document revision `r0.3`.

The owner-approved base-v0.1 PCA/SVD application rule is now explicit:
application projects any admitted input or fixed-state companion through the
frozen realized subspace and metric with linear coefficient recomputation.
The exact temporal-left, detector-right, two-sided, detector/time-specific, or
general vectorized acting space is realized-family state. Frozen numerical
component subtraction is a separately named affine family with identity
derivative, not the base PCA/SVD response rule.

The bounded r0.3 pass also separates the latent correlated component from its
fitted estimate, names the complete admitted upstream response ending on the
CAL grid, defines when a fit-excluded occurrence is application-available,
adds the scientist-facing validation-study matrix and estimator orientation,
explains surrogate purpose, and aligns the lifecycle wording with RTC. The
companion specification mechanically satisfies the exact local/chain/procedure
response, state-comparison, lifecycle, source/surrogate, bias, and identifier
obligations named by the r0.2 review.

The next gate is scientific-owner review of the r0.3 rationale and an explicit
rationale-freeze disposition, together with review of the 12
open/known-but-not-supplied/deferred entries in
[`AUTHOR_DRAFT_DECISIONS.md`](AUTHOR_DRAFT_DECISIONS.md). None blocks the
structural contract; each blocks only its named automatic policy, numerical
product, response, coefficient use, or evidentiary claim. `PTC-OWNER-Q001`
remains decided: conditioned `r` is diagnostic-only and inert/advisory in base
v0.1. `PTC-OWNER-Q002` is also decided: base PCA/SVD application is frozen-
subspace projection with the exact acting space declared by the realized
family. Final pagination and cosmetic layout polish are intentionally deferred
until the final editorial revision.

After the rationale disposition, the next scientific task is the cross-package
RTC--CAL--PTC response and lifecycle coherence profile, not another broad PTC
rewrite.

No implementation conformity, validation, achieved performance,
production-readiness, or scientific-freeze claim is made by this Stage B
draft.
