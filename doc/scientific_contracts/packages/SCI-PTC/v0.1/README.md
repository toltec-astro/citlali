# SCI-PTC — Correlated-Mode Cleaning And Detector Coefficients

Status: Scientific authority frozen; implementation conformity not yet assessed
under this contract.

Scientific contract version: `v0.1`

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
- [`CROSSWALK.md`](CROSSWALK.md): exact 139-row mapping for 89 requirements and
  50 falsifiable predictions
- [`AUTHOR_DRAFT_DECISIONS.md`](AUTHOR_DRAFT_DECISIONS.md): 27 author choices,
  the decided Q001 and Q002 dispositions, and 12 still-open,
  known-but-not-supplied, or deferred owner entries
- [`SCIENTIFIC_OWNER_DECISION_LEDGER.md`](SCIENTIFIC_OWNER_DECISION_LEDGER.md):
  manager-facing decision summary that refers to the detailed author ledger
  rather than duplicating it
- [`SCIENTIFIC_OWNER_FREEZE_R0.4.md`](SCIENTIFIC_OWNER_FREEZE_R0.4.md): exact
  owner freeze, retained open states, claim boundary, and change-control rule
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
- `pdf/`: the canonical frozen 11-page scientific rationale and 22-page
  engineering conformance PDFs

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
atmosphere science, raw-`Delta f/f` Beammap processing, AST coordinates, shared
VAL interchange/evaluation machinery, downstream named-use admission policy,
MAP estimation, NOI empirical uncertainty, FRUIT
recurrence, or FLT map filtering.

## Frozen Authority And Next Gate

The owner-approved packet and the r0.1/r0.2 reviews have been consolidated into
one shared normative core containing 41 definitions, 25 numbered equations,
29 assumptions, 89 sequential requirements, and 50 sequential falsifiable
predictions. Every r0.2 normative ID is preserved; r0.3 appended only the
fit-excluded application-availability definition, assumption, requirement,
and prediction plus the complete-upstream-response definition. The engineering
view imports that authority exactly once; the standalone rationale explains
the science without reproducing the full register. Packet-hash, identifier,
crosswalk, audience-separation, PDF coverage, and content checks pass at
document revision `r0.4`.

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

The bounded r0.4 pass makes producer causes cumulative without erasure and
assigns admission to the owner of each exact named use. PTC owns its local
support composites; MAP and other downstream owners own their admission
rules; VAL owns only shared types, knowledge/cause-preservation semantics,
provenance, evaluation machinery, and vocabulary. PTC-local eligibility is
the base predicate conjoined with every applicable permission predicate; one
exclusion cannot be rescued, and an unknown required predicate produces
`decision_unavailable`. The same pass resolves centering as nonrestoring:
detector output is `P(x-lambda)`, fixed-state response freezes `lambda`, and a
full-procedure response re-estimates, records, and again discards it. All 139
normative rows now carry nonblank resolved rationale locators.

Grant Wilson froze this exact v0.1/r0.4 package as scientific authority on
`2026-08-20`. The six open, four known-but-not-supplied, and two deferred
detailed-ledger entries retain their recorded states. None blocks the frozen
structural contract; each blocks only its named automatic policy, numerical
product, response, coefficient use, adjacent-owner input, or evidentiary claim.
`PTC-OWNER-Q001` remains decided: conditioned `r` is diagnostic-only and
inert/advisory in base v0.1. `PTC-OWNER-Q002` remains decided: base PCA/SVD
application is frozen-subspace projection with the exact acting space declared
by the realized family.

The next scientific task is the cross-package RTC--CAL--PTC response and
lifecycle coherence review, not another broad PTC rewrite. That review cannot
silently amend this frozen authority. Any substantive correction or newly
resolved open choice requires explicit owner action and a versioned successor
or formally reopened revision.

No implementation conformity, representation/response fidelity, validation,
achieved performance, science-qualification, or production-readiness claim is
made by this scientific-authority freeze.
