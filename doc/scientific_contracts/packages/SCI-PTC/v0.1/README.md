# SCI-PTC — Correlated-Mode Cleaning And Detector Coefficients

Status: SCI-PTC v0.1/r0.5 scientific authority frozen by Grant Wilson on
`2026-08-23`; implementation conformity not yet assessed under this contract.

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
- [`CROSSWALK.md`](CROSSWALK.md): exact 159-row mapping for 99 requirements and
  60 falsifiable predictions
- [`AUTHOR_DRAFT_DECISIONS.md`](AUTHOR_DRAFT_DECISIONS.md): 38 author choices
  and the 15-entry detailed owner ledger
- [`SCIENTIFIC_OWNER_DECISION_LEDGER.md`](SCIENTIFIC_OWNER_DECISION_LEDGER.md):
  manager-facing decision summary that refers to the detailed author ledger
  rather than duplicating it
- [`R0.5_CANDIDATE_REVIEW_COVER.md`](R0.5_CANDIDATE_REVIEW_COVER.md): exact
  content-bound candidate identity and promotion history
- [`CHANGE_LOG_R0.5.md`](CHANGE_LOG_R0.5.md): bounded r0.4-to-r0.5 scientific
  changes and preserved claim boundary
- [`CONSISTENCY_REPORT_R0.5.md`](CONSISTENCY_REPORT_R0.5.md): package-local
  implementation-blind source and audience-view consistency review
- [`SCIENTIFIC_OWNER_FREEZE_R0.5.md`](SCIENTIFIC_OWNER_FREEZE_R0.5.md): active
  owner freeze, retained open states, claim boundary, and change-control rule
- [`FREEZE_VERIFICATION_R0.5.md`](FREEZE_VERIFICATION_R0.5.md): status-only
  source/PDF verification and all-page visual-QA record
- [`SCIENTIFIC_OWNER_FREEZE_R0.4.md`](SCIENTIFIC_OWNER_FREEZE_R0.4.md):
  superseded r0.4 freeze retained for provenance
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
- `pdf/`: the canonical frozen r0.5 rationale and engineering conformance PDFs,
  with exact page counts, hashes, and all-page inspection recorded

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

The active r0.5 shared core contains 45 definitions, 25 numbered equations,
35 assumptions, 99 sequential requirements, and 60 sequential falsifiable
predictions. Every preserved normative identifier retains its earlier role;
the bounded successor appends only the exact repair and ordinary-route
authority recorded in the r0.4-to-r0.5 change map. The engineering view imports
the six shared modules exactly once, and the standalone rationale explains the
science without reproducing the full formal register.

R0.5 makes the complete affine application operator primary, defines total
removal by input-output identity, and distinguishes learned location,
correlated-subspace removal, and total removed signal. It replaces the
incomplete named-use truth rule with total T/F/U/C evaluation, preserves
classification without making classification a global ban on PTC mathematics,
and reserves distinct permission propositions for learning, application,
retention, QC, response, and other uses.

The ordinary route selects exactly one configured array- or network-level PCA
grouping, an explicit strictly positive independently feasible rank per group,
one immutable-CAL-parent fit, zero support-changing refinement, and exact
group-local fixed-state kernel propagation. Its detector-right mask-aware
application is authorized only where the finite time-local normal matrix has
full configured rank under frozen tolerance. A deficient group-time is
unavailable for both data and kernel, with no partial-rank, interpolation,
numerical-zero, route-conversion, or cross-group fallback.

Disabled PTC is a distinct RTC-terminal export workflow: Citlali publishes the
complete RTC product and terminates successfully before CAL, PTC, or MAP.
There is no PTC-disabled map and no inferred CAL-to-MAP fallback. MAP authority
remains deferred and outside this freeze.

Grant Wilson approved the complete content-bound candidate at commit
`8f0ecccfacbdce0543141c4289ec06c702065f5e` and authorized its freeze on
`2026-08-23`. The 15 detailed owner
entries retain five decided, three open, four known-but-not-supplied, and three
deferred states. None of the unresolved entries blocks the frozen ordinary
structural route; each blocks only its named automatic-selection, optional
diagnostic, stronger response/covariance, source-protection, MAP-facing, or
evidentiary claim.

The next integration work remains timestream-only: stabilize CAL and the
required coordinate/external producer boundaries, then bind the exact frozen
PTC and CAL sources into VAL and run the authorized clean-room horizontal
re-audit. This freeze alone does not close `F-001`, `F-002`, `F-009`, or
`F-011`; closure requires the re-audit result and recorded closure commit.

No implementation conformity, representation/response fidelity, validation,
achieved performance, science qualification, production readiness, or MAP
availability is established by this scientific-authority freeze. Any future
substantive correction or newly resolved open choice requires explicit owner
action and a versioned successor or formally reopened revision.
