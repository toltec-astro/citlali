# SCI-RTC — Raw-Timestream Conditioning And Temporal Response

Status: Stage B v0.1/r0.1 implementation-blind two-view draft manager-reviewed
and ready for scientific-owner review; scientific authority not approved or
frozen

Version: `v0.1`

## Program Adherence And Prior-Work Recovery

This package is governed by the
[Citlali Scientific Contract Library Program](../../../README.md) and the
[CAL/MAP pilot process review](../../../PILOT_PROCESS_REVIEW_2026-08-16.md).
Work began with the package's [`PRIOR_WORK.md`](PRIOR_WORK.md) recovery record.
That record reuses the frozen implementation-independent RTC core, later owner
decisions, the phase-zero downsampling amendment, and the learned-sampling
design instead of repeating their derivations.

Implementation, audits, repairs, re-audits, tests, validation, Unity evidence,
and production status remain quarantined in
[`INTERNAL_DOSSIER.md`](INTERNAL_DOSSIER.md). They are not proposed author
inputs.

## Current Gate

- Package selection: approved by Grant Wilson on `2026-08-17`.
- Stage A prior-work recovery: complete and owner-reviewed.
- Sanitized Scope Brief: approved `2026-08-17`, including the owner-modified
  `flxscale` donor-scaling rule in `RTC-SCOPE-D004`.
- Exact content-bound author packet: approved `2026-08-17`.
- Implementation-blind Stage B authorship: complete from that packet only.
- Manager review: complete; r0.1 accepted for scientific-owner review.
- Scientific authority, implementation conformity, validation, and production
  promotion: not established.

The next gate is scientific-owner review of the r0.1 two-view draft, its 18
author decisions, and the decomposed owner-decision register.

## Contents

- [`PRIOR_WORK.md`](PRIOR_WORK.md): exact recovery and disposition record
- [`INTERNAL_DOSSIER.md`](INTERNAL_DOSSIER.md): quarantined
  implementation-informed scope evidence
- [`SCOPE_BRIEF.md`](SCOPE_BRIEF.md): owner-approved sanitized author input
- [`AUTHOR_SUPERSESSION_COVER.md`](AUTHOR_SUPERSESSION_COVER.md): approved
  binding cover for the reusable RTC core
- [`AUTHOR_CONVENTIONS_AND_OWNERSHIP.md`](AUTHOR_CONVENTIONS_AND_OWNERSHIP.md):
  sanitized conventions and package boundaries
- [`AUTHOR_PACKET_MANIFEST.md`](AUTHOR_PACKET_MANIFEST.md): approved exact
  allowed and prohibited inputs
- [`AUTHOR_DRAFT_DECISIONS.md`](AUTHOR_DRAFT_DECISIONS.md): 18 bounded author
  presentation and consolidation decisions
- [`MANAGER_REVIEW_R0.1.md`](MANAGER_REVIEW_R0.1.md): independence,
  structural, scientific, build, and visual-QA review
- [`DECISION_LOG.md`](DECISION_LOG.md): package-selection and approved scope
  decisions
- [`SCIENTIFIC_OWNER_DECISION_LEDGER.md`](SCIENTIFIC_OWNER_DECISION_LEDGER.md):
  23 open, one conditional, and four deferred numerical/policy decisions
- [`CROSSWALK.md`](CROSSWALK.md): exact shared-core and packet traceability
- `src/`: one six-file shared core and the two audience views
- `pdf/`: canonical 25-page rationale and 17-page engineering draft outputs

## Protected Boundary

SCI-RTC begins with an admitted aligned primary detector stream and the exact
upstream identity, time-grid, coordinate, calibration, validity, and support
state required by the selected product role. It owns the ordered application
of raw-timestream conditioning, its temporal and detector-mixing response,
support and influence accounting, phase-zero sampling, and its atomic output
bundle.

It does not derive ALIGN timing, AST coordinates, BEAM calibration factors,
the CAL atmosphere operator, PTC correlated-mode cleaning or weights, VAL
eligibility policy, MAP estimation, FLT map filtering, or FRUIT recurrence.
It also does not silently equate the Beammap raw `Delta f/f` signal boundary
with a calibrated `mJy/beam` path.

## Authority And Status

This package contains an owner-approved Stage A scope/packet and a
manager-reviewed Stage B r0.1 author draft. Neither the author nor the manager
approves that draft as the SCI-RTC v0.1 scientific authority. Current
application and production behavior retain their existing repository status
until a later, separately authorized conformity and validation program
assesses them.
