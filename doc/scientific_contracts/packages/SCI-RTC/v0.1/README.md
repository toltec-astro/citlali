# SCI-RTC — Raw-Timestream Conditioning And Temporal Response

Status: Stage A scope and exact author packet approved; implementation-blind
Stage B drafting authorized; scientific authority not yet approved or frozen

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
- Implementation-blind Stage B authorship: authorized from that packet only.
- Scientific authority, implementation conformity, validation, and production
  promotion: not established.

The next gate is a manager-reviewed implementation-blind Stage B two-view
contract draft; that draft will return for scientific-owner review.

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
- [`DECISION_LOG.md`](DECISION_LOG.md): package-selection and approved scope
  decisions
- [`SCIENTIFIC_OWNER_DECISION_LEDGER.md`](SCIENTIFIC_OWNER_DECISION_LEDGER.md):
  open numerical and policy questions
- [`CROSSWALK.md`](CROSSWALK.md), `src/`, and `pdf/`: reserved canonical
  Stage B locations; no contract or PDF exists yet

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

This is an owner-approved Stage A scope package and author packet. That
approval authorizes drafting; it does not approve the future Stage B draft as
the SCI-RTC v0.1 scientific authority. Current application and
production behavior retain their existing repository status until a later,
separately authorized conformity and validation program assesses them.
