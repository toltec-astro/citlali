# SCI-BEAM — Detector Beam Inference, Calibration Candidates, QC, And Products

Status: Stage B implementation-blind draft `r0.1` ready for scientific-owner review

Version: `v0.1`

## Program Adherence And Prior-Work Recovery

This package is governed by the
[Citlali Scientific Contract Library Program](../../../README.md) and the
[CAL/MAP pilot process review](../../../PILOT_PROCESS_REVIEW_2026-08-16.md).
Work began with the package's [`PRIOR_WORK.md`](PRIOR_WORK.md) recovery record.
The approved [`SCOPE_BRIEF.md`](SCOPE_BRIEF.md) identifies reusable material,
genuinely new work, approved author references, and information-firewall
exclusions.

Grant Wilson approved the Scope Brief, `BEAM-SCOPE-D001--D012`, and the exact
three-part author packet on `2026-08-16`.

## Current Gate

- Stage A recovery and boundary drafting: complete.
- Separation from active ALIGN/AST work: confirmed as a process boundary.
- Scientific-owner Scope Brief approval: complete.
- Content-bound implementation-blind author packet: approved.
- Implementation-blind Stage B author: complete.
- Shared normative core: 46 requirements and 24 falsifiable predictions.
- Scientific rationale and engineering conformance PDFs: rendered and
  manager-reviewed at document revision `r0.1`.
- Exact approved-decision traceability: manager-integrated after author freeze.
- Scientific-owner review: pending, with seven substantive owner questions.
- Implementation conformity, validation, and production promotion: outside
  this stage.

## Contents

- [`PRIOR_WORK.md`](PRIOR_WORK.md): exact prior-work recovery and disposition
- [`INTERNAL_DOSSIER.md`](INTERNAL_DOSSIER.md): quarantined implementation-informed scope evidence
- [`SCOPE_BRIEF.md`](SCOPE_BRIEF.md): owner-approved Stage B author input
- [`AUTHOR_PACKET_MANIFEST.md`](AUTHOR_PACKET_MANIFEST.md): exact allowed and prohibited author inputs
- [`AUTHOR_CONVENTIONS_AND_OWNERSHIP.md`](AUTHOR_CONVENTIONS_AND_OWNERSHIP.md): sanitized stable conventions and interfaces
- [`AUTHOR_PRIMARY_REFERENCE_BOUNDARY.md`](AUTHOR_PRIMARY_REFERENCE_BOUNDARY.md): bounded primary TolTEC context
- [`DECISION_LOG.md`](DECISION_LOG.md): approved process and scientific-scope decisions
- [`CROSSWALK.md`](CROSSWALK.md): exact 70-row normative and approved-decision traceability
- [`SCIENTIFIC_OWNER_DECISION_LEDGER.md`](SCIENTIFIC_OWNER_DECISION_LEDGER.md): approved scope decisions and retained Stage B owner questions
- [`AUTHOR_DRAFT_DECISIONS.md`](AUTHOR_DRAFT_DECISIONS.md): implementation-blind author choices, questions, tensions, and unavailable claims
- [`MANAGER_REVIEW_R0.1.md`](MANAGER_REVIEW_R0.1.md): bounded correction, firewall, traceability, and QA review
- `src/`: canonical shared core and the two document views
- `pdf/`: rendered `r0.1` scientific-rationale and engineering-conformance views

## Protected Boundary

SCI-BEAM is not ALIGN/AST. It may condition scientific claims on an externally
declared coordinate relation, frame, identity binding, validity state, and
uncertainty. It may not establish physical timing, absolute pointing,
astrometric correction, or detector-coordinate truth. Active ALIGN work is not
an author reference.

Citlali owns reduction behavior and product conventions. TolAPT owns
producer-side soft beammap priors and matched/reference APT construction.
`toltec_beammap` owns downstream beammap analysis, calibration use, APT
diagnostics, and sensitivity utilities. SCI-BEAM must specify interfaces
without absorbing those repositories' separate authorities.

## Authority And Status

No accepted or frozen SCI-BEAM v0.1 scientific authority exists yet. The
Stage B `r0.1` artifacts are a review draft, not an accepted contract. The
historical `SCI-BEAM-001` audit
inventory was never launched and has no independent core or audit verdict.
Current products and algorithms remain governed by their existing repository
documents and production decisions until a later accepted contract says
otherwise.
