# SCI-BEAM — Detector Beam Inference, Calibration Candidates, QC, And Products

Status: Stage A complete; sanitized scope awaiting scientific-owner approval

Version: `v0.1`

## Program Adherence And Prior-Work Recovery

This package is governed by the
[Citlali Scientific Contract Library Program](../../../README.md) and the
[CAL/MAP pilot process review](../../../PILOT_PROCESS_REVIEW_2026-08-16.md).
Work began with the package's [`PRIOR_WORK.md`](PRIOR_WORK.md) recovery record.
The draft [`SCOPE_BRIEF.md`](SCOPE_BRIEF.md) identifies reusable material,
genuinely new work, proposed author references, and information-firewall
exclusions.

Do not launch scientific authorship until Grant Wilson approves that Scope
Brief and exact author packet. Agreement to the program workflow did not
pre-approve the scientific substance drafted here.

## Current Gate

- Stage A recovery and boundary drafting: complete.
- Separation from active ALIGN/AST work: confirmed as a process boundary.
- Scientific-owner Scope Brief approval: required.
- Implementation-blind Stage B author: not commissioned.
- Normative scientific contract, engineering conformance view, and PDFs: not
  authored.
- Implementation conformity, validation, and production promotion: outside
  this stage.

## Contents

- [`PRIOR_WORK.md`](PRIOR_WORK.md): exact prior-work recovery and disposition
- [`INTERNAL_DOSSIER.md`](INTERNAL_DOSSIER.md): quarantined implementation-informed scope evidence
- [`SCOPE_BRIEF.md`](SCOPE_BRIEF.md): sanitized draft for owner approval
- [`DECISION_LOG.md`](DECISION_LOG.md): approved process decisions and pending scope decisions
- [`CROSSWALK.md`](CROSSWALK.md): Stage A traceability scaffold
- [`SCIENTIFIC_OWNER_DECISION_LEDGER.md`](SCIENTIFIC_OWNER_DECISION_LEDGER.md): owner questions that gate or limit Stage B
- `src/`: canonical Stage B source paths, presently blocked placeholders
- `pdf/`: rendered outputs, presently unavailable because Stage B has not begun

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

No SCI-BEAM v0.1 scientific authority exists yet. The Stage A files define a
proposed author task, not a contract. The historical `SCI-BEAM-001` audit
inventory was never launched and has no independent core or audit verdict.
Current products and algorithms remain governed by their existing repository
documents and production decisions until a later accepted contract says
otherwise.
