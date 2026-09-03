# SCI-BEAM — Beammap Analysis, Effective PSF, Calibration, Sensitivity, And APT

Status: **Scientific authority frozen; implementation conformity not yet assessed under this contract.**

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
- Implementation-blind Stage B `r0.1` author and manager review: complete and
  retained as draft history.
- Scientific-owner `r0.2` revision directive: approved on `2026-08-17`.
- Final bounded `r0.3` scientific-owner review: approved and incorporated on
  `2026-08-17`.
- Shared normative core: 46 requirements and 24 falsifiable predictions.
- Formal Scientific/Engineering Contract and separate science-team rationale:
  frozen at `r0.3` after source, compilation, PDF, and page-by-page visual QA.
- Open scientific-owner decisions: nine, limited to sensitivity, adequacy,
  PSF accuracy/wings, pivot registration, and downstream kernel qualification.
- Implementation conformity, validation, and production promotion: outside
  this stage.
- No further editorial round. Revision is allowed only under the four
  contract-library stopping-rule triggers.

## Contents

- [`PRIOR_WORK.md`](PRIOR_WORK.md): exact prior-work recovery and disposition
- [`INTERNAL_DOSSIER.md`](INTERNAL_DOSSIER.md): quarantined implementation-informed scope evidence
- [`SCOPE_BRIEF.md`](SCOPE_BRIEF.md): owner-approved Stage B author input
- [`AUTHOR_PACKET_MANIFEST.md`](AUTHOR_PACKET_MANIFEST.md): exact allowed and prohibited author inputs
- [`AUTHOR_CONVENTIONS_AND_OWNERSHIP.md`](AUTHOR_CONVENTIONS_AND_OWNERSHIP.md): sanitized stable conventions and interfaces
- [`AUTHOR_PRIMARY_REFERENCE_BOUNDARY.md`](AUTHOR_PRIMARY_REFERENCE_BOUNDARY.md): bounded primary TolTEC context
- [`DECISION_LOG.md`](DECISION_LOG.md): approved process and scientific-scope decisions
- [`SCIENTIFIC_OWNER_DIRECTIVE_R0.2.md`](SCIENTIFIC_OWNER_DIRECTIVE_R0.2.md): governing substantive revision decisions
- [`SCIENTIFIC_OWNER_REVIEW_R0.3.md`](SCIENTIFIC_OWNER_REVIEW_R0.3.md): final bounded corrections and freeze authorization
- [`CROSSWALK.md`](CROSSWALK.md): exact 70-row rationale-to-contract traceability
- [`SCIENTIFIC_OWNER_DECISION_LEDGER.md`](SCIENTIFIC_OWNER_DECISION_LEDGER.md): three stable groups and nine atomic open decisions
- [`CHANGE_LOG_R0.2.md`](CHANGE_LOG_R0.2.md): r0.1 clause-to-r0.2 disposition map
- [`CHANGE_LOG_R0.3.md`](CHANGE_LOG_R0.3.md): bounded r0.2-to-r0.3 correction map
- [`CROSS_DOCUMENT_FOLLOWUP_R0.2.md`](CROSS_DOCUMENT_FOLLOWUP_R0.2.md): required future adjacent-authority amendments
- [`CROSS_DOCUMENT_FOLLOWUP_R0.3.md`](CROSS_DOCUMENT_FOLLOWUP_R0.3.md): bounded final-review follow-up delta
- [`CONSISTENCY_REPORT_R0.2.md`](CONSISTENCY_REPORT_R0.2.md): unit, ownership, traceability, compilation, and visual QA record
- [`CONSISTENCY_REPORT_R0.3.md`](CONSISTENCY_REPORT_R0.3.md): final freeze-gate consistency and rendered-PDF QA record
- [`AUTHOR_DRAFT_DECISIONS.md`](AUTHOR_DRAFT_DECISIONS.md): implementation-blind author choices, questions, tensions, and unavailable claims
- [`MANAGER_REVIEW_R0.1.md`](MANAGER_REVIEW_R0.1.md): bounded correction, firewall, traceability, and QA review
- `src/`: canonical shared core and the two document views
- `pdf/`: final revision-bearing `v0.1/r0.3` rationale and formal contract outputs; r0.2 pilot filenames retained as history

## Protected Boundary

SCI-BEAM derives observation-local raw and horizon-derotated relative detector
coordinates under an externally declared WCS and field-rotation convention.
It does not establish absolute boresight, physical rotation pivot, telescope
pointing error, or pointing-model correction. Bracketing pointing and science
observations require the same immutable APT artifact and AST convention
unless a separately authorized transform proves equivalence.

SCI-BEAM is the desired scientific authority for complete Beammap analysis and
the source Beammap APT, including accepted `flxscale` and `sens`. TolAPT owns
the supplied soft-prior producer contract; TolProj owns approved target
association or child-APT transformation; SCI-CAL later consumes the selected
factor and applies target atmosphere. Present repository implementations and
schemas remain unassessed and are not reconciled in this authorship task.

## Authority And Status

SCI-BEAM v0.1/r0.3 is the frozen implementation-independent scientific
authority. It is not an implementation verdict. The
historical `SCI-BEAM-001` audit
inventory was never launched and has no independent core or audit verdict.
Current products and algorithms remain governed by their existing repository
documents and production decisions until a later conformity and validation
program establishes their relationship to this frozen authority.
