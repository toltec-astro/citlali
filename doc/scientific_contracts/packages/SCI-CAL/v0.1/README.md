# SCI-CAL — Detector Calibration, Atmospheric Extinction, And Signal Transfer

Status: implementation-blind contract draft ready for owner scientific review;
not frozen

Version: `0.1` scope (`2026-08-16`)

## Program Adherence And Prior-Work Recovery

This package is governed by the
[Citlali Scientific Contract Library Program](../../../README.md). Work began
with the package's [`PRIOR_WORK.md`](PRIOR_WORK.md) recovery record. The
owner-approved [`SCOPE_BRIEF.md`](SCOPE_BRIEF.md) states the reused scientific
work, remaining questions, approved author references, and information-
firewall exclusions.

Earlier CAL reasoning is being consolidated rather than repeated. The frozen
implementation-independent CAL core and the applicable owner decisions are
reused; later identity and accuracy amendments supersede narrower earlier
statements. Implementation traces, audit findings, repairs, tests, reductions,
and conformity claims remain in [`INTERNAL_DOSSIER.md`](INTERNAL_DOSSIER.md)
and are excluded from scientific authorship.

`SCI-CAL` is the durable library family. Historical records named
`SCI-CAL-001` are predecessor scientific and audit material classified in the
recovery record; this package does not silently rename their version-specific
conformity claims into current authority.

Grant approved the Scope Brief, all five scope decisions, and the exact
four-item author-reference packet on `2026-08-16`.

## Current Contents

- [`PRIOR_WORK.md`](PRIOR_WORK.md): Stage A recovery, classification, and
  disposition
- [`INTERNAL_DOSSIER.md`](INTERNAL_DOSSIER.md): implementation-informed
  ownership and dependency map; permanently outside the author packet
- [`SCOPE_BRIEF.md`](SCOPE_BRIEF.md): sanitized eleven-section owner-review
  authority
- [`DECISION_LOG.md`](DECISION_LOG.md): concise approved scope decisions
- [`AUTHOR_PACKET_MANIFEST.md`](AUTHOR_PACKET_MANIFEST.md): exact allowed
  inputs and firewall exclusions for the isolated scientific author
- [`AUTHOR_SUPERSESSION_COVER.md`](AUTHOR_SUPERSESSION_COVER.md): binding
  limitations and supersessions for the reusable independent core
- [`AUTHOR_CONVENTIONS_AND_OWNERSHIP.md`](AUTHOR_CONVENTIONS_AND_OWNERSHIP.md):
  sanitized stable conventions and inter-package responsibilities
- [`AUTHOR_DRAFT_DECISIONS.md`](AUTHOR_DRAFT_DECISIONS.md): ten derived
  scientific decisions and one unresolved owner question
- [`CROSSWALK.md`](CROSSWALK.md): all 50 requirements traced to the
  scientist-facing authority and implementation-independent observables
- `src/common/`: shared notation, definitions, assumptions, equations,
  requirements, and edge predictions
- [`src/scientific-rationale.tex`](src/scientific-rationale.tex): canonical
  source for the scientist-facing view
- [`src/engineering-conformance.tex`](src/engineering-conformance.tex):
  canonical source for the engineering-facing view
- [`pdf/SCI-CAL-SCIENTIFIC-RATIONALE-v0.1-DRAFT.pdf`](pdf/SCI-CAL-SCIENTIFIC-RATIONALE-v0.1-DRAFT.pdf):
  24-page scientist-facing draft
- [`pdf/SCI-CAL-ENGINEERING-CONFORMANCE-v0.1-DRAFT.pdf`](pdf/SCI-CAL-ENGINEERING-CONFORMANCE-v0.1-DRAFT.pdf):
  21-page engineering-facing draft

## Next Gate

Grant reviews the scientist-facing draft as the primary artifact, the ten
derived scientific decisions, and `SCI-CAL-OWNER-Q001`: whether to approve one
content-bound atmosphere-operator record with exact nodes, ordinate
orientation, closed numeric support/seam rules, and model/passband provenance.
Until that record is supplied, numeric atmosphere evaluation, calibrated
numeric output, and even a numeric science-qualification-eligible disposition
remain unavailable.

After owner scientific review and revision, a fresh implementation-blind
consistency review must compare the two views and crosswalk before freezing.
This draft does not approve an implementation, run validation, or change
production status.
