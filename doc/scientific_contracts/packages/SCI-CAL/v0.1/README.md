# SCI-CAL — Detector Calibration, Atmospheric Extinction, And Signal Transfer

Status: scientist-facing v0.2 major revision ready for owner scientific
review; engineering v0.1 unchanged; not frozen

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
- [`SCIENTIFIC_RATIONALE_V0.2_CHANGELOG.md`](SCIENTIFIC_RATIONALE_V0.2_CHANGELOG.md):
  major narrative and scientific-clarification changes from v0.1
- [`SCIENTIFIC_OWNER_DECISIONS_V0.2.md`](SCIENTIFIC_OWNER_DECISIONS_V0.2.md):
  nine unresolved scientific-owner decisions exposed by the review
- [`SCIENTIFIC_ENGINEERING_CONSISTENCY_V0.2.md`](SCIENTIFIC_ENGINEERING_CONSISTENCY_V0.2.md):
  manager consistency check and `nw10` authority finding
- [`SCIENTIST_CROSSWALK_V0.2.md`](SCIENTIST_CROSSWALK_V0.2.md): grouped routing
  of every assumption, requirement, and edge prediction
- [`CROSSWALK.md`](CROSSWALK.md): all 50 requirements traced to the
  scientist-facing authority and implementation-independent observables
- `src/common/`: shared notation, definitions, assumptions, equations,
  requirements, and edge predictions
- [`src/scientific-rationale.tex`](src/scientific-rationale.tex): canonical
  source for the scientist-facing view
- [`src/engineering-conformance.tex`](src/engineering-conformance.tex):
  canonical source for the engineering-facing view
- [`pdf/SCI-CAL-SCIENTIFIC-RATIONALE-v0.1-DRAFT.pdf`](pdf/SCI-CAL-SCIENTIFIC-RATIONALE-v0.1-DRAFT.pdf):
  archived 24-page scientist-facing predecessor reviewed by Grant
- [`pdf/SCI-CAL-SCIENTIFIC-RATIONALE-v0.2-DRAFT.pdf`](pdf/SCI-CAL-SCIENTIFIC-RATIONALE-v0.2-DRAFT.pdf):
  revised 14-page science-team rationale, including formal appendices
- [`pdf/SCI-CAL-ENGINEERING-CONFORMANCE-v0.1-DRAFT.pdf`](pdf/SCI-CAL-ENGINEERING-CONFORMANCE-v0.1-DRAFT.pdf):
  21-page engineering-facing draft

## Next Gate

Grant reviews the v0.2 scientist-facing rationale and the nine-item scientific
owner-decision register. The former `SCI-CAL-OWNER-Q001` is preserved as
`SCI-CAL-OWNER-Q06`; the review also exposed missing authority for the physical
definition of `xs`, baseline and pipeline ordering, `flxscale` derivation and
transfer, broadband photometric convention, opacity-policy rationale,
numerical uncertainty products, and science-qualification criteria.

After owner scientific approval, a fresh implementation-blind consistency
review must compare the two views and crosswalk before freezing.
This draft does not approve an implementation, run validation, or change
production status.
