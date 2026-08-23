# SCI-CAL — Detector Calibration, Atmospheric Extinction, And Signal Transfer

Status: scientific authority frozen at v0.1/r0.5-r0.4 by Grant Wilson on
`2026-08-23`; implementation conformity, validation evidence, and
achieved-performance acceptance remain unestablished

Scientific contract version: `v0.1` (`2026-08-16`)

Active science-rationale revision: `r0.5`

Active engineering-conformance revision: `r0.4`

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
- [`ENGINEERING_CONFORMANCE_R0.2_CHANGELOG.md`](ENGINEERING_CONFORMANCE_R0.2_CHANGELOG.md):
  bounded repair aligning the engineering view with science rationale r0.3
- [`ENGINEERING_CONFORMANCE_R0.2_BUILD_REVIEW.md`](ENGINEERING_CONFORMANCE_R0.2_BUILD_REVIEW.md):
  mechanical, PDF, and visual QA for the repaired engineering artifact
- [`SCIENTIFIC_RATIONALE_R0.4_CHANGELOG.md`](SCIENTIFIC_RATIONALE_R0.4_CHANGELOG.md):
  bounded science-view corrections from the implementation-blind review
- [`ENGINEERING_CONFORMANCE_R0.3_CHANGELOG.md`](ENGINEERING_CONFORMANCE_R0.3_CHANGELOG.md):
  bounded engineering-view corrections from the implementation-blind review
- [`SCIENTIFIC_OWNER_DECISIONS_R0.5.md`](SCIENTIFIC_OWNER_DECISIONS_R0.5.md):
  approved Q01--Q09 scientific dispositions, exact atmosphere authority, and
  concrete closure/transfer validation workflow
- [`SCIENTIFIC_RATIONALE_R0.5_CHANGELOG.md`](SCIENTIFIC_RATIONALE_R0.5_CHANGELOG.md):
  owner-decision integration in the scientist-facing view
- [`ENGINEERING_CONFORMANCE_R0.4_CHANGELOG.md`](ENGINEERING_CONFORMANCE_R0.4_CHANGELOG.md):
  corresponding normative engineering repair
- [`SCIENTIFIC_ENGINEERING_CONSISTENCY_R0.5.md`](SCIENTIFIC_ENGINEERING_CONSISTENCY_R0.5.md):
  implementation-blind consistency assessment of the r0.5/r0.4 pair
- [`SCIENTIFIC_ENGINEERING_R0.5_R0.4_BUILD_REVIEW.md`](SCIENTIFIC_ENGINEERING_R0.5_R0.4_BUILD_REVIEW.md):
  mechanical, PDF, and visual QA for the owner-decision revision
- [`SCIENTIFIC_OWNER_FREEZE_R0.5.md`](SCIENTIFIC_OWNER_FREEZE_R0.5.md):
  exact owner approval, frozen authority, claim boundary, and change control
- [`FREEZE_VERIFICATION_R0.5.md`](FREEZE_VERIFICATION_R0.5.md): status-only
  source, artifact, mechanical, and visual freeze verification
- [`SCIENTIFIC_ENGINEERING_R0.4_R0.3_BUILD_REVIEW.md`](SCIENTIFIC_ENGINEERING_R0.4_R0.3_BUILD_REVIEW.md):
  mechanical, PDF, and visual QA for the repaired canonical pair
- [`CROSSWALK.md`](CROSSWALK.md): all 50 requirements traced to the
  scientist-facing authority and implementation-independent observables
- `src/common/`: shared notation, definitions, assumptions, equations,
  requirements, and edge predictions
- [`src/scientific-rationale.tex`](src/scientific-rationale.tex): canonical
  source for the scientist-facing view
- [`src/engineering-conformance.tex`](src/engineering-conformance.tex):
  canonical source for the engineering-facing view
- [`pdf/SCI-CAL-SCIENTIFIC-RATIONALE-v0.1.pdf`](pdf/SCI-CAL-SCIENTIFIC-RATIONALE-v0.1.pdf):
  stable canonical filename for the active scientist-facing r0.5 PDF
- [`pdf/SCI-CAL-ENGINEERING-CONFORMANCE-v0.1.pdf`](pdf/SCI-CAL-ENGINEERING-CONFORMANCE-v0.1.pdf):
  stable canonical filename for the active engineering-facing r0.4 PDF
- [`pdf/README.md`](pdf/README.md): canonical frozen PDF identities, page
  counts, digests, and claim boundary
- [`pdf/SCI-CAL-SCIENTIFIC-RATIONALE-v0.1-DRAFT.pdf`](pdf/SCI-CAL-SCIENTIFIC-RATIONALE-v0.1-DRAFT.pdf):
  archived 24-page scientist-facing predecessor reviewed by Grant
- [`pdf/SCI-CAL-SCIENTIFIC-RATIONALE-v0.2-DRAFT.pdf`](pdf/SCI-CAL-SCIENTIFIC-RATIONALE-v0.2-DRAFT.pdf):
  revised 14-page science-team rationale, including formal appendices
- [`pdf/SCI-CAL-ENGINEERING-CONFORMANCE-v0.1-DRAFT.pdf`](pdf/SCI-CAL-ENGINEERING-CONFORMANCE-v0.1-DRAFT.pdf):
  archived 21-page engineering-facing predecessor

The archived r0.1/r0.2 filenames above predate the adopted contract/document
version-axis rule and are retained as historical artifact names. They do not
identify contract versions later than v0.1.

## Consistency Artifacts

- [`SCIENTIFIC_RATIONALE_R0.3_CHANGELOG.md`](SCIENTIFIC_RATIONALE_R0.3_CHANGELOG.md): bounded ownership, ordering, atmosphere, terminology, versioning, and governance corrections
- [`SCIENTIFIC_OWNER_DECISION_LEDGER.md`](SCIENTIFIC_OWNER_DECISION_LEDGER.md): active Q01--Q09 authority, status, evidence, blocked-product, resolution, and affected-document ledger
- [`SCIENTIFIC_ENGINEERING_CONSISTENCY_R0.3.md`](SCIENTIFIC_ENGINEERING_CONSISTENCY_R0.3.md): corrected comparison of science rationale r0.3 and engineering conformance r0.2 across ownership, factor direction, orientation, units, ordering, claim layers, and version axes
- [`SCIENTIST_CROSSWALK_R0.3.md`](SCIENTIST_CROSSWALK_R0.3.md): refreshed grouped routing of every assumption, requirement, and edge prediction
- [`SCIENTIST_CROSSWALK_R0.4.md`](SCIENTIST_CROSSWALK_R0.4.md): corrected grouped routing for the r0.4/r0.3 pair
- [`SCIENTIFIC_ENGINEERING_CONSISTENCY_R0.4.md`](SCIENTIFIC_ENGINEERING_CONSISTENCY_R0.4.md): fresh high-effort implementation-blind consistency-review pass for the repaired pair
- [`pdf/SCI-CAL-SCIENTIFIC-RATIONALE-v0.1-r0.3-DRAFT.pdf`](pdf/SCI-CAL-SCIENTIFIC-RATIONALE-v0.1-r0.3-DRAFT.pdf): final-form science-team rationale and frozen package-template model

## Frozen Authority And Next Evidence Gate

Q01--Q09 are scientifically dispositioned, and the r0.5/r0.4 pair is the
frozen SCI-CAL v0.1 scientific authority. The next evidence gate is to execute
the realizable workflow: validate Beammap-derived source APTs with
`toltec_beammap`, enter accepted APTs into `apt_library`, match them through
TolProj, and run ordinary science reductions for same-Beammap closure and,
where an adequate independent flux exists, associated-pointing transfer.
Results are reported by array without an arbitrary matrix or sample minimum.

This freeze does not approve an implementation, establish implementation
conformity, execute scientific validation, or change production status. The
1%, 5%, and 5--10% figures remain reporting benchmarks;
achieved-performance acceptance is an owner decision based on the evidence
actually achieved. Future substantive scientific change requires a versioned
successor or formally reopened revision; later evidence may attach without
silently changing this authority.
