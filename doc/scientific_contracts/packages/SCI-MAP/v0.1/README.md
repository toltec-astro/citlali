# SCI-MAP — Ordinary Mapmaking And Observation Coaddition

Status: science-team rationale r0.2 first editing pass complete; formal
contract preserved; scientific-owner/voice review required; scientific
authority not frozen

Scientific contract scope: `v0.1`, owner-approved (`2026-08-16`)

## Program Adherence And Prior-Work Recovery

This package is governed by the
[Citlali Scientific Contract Library Program](../../../README.md). Work began
with the package's [`PRIOR_WORK.md`](PRIOR_WORK.md) recovery record. The
owner-approved [`SCOPE_BRIEF.md`](SCOPE_BRIEF.md) identifies the scientific
work already available, the package boundary, the exact approved author
references, and the information-firewall exclusions.

The recovery reuses the frozen implementation-independent SCI-MAP-001 core,
the later owner-approved whole-bundle/coaddition decisions, and the registered
weighted-normalization reasoning. It does not repeat the prior derivation or
promote the SCI-MAP-001/002/003 audits, repairs, tests, reductions, or
validation results into scientific authority.

Grant approved this opening, `MAP-SCOPE-D001--D006`, and the exact three-part
author packet on `2026-08-16`. The approved packet is content-bound in
[`AUTHOR_PACKET_MANIFEST.md`](AUTHOR_PACKET_MANIFEST.md).

## Approved v0.1 Boundary

The package covers:

- ordinary positive-coefficient normalized gridding of admitted Stokes-I
  detector samples into an observation map;
- the shared raw-map identity, signal, normalization, response/kernel,
  conditional-uncertainty, support, validity, and provenance vocabulary; and
- atomic centered-integer common-grid coaddition of compatible admitted
  observation-map bundles.

The proposal excludes the separate signed-coefficient JINC estimator
(`SCI-MAP-002`), the OOF residual transfer product (`SCI-MAP-003`),
maximum-likelihood mapmaking, RTC/PTC estimation, calibration, validity-policy
production, astrometry, empirical-noise construction, map filtering, source
fitting, Beammap inference, and fruit-loop feedback.

## Stage A And Author-Packet Contents

- [`PRIOR_WORK.md`](PRIOR_WORK.md): verified discovery, classification,
  disposition, and anti-repetition record
- [`INTERNAL_DOSSIER.md`](INTERNAL_DOSSIER.md): implementation-informed
  scope and dependency evidence; permanently excluded from authorship
- [`SCOPE_BRIEF.md`](SCOPE_BRIEF.md): owner-approved author input
- [`AUTHOR_SUPERSESSION_COVER.md`](AUTHOR_SUPERSESSION_COVER.md): approved
  cover limiting and updating the reusable SCI-MAP-001 core
- [`AUTHOR_CONVENTIONS_AND_OWNERSHIP.md`](AUTHOR_CONVENTIONS_AND_OWNERSHIP.md):
  approved sanitized convention and responsibility extract
- [`AUTHOR_PACKET_MANIFEST.md`](AUTHOR_PACKET_MANIFEST.md): exact allowed
  inputs, hashes, firewall, and deliverables
- [`DECISION_LOG.md`](DECISION_LOG.md): concise record of the six approved
  scope decisions

A fresh implementation-blind GPT-5.6 Ultra author was dispatched from the
content-bound packet on `2026-08-16`. Scope approval did not approve the
resulting contract.

## Preserved Formal Contract And Science-Team Rationale r0.2

The manager-reviewed r0.1 draft supplied one shared canonical LaTeX authority
and two rendered views. The first scientific editing round found the formal
science strong but required the SCI-CAL house-model genre separation. The
package now exposes:

- [formal scientific/engineering contract](pdf/SCI-MAP-FORMAL-SCIENTIFIC-ENGINEERING-CONTRACT-v0.1.pdf),
  preserving all canonical equations, 52 requirements, 25 predictions, exact
  support/state semantics, provenance, decision register, and conformance
  routing;
- [science-team rationale r0.2](pdf/SCI-MAP-SCIENTIFIC-RATIONALE-v0.1.pdf),
  a scientist-facing account of the estimator, response, uncertainty,
  support/validity, coaddition, WCS, products, and validation without the full
  requirement or prediction inventories;
- [engineering conformance specification](pdf/SCI-MAP-ENGINEERING-CONFORMANCE-v0.1.pdf)
  (18 pages);
- [`src/common/`](src/common/), containing the required canonical
  `notation`, `definitions`, `equations`, `assumptions`, `requirements`, and
  `edge_cases` modules;
- [`src/scientific-rationale.tex`](src/scientific-rationale.tex), the stable
  scientist-facing source filename;
- [`src/engineering-conformance.tex`](src/engineering-conformance.tex), the
  stable engineering-facing source filename;
- [`src/formal-scientific-engineering-contract.tex`](src/formal-scientific-engineering-contract.tex),
  the preserved complete formal view created by the first editing round;
- [`CONTRACT_INCONSISTENCY_AND_PROPOSED_AMENDMENT_R0.2.md`](CONTRACT_INCONSISTENCY_AND_PROPOSED_AMENDMENT_R0.2.md),
  recording `SCI-MAP-CI-001` and the pending dimensionless `coverage_cut`
  correction without changing normative text;
- [`SCIENTIST_CROSSWALK_R0.2.md`](SCIENTIST_CROSSWALK_R0.2.md), the grouped
  rationale/formal routing;
- [`SCIENCE_TEAM_RATIONALE_R0.2_CHANGELOG.md`](SCIENCE_TEAM_RATIONALE_R0.2_CHANGELOG.md);
- [`SCIENTIFIC_FORMAL_CONSISTENCY_R0.2.md`](SCIENTIFIC_FORMAL_CONSISTENCY_R0.2.md);
- [`CROSSWALK.md`](CROSSWALK.md), with complete requirement and prediction
  routing;
- [`SCIENTIFIC_OWNER_DECISION_LEDGER.md`](SCIENTIFIC_OWNER_DECISION_LEDGER.md),
  preserving OD-001--007 and appending OD-008--009; and
- [`AUTHOR_DRAFT_DECISIONS.md`](AUTHOR_DRAFT_DECISIONS.md), recording bounded
  author choices without silently resolving owner authority.

The six common modules define 52 stable requirements and 25 falsifiable
predictions. The retained r0.1 shared-authority filename is a compatibility
wrapper over those modules, not a second source of science. Normalized map and coadd vectors contain exactly the rows
authorized by the effective support policy; unsupported full-grid storage is
not promoted to zero-valued scientific output. The engineering view imports
all normative science from the shared authority and adds only evidence,
execution, traceability, and result procedure.

The r0.2 pass consulted only approved scientific ownership and representation
authorities needed to avoid reopening resolved work. Approved PTC D004 fixes
the ordinary coefficient as a declared scalar analysis/gridding coefficient,
not precision by default. Accepted ADR 0009 and its 2026-08-05 owner amendment
are the exact WCS/FITS and 0.1-arcsec serialization authority. Projection
normalization/boundary rules and any canonical-grid preparation or future
reprojection/mosaicking owner remain genuine gaps, now OD-008 and OD-009.

No implementation candidate was selected or inspected, and no validation,
reduction, Unity execution, or production-status decision was performed.

## Next Gate

The next gate is one scientific-owner and voice review for rationale r0.3,
including disposition of `SCI-MAP-CI-001` and any decisions the owner is ready
to answer in `SCI-MAP-OD-001--009`. Later revision requires an owner decision,
a normative contract change, new validation evidence, or a newly identified
scientific inconsistency. Implementation conformance and validation remain
later, separate gates.
