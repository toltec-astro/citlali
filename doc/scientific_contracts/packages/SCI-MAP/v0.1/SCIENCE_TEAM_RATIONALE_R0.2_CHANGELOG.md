# SCI-MAP v0.1 Science-Team Rationale r0.2 Change Log

Date: `2026-08-16`

Scope: major genre separation after first scientific editing review. No
implementation inspection, validation execution, reduction, Unity work, or
production claim was performed.

## Changes

- Preserved the r0.1 canonical equations, 52 requirements, 25 predictions,
  exact state semantics, provenance, and conformance machinery in a separately
  identified formal scientific/engineering contract.
- Created a scientist-facing rationale organized around the estimator,
  projection, response, uncertainty, user-facing support/validity facts,
  coaddition, products, validation, and open decisions.
- Put conditional linearity on the first page and made the inherited
  SCI-CAL, ALIGN/AST, PTC, and VAL status explicit.
- Imported the approved PTC D004 coefficient meaning: scalar analysis and
  gridding coefficients with declared family, units, normalization scope,
  lifecycle, and factors; never precision unless separately proved.
- Added the principal fractional-projection teaching figure and the two-sample
  worked example where `Q=1`, variance is `1/2`, and marginal inverse variance
  is `2`.
- Translated the eight formal facts into science-user questions and explained
  the two-threshold policy qualitatively, including its global-population
  dependence.
- Added the complete-bundle/common-grid coadd figure and explained odd-shape
  rejection and the boundary between ordinary coaddition and future
  reprojection/mosaicking.
- Cited accepted ADR 0009 and its 2026-08-05 owner amendment as the exact
  authority for the binary64 typed/sidecar WCS, FITS representation, and
  0.1-arcsec serialization bound.
- Removed full requirement and prediction inventories, exact machine-state
  vocabulary, indexing mechanics, numerical tolerance construction, hashes,
  and publication-state mechanics from the science-facing main text.
- Recorded `SCI-MAP-CI-001`, the dimensional inconsistency in the existing
  `coverage_cut` clauses, and prepared an exact owner amendment without
  changing normative text before approval.
- Preserved OD-001--OD-007 and appended OD-008 for projection normalization
  and boundary convention and OD-009 for canonical-grid preparation and
  future reprojection/mosaicking ownership.

## Planned next pass

One scientific-owner and voice review is planned for r0.3. Later changes
require an owner decision, a normative contract change, new validation
evidence, or a newly identified scientific inconsistency.

