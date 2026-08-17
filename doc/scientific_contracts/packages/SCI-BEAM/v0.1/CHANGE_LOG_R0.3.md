# SCI-BEAM v0.1 — Change Log r0.2 to r0.3

Status: final bounded scientific-owner review pass

Revision date: 2026-08-17

- Made NEFD-like sens finite and strictly positive when available by using
  abs(flxscale) while retaining signed flxscale.
- Separated map-fit Jacobian/covariance from derived calibration/sensitivity
  propagation and retained material cross-stage dependence.
- Added the explicit centroid-to-detector sign/frame transformation and
  parent-sample contribution provenance for effective derotation.
- Strengthened the pointing-transfer rule from a coordinate realization to
  the same immutable APT artifact.
- Added the bounded soft-prior/convergence paragraph to the rationale.
- Added stable decision groups SCI-BEAM-OD-001--003 across the rationale,
  formal contract, crosswalk, and atomic owner ledger.
- Replaced the two-page automatic contents with a compact single-page list.
- Required the source flux directly in TOA mJy per fixed nominal
  beam and removed defensive unit/double-factor argumentation.
- Retained all 46 requirement and 24 prediction IDs; their corrected r0.3
  meanings supersede the corresponding r0.2 draft text.
- Advanced document revision metadata and final output filenames to r0.3
  without changing scientific contract version v0.1.
- Did not inspect or change implementation, current APT storage, audits,
  repairs, tests, validation evidence, or production behavior.
