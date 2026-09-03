# Timestream Boundary Authority v0.1

Status: owner-approved boundary authority; implementation conformity not
assessed

Prepared: `2026-08-23`

Scientific owner: Grant Wilson

This directory contains package-neutral boundary records produced by WP-2 of
the timestream closure program. They compose the frozen or current package
authorities named in each record; they do not replace those authorities or
silently change package mathematics.

The approved records are:

- [`SCI-RTC_TO_SCI-AST_SAMPLE_GRID_BOUNDARY.md`](SCI-RTC_TO_SCI-AST_SAMPLE_GRID_BOUNDARY.md),
  identity `SCI-RTC_TO_SCI-AST_SAMPLE_GRID_BOUNDARY v0.1/r0.1`;
- [`DETECTOR_GEOMETRY_FIELD_ROTATION_BOUNDARY.md`](DETECTOR_GEOMETRY_FIELD_ROTATION_BOUNDARY.md),
  identity `DETECTOR_GEOMETRY_FIELD_ROTATION_BOUNDARY v0.1/r0.1`; and
- [`TIMESTREAM_EXPOSURE_LINEAGE_BOUNDARY.md`](TIMESTREAM_EXPOSURE_LINEAGE_BOUNDARY.md),
  identity `TIMESTREAM_EXPOSURE_LINEAGE_BOUNDARY v0.1/r0.1`.

The approved scientific decisions and their exact realization are
mapped in
[`WP2_BOUNDARY_CANDIDATE_CHANGE_MAP.md`](WP2_BOUNDARY_CANDIDATE_CHANGE_MAP.md).
[`SCIENTIFIC_OWNER_APPROVAL_2026-08-23.md`](SCIENTIFIC_OWNER_APPROVAL_2026-08-23.md)
records exact packet approval, and
[`SOURCE_MANIFEST.md`](SOURCE_MANIFEST.md) content-binds the approved bytes
after focused verification.

No artifact in this directory claims implementation conformity, numerical or
observational validation, achieved performance, production readiness, or MAP
availability. MAP projection, weighting, retained/projected exposure,
coaddition, and reprojection remain deferred.
