# SCI-POINT Proposed Author Conventions And Ownership

Identity: `SCI-POINT_AUTHOR_CONVENTIONS v0.1/r0.3`

Status: sanitized Stage A candidate; exact packet-byte approval pending

This document contains no implementation, configuration, audit, test,
validation, reduction, or production evidence. It is a proposed compact input
for a future implementation-blind scientific-contract author.

## Fixed Package Boundary

SCI-POINT concerns one known, isolated, bright pointing source on one exact
observation-local per-array map parent. It does not perform blind detection,
deblending, catalog construction, per-detector Beammap inference, OOF optical
inference, mapmaking, filtering, feedback, calibration, or correction
application.

## Parent Ownership

The exact parent producer owns its signal estimand, unit, calibration, WCS,
grid, frame, support, validity, response, covariance, null-space, lifecycle,
and provenance. POINT must bind and consume those meanings without redefining
them. Owner-approved ODQ-003 makes MAP, JINC, FLT-FIXED, and FLT-MATCHED
eligible as distinct routes; they are not interchangeable and POINT may not
select, substitute, equate, or fall back among them automatically. A terminal
FRUIT result retains one of those exact map types and adds complete FRUIT
terminal/generation ancestry; FRUIT is not another parent type. Coadds remain
outside base v0.1.

## Coordinate Conventions

- Authoritative expected source position, parent-map WCS/reference origin,
  requested/effective search center, and fitted centroid are four distinct
  objects.
- Base v0.1 publishes displacement in an exact declared two-axis AltAz
  tangent basis in arcseconds; axis labels alone do not establish raw
  delta-Az, cross-elevation, handedness, or sign.
- Continuous fitted map coordinates must be transformed with the exact parent
  WCS/pixel metric and declared handedness/orientation.
- The Gaussian is evaluated in the exact physical tangent-plane metric unless
  the coordinate boundary proves storage-pixel distance equivalent.
- `Delta_POINT = fitted centroid - authoritative expected source position` in
  that basis. Search/fallback centers do not define the zero point.
- A fitted displacement is not an absolute spherical coordinate and not an
  applied correction.
- Measurement-to-correction sign, selection, interpolation, and application
  are separately owned.

## Fit And Shape Conventions

- The generic model may use a symmetric positive-definite shape matrix. Exact
  published width and orientation conventions remain unavailable until the
  separately approved compatibility-method record exists.
- Source model, background treatment, weights/covariance, seed/search, fit
  domain, parameter constraints, and response are part of method identity.
- Under ODQ-005, the established configurable expected-center/central-search,
  weighted-peak initialization, global fallback, bounded fit domain, and
  amplitude/width/angle constraints are preserved. Requested, effective, and
  realized values or named states are distinct; every realized fallback and
  sentinel resolution is explicit.
- Fitted widths and angle describe the effective source shape under the exact
  parent response and fit model; they do not automatically establish an
  intrinsic beam.
- A parameter that hits a bound remains a bound-censored or measurement-limited
  result unless a named acceptance policy says otherwise.
- Exact circularity makes orientation undefined by rotational symmetry; a
  finite placeholder angle is not a physical measurement.
- Fitted amplitude, widths, and angle are required fit-result components and,
  together with centroid and fit state, telescope/observing-condition quality-
  control metrics. Quality-control use preserves exact parent response,
  calibration, support, constraint, and uncertainty state and does not by
  itself establish a unique physical cause.

## Uncertainty Conventions

Formal fit covariance or marginal errors, coordinate-transformation
uncertainty, pointing-correction uncertainty, calibration uncertainty,
empirical repeatability, and NOI uncertainty are different products. Missing
covariance is unknown/unavailable, not zero, diagonal, or evidence of
independence. ODQ-007 requires the established marginal formal parameter
errors only after the distinct exact formal-error method is owner-approved;
they are currently unavailable. Joint covariance may remain unavailable
without invalidating an otherwise authorized fit. Later versioned companions
do not alter the original fit product's claims.

## Per-Array Atomicity

Each requested observation-array-parent-method fit has an independent
producer lifecycle and component-identifiability state. One array's failure does not
erase sibling results, and file/table co-location does not merge their
scientific atomicity. POINT neither imputes a missing array nor publishes an
observation-wide success result. The downstream pointing-support producer owns
any partial-array aggregate admission policy.

## Diagnostic Conventions

- legacy `sig2noise`: conditionally fitted amplitude divided by an exactly
  bound full-map RMS; unavailable until that RMS population and denominator
  method are bound; dynamic range, not significance;
- `fitted_amplitude_over_full_map_rms`: canonical name for that descriptive
  ratio; legacy `sig2noise` is an exact alias only after the RMS method is
  approved;
- `fit_sig2noise`: conditionally fitted amplitude divided by a positive finite
  formal fitted-amplitude error from the exact formal-error method; currently
  unavailable; formal standardization only unless separately justified and
  validated.

## Adjacent Ownership

- SCI-BEAM owns per-detector Beammap fits, effective PSF, sensitivity, and APT.
- CAL/TolProj owns authorized photometric transfer using POINT amplitude.
- The pointing-support producer owns correction-record construction/selection
  and its displacement-admission policy under the final owner boundary.
- The named telescope/observing QC process owns parameter admission,
  references, thresholds, comparison/aggregation, and actions for its use.
- AST owns conforming correction application and coordinate realization.
- NOI owns empirical uncertainty methods.
- POINT owns only its per-array fit-result completeness policy. CAL/TolProj
  owns photometric-transfer amplitude admission. VAL registers and evaluates
  every exact named-use profile but authors none of them.

One immutable POINT result may be ineligible for one named use while separately
eligible for a diagnostic-display use with prescribed action
`diagnostic_display_only`. That action is not an eligibility value or a
universal result flag. No whole-observation or aggregate profile enters base
v0.1.

## Nonclaims

No scientific contract alone proves implementation conformity, response or
covariance fidelity, uncertainty coverage, achieved pointing accuracy,
validation, performance, readiness, production suitability, or Unity state.
