# SCI-BEAM v0.1 — Sanitized Conventions And Ownership

Status: owner-approved author reference

Approval date: `2026-08-16`

This document contains only stable scientific conventions and abstract
producer/transformer/consumer boundaries relevant to detector beam inference.
It contains no Citlali, TolAPT, or `toltec_beammap` source behavior; no audit
finding, repair, test, validation result, current A/B outcome, or production
status; and no active ALIGN/AST material.

Approved source classes:

- current Citlali scientific frame, unit, identity, validity, and one-way-state
  conventions, sanitized through the approved Scope Brief;
- the owner-approved TolAPT soft-prior producer boundary, without producer
  implementation or readiness claims;
- the SCI-CAL and SCI-MAP conditional interfaces summarized below, without
  importing their implementation, audits, or unresolved scientific authority;
  and
- owner-approved repository ownership decisions consolidated in
  `BEAM-SCOPE-D012`.

## Capability And Quantity Boundary

- V0.1 concerns observation-local detector source/beam inference and its
  result/QC bundle.
- The admitted model begins with an elliptical two-dimensional beam convolved
  with a separately declared calibrator brightness model and explicitly
  bounded background terms.
- Point-source treatment is a declared limiting case of the calibrator model.
- Fitted source centroid, detector relative coordinate, telescope pointing,
  physical timing, and absolute astrometry are distinct quantities.
- Fitted amplitude, source-model flux, response, candidate conversion factor,
  promoted calibration, and sensitivity are distinct quantities and authority
  states.
- V0.1 may publish a typed detector-calibration candidate. It does not promote
  that candidate to SCI-CAL authority and does not own a detector-sensitivity
  result without a later exact convention for noise, time, atmosphere,
  calibration, and bandwidth.

## Identity And Indexing

- Array identity is one of `a1100`, `a1400`, and `a2000`. Array ID, array name,
  array index, network ID, detector/acquisition identity, map index, candidate
  slot, and container position are distinct.
- Network IDs remain explicit across subsets; they are not reconstructed from
  local order.
- Detector-resolved products use an explicit artifact/occurrence-scoped stable
  identity relation. A row number, `det_N` label, candidate slot, array, or
  network is not by itself an external detector identity.
- A UID spelling is meaningful only with the artifact and binding that define
  it; equality of local key text does not establish universal detector identity.
- FITS/WCS pixels are one-based. In-memory map, detector, candidate, iteration,
  and sample indices are zero-based unless a field explicitly states otherwise.

## Frames, Shapes, And Orientation

- Beammap detector maps use a declared AltAz tangent plane about the Beammap
  source. Spatial axes are azimuth and elevation offsets in arcseconds.
- Persisted WCS is the pixel-to-coordinate authority. Array memory order does
  not determine axis sign, handedness, wrapping, or orientation.
- Fitted centroids and beam-shape parameters remain in the admitted relative
  frame unless an upstream authority explicitly supplies a valid transformation.
- The author must choose and state one nondegenerate ellipse convention,
  including major/minor ordering, angle zero, positive direction, periodicity,
  and circular-limit behavior.
- Detector signal/map, fit/QC, result/APT, and optional TOD companions preserve
  explicit shape, identity, and parent relations. Equal row counts or slot
  positions do not prove correspondence.

## Units And Statistical Labels

- The active admitted Beammap map signal and unit-bearing kernel boundary is
  `mJy/beam`. This recorded unit does not by itself prove absolute calibration
  or complete response fidelity.
- A map coefficient with inverse-square signal units has dimensional meaning;
  it is not automatically statistical precision.
- Noise variance and parameter covariance carry the square of their associated
  quantity units. Cross-covariances carry product units.
- Fitted offsets and beam widths use arcseconds. Angle units are explicit.
  Frequency uses Hz. Time or exposure quantities name their exact accounting
  convention.
- A signal-to-noise label requires a declared estimator and uncertainty
  calibration. A formal standardized residual or amplitude is not empirical
  significance by naming alone.

## Validity, Missing Data, And State

- Missing, disabled, automatic, unsupported, rejected, invalid, failed,
  non-converged, and unavailable are semantic states, not undocumented numeric
  sentinels.
- Invalid input is excluded before numerical payload evaluation. Multiplication
  by zero is not a non-finite mask.
- A finite parameter, positive amplitude, converged optimizer, low residual, or
  binary flag cannot independently establish BEAM scientific validity.
- Fit attempt, candidate selection, optimizer completion, parameter
  admissibility, convergence, QC disposition, calibration candidacy, and
  product publication are distinct states with explicit causes.
- Requested, effective, observation-resolved, and realized state flows one way.
  A later observation cannot inherit an earlier source, flux, prior, fit,
  convergence, QC, or validity state.
- A required result/QC/provenance failure propagates. An optional map or TOD
  companion cannot substitute for a required scientific result.

## Soft-Prior Boundary

- TolAPT owns production of the soft Beammap prior and its producer-side
  artifact contract.
- The admitted prior identity is array/network/slot-local spatial expectation,
  not exact detector UID, measured position, or design-to-measured truth.
- In v0.1 a compatible soft prior may initialize a candidate and define bounded
  candidate gating. It does not enter as an unconditional exact assignment or
  veto.
- Prior frame, units, producer/version, reliability class, location, scale or
  covariance, and compatibility decision are explicit.
- Prior influence and the selected candidate route are recorded. A strong
  observed compatible source may defeat a wrong soft prior.
- Missing, incompatible, weak, or unsuccessful priors require a declared blind
  fallback rather than fabricated prior success.

## Producer–Transformer–Consumer Responsibilities

### Upstream producers

- **TolProj/TolTECA photometry boundary** selects the calibrator, supplies
  immutable source identity, and supplies the declared per-array source
  flux/brightness model with its convention and uncertainty. BEAM does not
  select a catalog calibrator or silently update the source model.
- **SCI-CAL** conditionally supplies calibrated signal meaning, unit,
  calibration state, response basis, uncertainty status, and lineage. Its
  scientific authority remains limited by its own recorded owner decisions.
  BEAM consumes that exact meaning and may return only a typed calibration
  candidate for separate CAL promotion.
- **ALIGN/AST** supplies the admitted sample/map coordinate relation, frame,
  detector binding, coordinate validity, and uncertainty. BEAM does not infer
  physical timing, absolute pointing, detector-coordinate truth, or absolute
  astrometric correction from a relative centroid.
- **RTC/PTC** supply conditioned signal, causal validity, response status,
  analysis coefficient, and covariance status. BEAM cannot relabel a partial
  or unavailable upstream response as the complete realized beam.
- **VAL** supplies upstream sample/detector eligibility, flag precedence,
  non-finite policy, and cause-specific unavailable/failure state.
- **SCI-MAP** conditionally supplies a complete detector map bundle with its
  estimator, WCS, support, validity, unit, response companion or unavailable
  state, covariance status, and parentage. BEAM does not redefine that map.

### SCI-BEAM transformer

- SCI-BEAM owns the observation-local source/beam forward model, admitted fit
  support, likelihood/objective, parameter/covariance statement, prior
  influence, locator/measurement iteration, convergence state, model-specific
  QC causes, and atomic result identity.
- SCI-BEAM may transform an admitted source model and fitted amplitude into a
  typed calibration candidate under an exact normalization and uncertainty
  statement. It does not promote the factor or define downstream sensitivity.
- SCI-BEAM's internal loop is one observation-local estimator. It does not own
  general science feedback, learning, restart, or cross-product recurrence.

### Downstream consumers

- **TolAPT** owns matched/reference APT construction and later soft-prior
  production. It may consume BEAM results only through explicit identities and
  may not retroactively redefine an observation-local fit.
- **`toltec_beammap`** owns downstream calibration analysis, APT diagnostics
  and updates, planet workflows, and sensitivity utilities. It preserves BEAM
  result identity, validity, uncertainty, and limitations.
- **SCI-CAL** separately owns any promotion of a BEAM calibration candidate.
- **FRUIT** owns general feedback, recurrence, learning, restart identity, and
  stopping policy outside the observation-local BEAM estimator.

No artifact from one repository silently supersedes another repository's
authority. This extract is content-bound in
[`AUTHOR_PACKET_MANIFEST.md`](AUTHOR_PACKET_MANIFEST.md).
