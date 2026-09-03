# SCI-POINT v0.1 — Implementation-Blind Author Scope Brief

Identity: `SCI-POINT_AUTHOR_SCOPE v0.1/r0.3`

Status: closed conditional generic-contract candidate; numerical method
authorities intentionally unavailable; exact packet approval and Stage B
launch pending

## Assignment

Author one shared SCI-POINT v0.1 scientific core and two views:

1. a scientist-readable rationale; and
2. an engineering-conformance specification importing the same core.

SCI-POINT fits one known, isolated, bright Pointing source on one exact
observation-local per-array map product. Its primary scientific result is the
measured source displacement in the declared AltAz tangent basis. It also
publishes the complete six-parameter fit, formal uncertainty state, support,
method, parent, and use-specific eligibility facts.

The author shall formalize the approved compatibility-method family, package
boundaries, conditional product meanings, and exact unavailable method-record
requirements. It shall not invent the unavailable numerical procedure.

## Exact Scientific Operation

The base method identity is
`POINT-FIT/ELLIPTICAL-GAUSSIAN-COMPATIBILITY@1`, the established
zero-background six-parameter elliptical Gaussian with:

- fitted amplitude;
- two continuous centroid coordinates;
- two fitted widths; and
- fitted orientation angle.

The symbolic source-model family is defined without inventing missing
numerical behavior. `POINT-COMPATIBILITY-METHOD v0.1`, which must state the
model, parameter ordering, width convention,
orientation gauge and degeneracies, map weighting or covariance use, support,
initialization/search, constraints, failure behavior, response interpretation,
and numerical solution procedure, is currently
`unavailable_pending_separate_owner_approval`. The distinct
`POINT-FORMAL-ERROR-METHOD v0.1` is also
`unavailable_pending_separate_owner_approval`. The scientifically distinct
`POINT-FULL-MAP-RMS-METHOD v0.1` is likewise unavailable and blocks only the
descriptive dynamic-range diagnostic. No additional source-model family is
part of base v0.1. SCI-VAL policy-profile terminology remains separate.

## Inputs

Every requested fit binds:

- observation and array identity;
- exact observation-local MAP, JINC, FLT-FIXED, or FLT-MATCHED parent family,
  product, method, version, and generation;
- complete terminal FRUIT ancestry when FRUIT produced the terminal map;
- parent estimand, signal unit, calibration, normalization, WCS, grid, AltAz
  tangent basis, pixel metric, orientation, handedness, support, validity,
  missing/non-finite policy, response, covariance/uncertainty state,
  null-space, and provenance;
- authoritative expected source position with exact source/target/ephemeris
  authority, convention, time/epoch, frame, atmospheric/aberration treatment,
  morphology/reference origin, uncertainty, validity, and availability;
- parent-map WCS/reference origin;
- requested and effective algorithmic search center, separately from the
  expected source position; and
- requested fit method and all requested configuration states.

The fitted source centroid is a fourth distinct object. In the exact declared
AltAz tangent basis,

`Delta_POINT = fitted centroid - authoritative expected source position`.

When the tangent-coordinate origin is the expected source position,
`Delta_POINT` equals the fitted centroid coordinate. The search center affects
search/initialization only and never changes the displacement zero point.
POINT owns this measurement sign; the pointing-support producer owns the
measurement-to-correction sign.

MAP, JINC, FLT-FIXED, and FLT-MATCHED are four eligible but non-equivalent
routes. POINT may not choose, substitute, equate, or fall back among them.
FRUIT is ancestry on one of those terminal map types, not a fifth parent.
Coadds and intermediate FRUIT iterations are outside base v0.1.

## Search, Support, And Constraint State

Preserve the established configurable:

- expected center and central search domain;
- weighted-peak initialization;
- global-search fallback;
- bounded fit domain; and
- amplitude, fitted-width, and orientation-angle constraints.

Requested, effective, and realized values or named states are distinct method
identity. Every realized fallback is reported. Numeric sentinels resolve to
explicit named effective states. The contract shall neither freeze one
universal numerical configuration nor authorize a new algorithm or silent
replacement.

## Products And Claims

Each requested observation-array-parent-method fit is independently atomic.
Producer lifecycle, component identifiability, and named-use disposition are
separate from the four named-use evaluation axes: request, applicability,
eligibility, and realization. `diagnostic_only` is not an eligibility value or
producer state; it is represented as prescribed consumer action
`diagnostic_display_only` after those axes are evaluated.

When the exact compatibility method and required boundaries are available, a
realized numerical fit may contain:

- measured two-component tangent-plane displacement in arcseconds;
- fitted amplitude in the exact parent product's unit and response;
- two effective fitted widths and orientation under the exact processed-map
  response;
- marginal formal parameter errors only when the distinct exact formal-error
  method is available, and honest joint-covariance availability;
- fit support, constraints, requested/effective/realized state, parent and
  method identity; and
- the declared legacy dynamic-range and formal-standardization diagnostics
  where those quantities are produced.

`POINT-SOURCE-ASSOCIATION-STATE` is a first-class role applying to every
search, fallback, restart, or retry branch. Source-attributed amplitude and
displacement require `established` association. Known, isolated, bright, and
approximately centered are typed applicability facts, not informal labels or
implementation defaults.

Centroid displacement is the primary pointing measurement. Fitted centroid,
amplitude, widths, angle, and fit state are also telescope/observing-condition
quality-control metrics. Amplitude is not automatically universal flux.
Effective fitted shape is not automatically an intrinsic telescope beam,
detector PSF, or SCI-BEAM result. The metrics alone do not identify a unique
physical cause for a deviation.

## Uncertainty

Marginal formal parameter errors are unavailable until
`POINT-FORMAL-ERROR-METHOD v0.1` is separately recovered, content-bound, and
owner-approved. Once available, publish them with their method, assumptions,
domain, conditioning, and limitations. Full joint parameter covariance may
remain unavailable. Absence of either is not zero, diagonal covariance, or
evidence of independence and does not by itself invalidate an otherwise
authorized fit.

Later joint-covariance, astrometric, empirical-repeatability, or NOI
uncertainty estimates are separately versioned companions to the immutable
POINT result. They do not rewrite the original uncertainty claim.

## Named Uses And Policy Ownership

Keep four named-use policies distinct:

1. POINT owns per-array fit-result completeness.
2. The pointing-support producer owns displacement admission for correction
   construction.
3. The named telescope/observing QC process owns parameter admission,
   references, thresholds, comparisons, aggregation, and actions.
4. CAL or TolProj owns amplitude admission for photometric transfer.

The same result may pass one use, fail another, and be diagnostic-only for a
third. Diagnostic-only is use-specific, not a universal bad-result flag. VAL
registers/evaluates exact profiles and authors none of them. The author may
propose collision-free identifiers and mechanics for final owner review. No
whole-observation or cross-array aggregate profile belongs to POINT v0.1.

## Downstream Boundary

SCI-POINT ends with authoritative per-array measurements. It does not form a
cross-array aggregate and does not convert measured displacement into an
applied correction. The named pointing-support producer owns any aggregation,
member/weight/covariance/failure policy, measurement-to-correction sign,
telescope user/paddle-offset composition, record selection, native support,
and correction-record publication. AST owns conforming application.

Failure of one array does not erase sibling results. POINT never imputes a
missing array or reports a whole-observation success state. A downstream
producer that admits a partial array set owns and records that exact policy and
membership.

## Non-Goals And Claim Ceiling

SCI-POINT v0.1 does not perform blind detection, deblending, cataloging,
blank-field faint-source fitting, per-detector Beammap inference, OOF optical
inference, mapmaking, filtering, FRUIT recurrence, calibration, empirical
uncertainty inference, correction selection, interpolation, or application.

The contract establishes scientific meaning only. It shall not claim
implementation conformity, observational validation, uncertainty coverage,
achieved response, pointing accuracy, performance, readiness, production
suitability, or Unity state.

## Required Stage B Deliverables

The author shall return:

- one shared normative scientific core;
- scientist-readable scientific rationale and engineering-conformance views;
- stable requirement IDs and falsifiable prediction IDs with complete
  traceability;
- exact source-reference/displacement and AST tangent-basis boundaries;
- a symbolic invariant Gaussian model plus explicit unavailability of every
  missing legacy numerical convention;
- separate required-record and dependency definitions for
  `POINT-COMPATIBILITY-METHOD`, `POINT-FORMAL-ERROR-METHOD`, and
  `POINT-FULL-MAP-RMS-METHOD`;
- an exact product-and-claim dependency matrix and parent-signal-role table;
- route-specific compatibility and finite terminal-FRUIT ancestry envelopes;
- separate producer lifecycle, component-identifiability, and named-use axes;
- exact conditional diagnostic formulas and uncertainty-budget components;
- explicit typed unavailable states and edge cases;
- draft exact named-use profile identifiers/mechanics under the approved owner
  boundaries; and
- source/build manifests and draft PDFs.

If this exclusive packet is scientifically insufficient, the author shall
return one precise question and stop. The author may not inspect a prohibited
source, infer an implementation default, or broaden the package.
