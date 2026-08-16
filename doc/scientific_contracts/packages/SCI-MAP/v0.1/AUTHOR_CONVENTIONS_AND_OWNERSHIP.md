# SCI-MAP v0.1 — Draft Sanitized Conventions And Ownership

Status: draft for owner review; not authorized for an author packet

This document contains only stable scientific conventions and abstract
producer/transformer/consumer boundaries relevant to ordinary mapmaking and
observation coaddition. It contains no Citlali source behavior, audit finding,
repair, test, validation result, or production-status claim.

Proposed sources:

- `origin/codex/refactor-mainline@46ad23888a40f5102cdfd50c06e49a549bdf8a20:doc/SCIENTIFIC_CONVENTIONS.md`,
  content SHA-256
  `1970d7e31ccbcf77f890ea7c0854fde59d25b2fc745f909a74150360605d3049`;
- `doc/scientific_contracts/packages/SCI-CAL/v0.1/` at
  `b237c5f0be3b8558be58b8501faa9618e8061c12`, solely for the
  owner-reviewed CAL-to-MAP boundary; and
- owner-approved MAP decisions consolidated in the Scope Brief and
  supersession cover.

## Capability And Quantity Boundary

- V0.1 is Stokes I only.
- The ordinary input is the calibrated `xs` detector stream supplied under
  SCI-CAL. No other measured stream inherits that meaning.
- The active calibrated map signal boundary is `mJy/beam`. Accepted tokens
  for other unit families do not establish their conversions or scientific
  meaning.
- Mapmaking consumes calibrated samples and produces a map-domain estimator.
  It does not produce calibration, pointing, timestream cleaning, empirical
  noise, filtered response, source inference, or feedback science.

## Identity

- Array identity is one of `a1100`, `a1400`, and `a2000`. Array ID,
  array name, array index, network ID, detector/acquisition identity, map
  group, map index, Stokes component, and container position are distinct.
- A map index is a zero-based position in a reduction-local collection. Its
  scientific identity is resolved through explicit group, array/network/
  detector as applicable, Stokes, frequency when applicable, estimator,
  observation/coadd role, and product identity.
- Stokes I has index 0 and label `I`. Reserved legacy Q/U slots are not
  enabled scientific authority.
- `obsnum` is external observation identity. An observation's position in a
  reduction request is not that identity.
- Artifact-local detector UID or a row number is not a persistent universal
  detector namespace. Cross-artifact correspondence requires an explicit
  occurrence/product-scoped relation. Equal row positions or equal local-key
  spellings are insufficient unless their binding is proven for the named
  artifact.
- FITS/WCS pixel coordinates are one-based. In-memory map row/column indices
  are zero-based.

## Shape, Ordering, And Frames

- The admitted primary timestream shape is samples by detectors.
- Map products are collections of two-dimensional spatial planes with explicit
  group and Stokes identity.
- Science maps use a declared equatorial J2000 TAN WCS with degree-valued
  spatial axes.
- Pointing and OOF maps use a declared AltAz tangent plane with azimuth and
  elevation offsets in arcseconds.
- FITS/WCS is the persisted pixel-to-coordinate authority. Memory order does
  not determine axis sign, handedness, orientation, or wrapping.
- Coaddition requires exact compatible scientific identity and permits only
  the owner-approved centered integer relation between observation and coadd
  shapes. A WCS tolerance for physical serialization is not authority for
  reprojection or a different sky grid.

## Units And Statistical Labels

- Signal and its response/kernel companion carry the admitted calibrated
  signal unit when the kernel is a unit-bearing realized template response.
- The ordinary map coefficient is recorded with its declared gridding unit.
  An inverse-square signal unit is dimensional information, not proof of
  statistical precision.
- Conditional covariance has the square of the signal unit. A formal diagonal
  weight is inverse square only under its named covariance assumptions.
- Exposure carries a declared time/accounting unit. Detector-seconds are not
  unique wall-clock integration time.
- Hit and observation-count products are integer counts with their exact
  population stated.
- Support and validity states are dimensionless logical facts.
- `sig2noise` is reserved for an empirically calibrated estimator. Signal
  multiplied by the square root of a merely formal weight is a formal
  standardized signal, not statistical significance.

## Validity, Missing Data, And State

- Missing, disabled, automatic, unavailable, unsupported, invalid, and failed
  are semantic states, not undocumented numerical sentinels.
- Upstream sample/detector eligibility is distinct from mapmaker numerical
  support, science-policy support, and final raw map validity.
- A raw map value may be non-finite outside valid support. A finite value does
  not establish validity.
- Invalid contributions are excluded before their numerical payload is
  evaluated. Zero multiplication is not a non-finite mask.
- A required product, companion, or write failure propagates; a completion
  marker cannot override missing required output.
- Configuration and provenance flow one way:
  requested to effective to observation-resolved to realized. Later execution
  does not rewrite the accepted request or a prior observation's state.

## Producer–Transformer–Consumer Responsibilities

### Upstream producers

- **SCI-CAL** owns the calibrated sample meaning, unit, calibration
  quality/validity, conditional uncertainty transfer, response basis, and
  lineage. MAP applies that selected meaning without reinterpretation.
- **ALIGN/AST** owns sample time/alignment, coordinate meaning, projected
  position, frame/WCS, and astrometric uncertainty.
- **PTC** owns conditioned-sample meaning and the production, normalization,
  lifecycle, and covariance status of analysis coefficients.
- **VAL** owns upstream sample/detector eligibility, flag precedence,
  non-finite policy, and cause-specific unavailability/failure state.

### SCI-MAP transformer

- SCI-MAP owns ordinary positive-coefficient sample-to-map transformation,
  response propagation through that operator, conditional covariance
  equations, map-specific support/final validity, complete raw map identity,
  and compatible ordinary observation coaddition.
- It may transform a selected upstream quantity and preserve its lineage. It
  may not redefine the producer's physical meaning.

### Downstream consumers

- **NOI** owns noise-realization generation, empirical covariance/weight, and
  significance calibration.
- **FLT** owns map filtering and its output support, response, covariance, and
  validity while preserving immutable raw MAP validity and parentage.
- **SRC**, **MODE**, and **BEAM** own source, Pointing/OOF, and Beammap
  inference and their fit uncertainty/validity.
- **FRUIT** owns feedback, recurrence, iteration/restart identity, and stopping
  rules.
- Consumers may not reinterpret a normalization coefficient as precision,
  promote a raw-invalid pixel, discard response/parent identity, or claim an
  unavailable unit/covariance/significance.

## Separate Map-Adjacent Packages

- MAP-002/JINC owns signed coefficients, analytic and subpixel JINC response,
  method-specific normalization/conditioning, coverage, validity, and
  products. Ordinary positive-coefficient rules do not apply by analogy.
- MAP-003 owns OOF residual transfer estimation, exact tracer/final-response
  parentage, frequency-domain validity, and any future LMTOOF consumer
  boundary.
- Maximum-likelihood mapmaking, mosaicking/reprojection, filtering, noise,
  fitting, Beammap inference, and feedback require their own scientific
  authority.

After owner approval this extract will be content-hashed and listed in the
author-packet manifest. Until then it is a review draft only.
