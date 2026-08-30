# SCI-FLT v0.1 Author Conventions And Ownership

Status: sanitized SCI-FLT-FIXED author input; exact-packet owner approval and
Stage B launch required

## Program Adherence And Prior-Work Recovery

The future author must work only from the exact SHA-bound packet. The author
must not inspect or infer from Citlali implementation, configuration, schemas,
tests, audits, repairs, validation, reductions, defaults, historical behavior,
the internal dossier, or current draft NOI Stage B material.

The future task is fresh scientific authorship, not repair of an existing algorithm.
Recovered mathematics is a candidate to evaluate, not a required conclusion.

## Required Conventions

- State scientific identity, estimand, units, coordinate frame/grid, shape,
  indexing, support, validity, and missing/non-finite policy.
- Distinguish parent product, requested method, effective plan,
  observation-resolved operator state, applied transformation, learned-state
  generation, and realized successor product.
- Distinguish stored transformed amplitude, response-corrected amplitude,
  fitted/template amplitude, uncertainty, covariance, standardized signal, and
  statistical significance.
- Distinguish numerical support, scientific validity, confidence, and
  publication/admission policy.
- Treat unknown or unavailable information honestly. Unknown covariance is
  not zero; invalid support is not infinite precision; finite output is not
  automatic scientific validity.
- Bind every successor product to one immutable parent and one exact applied
  transformation generation.
- Treat observation and coadd parents separately unless an exact relation is
  derived and approved.
- Keep fixed-state, successor-update, and per-member-relearned methods
  separate.
- Identify source imprint, dependence, bias, and conditioning introduced by
  learned templates, masks, positions, priors, thresholds, or noise models.

## Ownership Rules

- MAP/JINC own the parent estimand and parent product claims.
- CAL owns absolute calibration, passband/color correction, and calibration
  covariance.
- SCI-FLT-FIXED owns the exact strict-linear transformation,
  response, transformed support/validity, lifecycle, and failure policy.
- SCI-NOI owns randomization design and empirical uncertainty/covariance/
  standardized-signal inference. NOI applies but never chooses the filter.
- SCI-BEAM and future source/mode contracts own source-fit and Pointing/OOF
  interpretation.
- SCI-FRUIT owns iterative source-model feedback and lifecycle.
- SCI-VAL evaluates owner-approved named-use policy and does not author it.
- RTC temporal filtering is outside map-domain FLT.

## Normative Output Expectations

The future author should provide a shared normative core and two views:

1. a scientist-facing rationale that explains the estimand, method classes,
   response, support/validity, uncertainty boundary, and use limits without an
   engineering inventory; and
2. a formal scientific-engineering contract with stable requirement and
   prediction identifiers.

Both views must import the same definitions, equations, assumptions,
requirements, and edge cases. Scientific authority must remain separate from
implementation conformity, validation, calibration evidence, achieved
performance, readiness, and production status.

## Forbidden Inferences

The author may not infer:

- one generic filter contract from shared software vocabulary;
- fixed scientific identity from a fixed numerical call when state was learned;
- equivalence from a shared FFT or convolution mechanism;
- Wiener-to-low-pass fallback equivalence;
- matched estimation from source-shaped smoothing;
- full covariance from a variance plane;
- precision or significance from a weight/denominator label;
- absolute photometry from a map unit or response kernel;
- filter/coadd commutation;
- signal/randomization parity from a same-name code path; or
- downstream Beammap, Pointing, OOF, source-fit, NOI, or FRUIT admission from
  product availability.

## Gate

The author must stop if the content-bound owner decision record or any exact
packet object/hash is unavailable. Open questions may not be resolved by
looking at implementation or historical validation.
