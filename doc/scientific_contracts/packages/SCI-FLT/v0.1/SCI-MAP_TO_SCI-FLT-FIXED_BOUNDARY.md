# SCI-MAP To SCI-FLT-FIXED Boundary

Boundary identity: `SCI-MAP_TO_SCI-FLT-FIXED v0.1/r0.1`

Status: sanitized Stage A boundary awaiting exact-byte owner approval; no
numerical parent is made available

Scientific owner: Grant Wilson

## Frozen Source Binding

This boundary is a compact extract from frozen `SCI-MAP v0.1/r0.7.1`. MAP
remains the authority for the parent estimand and product. The exact frozen
source manifest has SHA-256
`bd3f172f8bb1e17bf95fde034ad76632439c766ef89b0a992361b9e6d79ada0a`;
the shared MAP requirements have SHA-256
`68acf81d6c27788495cc680a1819da5f269616493b9aba5a6e0a1d1058ba5fa7`.

## Parent Roles

One FLT method instance binds exactly one of:

1. a complete base/unfiltered MAP observation bundle; or
2. a complete base/unfiltered centered-integer common-grid MAP coadd bundle.

The roles are never interchangeable. The parent binding includes exact MAP
package/revision, estimator and application generation, observation or coadd
membership, stable array/group, signal quantity, unit and originating nominal-
beam convention, target WCS/frame/topology/grid/pixel metric/shape/row domain,
normalization and coefficient meaning, support-authorized rows, product-local
validity and causes, response class and availability, covariance representation
and omitted terms, null/additive-reference state, exposure and causal-influence
facts, lifecycle, failure, and immutable provenance.

A similarly shaped map, equal-looking WCS, filename, or finite payload is not
a substitute for this exact parent identity.

## Quantity, Units, Beam, And Exposure

The exact MAP signal is the frozen calibrated-`x`-derived nonpolarimetric
total-intensity-equivalent quantity in `mJy/beam`, with its originating fixed
nominal beam/template and calibration lineage. SCI-FLT-FIXED computes output
units from the exact operator, retains that nominal-beam identity, and types
the filter-composed response separately. It does not relabel the result
`mJy/filtered-beam` or create an absolute, point-source, aperture, integrated-
flux, or extended-source claim.

Parent exposure remains physical parent lineage. FLT may publish input-to-
output influence, but filtering creates no physical exposure and cannot
convolve an exposure plane into a new physical-exposure claim.

## Response, Covariance, And Validity

MAP response, covariance, support, and validity arrive exactly in their frozen
states, including honest partial or unavailable states. SCI-FLT-FIXED may
apply its exact linear operator to an available compatible response or
covariance; it cannot manufacture a missing object or strengthen the parent
claim. Parent validity is necessary input information but does not itself
establish FLT input admission or FLT-local output validity.

## Observation And Coadd Order

Filtering an observation and filtering a coadd are distinct methods and
successor products. SCI-FLT-FIXED v0.1 does not coadd. It does not assume
`L(Coadd(m_o)) = Coadd(L(m_o))`; any bounded proved compatibility would require
a separate exact record.

## Current Numerical Availability

Frozen MAP r0.7.1 does not presently authorize a general numerical ordinary
MAP route. It remains gated by the exact PTC-owner-selected MAP-facing
coefficient family/value/QC state and the exact owner-admitted numerical
`coverage_cut` state/value and failure policy. Until those gates and runtime
parents exist, the numerical MAP parent is typed unavailable for FLT.

## Fail-Closed Join

Missing or conflicting parent role, generation, quantity, unit/beam, WCS/grid,
row domain, support/validity, response/covariance state, exposure, lifecycle,
failure, or provenance makes the FLT route unavailable. No fallback, inferred
default, or same-name equivalence is allowed.

This boundary establishes no implementation conformity, validation,
calibration, response/covariance fidelity, performance, readiness, freeze, or
production authorization.
