# SCI-FLT-INF-ODQ-007 scientific-owner approval

Decision identity: `SCI-FLT-INF-ODQ-007`

Date: `2026-08-31`

Scientific owner: Grant Wilson

Status: approved and closed; complete-support-only base method selected and
adaptive edge/background conditioning deferred

## Selected base support method

Base v0.1 admits only the **complete-support** matched-template estimator. A
location `x` is scientifically available only when the complete declared
influence support of the ODQ-005 template, eventual ODQ-004 weighting
operator, ODQ-006 exact or conformant approximate realization, and boundary
convention:

1. lies inside the exact immutable parent domain;
2. is admitted for this exact FLT input use;
3. has finite, available payload wherever the operator requires it; and
4. satisfies every required parent, template, weighting, support, and validity
   predicate.

This rule is location-based. The entire stored parent map need not be globally
valid when the exact operator has a bounded local influence support; each
output location is decided from its own complete support. If the selected
operator has global or otherwise nonlocal support, that full influence domain
must be honored rather than replaced by an assumed finite stencil.

The exact support and boundary convention are part of the realized method
identity. Observation-parent and coadd-parent support are evaluated separately
and retain their distinct ODQ-003 identities.

## Missing, nonfinite, and invalid inputs

Base v0.1 performs no partial-support estimation, support renormalization,
truncated estimation, zero/constant/reflected/periodic boundary extension,
wrap, clamp, mirror, edge completion, interpolation, imputation, replacement,
background estimation or subtraction, learned taper, or signal-derived support
selection. It does not redefine `N`, `D`, `Q`, or the template over only the
surviving samples at an affected location.

A parent-authoritative support or validity predicate may be consumed as an
exact input fact; consuming that fact is not filter-owned support learning.
A missing, nonfinite, invalid, out-of-domain, or otherwise inadmissible
required input makes the affected output location unavailable. It does not
become a zero signal, zero template amplitude, or successful partial estimate.
Required-product failure remains governed by the later lifecycle decision.

Parent-shaped storage does not promote an unavailable location into the
scientific output vector. The realized product must bind the exact admitted
support and distinguish scientific validity from numerical computability;
the detailed product representation is deferred to ODQ-013.

## Numerical fill and conservative erosion

Padding or fill may be used solely as a numerical boundary device when the
realization establishes, and validation can verify, that no scientifically
admitted output depends on any invented or replaced value. The admitted region
must be conservatively eroded by the complete declared influence support so
that every admitted location satisfies the complete-support rule.

This independently adopts the useful principle in historical FLT-D001, but it
does not adopt historical median fill, thresholds, taper, erosion distance, or
implementation mechanics. A global/nonlocal operator cannot claim an
unaffected interior merely by applying a finite guard band; if exclusion of
fill influence cannot be established for the exact operator and selected
ODQ-006 conformance envelope, that numerical route is unavailable.

Fill-influenced or padding-influenced locations are invalid for scientific
use and receive no amplitude, weighting, uncertainty, significance,
photometry, morphology, confidence, or feedback interpretation. No covariance
calculation for those invalid locations is required or authorized by this
decision.

## Deferred adaptive method

Adaptive edge/background conditioning is not part of base v0.1. A method that
learns or derives a support mask, coverage refinement, background/fill level,
window, taper, or other edge state from the target parent is a separately
identified future scientific method or preprocessing contract.

If later commissioned, that method must bind its exact learning facts and
population, immutable learned-state generation, conditional apply operator,
fill/background/taper law, complete influence footprint, eroded admitted
region, fixed-state and full-procedure response, covariance and uncertainty,
missing/nonfinite behavior, validity, NOI member parity, lifecycle, and
failure. It may not be introduced as an implementation option of the selected
base estimator.

## NOI consequence

The selected complete-support and boundary identity is consequential method
state and must be applied consistently to the science product and every
admitted transformed NOI member. This decision selects no NOI generation
graph or uncertainty estimand; ODQ-010 retains those choices. Any future
member-specific support relearning is a separate NOI-GEN method and cannot be
mixed with a fixed-support member population.

## Consequences

- `SCI-FLT-INF-ODQ-007` is approved and closed.
- Base v0.1 uses complete-support-only output admission.
- Missing, nonfinite, invalid, or out-of-domain required inputs make affected
  locations unavailable, never zero or partially renormalized estimates.
- Numerical fill is allowed only when provably excluded from the admitted
  region by conservative erosion over the complete influence support.
- Learned support, background, fill, windows, and tapers are deferred to a
  separately identified future method.
- ODQ-008 response, units, beam, and output interpretation is the next owner
  gate.

## Nonclaims

This approval selects no numerical support extent, guard-band size, erosion
distance, fill value, missing-value representation, FFT boundary convention,
coverage threshold, validity schema, adaptive-edge method, response or beam
product, uncertainty, NOI generation, public product bundle, implementation,
conformity, validation, performance, readiness, production, freeze, or Unity
action. It changes no SCI-FLT-FIXED or frozen SCI-NOI byte.
