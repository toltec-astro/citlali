# SCI-MAP To SCI-POINT Boundary

Identity: `SCI-MAP_TO-SCI-POINT-BOUNDARY-REQUIREMENTS v0.1/r0.3`

Status: draft boundary requirements; exact parent authority/version/source
digest not bound; owner approval pending; numerical route unavailable

The boundary admits only signal role
`MAP-SIGNAL/OBSERVATION-LEVEL-NORMALIZED@1`: an exact observation-level
normalized MAP signal, never a coadd. The instantiated boundary binds exact
observation and array, immutable MAP product,
estimand/signal role, method/version/generation, unit/calibration,
WCS/grid/frame/tangent basis, support/validity, missing/non-finite policy,
response, covariance/uncertainty, null/additive-reference state, phase/origin,
lifecycle, and provenance.

Instantiation additionally requires exact parent package/version/source digest,
owner approval, and compatibility/supersession state.

It also binds source morphology -> MAP response -> expected processed-map
profile -> Gaussian compatibility, including response center, model mismatch,
expected centroid bias or typed unavailability, fit-domain relation, and
displacement-claim status.

Current numerical state:
`unavailable_pending_method_and_route_binding`. MAP eligibility supplies no
automatic fit, zero bias, response fidelity, or fallback to another route.
