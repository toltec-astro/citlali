# SCI-FLT-FIXED To SCI-POINT Boundary

Identity: `SCI-FLT-FIXED_TO-SCI-POINT-BOUNDARY-REQUIREMENTS v0.1/r0.3`

Status: draft boundary requirements; exact parent authority/version/source
digest not bound; owner approval pending; numerical route unavailable

The boundary admits only signal role
`FLT-FIXED-SIGNAL/TRANSFORMED-MAP@1`. Response, covariance, support, exposure,
operator, and state records are companions rather than alternate signal
inputs. The instantiated boundary binds exact observation and array, immutable filtered product and
its MAP/JINC parent, filter operator/kernel and version, normalization,
method/generation, unit/calibration, WCS/grid/frame/tangent basis,
support/validity, edge state, missing/non-finite policy, response,
covariance/uncertainty, null/additive-reference state, phase/origin,
lifecycle, and provenance.

Instantiation additionally requires exact parent package/version/source digest,
owner approval, and compatibility/supersession state.

It also binds source morphology -> parent response -> fixed filter response ->
expected processed-map profile -> Gaussian compatibility, including response
center, model mismatch, expected centroid bias or typed unavailability,
fit-domain relation, and displacement-claim status.

Current numerical state:
`unavailable_pending_method_and_route_binding`. The filtered route is not
silently equivalent to its unfiltered parent and provides no automatic
substitute or fallback.
