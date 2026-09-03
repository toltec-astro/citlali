# SCI-FLT-MATCHED To SCI-POINT Boundary

Identity: `SCI-FLT-MATCHED_TO-SCI-POINT-BOUNDARY-REQUIREMENTS v0.1/r0.3`

Status: draft boundary requirements; exact parent authority/version/source
digest not bound; owner approval pending; numerical route unavailable

The boundary admits only signal role
`FLT-MATCHED-SIGNAL/MATCHED-TEMPLATE-AMPLITUDE-FIELD@1`. Template,
normalization, response, covariance, support, and state records are companions
rather than alternate signal inputs. The instantiated boundary binds exact observation and array, immutable matched product and
its parent, template and covariance/noise selector, complete support,
normalization, method/version/generation, output unit/calibration,
WCS/grid/frame/tangent basis, validity/edge state, missing/non-finite policy,
response, covariance option, null/additive-reference state, phase/origin,
lifecycle, and provenance.

Instantiation additionally requires exact parent package/version/source digest,
owner approval, and compatibility/supersession state.

It also binds source morphology -> parent/template/noise/phase response ->
expected matched amplitude-map profile -> Gaussian compatibility, including
response center, model mismatch, expected centroid bias or typed
unavailability, fit-domain relation, and displacement-claim status.

Current numerical state:
`unavailable_pending_method_and_route_binding`. A matched-filtered peak is a
distinct route and provides no automatic substitute or fallback.
