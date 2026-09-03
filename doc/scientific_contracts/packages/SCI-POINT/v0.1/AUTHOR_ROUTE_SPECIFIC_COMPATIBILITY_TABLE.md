# SCI-POINT Route-Specific Source-Model Compatibility

Identity: `SCI-POINT_ROUTE_COMPATIBILITY_TABLE v0.1/r0.3`

| Route boundary | Required response chain | Current numerical state |
| --- | --- | --- |
| `SCI-MAP_TO-SCI-POINT-BOUNDARY-REQUIREMENTS` | source morphology -> exact MAP response -> expected processed-map profile -> Gaussian compatibility model | `unavailable_pending_method_and_route_binding` |
| `SCI-JINC_TO-SCI-POINT-BOUNDARY-REQUIREMENTS` | source morphology -> exact signed/normalized JINC response -> expected processed-map profile -> Gaussian compatibility model | `unavailable_pending_method_and_route_binding` |
| `SCI-FLT-FIXED_TO-SCI-POINT-BOUNDARY-REQUIREMENTS` | source morphology -> exact MAP/JINC parent response -> fixed filter response/edge state -> processed profile -> Gaussian model | `unavailable_pending_method_and_route_binding` |
| `SCI-FLT-MATCHED_TO-SCI-POINT-BOUNDARY-REQUIREMENTS` | source morphology -> exact parent/template/noise/phase response -> matched amplitude-map profile -> Gaussian model | `unavailable_pending_method_and_route_binding` |

Each record must publish response-center relation, source-reference origin,
model-mismatch state, expected centroid bias or typed unavailability,
fit-domain relation, and displacement-claim status. Recognizing a route does
not prove Gaussian adequacy, zero centroid bias, equal response, or numerical
availability. No route substitutes or falls back to another.
