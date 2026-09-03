# SCI-POINT v0.1 Author Parent Route And Claim Matrix

Identity: `SCI-POINT_PARENT_ROUTE_MATRIX v0.1/r0.3`

Status: scientifically eligible families; numerical availability not established

| Parent family | POINT interpretation | Required bound state | Claim boundary |
| --- | --- | --- | --- |
| observation-local SCI-MAP per-array map | `MAP-SIGNAL/OBSERVATION-LEVEL-NORMALIZED@1` | exact product, estimand, unit, WCS, support, validity, response, covariance, calibration, generation | never a coadd; claims remain conditional on ordinary MAP state |
| observation-local SCI-JINC per-array map | `JINC-SIGNAL/NORMALIZED-JINC-MAP@1` | exact normalization, coefficient, support, response/covariance, generation, unit | `N`, `C`, `Q`, and coefficient-squared accounting are companions; never silently equivalent to MAP |
| observation-local SCI-FLT-FIXED product | `FLT-FIXED-SIGNAL/TRANSFORMED-MAP@1` | exact parent plus filter operator/kernel, normalization, response, edge/support, covariance state | response/covariance/support/exposure/operator records are not alternate signal inputs |
| observation-local SCI-FLT-MATCHED product | `FLT-MATCHED-SIGNAL/MATCHED-TEMPLATE-AMPLITUDE-FIELD@1` | exact template, complete support, output unit, response, covariance option, phase/origin | template/normalization/response/covariance/state records are not alternate signal inputs |
| NOI standardized-signal companion | search/QC diagnostic scale only | exact signal parent and NOI uncertainty product | not a POINT amplitude/displacement parent in base v0.1 |

For each eligible parent family, the exact source morphology, parent response,
expected processed-map source profile, and Gaussian compatibility relation
must be bound separately. Eligibility does not prove that one Gaussian is
adequate for every route. Until that relation and the exact compatibility
method are approved, every numerical route has state
`unavailable_pending_method_and_route_binding`.

## FRUIT Ancestry

A terminal product created through FRUIT is consumed under its exact MAP,
JINC, FLT-FIXED, or FLT-MATCHED type and binds complete FRUIT method, terminal
iteration, generation, response, support, uncertainty, and lineage. FRUIT is
not a fifth route; intermediate iterations are excluded.

## Route Rule

For any requested route, POINT claims only fitted quantities under the exact
parent's response, support, frame, unit, calibration, and uncertainty state.
POINT never automatically chooses, substitutes, equates, or falls back among
routes. Unavailability of one eligible route does not authorize another.

Coadd parents are outside base v0.1.
