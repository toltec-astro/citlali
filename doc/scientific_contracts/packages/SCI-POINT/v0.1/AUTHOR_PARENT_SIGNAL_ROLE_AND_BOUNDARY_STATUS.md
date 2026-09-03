# SCI-POINT Parent Signal Roles And Boundary Status

Identity: `SCI-POINT_PARENT_SIGNAL_ROLES v0.1/r0.3`

These are exact role names for Stage B authorship, not source-bound numerical
instances. Every current route is unavailable because exact parent authority,
version, source digest, owner approval, and compatibility binding are absent.

| Parent family | Exact POINT signal role | Explicit non-signal companions |
| --- | --- | --- |
| MAP | `MAP-SIGNAL/OBSERVATION-LEVEL-NORMALIZED@1` | coadds, exposure/support, response, covariance, and state records are not alternate POINT signal inputs |
| JINC | `JINC-SIGNAL/NORMALIZED-JINC-MAP@1` | `N`, `C`, `Q`, coefficient-squared temporal accounting, response, covariance, and state records are companions, not POINT signal parents |
| FLT-FIXED | `FLT-FIXED-SIGNAL/TRANSFORMED-MAP@1` | response, covariance, support, exposure, operator, and state records are companions, not alternate signal inputs |
| FLT-MATCHED | `FLT-MATCHED-SIGNAL/MATCHED-TEMPLATE-AMPLITUDE-FIELD@1` | template, normalization, response, covariance, support, and state records are companions, not alternate signal inputs |

When instantiated, every boundary must bind exact package/version/source
digest, signal role, observation/array, WCS/grid/frame/tangent relation, unit/
calibration, support/validity, response/covariance status,
null/additive-reference state, lifecycle/provenance, and compatibility/
supersession state. The Stage B author may not open adjacent packages to
recover another role name or missing boundary fact.
