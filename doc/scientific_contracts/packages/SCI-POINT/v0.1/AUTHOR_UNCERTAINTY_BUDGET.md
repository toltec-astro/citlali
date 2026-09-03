# SCI-POINT Uncertainty Budget

Identity: `SCI-POINT_UNCERTAINTY_BUDGET v0.1/r0.3`

| Component | Owner/source | Base state and rule |
| --- | --- | --- |
| conditional formal fit error | `POINT-FORMAL-ERROR-METHOD` | unavailable pending separate approval |
| joint parameter covariance | exact formal-error or separate covariance authority | unavailable unless explicitly supplied |
| source-reference/ephemeris uncertainty | source-reference producer | required boundary state; not formal fit error |
| WCS/tangent-coordinate uncertainty | AST/coordinate boundary | required boundary state; not optimizer covariance |
| parent response/model-mismatch uncertainty | exact parent and route compatibility record | typed separately; expected centroid bias may be unavailable |
| calibration uncertainty | parent/CAL authority | separate from fit amplitude error |
| empirical pointing repeatability | separately authorized empirical process | unavailable without that authority |
| NOI uncertainty | NOI companion | separate versioned product |
| cross-array dependence | downstream pointing-support producer/input authority | POINT does not aggregate or infer it |
| pointing-correction uncertainty | pointing-support producer and AST use | downstream; not measurement uncertainty by implication |

Components may not be combined unless every required variance and
cross-covariance for the declared operation is available. If fit and published
tangent bases differ, exact uncertainty transformation generally requires
joint centroid covariance; transformed marginals alone are not exact under a
non-axis-aligned Jacobian. Preferred generic formulation evaluates the model
directly in the declared tangent coordinates.
