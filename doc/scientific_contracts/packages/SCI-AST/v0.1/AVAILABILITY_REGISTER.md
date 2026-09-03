# SCI-AST v0.1 Stage B r0.3 Availability Register

Status: prospective typed availability; not an achieved conformance,
validation, or readiness record

| Fact or claim | Availability rule | Exact cause or owner gate | Upstream facts preserved on failure |
| --- | --- | --- | --- |
| ALIGN-grid direction | Available only with stable occurrence `(o,s)`, exact ALIGN mapping/source relation, observing state, correction, selected geometry/rotation, and output frame | Five cause types remain distinct; Q001 and Q006 gate unresolved fields/geometry | Exact ALIGN/readout producer facts not rewritten |
| Tangent coordinate | Requires available direction, upstream-selected physical center, and tangent projection | Q002 gates family center owner; TAN invalidity has its own reason | Direction remains available |
| Continuous pixel | Requires available tangent role and exact nonsingular WCS | Unsupported/unavailable WCS and center plan are typed | Direction and tangent remain available |
| Nominal pixel | Requires finite continuous pixel plus exact rounding, dimensions, and bounds convention | Non-finite/out-of-rule/bounds causes are distinct | Continuous pixel, tangent, and direction remain available |
| RTC coordinate/response | Requires appropriate upstream coordinate role and complete exact RTC parent under `SCI-AST:rtc_output_grid_coordinates@1` | Q008 is closed for the ordinary RTC-grid science chain; missing RTC facts are dependency-limited and do not authorize reconstruction | Unrelated ALIGN-grid roles remain available |
| Base pre-MAP AST facts | Continuous pixel, optional nominal containing pixel, and bounds state only | Missing exact coordinate/WCS parent makes only affected fact unavailable | No kernel-dependent neighborhood or estimator support is emitted |
| Exact `G_pi` | Available only for a complete exact MAP-owned deposition request and all AST/RTC/MAP parents | Missing kernel/support/normalization/boundary/conservation/response/covariance/plan fact refuses materialization | Base AST coordinate roles remain unchanged |
| Map-center Jacobian | Unavailable until exact family/domain/codomain/direction/axes/units request | Q007 | Coordinate roles remain unchanged |
| Map-center covariance total | Formed only with all claimed components, cross terms, conditioning, and combination assumptions | Q003; unavailable is not zero | Available individual terms retain their states |
| Quantitative approximation/time/precision adequacy | Unavailable without preregistered domain, metric, tolerance, owner, oracle, and failure rule | Q004 | Mathematical identities and qualitative contract remain available |
| MAP-003 retained-grid interface | Deferred | Q005 | Generic AST/MAP ownership boundary remains available |
| Ordinary nonpolarimetric coordinate path | Admitted within the existing coordinate contract | AST does not interpret polarization; optional HWPR timing may remain only as an ALIGN-parent fact | Coordinate parentage is unchanged by the optional timing fact |
| Demodulation, polarization calibration/response, or Stokes reconstruction | Not authorized by SCI-AST v0.1 | Outside the ordinary nonpolarimetric coordinate path; raw KID `x` is not Stokes I | Readout and coordinate producer facts retain their exact non-Stokes identities |
| Scientific approval, implementation conformity, empirical adequacy, validation, freeze, observational performance, production readiness | Unassessed | Requires separate authorized evidence/review outside this draft | Document build and visual QA do not change these states |

The canonical vocabulary remains `available`, `available_conditional`,
`unavailable_input`, `unavailable_authority`, `unavailable_unsupported`,
`not_applicable`, and `not_persisted_standard`, always with an exact reason.
