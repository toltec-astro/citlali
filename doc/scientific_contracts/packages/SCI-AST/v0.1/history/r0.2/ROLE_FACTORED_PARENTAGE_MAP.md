# SCI-AST v0.1 Stage B r0.2 Role-factored Parentage Map

Status: targeted author draft; the identities below are prospective scientific
authority, not implementation or validation evidence

| Role | Exact parent extension | Independent availability consequence |
| --- | --- | --- |
| Detector sky direction | `(observation, detector occurrence, stable ALIGN slot (o,s), SCI-ALIGN_TO_SCI-AST v0.1/r0.1 relation/plan/grid/source, observing state, pointing correction, geometry/rotation realization, output frame)` | Does not depend on WCS or pixel materialization. A WCS failure cannot erase a valid direction. |
| Tangent-plane coordinate | `(Pi_direction, selected physical center, tangent projection identity)` | Does not depend on WCS. A WCS or pixel failure cannot erase a valid tangent fact. |
| Continuous FITS pixel | `(Pi_tangent, exact immutable WCS identity)` | WCS failure makes this role unavailable but leaves direction and tangent roles intact. |
| Nominal discrete pixel | `(Pi_pixel, rounding rule, dimensions, bounds convention)` | Rounding/bounds failure leaves the continuous pixel and upstream roles intact. |
| RTC-output-grid coordinate | `(appropriate direction/tangent/pixel role, exact RTC product, plan, grid, stable output n, representative stable ALIGN slot, output time, phase/delay, segment, decimation, support, response and correction state)` | Missing RTC facts affect only the dependent RTC role/response/delegated deposition, not unrelated ALIGN-grid roles. |

## Atomicity And Failure

Each available role atomically publishes its value, role validity and exact
reason, layered parent, uncertainty/Jacobian availability, and four-stage
provenance. A product family may separately require an atomic complete product
and fail that requested product if a role is missing. Whole-product failure
does not retroactively delete valid upstream scientific facts or rewrite their
causes.

The canonical locations are Equations `align-parent`, `tangent-parent`,
`pixel-parent`, `nominal-pixel-parent`, `rtc-parent`, and `atomic-bundle`, with
normative coverage in REQ-056-060 and REQ-073-077 and falsifiers in PRED-025,
030, 035, and 036.
