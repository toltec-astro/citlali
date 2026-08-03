# SCI-MAP-002 JINC conditioning owner decision — 2026-08-03

Status: owner approved contract clarification; no implementation work
authorized

Package: `SCI-MAP-002`

Decision ID: `SCI-MAP-002-D003-CONDITIONING-001`

Authority: project owner

## Decision

JINC support and formal-weight conditioning must be invariant under a change
of signal units. The current absolute `|C| > 1e-8` gate and `Q` floor are not
the future contract.

For each pixel, define the dimensionless cancellation ratio

\[
\rho = \frac{|\sum_i q_i c_i|}{\sum_i |q_i c_i|}.
\]

Future JINC admission must require finite contributors, finite accumulated
`C` and `Q`, and `Q > 0`. Exact cancellation is invalid. A finite,
nonzero `C` is treated as numerically unresolved only when `rho` is below a
documented floating-point error bound derived from the realized summation
method and contributor count. The bound, summation method, and realized
admission policy must be serialized.

When cancellation is numerically resolved, retain the pixel even if its
formal weight is small; the estimator `C^2/Q` is the correct expression of
that low formal precision. Stable internal scaling may prevent overflow or
underflow, but it must preserve the estimator and may not introduce a
unit-bearing `Q` floor. This decision does not require a per-pixel diagnostic
product or per-sample identifier.

Future validation must cover exact cancellation, numerically unresolved
near-cancellation, resolved low-weight cancellation, unit scaling, and
extreme finite `Q`. No code change, Unity evidence, repair, re-audit, or
production-status change is authorized.
