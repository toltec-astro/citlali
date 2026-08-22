# SCI-AST v0.1 Stage B r0.3 Notation and Symbol Change Map

Status: targeted author-draft traceability; not scientific approval,
implementation conformity, validation, freeze, or readiness.

| Symbol or identity | r0.3 meaning | Guardrail |
| --- | --- | --- |
| `s` | stable SCI-ALIGN detector-reference slot, stable as `(o,s)` | Never reconstructed from local row `j`. |
| `j` | local storage row only | Never an external scientific parent. |
| `n` | stable SCI-RTC output sample | Binds exact RTC product/grid parent. |
| `d` | detector occurrence or stable detector identity | Not a storage row. |
| `p` | map pixel | Never a detector. |
| `x,r` | paired KID readout coordinates | Never sky-vector or TAN-axis symbols. |
| `u_sky` | unit sky direction | Carries declared spherical frame. |
| `t`, `rho`, `v_hat` | exact tangent vector, its norm, and unit tangent direction | `v_hat=t/rho` only for `rho>0`; zero returns `u_sky`. |
| `theta^A_ds` | AST coordinate on exact ALIGN grid | Parent is `(o,d,s)` plus role-specific facts. |
| `theta^RTC_dn` | AST coordinate on exact RTC output grid | Stable role is `SCI-AST:rtc_output_grid_coordinates@1`; phase-zero equality never erases RTC parent. |
| `B^AST_i` | base pre-MAP continuous pixel, optional nominal pixel, and bounds tuple | Contains no kernel-dependent neighborhood or estimator support. |
| `G_pi` | MAP-owned sample-to-pixel deposition/gridding operator | AST may materialize only for an exact MAP-owned request. |

The shared import identity is exactly `SCI-ALIGN_TO_SCI-AST v0.1/r0.1`.
No stable requirement or prediction identifier was renumbered.
