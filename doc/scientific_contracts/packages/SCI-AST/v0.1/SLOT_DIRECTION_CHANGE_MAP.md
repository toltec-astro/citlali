# SCI-AST v0.1 Stage B r0.3 Slot And Direction Change Map

Status: targeted author-draft traceability; not scientific approval,
implementation conformity, validation, freeze, or readiness

## Canonical Symbol Changes

| Prior draft use | r0.3 canonical use | Scope and invariant |
| --- | --- | --- |
| `j` as stable ALIGN slot | `s` as stable detector-reference slot | Stable identity is `(observation,s)`; it is independent of storage row and numerical array position. |
| `j` in correction/geometry time subscripts | `s` | `c_s`, `m_s`, `t_s`, `xi_ds`, `B_ds`, and every ALIGN-grid parent bind the stable slot. |
| `theta^A_dj` | `theta^A_ds` | Applies to definitions, parent equations, grid-role equations, requirements, predictions, response support, and prose. |
| `j_mem` as local column | local row `j`, local column `h` | `j` is local storage row only and is never external identity. FITS relation is `q_1=h+1`, `q_2=j+1`. |
| `n` | `n` | Unchanged: stable RTC output-sample identity. |
| sky vector `r` | unit sky direction `u_sky` | Applied to direction/basis, exponential map, boresight/detector/frame direction, TAN, Jacobian, covariance-domain prose, and predictions. |
| TAN axes `x,y` | `zeta_1,zeta_2` | Prevents TAN axes from colliding with readout `x`; `w` remains the unit-bearing tangent-plane coordinate. |
| paired KID readout `x,r` | paired KID readout `x,r` | Exclusively reserved for Tune/readout and ALIGN signal-coordinate relations. Neither coordinate is thereby a Stokes parameter; raw KID `x` is not identified as Stokes I. |

## Cross-domain Audit

| Domain | r0.3 binding |
| --- | --- |
| Exact ALIGN import | Profile `SCI-ALIGN_TO_SCI-AST v0.1/r0.1`; reference-grid transfer binds stable `(o,s)`. |
| Pointing support | Interpolation is indexed by stable slot occurrence time (`lambda_s`, `c_s`), never local row. |
| Detector geometry | Geometry realization and ordered direction composition use `xi_ds` and `u_sky,ds`. |
| Frame/TAN | Unit directions use `u_sky`; TAN dimensionless axes use `zeta_1,zeta_2`; `x/r` do not appear as sky variables. |
| WCS/pixels | `w` and `q` retain their types; local `(j,h)` never enters external parentage. |
| RTC | `theta^RTC_dn` extends the appropriate `theta^A_ds` role; representative ALIGN identity is stable slot, not row. |
| Response | Full support is `(L_ns,{theta^A_ds})`; direct angular signal filtering remains prohibited. |
| Predictions | PRED-018, 022, and 035-037 carry the corrected topology, sky symbol, stable slot, and response fixtures. |

No stable requirement or prediction identifier was renumbered. This map
records a semantic identity repair, not evidence that any representation or
implementation conforms.
