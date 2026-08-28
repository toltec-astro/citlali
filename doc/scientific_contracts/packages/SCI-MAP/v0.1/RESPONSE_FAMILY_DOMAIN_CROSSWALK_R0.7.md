# SCI-MAP v0.1 r0.7 Response-Family And Domain Crosswalk

For selected frozen plan `Pi`, the one MAP operator is
`A_MAP,Pi == A_out == D_Q,out^-1 J_out G Omega`. It binds the admitted
PTC-sample input domain, support-authorized output rows, coefficient
generation, one-hot projection plan, support policy, WCS/grid, immutable
parents, and lifecycle generation.

| Response object | State resolution | Equation and domain | Permitted claim |
| --- | --- | --- | --- |
| Fixed-state upstream response | PTC/MAP membership, masks, groups, rank, loading subspace, coefficients, support, and projection fixed | `delta m=A_MAP,Pi H_fixed,Theta delta s`; exact source domain to exact MAP output rows | Linear conditional response only for the bound state/basis |
| PTC full-procedure response | PTC procedure rerun on `s` and `s+delta s`; `alpha` binds amplitude, side, source state, location, and domain | `Delta z_PTC-FP=P_PTC(s+delta s;alpha)-P_PTC(s;alpha)` plus separately typed `Delta S_PTC-FP`; if MAP state is fixed, `Delta m_PTC-FP=A_MAP,Pi Delta z_PTC-FP` | Bounded finite-difference family, not one `H` matrix; state record is not covariance |
| PTC+MAP re-resolved procedure response | PTC and MAP membership/coefficient/projection/support may change; exact `Pi_0` and `Pi_+` retained | difference of complete PTC+MAP re-resolved products with both states recorded | Procedure response only; not the fixed MAP operator and not a whole-chain claim |
| Whole-chain RTC-to-CAL-to-PTC-to-MAP response | Would require a separately authorized rerun of RTC, CAL, AST-dependent selection, PTC, MAP admission, coefficient/support resolution, and MAP from exact immutable parents | unavailable in r0.7 | Reserved name; no numerical or identity claim |
| Realized PTC-grid companion | Begins in PTC output-sample domain | fixed MAP operator applied exactly once | Companion response on exact common membership; never passed through upstream response again |

Missing response for any signal member never creates a hidden smaller signal
membership. A response-bearing role fails or reports unavailable according to
its exact policy; a response-independent base role may preserve the numerical
signal with honest unavailable response.
