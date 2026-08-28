# SCI-MAP v0.1 r0.6 Response-Role Crosswalk

| Response object | State resolution | Equation and domain | Permitted claim |
| --- | --- | --- | --- |
| Fixed-state upstream response | PTC/MAP membership, masks, groups, rank, loading subspace, coefficients, support, and projection fixed | `delta m=A_MAP,Pi H_fixed,Theta delta s`; exact source domain to exact MAP output rows | Linear conditional response only for the bound state/basis |
| PTC full-procedure response | PTC procedure rerun on `s` and `s+delta s`; family parameter `alpha` binds amplitude, side, source state, location, and domain | `Delta z_FP=P_PTC(s+delta s;alpha)-P_PTC(s;alpha)` plus separate typed state-change record; if MAP state is fixed, `Delta m_FP=A_MAP,Pi Delta z_FP` | Bounded finite-difference family, not one `H` matrix |
| Whole-chain re-resolved response | PTC and MAP membership/coefficient/projection/support may change | difference of complete re-resolved MAP products with each state recorded | Procedure response only; not the fixed MAP operator |
| Realized PTC-grid companion | Begins in PTC output-sample domain | fixed MAP operator applied exactly once | Companion response on exact common membership; never passed through upstream response again |

Missing response for any signal member never creates a hidden smaller signal
membership. A response-bearing role fails or reports unavailable according to
its exact policy; a response-independent base role may preserve the numerical
signal with honest unavailable response.
