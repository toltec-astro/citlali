# SCI-MAP v0.2/r0.4 response-family-qualified coadd amendment

Status: **CANDIDATE scientific amendment; no response product or fidelity
result is asserted**.

For one exact response family `R`, `T_obs^(R)` is the ordered stack of exact
compatible member responses `R_o^(R)`, and

`R_coadd,out^(R) = B_out T_obs^(R)`

only when every member has the same exact response-family identity; source
domain, perturbation definition, basis, class, unit, normalization,
reference/gauge/null state, WCS/grid, and row maps are compatible; coadd
membership, centered-integer placement, `B_out`, support, and coefficient
state are fixed for the query; and an exact mathematical composition theorem
authorizes propagation through that fixed linear coadd operator.

`SCI-MAP:response_bearing_coadd@1` binds the exact family it carries. A
fixed-state response, PTC full-procedure finite difference, PTC+MAP re-resolved
response, coadd re-resolved/full-procedure response, and whole-chain response
are distinct families and cannot enter one unqualified stack. A finite
difference is not promoted to a Jacobian.

If a perturbation may change observation admission, coadd membership, support,
placement, coefficients, product role, or the coadd plan, it requires a
separate coadd full-procedure or coadd re-resolved family and its own complete
`SCI-MAP-FULL-PROCEDURE-COMPARISON-SPACE` record. That coadd-procedure response
and the whole-chain response are separately unavailable until authorized.
