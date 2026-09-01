# SCI-MAP_TO_SCI-FLT-MATCHED v0.1/r0.4

Status: second-review boundary repair draft; no realized input, implementation,
conformity, validation, or production claim

Producer: `SCI-MAP`

Consumer: `SCI-FLT-MATCHED`

## Admitted parent role

The producer supplies exactly one immutable normalized ordinary-MAP observation
bundle or one immutable normalized ordinary-MAP coadd bundle. The boundary
carries parent identity, class, generation, grouping, signal quantity and
units, WCS/frame/grid/shape/index/order/pixel-center convention, valid support,
missing/nonfinite policy, response identity, covariance availability and
population/conditioning identity, beam and CAL lineage, lifecycle, provenance,
and exact coadd membership when applicable.

`SCI-FLT-MATCHED` consumes but does not rewrite these facts. It derives one
base-v0.1 output anchor at each exact parent pixel center. For each anchor and
selected route, FLT declares `D_loc(p)` and `E_p` before local weight/operator
construction. Final `S_apply(p)` is derived only after the exact row exists.
MAP payload coordinates with final exact-zero row coefficients are not
dereferenced during Apply, while MAP authority and causal operator/state
lineage for any coordinate that influenced construction remain preserved.

Every fixed-state expectation, variance, reference covariance, and realized
covariance conditions on the complete frozen FLT condition `h=(g,theta)`.
Where a covariance scope is selected, FLT declares one fixed selector `P_C`
and requires `P_C F_g(m)` to be a finite numerical random vector almost surely
under the exact parent law. Success conditioning, censoring, pairwise deletion,
or a draw-dependent domain does not reinterpret MAP covariance authority.

Observation and coadd parents remain separate; no commutation, state sharing,
or FLT coaddition follows. Missing ordinary-MAP authority, invalid lifecycle,
unavailable required payload, or unresolved required facts make only the
dependent FLT route unavailable. An AO choice cannot manufacture or repair a
MAP parent. JINC and other derived parents are rejected by this boundary.
