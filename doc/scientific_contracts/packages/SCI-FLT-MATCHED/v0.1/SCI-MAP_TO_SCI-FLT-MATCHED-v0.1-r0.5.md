# SCI-MAP_TO_SCI-FLT-MATCHED v0.1/r0.5

Status: final targeted type-closure boundary draft; no realized input,
implementation, conformity, validation, or production claim

Producer: 'SCI-MAP'

Consumer: 'SCI-FLT-MATCHED'

Provisional package-title status: 'optimality_status=not_claimed'; this
boundary title and method ID are not optimality evidence.

MAP supplies exactly one immutable normalized ordinary-MAP observation bundle
or ordinary-MAP coadd bundle. It binds parent identity/class/generation,
grouping, signal quantity and units, WCS/frame/grid/shape/index/order/pixel
centers, the row-identity/fact domain 'S_parent_fact', numerical payload domain
'D_m', valid support, missing/nonfinite policy, response and covariance
authority, beam/CAL lineage, lifecycle, provenance, and exact coadd membership
when applicable.

FLT consumes but does not rewrite those facts. For each parent-pixel anchor it
predeclares 'D_loc(p)' and 'E_p', constructs 'ell_p*' and
'c_p=E_p^T ell_p*', and numerically applies only coordinates in
'S_apply(p)={q:c_pq!=0}'. Every active coordinate must lie in 'D_m'. A
construction-only coordinate with exact-zero final coefficient needs no
payload dereference; exact zero is not a numerical tolerance.

All U1 statements condition only on pre-draw
'h_pre=(g_resolved,theta)'. Attempt, outcome, realization, and publication
facts are provenance. Observation and coadd remain separate; no commutation,
state sharing, FLT coaddition, or fallback follows. JINC and other derived
parents are rejected.
