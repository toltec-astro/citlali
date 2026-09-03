# SCI-MAP_TO_SCI-FLT-MATCHED v0.1/r0.6

Status: frozen v0.1/r0.6 stochastic-model/observed-payload scientific boundary;
no realized input, implementation, conformity, validation, or production claim

Producer: `SCI-MAP`

Consumer: `SCI-FLT-MATCHED`

Selected human title: **Matched-template map amplitude estimation**. Title and
method ID are not optimality evidence; each realization carries its exact
`optimality_status`.

MAP supplies exactly one immutable normalized ordinary-MAP observation bundle
or ordinary-MAP coadd bundle. It binds parent identity/class/generation,
grouping, signal quantity and units, WCS/frame/grid/shape/index/order/pixel
centers, the row-identity/fact domain `S_parent_fact`, valid support,
missing/nonfinite policy, response and covariance authority, beam/CAL lineage,
lifecycle, provenance, and exact coadd membership when applicable.

For stochastic statements, the boundary additionally identifies the exact
authorized coordinate domain `D_model`, random vector `M:D_model->R`,
stochastic-law/population identity, and
`C_parent|h_pre=Cov[M|h_pre]`. For an observed realization it separately binds
the numerical-payload domain `D_m` and immutable payload `m_obs:D_m->R`.
`S_parent_fact`, `D_model`, and `D_m` are distinct facts and need not be equal.

FLT consumes but does not rewrite those facts. For each parent-pixel anchor it
predeclares `D_loc(p)` and `E_p`, constructs `ell_p*` and
`c_p=E_p^T ell_p*`, and numerically applies only coordinates in
`S_apply(p)={q:c_pq!=0}`. Every active coordinate must lie in `D_m`. For
AO-001-A, every construction coordinate must lie in `D_model` and carry the
complete model/covariance authority. An exact-zero final coefficient removes
an observed-payload dereference only; it does not erase a construction-domain
template, model, or covariance dependency. Exact zero is not a tolerance.

All U1 statements condition only on pre-draw
`h_pre=(g_resolved,theta)`. The numerical payload and any digest fixing the
draw, execution success, draw-dependent identity/domain, publication,
censoring, and pairwise deletion are excluded. Observation and coadd remain
separate; no commutation, state sharing, FLT coaddition, or fallback follows.
JINC and other derived parents are rejected.
