# Stochastic-model and observed-payload domain amendment — r0.6

Status: frozen normative micro-repair

The contract distinguishes:

- `S_parent_fact`: exact parent row/fact domain;
- `D_model`: coordinate domain of one authorized stochastic model;
- `M:D_model->R`: parent random vector under one exact declared law;
- `C_parent|h_pre=Cov[M|h_pre]`;
- `D_m`: parent-fact coordinates carrying admitted finite observed payload; and
- `m_obs:D_m->R`: one immutable observed MAP payload.

For `AO-001-A`, `D_loc(p) subset D_model`, `M_p=E_p M`, and
`C_p=Cov[M_p|h_pre]=E_p C_parent|h_pre E_p^T`. Missing model or covariance
authority anywhere on `D_loc(p)` makes the GLS theorem, optimality claim, and
`v_GLS,reference=d_p^-1` unavailable.

Actual application requires `S_apply(p) subset D_m` and uses only
`sum_{q in S_apply(p)} c_pq m_obs_q`. A construction coordinate may lack
observed payload only when its exact final application coefficient is zero and
all required construction authority exists. This is neither imputation nor a
partial-support estimator.
