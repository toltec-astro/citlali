# SCI-NOI v0.1 r0.5 Normative Notation
Module identity: `SCI-NOI_NORMATIVE_NOTATION v0.1/r0.5`
Scientific owner: Grant Wilson. Status: proposed final; not frozen.

| Symbol | Exact meaning |
| --- | --- |
| `i,d,h,b,p` | Exact retained occurrence, observation-scoped detector/channel, exact network, member, and frozen MAP row identities. |
| `Z_i^PTC,gamma_i,G_pi,a_pi` | PTC signal, exact MAP-facing coefficient, frozen projection, and positive contribution `G_pi gamma_i`. |
| `C_p,Q_p` | Exact contributing occurrences and unsigned normalization `sum_i a_pi`. |
| `beta_d,D_h^+` | Exact detector mass and `{d: beta_d>0 and exact h(d)=h}`. |
| `epsilon_bd,tau_h` | Active detector sign and exact no-default tolerance in `[0,1)`. |
| `(n_x,q_x),decode` | Canonical reduced arbitrary-precision rational for design quantity `x` and its lossless decoding; `q_x>0`, gcd one, zero only `(0,1)`. |
| `A_h,A` | Exact admitted assignment set for active stratum `h`, and `A=product_h A_h` under canonical stratum concatenation. |
| `K,F(K,plan)` | Bound opaque key and deterministic replay map from key/counters to the realized assignment sequence. |
| `R_b` | Conjunction of binding completion and UNC-member-admission facts for member `b`; fixed before assignment or invariant under it. |
| `B_resolved,A_UNC` | Actual resolved member-identity set and initial-UNC admitted set; equality is required only when initial UNC is realized. |
| `D_epsilon_b` | Occurrence-diagonal expansion of the active assignment. |
| `M_b,m_MAP` | NOI member and independently governed ordinary SCI-MAP signal. |
| `Vhat_cond,sigma_cond,zeta_cond` | Conditional marginal second moment, its square-root scale, and unit-one standardized MAP. |
| `N_requested,N_resolved,N_completed,N_admitted` | Actual requested, resolved, completed, and initial-UNC admitted counts; failed/unavailable states may have unequal values. |
| `N_unique_sign,N_unique_orbit,r_sign,r_map` | Distinct assignments, complement orbits, uncentered sign rank, and named projected rank. |
| `I_attempt` | Execution-attempt identity, distinct from scientific identity. |
| `I_request,I_effective` | Exact request and effective-plan identities; absent for `not_requested`, both present for `explicitly_disabled`. |
| `initial_UNC_realized` | Typed realization state whose truth implies exact all-member set/count equality and estimator existence. |
