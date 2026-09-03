# SCI-NOI v0.1 r0.5 Normative Equations
Module identity: `SCI-NOI_NORMATIVE_EQUATIONS v0.1/r0.5`
Scientific owner: Grant Wilson. Status: proposed final; not frozen.

`SCI-NOI-EQ-001` `a_pi=G_pi gamma_i`.

`SCI-NOI-EQ-002` `Q_p=sum_{i in C_p} a_pi`.

`SCI-NOI-EQ-003` `beta_d=sum_p sum_{i in C_p,d(i)=d} a_pi`; `D_h^+={d: beta_d>0 and exact h(d)=h}`.

`SCI-NOI-EQ-004` `abs(sum_{d in D_h^+} epsilon_bd beta_d) <= tau_h sum_{d in D_h^+} beta_d`, `0<=tau_h<1`.

`SCI-NOI-EQ-005` `M_b(p)=[sum_{i in C_p} a_pi epsilon_b,d(i) Z_i^PTC]/Q_p`; the sign occurs exactly once only on `Z_i^PTC`, and every other MAP fact/gate is frozen.

`SCI-NOI-EQ-006` `m_MAP(p)=[sum_i a_pi Z_i^PTC]/Q_p` only when independently governed SCI-MAP is scientifically realized; NOI cannot manufacture it by all-plus-one assignment.

`SCI-NOI-EQ-007` `R_b^fixed=A_MAP,Pi D_epsilon_b H_PTC^fixed`; required response companions have identical membership/sign occurrence.

`SCI-NOI-EQ-008` `E_D[epsilon_bd]=0` for active `d`; `E_D[M_b(p)|fixed parent/operator]=0` on exact available rows.

`SCI-NOI-EQ-009` `initial_UNC_realized` implies `A_UNC=B_resolved` and `N_admitted=N_completed=N_resolved=N_requested>0`. No equality is asserted for a failed or unavailable UNC state.

`SCI-NOI-EQ-010` When `initial_UNC_realized`, `Vhat_cond(p)=(1/N_resolved) sum_{b in B_resolved} M_b(p)^2` on the exact common all-member domain. Otherwise no estimator or changed divisor exists.

`SCI-NOI-EQ-011` For canonical stratum order, `A=product_{h active} A_h`. Before key resolution, conditional on complete assignment-design resolution, the declared ideal model is `epsilon_1,...,epsilon_N iid ~ Uniform(A)`. After binding exact key `K`, the realized sequence is deterministically `epsilon_1,...,epsilon_N=F(K,plan)`.

`SCI-NOI-EQ-012` For rational left numerator `L_h=n_L/q_L`, total `T_h=n_T/q_T`, and `tau_h=n_tau/q_tau`, admission is exact integer comparison `n_L q_T q_tau <= n_tau n_T q_L` after nonnegative canonical reduction.

`SCI-NOI-EQ-013` `sigma_cond=sqrt(Vhat_cond)` and `zeta_cond=m_MAP/sigma_cond` on the exact finite-positive compatible support.

`SCI-NOI-EQ-014` `delta zeta_fixed_scale=diag(1/sigma_cond) delta m_MAP`, only with scale explicitly held fixed.

`SCI-NOI-EQ-015` `delta zeta=delta m_MAP/sigma_cond - m_MAP delta Vhat_cond/(2 sigma_cond^3)`; the second term requires separately authorized complete-procedure response.

`SCI-NOI-EQ-016` `decode(n_pi/q_pi)=a_pi^MAP` exactly, with the rational bound to the exact frozen MAP product/generation, `G_pi`, `gamma_i` family/generation/payload, representation source, and source digest.

`SCI-NOI-EQ-017` For every `e in A`, `P(R_b=1 | epsilon_b=e, fixed parent and plan)=P(R_b=1 | fixed parent and plan)`; deterministic predicates may satisfy this by exact invariance.
