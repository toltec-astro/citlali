# SCI-NOI v0.1 r0.2 Shared Normative Module: Notation

Module identity: `SCI-NOI_NORMATIVE_NOTATION v0.1/r0.2`

Status: implementation-blind Stage B draft authority; content-bound by
`SCI-NOI_NORMATIVE_MODULE_BINDING v0.1/r0.2`; not owner-accepted or frozen.

| Symbol | Exact meaning |
| --- | --- |
| `i` | One exact retained PTC occurrence, with immutable observation, detector/channel, stable RTC sample, product/application generation, segment, array/network/group, ancestry, time, validity, and provenance identity. |
| `d(i)` | Stable realized detector/channel identity owning occurrence `i`. |
| `d` | One stable realized detector/channel within one exact observation. It is the approved ordinary coherence unit. |
| `h(d)` | Stable readout-network stratum containing detector `d` for one exact observation. Missing or ambiguous membership is unavailable. |
| `b` | One admitted realization-member identity in one exact design generation. |
| `p` | One exact frozen MAP output-row/pixel identity on its declared WCS, frame, support, response, and lifecycle domain. |
| `Z_i^PTC` | Exact PTC transformed signal for occurrence `i`. The superscript and owner are mandatory. |
| `gamma_i` | Exact PTC-owner-selected MAP-facing coefficient for occurrence `i`, including family, generation, value, and QC state. |
| `G_pi` | Exact frozen MAP projection/gridding contribution from occurrence `i` to output row `p`. |
| `a_pi` | Exact frozen positive pre-normalization MAP contribution `G_pi gamma_i` on an admitted occurrence/row pair. |
| `C_p` | Exact frozen set of admitted PTC occurrences contributing to MAP output row `p`. |
| `Q_p` | Exact unsigned frozen MAP normalization `sum_{i in C_p} a_pi`. It is not a signal quantity and is never signed or relearned by NOI. |
| `beta_d` | Canonical r0.2 detector coefficient mass `sum_p sum_{i in C_p, d(i)=d} a_pi`. It replaces the r0.1 spelling `B_d` to avoid collision with member counts. |
| `epsilon_bd` | Canonical r0.2 sign assigned to detector `d` in member `b`, in `{-1,+1}`. It replaces the r0.1 spelling `s_bd`; source quantities never use this symbol. |
| `mathcal_D` | Complete finite assignment-design identity: target law, population, ordering, construction, counts, dependence, equivalence, rank, weights, keying, persistence, and generation. It replaces the ambiguous r0.1 bare `S`. |
| `M_b(p)` | NOI realization-map value for member `b` at row `p`, produced by the exact ordinary frozen MAP operator equation. It is not an ordinary MAP science product. |
| `m_MAP(p)` | Exact immutable normalized real-observation MAP signal, using the corresponding all-`+1` operation and retaining ordinary MAP product identity. It replaces the r0.1 spelling `q_MAP`. |
| `Vhat_cond(p)` | Initial zero-centered design-weighted conditional detector-sign-randomization second moment on the exact common all-member domain. It replaces the r0.1 spelling `V_hat_cond`. |
| `rho_cond(p)` | Proposed r0.2 successor spelling for the reciprocal `1/Vhat_cond(p)`. It is unavailable pending exact owner acceptance and successor profile/source binding. The owner-approved r0.18 legacy method identity remains immutable history. |
| `sigma_cond(p)` | Canonical positive denominator `sqrt(Vhat_cond(p))` on the exact finite-positive compatible domain. |
| `zeta_cond(p)` | Canonical r0.2 standardized product `m_MAP(p)/sigma_cond(p)`. It replaces the r0.1 spelling `S_cond` and has unit exactly `1`. |
| `N_requested` | Requested member count, preregistered before candidate or member output is inspected. |
| `N_resolved` | Member count in the successfully resolved design. |
| `N_completed` | Count of members that reached the producer's complete terminal state. |
| `N_admitted` | Count admitted to one exact named UNC method after member and ensemble policy evaluation. |
| `N_unique_sign` | Count of byte-identical-distinct canonical signed assignment vectors. |
| `N_unique_orbit` | Count of complement orbits `{epsilon,-epsilon}` on the identical ordered active domain. |
| `r_sign` | Exact rank of the method-declared sign-design operator on the active coherence domain. |
| `r_map` | Exact rank, or declared rank limit/unavailable state, after projection of the admitted design into the exact map-domain statistic or covariance representation. |
| `I_attempt` | Execution-attempt/evidence identity. It may change during an idempotent rerun without changing scientific member identity. |

Inactive r0.1 spellings appear only in the change map or exact immutable
upstream/profile quotations. They are not second active identities.
