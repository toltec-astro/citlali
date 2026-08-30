# SCI-NOI v0.1 r0.3 Shared Normative Module: Notation

Module identity: `SCI-NOI_NORMATIVE_NOTATION v0.1/r0.3`

Scientific owner: Grant Wilson

Status: implementation-blind Stage B draft authority; content-bound by
`SCI-NOI_NORMATIVE_MODULE_BINDING v0.1/r0.3`; not frozen.

| Symbol | Exact meaning |
| --- | --- |
| `i` | One exact retained PTC occurrence with immutable observation, detector/channel, stable sample, segment, array/network, time, validity, ancestry, and provenance identity. |
| `d(i)` | Stable observation-scoped detector/channel owning occurrence `i`. |
| `d` | One stable realized detector/channel in one exact observation; the ordinary coherence unit. |
| `h(d)` | Exact stable readout-network stratum of `d`; missing or ambiguous membership is unavailable for positive-mass detectors. |
| `b` | One admitted realization-member identity in one exact design generation. |
| `p` | One exact frozen MAP output row/pixel on its declared WCS, support, response, validity, and lifecycle domain. |
| `Z_i^PTC` | Exact PTC transformed signal occurrence. |
| `gamma_i` | Exact PTC-owner-selected MAP-facing coefficient, including family, generation, value, and QC state. |
| `G_pi` | Exact frozen MAP projection/gridding contribution. |
| `a_pi` | Exact frozen positive pre-normalization contribution `G_pi gamma_i`. |
| `C_p` | Exact frozen set of admitted PTC occurrences contributing to row `p`. |
| `Q_p` | Exact unsigned frozen MAP normalization `sum_{i in C_p} a_pi`; never signed or relearned by NOI. |
| `beta_d` | Detector coefficient mass `sum_p sum_{i in C_p,d(i)=d} a_pi`. |
| `D_h^+` | Exact active design population `{d : beta_d > 0 and exact h(d)=h}`. |
| `epsilon_bd` | Sign for active detector `d` in member `b`, in `{-1,+1}`. It is undefined for nonactive detectors. |
| `tau_h` | Exact plan-bound balance tolerance for active stratum `h`, in `[0,1)`, with no default. |
| `mathcal_D` | Complete finite design identity: law or measure, population/order, alphabet, probabilities/weights, relations, mechanics, counts, ranks/null spaces, reconstruction, lifecycle, generation, completion, and failure. |
| `D_epsilon_b` | Occurrence-domain diagonal operator obtained by expanding member `b`'s active detector assignment over the exact admitted occurrence population. |
| `M_b(p)` | NOI realization-map value produced by the exact frozen ordinary operator. |
| `m_MAP(p)` | Independently governed immutable ordinary SCI-MAP signal for the corresponding all-`+1` arithmetic, when scientifically realized. |
| `Vhat_cond(p)` | Product `conditional_detector_sign_randomization_marginal_second_moment` on the exact common all-member domain. |
| `rho_cond(p)` | Proposed reciprocal successor `1/Vhat_cond`; unavailable pending owner decision and exact successor binding. |
| `sigma_cond(p)` | Positive conditional scale `sqrt(Vhat_cond(p))`. |
| `zeta_cond(p)` | Standardized product `m_MAP(p)/sigma_cond(p)`, unit exactly `1`. |
| `N_requested` | Preregistered requested member count. |
| `N_resolved` | Successfully resolved member count. |
| `N_completed` | Members reaching complete producer terminal state. |
| `N_admitted` | Members admitted to one exact named UNC use. |
| `N_unique_sign` | Byte-identical-distinct canonical active assignment vectors. |
| `N_unique_orbit` | Distinct exact complement orbits `{epsilon,-epsilon}`. |
| `r_sign` | Uncentered active sign-matrix rank for the known-zero-center method. |
| `r_map` | Exact rank, rank limit, or unavailable state after named map-domain projection. |
| `I_attempt` | Execution-attempt/evidence identity; may change during an idempotent retry without changing scientific identity. |

Inactive r0.1/r0.2 spellings appear only in immutable source quotations or the
change map. They are not parallel active identifiers.
