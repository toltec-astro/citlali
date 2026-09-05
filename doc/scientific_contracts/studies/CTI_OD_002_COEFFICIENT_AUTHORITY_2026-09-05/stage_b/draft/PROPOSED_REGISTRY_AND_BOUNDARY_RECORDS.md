# Proposed Registry, profile, and boundary records

Draft authority: **SCI-PTC-COEFFICIENT-UNIFORM v0.1-draft.1 / r0.1**  
Status: **exact author proposal; unavailable for evaluation or use**

The exact draft source set is `src/common_core.tex` plus the six files under
`src/common/`. Their hashes are recorded in
`DRAFT_ARTIFACT_MANIFEST.sha256`. Owner adoption must bind a promoted,
immutable source digest; this draft reference does not activate a record.

## Proposed family record

| Field | Proposed value |
| --- | --- |
| Registry identity/version | `SCI-PTC:analysis_gridding_coefficients@draft-0.1` |
| Family identity/version | `SCI-PTC:uniform_constant@draft-0.1` |
| Scientific producer/package owner | `SCI-PTC` |
| Proposed actual scientific owner | Grant Wilson acting for SCI-PTC; confirmation remains open as UFC-Q-001 |
| Source identity | `SCI-PTC-COEFFICIENT-UNIFORM v0.1-draft.1 / r0.1`; exact draft file hashes in the artifact manifest |
| Logical index | Exact detector-time occurrence `i` |
| Broadcast | One exact scalar `1` may represent every and only exact identity in `D_(g,o)`; the complete identity relation is mandatory |
| Raw factor/statistic | Fixed `c*=1`; no measured statistic or configurable parameter |
| Unit | Dimensionless `1` |
| Normalization | Exact arithmetic-mean-to-one operator on `D_(g,o)` |
| Domain/population/support | One observation and coefficient generation; exactly the bound PTC product generation’s output support; finite and nonempty |
| Parent/product compatibility | Exact PTC transformed product/application generation, immutable CAL parent, observation, occurrence identities, and selection lineage; no inferred compatibility |
| Lifecycle | Requested, effective-policy, observation-resolved, selected, realized, published, and evaluated identities retained and one-way linked |
| Availability/QC | Typed coefficient availability plus exact `SCI-PTC:uniform_coefficient_handoff@draft-0.1` evaluation |
| Uncertainty | No sampling uncertainty in the exact constant; no assertion about physical, response, covariance, selection, representation, or performance uncertainty |
| Proposed SCI-MAP permission | Explicitly proposed, independently open under UFC-OD-008 |
| Proposed SCI-JINC permission | Explicitly proposed, independently open under UFC-OD-009 |
| Numerical use | Positive PTC analysis/gridding factor in each consumer’s existing arithmetic |
| Prohibited meanings | Precision, inverse variance, sensitivity, significance, loading, scatter, validity, exposure, response, covariance, independence, optimality, default, support restoration, or missing-value replacement |
| Missing/conflict rule | Typed unavailable or decision unavailable at the stated scope; no unity or alternate-family fallback |
| Compatibility/supersession | No alias and no retroactive change. Every material change creates a successor; cross-generation use needs an explicit record |

## Proposed complete VAL profile record

| Registry component | Proposed exact binding |
| --- | --- |
| `I_rho, v_rho` | `SCI-PTC:uniform_coefficient_handoff@draft-0.1` |
| `U_rho` named use | For one requested `(g,o,i,c)`, decide whether a realized PTC payload faithfully instantiates the exact uniform family and is authorized for handoff to named consumer `c` |
| `O_rho` actual owner | Proposed: Grant Wilson acting for SCI-PTC; owner confirmation required |
| `S_rho, h_rho` | `SCI-PTC-COEFFICIENT-UNIFORM v0.1-draft.1 / r0.1`; the promoted source digest must replace the draft artifact-manifest reference before activation |
| Bound common semantics | Admitted SCI-PTC common named-use fragment, exact digest `c1fc8370007b65307769fb966c8523251695924aaff84f3e5b4c89b6d3380b8c` |
| `D_rho` domain/object | One coefficient occurrence plus its generation-level scalar/broadcast, one observation, one exact PTC product/coefficient generation, and one named consumer |
| `A_rho` applicability | Applicable iff exact product exists, exact population is authoritatively nonempty, and `i` is a member; inapplicable iff authoritatively outside; otherwise applicability unknown |
| `G_rho` structural gate | Active exact Registry/source/owner plus complete consistent occurrence, observation, product, parent, family, generation, selection, support, scalar, broadcast, profile, consumer, scope, influence, and provenance binding |
| `R_rho` restrictions | UQ-01 through UQ-07 exactly as defined in `src/common/definitions.tex` |
| Predicate input/truth rules | Exact input set and T/F/U/C rules in `src/common/definitions.tex`; UQ-05 is the independent PTC payload-fidelity primitive |
| `X_rho` exceptions | None; every restriction and structural invariant is nonexceptionable |
| `Y_rho` response/uncertainty roles | Both `advisory`; exact states preserved, neutral for coefficient fidelity |
| `Gamma_rho` aggregation | None. The profile is occurrence-level; scalar-broadcast failure scope does not create an aggregate eligibility profile |
| `L_rho` lineage | Exact requested, effective-policy, observation-resolved, selected, coefficient-realized/published, and evaluation identities, bound one way |
| `M_rho` missing/conflict behavior | Structural missing/conflict gives applicability unknown or decision unavailable; false after valid establishment gives ineligible; unresolved non-gating predicate gives decision unavailable absent a false; all causes retained |
| Four axes | Independent requested, applicability, eligibility, and profile-realization axes, including uninstantiated eligibility |
| `J_rho` compatibility/supersession | No alias; any source, owner, use, domain, restriction, exception, role, missing rule, or lifecycle change creates a new immutable record |
| Registry status | Draft/unbound/unevaluable. It is not the unsupported `SCI-PTC:coefficient_qc_population@1` placeholder |

## Proposed boundary successor records

### MAP

Proposed identity:
`SCI-PTC_TO_SCI-MAP_BOUNDARY v0.2-draft.1/r0.1`.

The only scientific change is the coefficient-slot binding to the exact
uniform family, explicit SCI-MAP permission, selected family/generation,
dimensionless exact-unity payload, complete profile decision, and their
parent/support/provenance identities. MAP retains every existing admission,
classification, arithmetic, coordinate, placement, exposure, response,
uncertainty, support, and failure rule. The draft cannot make the route
available.

### JINC

Proposed identity:
`SCI-PTC_TO_SCI-JINC_BOUNDARY v0.2-draft.1/r0.1`.

The only scientific change is the coefficient-slot binding to the same exact
uniform family, separately explicit SCI-JINC permission, selected
family/generation, dimensionless exact-unity payload, complete profile
decision, and their parent/support/provenance identities. JINC retains
ownership of `kappa_ip`, signed `w_ip`, normalization, conditioning,
support, and bundle semantics. No base-v0.1 JINC response, covariance, or
uncertainty product is created. The draft cannot make the route available.

## NOI conditional binding

The proposed family is conditionally compatible only through the existing MAP
host route. Its exact rational coefficient factor is `1/1`, but NOI still
requires a canonical lossless rational source for the exact complete MAP
parent `a_pi = G_pi gamma_i` and must bind the MAP product/generation,
`G_pi`, family/generation/payload, representation source, and source digest.
Missing any binding leaves design resolution unavailable. No separate PTC
NOI-balance family, JINC-hosted NOI route, profile approval, or altered design
law is proposed.

