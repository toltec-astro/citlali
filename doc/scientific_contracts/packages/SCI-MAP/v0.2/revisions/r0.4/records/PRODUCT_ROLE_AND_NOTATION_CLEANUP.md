# SCI-MAP v0.2/r0.4 product-role and notation cleanup

Status: **CANDIDATE parity record**. The shared core is normative.

| Concept | Exact r0.4 token or notation | Disposition |
| --- | --- | --- |
| Base observation role | `SCI-MAP:base_observation_map@1` | New durable role identity. |
| Response-bearing observation role | `SCI-MAP:response_bearing_observation_map@1` | New durable role identity. |
| Covariance-qualified observation role | `SCI-MAP:covariance_qualified_observation_map@1` | New durable role identity. |
| Base coadd role | `SCI-MAP:base_coadd@1` | New durable role identity. |
| Response-bearing coadd role | `SCI-MAP:response_bearing_coadd@1` | New durable role identity. |
| Covariance-qualified coadd role | `SCI-MAP:covariance_qualified_coadd@1` | New durable role identity. |
| MAP lifecycle | `requested/effective/observation_resolved/applied/realized` | One exact ordered lifecycle. |
| MAP attempt outcome | `succeeded/failed/not_produced` | Independent of MAP product realization and ECS artifact state. |
| MAP product realization | Exact immutable product identity and contents, or absent | Failure evidence is not a realized MAP product. |
| MAP occurrence policy | `SCI-MAP:map_upstream_admission@2` | Scientific predicates and identity unchanged; new candidate source binding. |
| MAP coadd policy | `SCI-MAP:observation_coadd_admission@2` | Candidate successor because role-qualified response/covariance admission changes the profile record. Historical `@1` is immutable and no alias. |
| Candidate profile state | `candidate/pending/pending/pending/unavailable/unavailable` | Semantics, owner approval, Registry registration, source binding, evaluability, and object-decision axes respectively. |
| One-hot projection | `SCI-MAP:one_hot_containing_pixel@1` | Unchanged. |
| Equal-observation coefficient | `SCI-MAP:uniform_observation_coadd_coefficient@1` | Unchanged; exact dimensionless `u_op=1`. |
| PTC publication | Exact realized transformed PTC product/publication for the bound occurrence, product/application generation, and selected product role under the PTC-to-MAP boundary | Undefined publication symbol `q` removed. |
| Response accumulator | `K_p` | Reserved for the MAP response/kernel accumulator. |
| Stacked observation covariance | `bold Sigma^obs` | Replaces the colliding bold `K_p` GLS-boundary notation. |
| Response-family stack | `T_obs^(R)` | Ordered exact member responses from one compatible family `R`. |
| Response-bearing coadd | `R_coadd,out^(R) = B_out T_obs^(R)` | Only for a fixed coadd state under an exact composition theorem; finite differences are not presumed Jacobians. |
| Procedure comparison record | `SCI-MAP-FULL-PROCEDURE-COMPARISON-SPACE` | Stable exact cross-reference for every numerical full-procedure difference. |
| MAP plan | `Pi` with stage-qualified plan where needed | Immutable operation/parameter plan; never the output-row selector. |
| MAP output-row selector | `J_out` | Selects exactly `S_out(Pi_eff)`; never a plan. |
| Above-one cut authorization | `coverage_cut_above_one_override_requested` | Typed plan fact with identity, authority, scope, stage values, cause, and provenance. |
| Requirement result | `pass/fail/blocked/not_applicable/not_assessed` | Separate from evidence-artifact realization. |
| Evidence-artifact realization | `complete/incomplete/failed/not_produced` | Separate from requirement result and from VAL decision realization. |
| NOI marginal moment | `conditional_detector_sign_randomization_marginal_second_moment` | Exact restricted NOI identity retained. |

No token in this table establishes selection, realization, compatibility,
source closure, implementation availability, or a stronger scientific claim.
