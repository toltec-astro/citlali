# SCI-NOI v0.1 r0.4 Proposed Profile Successors

Record identity: `SCI-NOI_PROPOSED_PROFILE_SUCCESSORS v0.1/r0.4`

Scientific owner: Grant Wilson

Status: complete proposed policy bytes; not owner-approved; not Registry-bound;
not evaluable. These records do not modify or alias the immutable r0.18
profiles.

## SCI-NOI:generation_input_admission@2

| Field | Exact proposed value |
| --- | --- |
| Object and domain | One exact retained PTC occurrence in one exact observation and TolTEC array for `NOI-GEN/PTC-TO-FROZEN-MAP-CONDITIONAL-SIGN@1`. |
| Policy owner | Grant Wilson. |
| Scientific sources | r0.18 owner decisions; r0.3 owner supplement; r0.4 owner directive; byte-bound r0.4 normative modules. |
| Required facts | Explicit enabled request; positive integer `N_requested`; exact observation, array, occurrence, detector/channel, network, coefficient family/value/QC, canonical rational `a_pi`, stable identities/order, PTC validity/support/response, frozen MAP projection/denominator/gates, numerical `coverage_cut`, exact plan and generator/key identities, lifecycle and provenance. |
| Exclusions | Zero-mass occurrence from active signing; ambiguous positive-mass network identity; missing/conflicting/noncanonical fact; disabled request; cross-observation or cross-array ensemble; any changed parent/operator; any undocumented default or floating admission decision. |
| Exact action | Admit this exact occurrence only as a candidate input to the named ordinary GEN route. This action does not assign a sign, resolve a design, realize a member, admit UNC, or authorize STD. |
| Missing behavior | Missing profile authority or any required fact yields decision `unavailable`; explicit scientific nonmembership yields `ineligible`; disabled request yields `not_requested`. No empty success. |
| Lifecycle | Evaluate in a new profile-evaluation generation before assignment inspection; bind result to exact source, plan, parent, observation, and array; immutable after use. |
| Supersession | Proposed successor to `SCI-NOI:generation_input_admission@1`; no alias or backward reinterpretation. |

## SCI-NOI:uncertainty_member_admission@2

| Field | Exact proposed value |
| --- | --- |
| Object and domain | One exact resolved and completed NOI member for the base conditional marginal-second-moment estimator on its exact common-domain candidate. |
| Policy owner | Grant Wilson. |
| Scientific sources | r0.18 owner decisions; r0.3 owner supplement; r0.4 owner directive; byte-bound r0.4 normative modules. |
| Required facts | Exact GEN method/design/member identities; `N_resolved=N_requested>0`; member assignment and attempt identity; completion success; one-time sign application; exact parent/operator parity; support/WCS/response/unit; common-domain membership; duplicate and complement-orbit facts; QC, persistence, lifecycle, cause, and provenance. |
| Exclusions | Rejected candidates; unresolved, unrealized, incomplete, failed, or wrong-method members; hidden survivor selection; pairwise or missing-data subsets; changed operator/parent; missing profile authority or required fact. |
| Exact action | Admit this exact completed member as a candidate member for `conditional_detector_sign_randomization_marginal_second_moment` on the named common domain. Ensemble admission remains separate. |
| Missing behavior | Missing authority or facts yields `unavailable`; explicit policy failure yields `ineligible` and thereby makes the complete base ensemble unavailable. No survivor normalization. |
| Lifecycle | Evaluate once for the exact member/use generation after GEN completion and before ensemble admission; immutable result with exact cause. |
| Supersession | Proposed successor to `SCI-NOI:uncertainty_member_admission@1`; no alias or name-only compatibility. |

## SCI-NOI:uncertainty_ensemble_admission@2

| Field | Exact proposed value |
| --- | --- |
| Object and domain | The one exact complete resolved ensemble for the named base estimator and exact common all-member domain. |
| Policy owner | Grant Wilson. |
| Scientific sources | r0.18 owner decisions; r0.3 owner supplement; r0.4 owner directive; byte-bound r0.4 normative modules. |
| Required facts | `A_UNC=B_resolved`; `N_admitted=N_completed=N_resolved=N_requested>0`; every resolved member positively admitted by `uncertainty_member_admission@2`; exact common domain; equal weights `1/N_resolved`; known target center zero; conditional iid `Uniform(A)` law; all count/rank/orbit reports; estimator identity, target, units, representation, uncertainty state, lifecycle and provenance. |
| Exclusions | Failed design; partial or survivor ensemble; any ineligible/unavailable/incomplete/failed/unrealized member; rejected candidates; pairwise/missing-data/reweighted estimator; reciprocal, covariance, precision, physical-noise, or significance use. |
| Exact action | Admit the complete ensemble only to compute `Vhat_cond=(1/N_resolved) sum_b M_b^2` on the exact common domain. This action neither realizes the estimator nor authorizes STD or another use. |
| Missing behavior | Missing authority or required fact yields decision `unavailable`; there is no numerical ineligibility substitute, empty product, subset normalization, or divisor change. |
| Lifecycle | Evaluate once after all member decisions and before estimator realization; bind exact member set, domain, method, plan, target, and generation; immutable after use. |
| Supersession | Proposed successor to `SCI-NOI:uncertainty_ensemble_admission@1`; no rewrite of its r0.18 action. |

## SCI-NOI:standardization_admission@2

| Field | Exact proposed value |
| --- | --- |
| Object and domain | Exact independently realized `m_MAP` joined with exact compatible `Vhat_cond` for `zeta_cond=m_MAP/sqrt(Vhat_cond)` on the finite-positive support intersection. |
| Policy owner | Grant Wilson. |
| Scientific sources | r0.18 owner decisions; r0.3 owner supplement; r0.4 owner directive; byte-bound r0.4 normative modules. |
| Required facts | Independently governed SCI-MAP identity/admission; conditional-second-moment identity/admission; common immutable observed-parent ancestry; exact units, WCS, support/response intersection, finite-positive scale, numerator/scale dependence, response state and cause, lifecycle and provenance. |
| Exclusions | All-plus-one manufactured numerator; interpolation or support extension; implicit reciprocal product; JINC substitution; missing profile authority; missing/nonfinite/nonpositive parent; significance, probability, detection, false-alarm, completeness, purity, or catalog use. |
| Exact action | Permit only the unit-one product `zeta_cond` with claim “MAP signal standardized by the stated conditional detector-sign-randomization second-moment scale.” |
| Missing behavior | Missing authority or required fact yields `unavailable`. Full STD procedure response remains `unavailable` until separately authorized `delta Vhat_cond` exists. Fixed-scale response, if requested and authorized, remains separately labeled and cannot be relabeled full. |
| Lifecycle | Evaluate once for the exact numerator/scale/use generation; bind dependence, support, response state, and cause; immutable after use. |
| Supersession | Proposed successor to `SCI-NOI:standardization_admission@1`; no alias from r0.18 notation or action. |

No reciprocal-use successor is proposed. Every reciprocal, inverse-variance,
precision, or consumer-weight action remains unavailable pending a separate
exact owner decision, complete profile, and Registry/source binding.
