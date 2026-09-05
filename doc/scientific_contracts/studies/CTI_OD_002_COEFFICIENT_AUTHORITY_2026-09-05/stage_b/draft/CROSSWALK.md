# Requirement, prediction, and evidence crosswalk

Draft: **SCI-PTC-COEFFICIENT-UNIFORM v0.1-draft.1 / r0.1**  
Status: **proposed for owner review**

Both rendered views include the same six files under `src/common/`.
“Scientist-facing source” identifies the canonical rationale location;
“Engineering interpretation” states the corresponding eventual conformance
reading without adding science.

| Requirement | Scientist-facing source | Engineering interpretation |
| --- | --- | --- |
| UFC-REQ-001 | `definitions.tex`, Common registry-entry contract | Bind the exact draft family/registry identities and SCI-PTC ownership; reject any claim that the draft is active. Evidence: source and Registry binding inspection. |
| UFC-REQ-002 | `equations.tex`, Occurrence index and broadcast | Verify full occurrence identities and exact scalar-to-member coverage; row/shape order cannot substitute. Predictions: UFC-PRED-001, 009. |
| UFC-REQ-003 | `equations.tex`, Support and population | Verify equality of coefficient support, normalization population, and exact single-observation PTC output support. Predictions: UFC-PRED-003, 004. |
| UFC-REQ-004 | `equations.tex`, Fixed constant and normalization | Decode exact dimensionless one and calculate exact mean-one normalization for every nonempty population. Predictions: UFC-PRED-001, 002, 007. |
| UFC-REQ-005 | `equations.tex`, Fixed constant and normalization | Reject a parameter field or altered constant/operator/population as this family. Prediction: UFC-PRED-006. |
| UFC-REQ-006 | `equations.tex`, Fixed constant and normalization | Accept singleton support; fail closed on empty, duplicate, ambiguous, or conflicting populations without repair. Predictions: UFC-PRED-002, 003, 009. |
| UFC-REQ-007 | `equations.tex`, Support and population | Represent outside-support state as absence/inapplicability, never numeric zero. Predictions: UFC-PRED-003, 010. |
| UFC-REQ-008 | `assumptions.tex`, Immutable lifecycle and repetition | Verify all immutable parents and one-way lifecycle links; ensure no prior product/generation is mutated. Predictions: UFC-PRED-004, 005. |
| UFC-REQ-009 | `assumptions.tex`, Immutable lifecycle and repetition | Replay exact values and membership while keeping each realized generation distinct; prohibit borrowing. Prediction: UFC-PRED-005. |
| UFC-REQ-010 | `definitions.tex`, Proposed boundary/profile successors | Inspect separate MAP and JINC permission fields and independently exercise absence of each. Prediction: UFC-PRED-013. |
| UFC-REQ-011 | `definitions.tex`, Proposed complete coefficient-handoff profile | Bind and evaluate the complete narrow profile; reject the unsupported generic QC placeholder. Evidence: Registry/profile identity check. |
| UFC-REQ-012 | `definitions.tex`, Identity, use, object, applicability, inputs, and restrictions | Check every required Registry field, exact input, truth-domain rule, missing behavior, no-exception declaration, and lineage field. |
| UFC-REQ-013 | `definitions.tex`, Profile inputs; `assumptions.tex`, decision semantics | Exercise coefficient fidelity separately from sample validity, payload availability, retention, consumer admission, and numerical contribution. Prediction: UFC-PRED-010. |
| UFC-REQ-014 | `assumptions.tex`, Knowledge and four axes | Test T/F/U/C composition, cause preservation, false dominance, and unresolved conflict without short-circuit erasure. |
| UFC-REQ-015 | `assumptions.tex`, Knowledge and four axes | Verify independent R/A/E/Z storage and the uninstantiated eligibility state. |
| UFC-REQ-016 | `definitions.tex`, Restriction set | Verify the profile exposes no exception capable of changing a structural invariant or UQ-01--UQ-07. |
| UFC-REQ-017 | `assumptions.tex`, Payload order and failure scope | Exercise every typed missing/invalid/conflict/zero/non-finite/unrepresentable state and declared scope. Predictions: UFC-PRED-003, 007--009. |
| UFC-REQ-018 | `assumptions.tex`, Scientific assumptions; `definitions.tex`, Profile restrictions | Preserve response/uncertainty facts as advisory and verify no identity response or zero uncertainty appears. |
| UFC-REQ-019 | `definitions.tex`, Physical meaning and common entry contract | Check metadata and consumer handoff contain every prohibited interpretation and no precision/sensitivity semantics. |
| UFC-REQ-020 | `definitions.tex`, MAP boundary proposal | Verify only the coefficient facet changes; MAP retains admission/arithmetic and performs independent finite-positive classification. Predictions: UFC-PRED-007, 011. |
| UFC-REQ-021 | `definitions.tex`, JINC boundary proposal; `equations.tex`, consumer consequences | Verify JINC retains ownership and obtains `w_ip = kappa_ip` only after its gates; no new response/covariance role. Prediction: UFC-PRED-012. |
| UFC-REQ-022 | `definitions.tex`, NOI compatibility limit; `equations.tex`, consumer consequences | Bind exact MAP/coefficient parents and lossless rational identity; withhold NOI when any binding is missing. Prediction: UFC-PRED-015. |
| UFC-REQ-023 | `definitions.tex`, No inference or fallback | Exercise missing selection and all prohibited fallback routes, including PTC-disabled. Predictions: UFC-PRED-008, 014, 016. |
| UFC-REQ-024 | `assumptions.tex`, Immutable lifecycle and repetition | Demonstrate every material change creates a successor and never rewrites an earlier evaluation. Predictions: UFC-PRED-004, 005. |
| UFC-REQ-025 | `assumptions.tex`, Scientific validity and uncertainty; `edge_cases.tex`, Evidence required | Report algebraic, implementation, representation, observational, and readiness evidence separately; do not promote a lower layer. |

