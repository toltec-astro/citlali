# Proposed Registry, profile, and boundary records

Draft authority: **SCI-PTC-COEFFICIENT-UNIFORM v0.1-draft.1 / r0.3**  
Revision date: **2026-09-06**  
Status: **exact author proposal; inactive and unavailable for use**

The source identity is src/common_core.tex plus the six canonical files under src/common. Owner adoption must bind promoted immutable source bytes; neither this record nor the artifact manifest activates the proposal.

## Proposed family record

| Field | Proposed exact value |
| --- | --- |
| Registry identity | SCI-PTC:analysis_gridding_coefficients@draft-0.1 |
| Family identity | SCI-PTC:uniform_constant@draft-0.1 |
| Producer/package owner | SCI-PTC |
| Proposed actual owner | Grant Wilson acting for SCI-PTC; UFC-Q-001 remains open |
| Source identity | SCI-PTC-COEFFICIENT-UNIFORM v0.1-draft.1 / r0.3, bound by exact promoted hashes before activation |
| Rule version versus realization | The family version owns scalar, unit, normalization, support, representation permissions, and profile semantics. Application a, PTC product q, coefficient g, payload, request, and evaluation retain distinct immutable realization identities. |
| Indices | Detector-time occurrence i with stable full identity; spatial/map index p remains consumer-owned and never names a PTC product. |
| Raw and normalized coefficient | Exact c-star=1 and omega_i=1; dimensionless unit 1; analytic arithmetic-mean-to-one over the exact support. No literal population reduction is required. |
| Numerical parameters/defaults | None. No constant override, family default, or publication-size default is accepted. |
| PTC parent | One immutable completed PTC publication q bound to exact application, CAL parent, observation, selection lineage, and ordinary-route rank provenance where applicable. |
| Publication unit | q is explicitly tagged as a complete segment, complete scan, or bounded chunk wholly within one named scan or segment and one observation. It becomes complete only after all required components and links are durable and can then be handed off independently. |
| Authoritative target | R_q is the immutable authoritative target-support reference structurally bound to q by identity, source, generation, scope, and digest. It resolves to a finite, nonempty, identity-unique exact q output support. |
| Stored representation | Exact scalar plus stored compact R-hat_g independently matched to R_q, or complete M_g containing every and only each target identity exactly once. A compact form need not duplicate members; cardinality or order is insufficient for an enumerated form. |
| Shared association | Immutable scalar, unit, integrity, compatibility, family, and permission evaluations may be stored once at their exact scope. Each logical occurrence handoff references them and retains its own membership/request fact; no per-sample coefficient or profile array is required. |
| Uncertainty | The fixed coefficient introduces no sampling uncertainty and supplies no assertion about physical, response, covariance, selection, representation, or performance uncertainty. |
| Permissions | SCI-MAP and SCI-JINC permissions are separate open proposals under UFC-OD-008 and UFC-OD-009. |
| Prohibited meanings | Precision, inverse variance, sensitivity, significance, loading, scatter, validity, exposure, response, covariance, independence, optimality, policy default, support restoration, or missing-value replacement. |
| Missing/conflict rule | Missing or conflicting authoritative identity/binding prevents structural establishment. After valid target establishment, a wrong stored reference or missing/extra/repeated enumeration is payload F; absent comparison evidence is U and same-fact contradiction is C. No fallback. |
| Compatibility/supersession | No alias or retroactive change. A wrong pair in one request is local; a corrupt/misbound authoritative q-to-g object is shared. Changed rules create versioned successors; changed actual inputs create new realizations. |

## Proposed complete VAL profile record

| Registry component | Proposed exact binding |
| --- | --- |
| Identity | SCI-PTC:uniform_coefficient_handoff@draft-0.1 |
| Named use | For one explicit request naming (g,q,i,c), decide whether the handoff selects this family, binds exact parents/support, uses a faithful exact-unity payload, and is authorized for named consumer c in {SCI-MAP, SCI-JINC}. |
| Actual owner | Proposed Grant Wilson acting for SCI-PTC; UFC-Q-001 requires confirmation. |
| Source | SCI-PTC-COEFFICIENT-UNIFORM v0.1-draft.1 / r0.3; activation requires exact promoted source digest. |
| Bound common semantics | Admitted SCI-PTC common named-use fragment, exact digest c1fc8370007b65307769fb966c8523251695924aaff84f3e5b4c89b6d3380b8c. |
| Request rule | The explicit named-profile request sets R independently of family selection. No request creates no evaluation. A wrong selected value is a reachable UQ-01 F; missing value in a bound selection record is U; same-value contradiction is C. |
| Object | One coefficient realization g, its PTC publication q, exact occurrence i, and named consumer c, plus explicitly referenced immutable shared facts. |
| Applicability | Applicable when q and valid support exist and i is in support; inapplicable when authoritative identity puts i outside; otherwise unknown where the occurrence-membership question is well formed. |
| Structural gate | Exact active Registry/source/owner, application/CAL, observation, occurrence, q, authoritative q-to-g object, family, selection-record identity, R_q, payload-publication, target-broadcast, profile, named consumer, admitted VAL structural scope/influence, and immutable provenance bindings. Missing/conflicting structural identity makes the question unavailable. Permission evidence is assessed separately by UQ-06. |
| Payload fidelity | Independently assess Q_scalar, Q_unit, Q_integrity, and Q_member. Any independent F gives UQ-05 F; with none F, any C gives C; with none F/C, any U gives U; all T gives T. A same-fact contradiction remains one C. |
| Stored membership | Q_member checks R-hat_g against established R_q or M_g against every target identity. Wrong reference or missing/extra/duplicate enumerated membership is F after target establishment. |
| Permission | UQ-06 uses a complete authoritative record for the named consumer. Explicit denial/nonauthorization is consumer-local F; missing/unresolved evidence is U; contradiction is C. Payload and the other consumer are unchanged. An independent restriction F still makes the requested use ineligible while permission U or C and its cause remain recorded. |
| Restrictions | UQ-01--UQ-07 exactly as defined in src/common/definitions.tex. |
| Exceptions | None. Every restriction and structural invariant is nonexceptionable. |
| Advisory roles | Response and uncertainty states are retained, advisory, and neutral for coefficient fidelity. |
| Aggregation | Gamma_rho=none. The profile is occurrence-level; reuse or shared failure scope for an immutable scalar/fidelity result does not create an aggregate eligibility profile or rule. |
| Lineage | L_rho one-way links the exact requested, effective-policy, observation-resolved, selected, coefficient-realized/published, and evaluation identities. Sharing a fact does not collapse or reverse the lineage. |
| Four axes | R, A, E, and Z remain independent. A completed F gives realized/ineligible; a completed unresolved result may be realized/decision-unavailable. Failed Z means failure of the evaluation artifact itself. |
| Lifecycle | UQ-07 snapshots pre-existing facts, evaluates lineage/nonmutation/write-once conditions before sealing, and only then finalizes. Finalization failure invents neither a UQ-07 value nor an eligibility assertion. |
| Failure scope | Malformed occurrence or wrong otherwise-authoritative q,g request is local. Corrupt/misbound shared object affects all references. Payload F affects that payload's handoffs. Permission affects one named consumer. Every cause is retained. |
| Supersession | Any change to owner, source, named use, object, applicability, restriction, exception, role, truth rule, scope, or lifecycle creates an immutable successor. |
| Status | Draft, unbound, and unevaluable; it is not the unsupported generic SCI-PTC:coefficient_qc_population@1 placeholder. |

## Proposed boundary successors

### MAP

Proposed identity: **SCI-PTC_TO_SCI-MAP_BOUNDARY v0.2-draft.1/r0.3**.

Only the coefficient-slot binding changes: MAP may retrieve exact gamma_i=omega_i=1 after separate SCI-MAP permission and a requested, applicable, eligible, realized profile tied to exact family, q, g, support, payload, and provenance. MAP retains all existing admission, placement, exposure, coordinate, numerical classification, arithmetic, support, response, uncertainty, and failure rules. This inactive draft cannot make a route available.

### JINC

Proposed identity: **SCI-PTC_TO_SCI-JINC_BOUNDARY v0.2-draft.1/r0.3**.

Only the PTC-coefficient binding changes: JINC requires its own explicit SCI-JINC permission and the same exact identity/profile conditions. JINC retains ownership of spatial p, kappa_ip, signed w_ip, normalization, conditioning, support, gates, and bundle semantics. No response, covariance, uncertainty, or new bundle role is created. This inactive draft cannot make a route available.

## NOI conditional limit

Compatibility exists only through the already selected MAP host route. Although the exact coefficient factor is rational 1/1, NOI still requires the frozen lossless rational source for the exact complete MAP parent a_pi=G_pi gamma_i and binds the MAP product/generation, G_pi, family/generation/payload, representation source, and source digest. Missing any required binding leaves NOI design resolution unavailable. No PTC NOI-balance family, JINC-hosted NOI route, profile approval, or changed design law is proposed.
