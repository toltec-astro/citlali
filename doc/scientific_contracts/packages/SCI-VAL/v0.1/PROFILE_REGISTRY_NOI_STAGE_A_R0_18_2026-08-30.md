# SCI-VAL v0.1 — Owner-Bound Profile Registry

Registry identity: `SCI-VAL_PROFILE_REGISTRY v0.1/r0.3-map-r0.7.1-jinc-stage-a-q002-noi-stage-a-r0.18-2026-08-30`

Status: exact manifest-bound r0.3 contract-registry revision; mandatory
canonical atomic profile, five SCI-PTC named-use profiles, two versioned
SCI-MAP upstream records, one SCI-MAP aggregate coadd profile, one SCI-JINC
atomic admission profile, and four SCI-NOI profiles registered; two broad
SCI-PTC identities explicitly unsupported

Last updated: `2026-08-30` (SCI-NOI Stage A r0.18 binding successor)

## Registry Rule

The registry is a binding and replay mechanism. It does not write, approve,
or inherit scientific-use policy. A usable profile record must bind:

1. immutable registry key and profile version;
2. exact named scientific use;
3. actual scientific owner;
4. authoritative source and exact version or digest;
5. applicability domain and object type;
6. required restrictions and missing-fact behavior;
7. permitted exceptions, including any non-exceptionable invariants;
8. response and uncertainty roles, each selected from `structural_gate`,
   `required_permission`, `decisive_exclusion`, or `advisory`;
9. aggregation and propagation compatibility; and
10. supersession and incompatibility behavior.

Missing or conflicting binding information makes the profile unavailable for
evaluation. A reserved name is not a usable policy.

## Mandatory Canonical Registered Profile

| Field | Binding |
| --- | --- |
| Registry key | `SCI-VAL:independent_exposure@1` |
| Named use | `independent_exposure` |
| Scientific owner | Grant Wilson, as scientific owner of the owner-approved SCI-VAL v0.1 contract-level invariant |
| Authoritative source | `REVISION_DIRECTIVE_R0.3.md`, `VAL-R03-D001`, preserving the invariant approved in `REVISION_DIRECTIVE_R0.2.md`, `VAL-R02-D003`; formal clauses `SCI-VAL-REQ-019`, `SCI-VAL-REQ-043`, and `SCI-VAL-REQ-045`; exact adjacent-source compatibility is bound by `SCI-VAL_SOURCE_BINDING_REGISTER v0.1/r0.3-map-r0.7.1-2026-08-28`, SHA-256 `7b91a324f35196a8c8a6e23c8abbbf5322fc601798e36d4ac821907a6090eadf`, including frozen SCI-ALIGN r0.3, frozen SCI-RTC r0.12, and the approved WP-2 exposure-lineage boundary |
| Domain | Exact sample-detector occurrence with authoritative representative-source identity and origin state |
| Decisive invariant | An exact representative source that is synthesized or replaced is not an original independent astronomical exposure |
| Exception permission | None for the decisive invariant; an attempted exception makes the supplied policy invalid |
| Other restrictions | Must be supplied by the actual scientific owner in an immutable compatible successor record; VAL supplies none |
| Compatibility | Compatible only when the direct-origin invariant, domain identity, and no-exception rule are preserved exactly; a weakening requires a different named use and is not a compatible successor |
| Response/uncertainty roles | No role is imposed by this direct-origin invariant; any added compatible restriction must select only from the four closed roles and retain the same owner/source binding discipline |
| Aggregation and propagation compatibility | `atomic_only`. Aggregation and reverse propagation are `not_applicable` under this profile identity. Any future aggregate is a separately registered, owner-bound proposition under the Aggregate Profile Rule. Any propagation it authorizes creates a new derived proposition and lifecycle generation; it preserves the exact atomic source references and cannot rewrite the atomic decision or the original acquisition and valid-original facts |
| Missing/conflicting behavior | `applicability_unknown` plus `decision_unavailable` when structural/profile binding or required origin authority is unresolved; an authoritative synthesized/replaced origin is `ineligible` |

This registry entry names its scientific owner explicitly so that the
contract-level invariant is not misread as a VAL-authored downstream policy.
The `SCI-VAL` namespace identifies the registering package; it does not make
SCI-VAL Core the policy owner.

The former draft key `VAL.core.independent_exposure@1` is not registered and
has no compatibility alias. An input carrying that key is an unbound profile,
not a migration accepted by inference.

The atomic-only disposition does not define a generic usable exposure,
detector fraction, retained exposure, projected exposure, coadd exposure, or
threshold. Physical-acquisition and valid-original facts remain owned by
SCI-ALIGN and preserved through the approved timestream exposure-lineage
boundary. A future scientific use owns any additional use-qualified exposure
quantity it defines.

## SCI-PTC Common Named-Use Semantics

The five registered SCI-PTC profiles below bind the PTC-owned documentation
artifact
`SCI-PTC-COMMON-NAMED-USE-SEMANTICS-v0.1/r0.1`, SHA-256
`c1fc8370007b65307769fb966c8523251695924aaff84f3e5b4c89b6d3380b8c`.
The artifact is
`audits/SIX_PACKAGE_TIMESTREAM_CLOSURE_PROGRAM_55EFD8A/PTC_NAMED_USE_COMMON_SEMANTICS_R0.1.md`.

The fragment is not a VAL profile, cannot be evaluated independently, grants
no permission, and creates no runtime or serialization object. It requires
distinct named-use propositions, preservation without upgrade of producer
facts and classifications, and an explicit declaration of scientific
relevance before any fact may affect a decision. In particular, the mere
existence, availability, or unknown state of unrelated metadata has no
admission consequence.

## Registered SCI-PTC Named-Use Profiles

All five records bind frozen SCI-PTC v0.1/r0.5 through freeze-record SHA-256
`8357961a49272adc40e27a8aa9e760e0d01ff2419ae2c88a62c0f93c9f959e66`,
the exact adjacent meanings in `SCI-VAL_SOURCE_BINDING_REGISTER
v0.1/r0.3-map-r0.7.1-2026-08-28`, SHA-256
`7b91a324f35196a8c8a6e23c8abbbf5322fc601798e36d4ac821907a6090eadf`,
and their exact owner decisions in
`WP5_VAL_SCIENTIFIC_OWNER_DECISION_PACKET.md` at commit `44662a36b`, file
SHA-256 `9bc101e8447173836380e00ea58185fc2e67cbcbac5077ff1578ca5dc27139fd`.
They share evaluation mechanics, not permission. Every record below remains a
complete proposition.

### `SCI-PTC:basis_fit_admission@1`

| Field | Binding |
| --- | --- |
| Registry key | `SCI-PTC:basis_fit_admission@1` |
| Named use and action | `basis_fit_admission`; authorize whether one exact occurrence may influence ordinary-route detector centering and PCA basis/subspace estimation |
| Scientific owner | Grant Wilson, SCI-PTC scientific owner |
| Authoritative source | Frozen SCI-PTC clauses `SCI-PTC-DEF-007`, `SCI-PTC-REQ-012`, `SCI-PTC-REQ-016`, `SCI-PTC-REQ-023`, `SCI-PTC-REQ-031`, `SCI-PTC-REQ-090`--`SCI-PTC-REQ-094`, and `SCI-PTC-REQ-098`; `WP5-OWNER-D004` and `WP5-OWNER-D011`; exact common and source bindings above |
| Applicability and object | Requested ordinary configured-rank PCA route; atomic exact CAL sample-detector occurrence within one immutable PTC segment and one configured network or array group |
| Required restrictions | Exact CAL parent, detector/sample identity, group, segment, calibrated `x`, basis-fit support/mask, and every quantity explicitly used by the frozen basis estimator are resolved and numerically admissible. The occurrence lies within the exact group/segment, its required calibrated `x` is finite and available, and its basis-fit mask admits it. Direct ALIGN-synthesized or RTC-replaced origin is a decisive exclusion. No unrelated metadata is a restriction |
| Centering consequence | Eligibility supplies binary centering influence one and basis influence. Ineligibility supplies binary centering influence zero. The same admitted finite population defines the frozen ordinary \(\lambda_{g,d}\); VAL does not infer a different centering estimator |
| Classification consequence | CAL `engineering-only` is preserved and has no basis-fit admission consequence by itself; eligibility does not create CAL science qualification |
| Exceptions | None. Direct synthesized/replaced origin and failed required numerical inputs are non-exceptionable under this profile |
| Response/uncertainty roles | `advisory` for both. Their typed states remain visible, but numerical response or covariance availability is not required for basis fitting |
| Aggregation and propagation compatibility | `atomic_only`; no VAL aggregate or reverse propagation is defined. Consumption of eligible atomic occurrences by the exact PTC centering/basis estimator is the named action, not permission for another use |
| Missing/conflicting behavior | Missing/conflicting structural binding or applicability authority yields `applicability_unknown` and `decision_unavailable`. For an applicable requested occurrence, any decisive false restriction yields `ineligible`, all restrictions true yields `eligible`, and every remaining required U/C state yields `decision_unavailable`; all facts and causes are preserved |
| Action scope and nonimplication | Authorizes only centering and basis/subspace influence. It grants no loading-fit, operator-application, output-retention, coefficient/QC, response, empirical/simulation, exposure, or MAP permission |
| Supersession | A changed domain, estimator population, centering identity, restriction, exception, common-fragment digest, or source digest requires a new immutable profile version; no prior decision is rewritten |

### `SCI-PTC:loading_fit_admission@1`

| Field | Binding |
| --- | --- |
| Registry key | `SCI-PTC:loading_fit_admission@1` |
| Named use and action | `loading_fit_admission`; authorize whether one exact occurrence may influence estimation of a detector's coupling to an already fixed basis/template |
| Scientific owner | Grant Wilson, SCI-PTC scientific owner |
| Authoritative source | Frozen SCI-PTC clauses `SCI-PTC-REQ-012`, `SCI-PTC-REQ-016`, `SCI-PTC-REQ-019`, `SCI-PTC-REQ-029`, `SCI-PTC-REQ-031`, `SCI-PTC-REQ-053`, `SCI-PTC-REQ-090`--`SCI-PTC-REQ-092`, and `SCI-PTC-REQ-098`; `WP5-OWNER-D005` and `WP5-OWNER-D011`; exact common and source bindings above |
| Applicability and object | Requested ordinary configured-rank PCA route with a fixed model; atomic exact CAL sample-detector occurrence within one immutable PTC segment and configured network or array group |
| Required restrictions | Only inputs explicitly required by the frozen loading estimator may restrict admission: exact fixed basis/template identity, frozen centering state, detector/group/segment identity, loading support and mask, fitting coordinate, declared metric and gauge, and numerical quantities actually used to estimate that loading. Required inputs are resolved, available, and finite. Direct ALIGN-synthesized or RTC-replaced origin is a decisive exclusion. Reachable or optional metadata not used by the estimator has no admission consequence |
| Fixed-model consequence | The loading fit uses the already frozen \(\lambda_{g,d}\) and model. It cannot re-estimate centering, change the basis, alter basis membership, change configured rank, or change the resolved subspace. Basis-fit and loading-fit populations need not be identical |
| Classification consequence | CAL `engineering-only` is preserved and has no loading-fit admission consequence by itself; eligibility does not create CAL science qualification |
| Exceptions | None. Direct synthesized/replaced origin and failed required estimator inputs are non-exceptionable under this profile |
| Response/uncertainty roles | `advisory` for both; neither numerical response nor covariance availability is required to fit a loading |
| Aggregation and propagation compatibility | `atomic_only`; no VAL aggregate or reverse propagation is defined. The fitted loading retains exact basis/template, gauge, unit, group, support, lifecycle, and numerical-use identity and is not a weight, precision, significance, sensitivity, or independent-noise measure |
| Missing/conflicting behavior | Missing/conflicting structural binding or applicability authority yields `applicability_unknown` and `decision_unavailable`. For an applicable requested occurrence, any decisive false restriction yields `ineligible`, all restrictions true yields `eligible`, and every remaining required U/C state yields `decision_unavailable`; all facts and causes are preserved |
| Action scope and nonimplication | Authorizes only influence on the fitted detector loading. It grants no basis, application, output, coefficient/QC, response, empirical/simulation, exposure, or MAP permission |
| Supersession | A changed loading estimator, fixed-model meaning, domain, restriction, exception, common-fragment digest, or source digest requires a new immutable profile version; no prior decision is rewritten |

### `SCI-PTC:operator_application@1`

| Field | Binding |
| --- | --- |
| Registry key | `SCI-PTC:operator_application@1` |
| Named use and action | `operator_application`; authorize application of the exact resolved frozen operator at one configured group-time |
| Scientific owner | Grant Wilson, SCI-PTC scientific owner |
| Authoritative source | Frozen SCI-PTC clauses `SCI-PTC-REQ-012`, `SCI-PTC-REQ-029`, `SCI-PTC-REQ-031`, `SCI-PTC-REQ-083`, `SCI-PTC-REQ-089`--`SCI-PTC-REQ-095`, and `SCI-PTC-REQ-098`; `WP5-OWNER-D006` and `WP5-OWNER-D011`; exact common and source bindings above |
| Applicability and object | Requested ordinary configured-rank PCA route; exact coupled group-time application event and its target occurrence roles, network-time in network mode or array-time in array mode |
| Required restrictions | Resolved \(\Theta_g\), frozen \(\lambda_g\), strictly positive configured rank, fitted subspace and required loadings, metric, tolerance, generalized inverse, application support/mask, detector bindings, finite time-local coefficient-recomputation inputs, and every coordinate transform and boundary state explicitly used by the operator are available within the same configured group. The finite time-local normal matrix satisfies \(\operatorname{rank}(N_{g,t})=k_{{\rm req},g}\) under frozen \(\tau_g\) |
| Fail-closed guard | \(\operatorname{rank}(N_{g,t})<k_{{\rm req},g}\) makes the exact group-time application unavailable. No lower-rank, interpolated, zero-filled, reconstructed, or cross-group alternative may be labeled as the same operator. Failure is scoped to the configured group |
| Fit and classification consequence | Basis/loading-fit exclusion does not imply application exclusion when the frozen operator defines the value completely. Direct synthesized/replaced origin and CAL `engineering-only` are preserved producer facts but are not universal vetoes on performing the mathematics; application does not authorize ordinary output retention or relabel either fact |
| Exceptions | None for the full-rank guard, group identity, or required frozen-operator inputs; the ordinary route defines no numerical fallback |
| Response/uncertainty roles | `advisory` for complete-chain response and covariance. Their availability is not required for data application. Response-kernel action is governed separately by `SCI-PTC:response_companion@1` |
| Aggregation and propagation compatibility | Not an aggregate profile. The decision is group-time local and cannot be transferred across groups, times, lifecycle states, or operator identities. Reverse propagation is `not_applicable` |
| Missing/conflicting behavior | Missing/conflicting structural binding or applicability authority yields `applicability_unknown` and `decision_unavailable`. For an applicable requested group-time, any decisive false restriction yields `ineligible`, all restrictions true yields `eligible`, and every remaining required U/C state yields `decision_unavailable`; no numerical action is authorized unless eligible |
| Action scope and nonimplication | Authorizes only data application of the exact frozen operator. It grants no output-retention, coefficient/QC, response-companion, empirical/simulation, exposure, or MAP permission |
| Supersession | A changed operator, group mode, guard, domain, restriction, exception, common-fragment digest, or source digest requires a new immutable profile version; no prior decision is rewritten |

### `SCI-PTC:output_retention@1`

| Field | Binding |
| --- | --- |
| Registry key | `SCI-PTC:output_retention@1` |
| Named use and action | `output_retention`; decide whether one exact transformed occurrence is scientifically legitimate as a member of ordinary PTC science-signal support |
| Scientific owner | Grant Wilson, SCI-PTC scientific owner |
| Authoritative source | Frozen SCI-PTC clauses `SCI-PTC-REQ-012`, `SCI-PTC-REQ-016`, `SCI-PTC-REQ-021`, `SCI-PTC-REQ-039`, `SCI-PTC-REQ-069`--`SCI-PTC-REQ-074`, `SCI-PTC-REQ-088`, `SCI-PTC-REQ-090`--`SCI-PTC-REQ-096`, and `SCI-PTC-REQ-098`; `WP5-OWNER-D007` and `WP5-OWNER-D011`; exact common and source bindings above |
| Applicability and object | Requested ordinary configured-rank transformed calibrated-`x` route; atomic exact transformed sample-detector occurrence with exact CAL parent, PTC group-time application parent, segment, group, and output identity |
| Required restrictions | Exact group-time operator application was realized successfully; the transformed value is finite; the exact output support/mask admits the occurrence; and every output-specific required fact is resolved. Direct ALIGN-synthesized or RTC-replaced origin is a decisive exclusion from ordinary PTC science-signal support. A failed application supplies no retained substitute |
| Classification consequence | CAL `engineering-only` may be retained with the classification preserved. Retention does not create CAL science qualification or a general science-quality claim; later uses apply their own policies |
| Exceptions | None for realized application, finite transformed value, output mask, or direct synthesized/replaced exclusion. An ineligible calculated value may remain reachable only as explicitly flagged diagnostic/ineligible data, never as ordinary retained support |
| Response/uncertainty roles | `advisory` for both. Missing complete-chain response or covariance does not erase an otherwise valid transformed signal, and retention establishes no response-dependent or uncertainty-qualified claim |
| Aggregation and propagation compatibility | `atomic_only`; no VAL aggregate, generic usable exposure, or reverse propagation is defined. Retention cannot recreate or increase physical acquisition or valid-original exposure |
| Missing/conflicting behavior | Missing/conflicting structural binding or applicability authority yields `applicability_unknown` and `decision_unavailable`. For an applicable requested occurrence, any decisive false restriction yields `ineligible`, all restrictions true yields `eligible`, and every remaining required U/C state yields `decision_unavailable`; all facts and causes are preserved |
| Action scope and nonimplication | Authorizes only ordinary PTC science-signal membership. It neither prescribes serialization nor grants coefficient/QC, response, empirical/simulation, MAP contribution, exposure, beam, photometry, or later-use admission |
| Supersession | A changed transformed-product role, output support, domain, restriction, exception, common-fragment digest, or source digest requires a new immutable profile version; no prior decision is rewritten |

### `SCI-PTC:response_companion@1`

| Field | Binding |
| --- | --- |
| Registry key | `SCI-PTC:response_companion@1` |
| Named use and action | `response_companion`; authorize the existing fixed-state tracked-kernel propagation through PTC, not a new response computation |
| Scientific owner | Grant Wilson, SCI-PTC scientific owner |
| Authoritative source | Frozen SCI-PTC clauses `SCI-PTC-DEF-016`, `SCI-PTC-REQ-012`, `SCI-PTC-REQ-061`, `SCI-PTC-REQ-062`, `SCI-PTC-REQ-066`, `SCI-PTC-REQ-083`, `SCI-PTC-REQ-087`, `SCI-PTC-REQ-088`, and `SCI-PTC-REQ-097`; `WP5-OWNER-D009` and `WP5-OWNER-D011`; exact common and source bindings above |
| Applicability and object | An exactly requested compatible fixed-state response companion with declared parent and domain, coupled to one resolved PTC group-time. A CAL-grid companion selects the PTC-local role; a source-domain companion selects the complete-chain role. Full-procedure response is outside this profile |
| Required restrictions | The exact data operator application state is available. The companion uses the same group, coordinates, metric, support, masks, positive rank, subspace, tolerance, generalized inverse, time-local full-rank guard, detector classes, and boundaries as the data. Its domain and parent are resolved. A source-domain complete-chain role additionally requires exact admitted \(K_{\rm up\to CAL}\); a CAL-grid companion does not apply that operator again |
| Propagation consequence | The companion is acted on by existing \(J_{\Theta_g}[Y_g]\); it does not enter or alter learning or the science result, and frozen \(\lambda_g\) is not subtracted, re-estimated, or restored. A group-time unavailable for data is unavailable for its kernel, with no different support, lower rank, interpolation, or borrowed solve |
| Classification consequence | CAL `engineering-only` is preserved and does not by itself prohibit the exact tracked-kernel mathematics. Companion eligibility does not upgrade the input or establish science qualification |
| Pointing consequence | When the requested response domain requires pointing, exact pointing identity is a structural gate. Unrecoverable pointing makes the affected response role unavailable; PTC cannot reconstruct or replace it |
| Exceptions | None for parent/domain identity, identical frozen state, the full-rank guard, single application of each upstream operator, or the prohibition on companion influence over learning |
| Response/uncertainty roles | Response is `required_permission` for the exact requested companion role. Response uncertainty is `advisory` unless a separately named claimed use makes it required; no uncertainty is manufactured here |
| Aggregation and propagation compatibility | This profile authorizes only the existing forward fixed-state propagation. It requires response authority and exact recoverability, not dense kernel serialization. No aggregate or reverse propagation is registered |
| Missing/conflicting behavior | Missing/conflicting parent, domain, operator, group-time, required source-domain upstream response, or pointing gate yields `applicability_unknown` and `decision_unavailable`. For an applicable requested companion, any decisive false restriction yields `ineligible`, all restrictions true yields `eligible`, and every remaining required U/C state yields `decision_unavailable` |
| Action scope and nonimplication | Authorizes only the existing tracked-kernel role. It creates no response object, estimator, computation, sidecar, payload, PSF/beam recovery, photometric validity, MAP authority, science qualification, or full-procedure response |
| Supersession | A changed response role, operator/domain chain, restriction, exception, common-fragment digest, or source digest requires a new immutable profile version; no prior decision is rewritten |

## Registered SCI-MAP Named-Use Profile

### `SCI-MAP:map_upstream_admission@1`

| Field | Binding |
| --- | --- |
| Registry key | `SCI-MAP:map_upstream_admission@1` |
| Named use and action | `map_upstream_admission`; decide whether one exact realized PTC occurrence may enter the MAP-route-candidate population before pixel-specific placement and local numerical contribution gates |
| Scientific owner | Grant Wilson, SCI-MAP scientific-policy owner |
| Authoritative source | Owner directive `SCI-MAP v0.1 r0.5 TARGETED PTC-TO-MAP CROSS-PACKAGE CLOSURE DIRECTIVE`, SHA-256 `210e8beafe26381a7d35cf38bacab9a9d959646055635a7c1179e0729a3cfa9a`; `SCI-PTC_TO_SCI-MAP v0.1/r0.1`; frozen SCI-PTC v0.1/r0.5 freeze-record SHA-256 `8357961a49272adc40e27a8aa9e760e0d01ff2419ae2c88a62c0f93c9f959e66`; frozen SCI-AST v0.1/r0.3 source-manifest SHA-256 `b54b6013750540f28aad02339a60bf36078980dc53b132beab73069d66ef3601`; frozen SCI-CAL v0.1/r0.5-r0.4 freeze-record SHA-256 `413426f49edf1249f751a05bb8c6e9fd907b11e8da0530fe2da39814885efb22`; SCI-VAL Core r0.3 plus exact Registry revision `SCI-VAL_PROFILE_REGISTRY v0.1/r0.3-map-r0.7.1-2026-08-28` and exact source-binding revision `SCI-VAL_SOURCE_BINDING_REGISTER v0.1/r0.3-map-r0.7.1-2026-08-28` |
| Applicability and object | Requested ordinary positive-rank PTC-to-MAP route; one exact occurrence binding observation, detector occurrence/UID, stable RTC output sample `n`, PTC product/application generation, segment, array/network/group, and complete PTC/AST/CAL/RTC/ALIGN parent chain |
| Required permissions | Exact realized PTC product exists; `SCI-PTC:output_retention@1` is `eligible`; transformed signal and its source identity are available; an exact PTC-owner-selected MAP-facing coefficient family/value is available and its coefficient/QC disposition permits this use; the exact AST RTC-grid coordinate for the same `n` and parent chain is structurally bound; all required source and lifecycle bindings are compatible |
| Decisive exclusions | PTC-disabled, no-product, direct CAL input, inferred no-op PTC, PTC output-retention `ineligible`, direct synthesized or replaced representative source for the MAP signal use, incompatible signal/coordinate parent, incompatible generations, or a coefficient/QC disposition that forbids MAP use |
| Classification and influence behavior | CAL `engineering-only` may remain a route candidate only when PTC retained it and the classification is preserved; this creates no science-qualification claim. Direct synthesized/replaced origin is decisive for this profile. Transitive inherited influence and all other causes are preserved but have no universal veto: they affect this decision only when an exact required restriction names them |
| Response/uncertainty roles | `advisory` for base numerical MAP signal admission. Exact response class/state, conditional covariance, missing terms, limitations, and causes are carried. A later MAP claim may require them through a separate exact product/claim binding; this profile neither fabricates nor upgrades them |
| Exceptions | None for exact occurrence/parent/generation binding, realized positive-rank PTC route, output-retention permission, direct synthesized/replaced exclusion, coefficient identity/QC permission, or same-`n` AST binding |
| Missing/conflicting behavior | Missing or conflicting applicability, identity, parent, generation, required permission, coefficient family/value/QC state, or coordinate binding yields `applicability_unknown` and `decision_unavailable`. A decisive false restriction yields `ineligible`; all restrictions true yields `eligible`. Causes and scopes are retained |
| Lifecycle and consumer action | Evaluation binds requested, effective, observation-resolved, applied, and realized identities plus exact source/profile versions. `eligible` creates a MAP-route candidate only. MAP still evaluates signal finiteness, positive coefficient, `G_pi`, boundary, support, required companions, and final contribution; VAL performs no pixel placement or accumulation |
| Aggregation and propagation compatibility | `atomic_only`; no pixel, detector, observation, exposure, or coadd aggregate is implied. No reverse propagation or producer-fact rewrite is authorized |
| Supersession | Any changed source digest, occurrence domain, restriction, exception, response/uncertainty role, lifecycle, or direct/inherited influence rule requires a new immutable profile version and evaluation generation; prior decisions remain unchanged |

The `@1` record remains immutable for replay of r0.5 decisions. It is not a
compatibility alias for the r0.7.1 source binding. A consumer shall not
substitute `@1` for `@2` by name similarity, object shape, or numerical
agreement.

### `SCI-MAP:map_upstream_admission@2`

| Field | Binding |
| --- | --- |
| Registry key | `SCI-MAP:map_upstream_admission@2` |
| Named use and action | `map_upstream_admission`; decide whether one exact realized PTC occurrence may enter the MAP-route-candidate population before pixel-specific placement and local numerical contribution gates |
| Scientific owner | Grant Wilson, SCI-MAP scientific-policy owner |
| Authoritative source | SCI-MAP r0.7 directive SHA-256 `f7747eea28710d524e12c818b872ac3fcc49f413271f83c0644ae129949a8c8c`; r0.7.1 errata `FREEZE_ONLY_ERRATA_R0.7.1.md`; `SCI-PTC_TO_SCI-MAP v0.1/r0.1`; MAP scientist-readable profile; shared MAP r0.7.1 authority; original-footprint-coordinate boundary; frozen SCI-PTC v0.1/r0.5, SCI-AST v0.1/r0.3, and SCI-CAL v0.1/r0.5-r0.4 authorities; SCI-VAL Core r0.3 plus exact source-binding revision `SCI-VAL_SOURCE_BINDING_REGISTER v0.1/r0.3-map-r0.7.1-2026-08-28`, SHA-256 `7b91a324f35196a8c8a6e23c8abbbf5322fc601798e36d4ac821907a6090eadf`. Exact remaining digests are bound by `SCI-MAP_SOURCE_MANIFEST v0.1/r0.7.1` |
| Applicability and object | Requested ordinary positive-rank PTC-to-MAP route; one exact occurrence binding observation, detector occurrence/UID, stable RTC output sample `n`, PTC product/application generation, segment, array/network/group, and complete PTC/AST/CAL/RTC/ALIGN parent chain |
| Required permissions | Exact realized PTC product exists; `SCI-PTC:output_retention@1` is `eligible`; transformed signal and source identity are available; an exact PTC-owner-selected MAP-facing coefficient family/value is typed available; its coefficient/QC record is requested, applicable, eligible, and realized under the exact PTC-owned coefficient/QC profile for that family; exact AST RTC-grid coordinate for the same `n` and parent chain is structurally bound; all required boundary, source, and lifecycle bindings are compatible. The PTC coefficient/QC pass never grants, replaces, or rescues this MAP admission |
| Decisive exclusions | PTC-disabled, no-product, direct CAL input, inferred no-op PTC, retention `ineligible`, direct synthesized or replaced representative source for the MAP signal use, incompatible signal/coordinate parents, incompatible generations, coefficient/QC exclusion, or boundary/profile/source-version mismatch |
| Classification and influence behavior | CAL `engineering-only` may remain a route candidate only when PTC retained it and the classification is preserved; this creates no science-qualification claim. Direct synthesized/replaced origin is decisive for this profile. Transitive influence and all other causes are preserved but have no universal veto unless an exact required restriction names them |
| Response/uncertainty roles | `advisory` for base numerical MAP signal admission. Exact fixed-state, PTC full-procedure, or PTC+MAP re-resolved response class/state and conditional covariance limitations remain carried; whole-chain RTC-to-CAL-to-PTC-to-MAP response is unavailable here. A stronger product role applies its own exact requirement; this profile neither fabricates nor upgrades response or uncertainty |
| Exceptions | None for exact occurrence/parent/generation binding, realized positive-rank PTC route, output-retention permission, direct synthesized/replaced exclusion, coefficient identity/QC permission, same-`n` AST binding, or exact boundary/source/profile identity |
| Four named decision fields and pass projection | Request: `requested`/`not_requested`; applicability: `applicable`/`inapplicable`/`applicability_unknown`; eligibility: `eligible`/`ineligible`/`decision_unavailable`; realization: `realized`/`incomplete`/`failed`/`not_produced`. Only request=`requested`, applicability=`applicable`, eligibility=`eligible`, and realization=`realized` projects to Boolean MAP upstream-admission pass. Every other field combination is estimator nonmembership with complete causes retained |
| Missing/conflicting behavior | Missing or conflicting applicability, identity, parent, generation, required permission, coefficient family/value/QC state, coordinate, boundary, source, or profile binding yields `applicability_unknown` and `decision_unavailable` where evaluable. A decisive false restriction yields `ineligible`; all restrictions true yields `eligible`. Realization remains independent |
| Lifecycle and consumer action | Evaluation binds requested, effective, observation-resolved, applied, and realized identities plus exact source/profile versions. Passing creates a MAP-route candidate only. MAP still evaluates finiteness, positive coefficient, `G_pi`, boundary, support, required companions, and final contribution; VAL performs no pixel placement or accumulation |
| Aggregation and propagation compatibility | `atomic_only`; no pixel, detector, observation, exposure, or coadd aggregate is implied. No reverse propagation or producer-fact rewrite is authorized |
| Supersession | Any changed source digest, occurrence domain, restriction, exception, response/uncertainty role, lifecycle, or direct/inherited influence rule requires a new immutable profile version and evaluation generation; prior decisions remain unchanged |

### `SCI-MAP:observation_coadd_admission@1`

| Field | Binding |
| --- | --- |
| Registry key | `SCI-MAP:observation_coadd_admission@1` |
| Named use and action | `observation_coadd_admission`; atomically decide whether one complete base/unfiltered observation MAP bundle may enter the centered-integer equal-observation coadd before any coadd-owned state changes |
| Scientific owner | Grant Wilson, SCI-MAP scientific-policy owner |
| Authoritative source | SCI-MAP r0.7 directive SHA-256 `f7747eea28710d524e12c818b872ac3fcc49f413271f83c0644ae129949a8c8c`; r0.7.1 errata `FREEZE_ONLY_ERRATA_R0.7.1.md`; `SCI-MAP_COADD_PROFILES_R0.7.md`; shared MAP r0.7.1 authority; exact adjacent sources and boundaries in `SCI-VAL_SOURCE_BINDING_REGISTER v0.1/r0.3-map-r0.7.1-2026-08-28`, SHA-256 `7b91a324f35196a8c8a6e23c8abbbf5322fc601798e36d4ac821907a6090eadf`. Exact remaining digests are bound by `SCI-MAP_SOURCE_MANIFEST v0.1/r0.7.1` |
| Source atomic profile and aggregate population | Exact manifest-bound atomic MAP identity `SCI-MAP:map_upstream_admission@2`; aggregate object is one immutable complete observation MAP bundle, its exact support-authorized rows, and one requested centered-integer coadd plan. No aggregate rewrites atomic decisions |
| Four decision axes | Request: `requested`/`not_requested`; applicability: `applicable`/`inapplicable`/`applicability_unknown`; eligibility: `eligible`/`ineligible`/`decision_unavailable`; realization: `realized`/`incomplete`/`failed`/`not_produced`. Eligibility never substitutes for realization |
| Required restrictions | Compatible nonpolarimetric quantity and exact nominal beam; realized PTC route/application generation; exact manifest-bound MAP profile; complete AST frame/WCS and identical grid; centered-integer shape/reference-pixel relation; support policy and admitted `coverage_cut`; exact uniform observation coefficient; role-qualified response/covariance state; compatible null/additive-reference and removed-subspace state; exposure convention; required companions; lifecycle; and immutable parentage |
| Response and uncertainty roles | Response is `advisory` and unavailable-compatible for `SCI-MAP:base_signal_coadd_without_required_response@1`; it is `required_permission` for a response-bearing role with exact compatible source domain, basis, class, units, normalization, parent, and rows for every member. Covariance may be partial/unavailable for the base signal role; a covariance-qualified role requires every named block. No hidden subset, zero response, zero unknown block, or inferred independence is permitted |
| Exceptions | None for quantity/beam identity, exact grid/frame, centered-integer relation, required companions, source/profile generation, atomicity, or any restriction named required by the selected product role |
| Missing/conflicting behavior | Missing/conflicting object, plan, parent, source, profile, identity, or required role fact yields `applicability_unknown` and `decision_unavailable` where evaluable. A decisive incompatibility yields `ineligible`. Rejection preserves every cause and changes no coadd state |
| Aggregation and propagation compatibility | Authorizes only the centered-integer equal-observation aggregate and forward propagation of exact compatible companions. It authorizes no reverse propagation, crop, pad, interpolation, reprojection, GLS, or mosaic. Physical exposure uses observation-scoped unique-original union and own-coordinate placement; influence does not duplicate seconds |
| Lifecycle and consumer action | A realized eligible decision permits MAP to perform centered-integer placement, equal-observation accumulation, support resolution, exposure union, and MAP-local coadd validity. VAL performs no arithmetic and authors no MAP policy |
| Supersession | Any changed source digest, population, role, restriction, exception, aggregation rule, or missing/conflict behavior requires a new immutable profile version and decision generation; prior decisions remain unchanged |

## Registered SCI-JINC Named-Use Profile

### `SCI-JINC:jinc_map_contribution@1`

| Field | Binding |
| --- | --- |
| Registry key | `SCI-JINC:jinc_map_contribution@1` |
| Named use and action | `jinc_map_contribution`; decide whether one exact realized upstream detector-sample occurrence may be considered for JINC map contribution before the distinct rounded-center, sample-pixel geometry, finite-support, signed-coefficient, conditioning and fixed-bundle validity gates |
| Scientific owner | Grant Wilson, SCI-JINC scientific-policy owner |
| Authoritative source | Owner-approved SCI-JINC Stage A successor packet at commit `88dcce8b0f7b1d78053b25831b39cf370afd47cc`; author-packet manifest SHA-256 `52a8e843456a8cb033b7593d9b9f67fb83b0ee565c91c141d8e16d46b906140e`; exact scientist-readable profile SHA-256 `2db95da7e5d1b980df79993907d45ac0ababc3aa05c189bfb62dcf04ff2c2e8a`; exact `SCI-PTC_TO_SCI-JINC v0.1/r0.3` SHA-256 `5769d413460e931745e0d401ea432b12d1077c15466247c49caa71b997d4ab1e`; exact `SCI-AST_TO_SCI-JINC v0.1/r0.2` SHA-256 `efffa7059b59c89793fa1d523fb3bb48235f1ab55f7d55060af1600cbfd470a5`; Q002 approval at commit `ebc0e907fe96163e48818fec99e42cc272b2cfb4`, record SHA-256 `c70e8216e816a7f98486b4c61236acc49713a5ce1d6f5ba722ad6e015e0c7e9f`; frozen SCI-PTC v0.1/r0.5 and SCI-AST v0.1/r0.3 authorities; SCI-VAL Core r0.3 plus exact source-binding revision `SCI-VAL_SOURCE_BINDING_REGISTER v0.1/r0.3-map-r0.7.1-jinc-stage-a-q002-2026-08-28`, SHA-256 `0e7ca29ee2e9cd02fb1b76cf87cc64fce6164407a7801f9b9a105ca646317e88` |
| Applicability and object | Requested observation-level SCI-JINC route; one exact atomic occurrence binding observation, detector occurrence/UID, stable RTC output sample `n`, PTC product/application generation, segment, stable TolTEC array, exact JINC plan, target WCS and complete parent chain |
| Required permissions | Exact realized PTC product exists; `SCI-PTC:output_retention@1` is requested, applicable, eligible and realized; transformed signal and source identity are available; requested/effective/observation-resolved/realized identities select one exact registered PTC positive analysis/gridding family by user request or authorized versioned mode default; that family explicitly permits `SCI-JINC`; its compatible value is typed available; its separate coefficient/QC record permits JINC use; AST supplies an available valid coordinate realization associated with the exact same processed sample realization entering JINC; and all required source, ancestry, generation and lifecycle bindings are compatible |
| Decisive exclusions | PTC-disabled/no-product; direct CAL fallback or inferred no-op PTC; output-retention ineligible; direct synthesized or replaced representative source for ordinary JINC signal support; missing selection with no authorized default; unregistered family; missing `SCI-JINC` permission; unavailable or mismatched coefficient payload/QC; unavailable AST coordinate; missing, duplicate or ambiguous sample-coordinate association; incompatible ancestry or generations; absent stable array identity; or boundary/profile/source-version mismatch |
| Direct and inherited influence | Direct causes and complete transitive influence are preserved. Only a fact named by an exact restriction affects this profile. Reachability, inherited influence, a generic flag, or an optional unavailable fact has no universal veto |
| Response and uncertainty roles | `advisory` for both response and uncertainty in this atomic admission decision: their typed producer states remain neutrally preserved but neither gates admission nor creates a base-v0.1 JINC response, covariance, uncertainty or availability product. Missing state is not zero or identity, and admission neither fabricates nor upgrades it |
| Exceptions | None for exact processed-sample association and ancestry/generation compatibility, realized PTC route, output-retention permission, direct synthesized/replaced exclusion, registry/family identity, JINC consumer permission, coefficient identity/QC permission, AST coordinate availability/validity, stable array identity, or exact source/profile identity |
| Four named decision fields and pass projection | Request: `requested`/`not_requested`; applicability: `applicable`/`inapplicable`/`applicability_unknown`; eligibility: `eligible`/`ineligible`/`decision_unavailable`; realization: `realized`/`incomplete`/`failed`/`not_produced`. Only request=`requested`, applicability=`applicable`, eligibility=`eligible`, and realization=`realized` projects to Boolean JINC map-contribution sample-admission pass. Every other combination is estimator nonmembership with its exact causes retained |
| Missing/conflicting behavior | Missing or conflicting applicability, identity, ancestry, generation, selection/default authority, registration, named-consumer permission, coefficient family/value/QC, coordinate, exact sample association, boundary, source or registry binding yields `applicability_unknown` and `decision_unavailable` where evaluable. A duplicate or ambiguous sample-coordinate association makes the coordinate unavailable. A decisive false restriction yields `ineligible`; all restrictions true yields `eligible`. Causes and scopes remain exact. No row/order/time/tolerance/detector fallback, alternate-family fallback, or unity fallback is permitted |
| Lifecycle and consumer action | Evaluation binds requested, effective, observation-resolved, applied and realized identities plus exact source/profile versions. Passing admits the upstream sample for JINC consideration only. JINC still requires an authorized array-associated parameter-set identity, resolves the rounded cache center, applies the center-in-map occurrence gate, and only then evaluates sample-pixel finite support, point-phase `kappa_ip`, square placement, conditioning and fixed-bundle validity. Missing parameter-set authority makes the numerical route unavailable without a hidden default; outside-center, outside-support and contract-zero cases are ordinary no-contribution under their separately approved rules |
| Aggregation and propagation compatibility | `atomic_only`. No pixel, detector, observation, exposure or coadd aggregate and no reverse propagation are implied. Producer facts and earlier decisions remain immutable |
| Action scope and nonimplication | Authorizes only evaluation of JINC upstream sample admission. It grants no MAP permission, PTC coefficient-family authorization, TolTEC numerical parameter set, pixel placement, signed coefficient, accumulator, normalization, response/covariance/exposure product, bundle validity, implementation conformity, validation, performance or readiness claim |
| Supersession | Any changed source digest, occurrence domain, restriction, exception, response/uncertainty role, lifecycle, direct/inherited influence rule or consumer action requires a new immutable profile version and evaluation generation; prior decisions remain unchanged |

## Registered SCI-NOI Named-Use Profiles

The four records below bind the owner-approved policy/action bytes in
`SCI-NOI_VAL_PROFILE_DRAFTS.md`, SHA-256
`c89883d8c20f72aea05f0ae62464daee3a3ee6e81543ff313888ac318a192d6b`,
from the SCI-NOI v0.1 Stage A r0.18 packet at commit
`2f7076e0c7a51320413a86cc6be74c2d3e8f1537`. They also bind sanitized owner
decisions `SCI-NOI_OWNER_DECISIONS v0.1/r0.18`, SHA-256
`272ac939b8a7109a123073b1a39fcdd7ac4129c603683ee81257b94ab2f55a0b`,
ODQ-111 approval SHA-256
`4e377dba46f8aead91ce14291ff6ae41de46476ff6cf3eab732d3aa29b503e67`,
and exact adjacent sources through
`SCI-VAL_SOURCE_BINDING_REGISTER v0.1/r0.3-map-r0.7.1-jinc-stage-a-q002-noi-stage-a-r0.18-2026-08-30`,
SHA-256 `04eca2da9ce76afacf18ae90dc2dbcb702fedbf55e03acb28e14e7dbc459a7c3`.
VAL binds and evaluates these NOI-authored policies; it authors neither their
producer facts nor their consumer actions.

### `SCI-NOI:generation_input_admission@1`

| Field | Binding |
| --- | --- |
| Registry key | `SCI-NOI:generation_input_admission@1` |
| Named use and action | `generation_input_admission`; admit one exact object as a candidate occurrence for `NOI-GEN/PTC-TO-FROZEN-MAP-CONDITIONAL-SIGN@1` only |
| Scientific owner | Grant Wilson, SCI-NOI scientific-policy owner |
| Authoritative source | Exact SCI-NOI profile, owner-decision, ODQ-111, manifest, commit, and source-register bindings above; exact `SCI-PTC_TO_SCI-NOI-GEN v0.1/r0.6` and its bound frozen `SCI-PTC_TO_SCI-MAP v0.1/r0.1` source for the ordinary route |
| Applicability and object | One exact input occurrence or complete parent bundle named by one requested GEN method, including earliest immutable parent and insertion point |
| Required restrictions | Exact parent package/version, product/application generation, observation and stable realized detector/channel/array/readout-network/group identity, observation-scoped detector coherence, network-local coefficient-mass and complement-symmetry design identity, accepted exact finite-design mechanics, signal quantity/unit/beam, WCS/frame, coefficient/projection/normalization/support/boundary, parent validity/causes/influence, lifecycle, operator route, assignment method and immutable provenance required by the selected boundary are compatible and available |
| Decisive exclusions | Nonexistent or incomplete parent; typed parent unavailability; incompatible ancestry/generation; missing exact route/profile identity; unresolved required MAP/JINC numerical gate; absent required signal/support/coordinate/permission; or a parent restriction explicitly prohibiting GEN use |
| Exceptions | None for exact identity, ancestry, required parent completion, source binding, quantity/unit/beam, WCS/frame, lifecycle, or route compatibility. Any other exception requires a versioned NOI profile successor |
| Source-imprint role | Parent source content and causes remain immutable inputs; admission creates no source-cancellation claim |
| Response and uncertainty roles | `advisory` for both on the ordinary route. Exact producer states are preserved without upgrade; neither is a universal admission prerequisite unless a separately versioned GEN method/profile explicitly makes it one |
| Four named decision fields and action projection | Request: `requested`/`not_requested`; applicability: `applicable`/`inapplicable`/`applicability_unknown`; eligibility: `eligible`/`ineligible`/`decision_unavailable`; realization: `realized`/`incomplete`/`failed`/`not_produced`. Only the exact positive conjunction projects to the named candidate-occurrence action |
| Missing/conflicting behavior | Missing/conflicting required facts yield `applicability_unknown` and `decision_unavailable`; a decisive false restriction yields `ineligible`. No row/time/shape/filename/unit-only fallback is permitted |
| Lifecycle and consumer action | Evaluation binds requested/effective/observation-resolved/applied/realized GEN and parent generations plus exact profile/source versions. Passing admits only the exact candidate occurrence. NOI assignment, MAP conforming application, GEN completion/QC, ensemble completion and UNC admission remain separate |
| Aggregation and propagation compatibility | `atomic_only`. No detector, observation, ensemble or reverse-propagated decision is implied; another-use pass or generic flag has no veto or rescue effect |
| Action scope and nonimplication | Creates no assignment, realization, ordinary MAP science product, uncertainty, covariance, standardized signal, source-free claim, implementation conformity, validation, performance or readiness claim |
| Supersession | Any changed object/domain, source binding, required fact, restriction, exception, source-imprint/response/uncertainty role, missing/conflict behavior, propagation rule, lifecycle or consumer action requires a new immutable profile version and evaluation generation; prior decisions remain unchanged |

### `SCI-NOI:uncertainty_member_admission@1`

| Field | Binding |
| --- | --- |
| Registry key | `SCI-NOI:uncertainty_member_admission@1` |
| Named use and action | `uncertainty_member_admission`; admit one exact GEN member as a candidate member for one named UNC method only |
| Scientific owner | Grant Wilson, SCI-NOI scientific-policy owner |
| Authoritative source | Exact SCI-NOI profile, owner-decision, ODQ-111, manifest, commit, and source-register bindings above |
| Applicability and object | One atomic GEN realization member `b`, with immutable GEN-owned completion/QC facts, presented to one exact UNC method/use |
| Required restrictions | Exact member/ensemble IDs, GEN completion and terminal state, assignment/equivalence/duplicate state, earliest parent, fixed or relearned operator state, required operator terminal states, NOI realization-product identity, representation/persistence mode and realized state, required payload or exact regeneration/sufficient-statistic state, reproducibility and audit limits, unit/WCS/support, source-imprint/QC, lifecycle, causes, failure scope and provenance satisfy the exact UNC method's permitted terminal state, domain, response reference and parent/method compatibility |
| Decisive exclusions | Any admitted member of the GEN ensemble is incomplete, failed or unavailable; member/parent identity is missing or ambiguous; duplicate/equivalence state is disallowed; unit/WCS/support/response is incompatible; persistence/reconstruction/streaming/source-imprint/QC facts are inadequate; mode fallback is silent; method generations are mixed; or source binding is false or missing |
| Exceptions | None for admitted-member failure or a failed GEN ensemble. UNC policy cannot relabel completion, failure, duplicate/equivalence, support, QC or lifecycle facts |
| Source-imprint role | Consumed exactly from GEN; admission creates no source-cancellation, physical-noise or calibrated-null claim |
| Response and uncertainty roles | Response identity/compatibility is `structural_gate`; numerical response availability is not upgraded. Pre-existing uncertainty is `advisory`; member admission supplies no variance, covariance, physical-noise or adequacy claim |
| Four named decision fields and action projection | The four fields and values are exactly those above. Only `requested + applicable + eligible + realized` projects to candidate-member admission |
| Missing/conflicting behavior | Missing/conflicting required facts yield `applicability_unknown` and `decision_unavailable`; a decisive named-use restriction yields `ineligible`. No finite-payload, completion, generic-QC or another-use-pass rescue is permitted |
| Lifecycle and consumer action | Evaluation binds immutable GEN completion generation, requested UNC method/use and exact profile/source versions. Passing admits only this exact member as a candidate for that named UNC method; ensemble admission remains separately required |
| Aggregation and propagation compatibility | `atomic_only` member-to-one-UNC-use. No ensemble admission, producer-fact change, universal veto/rescue or reverse propagation is implied |
| Action scope and nonimplication | Does not admit an ensemble, realize an estimator, create an UNC product, or make an incomplete/failed member usable |
| Supersession | Any changed object/domain, source binding, required fact, restriction, exception, source-imprint/response/uncertainty role, missing/conflict behavior, propagation rule, lifecycle or consumer action requires a new immutable profile version and evaluation generation; prior decisions remain unchanged |

### `SCI-NOI:uncertainty_ensemble_admission@1`

| Field | Binding |
| --- | --- |
| Registry key | `SCI-NOI:uncertainty_ensemble_admission@1` |
| Named use and action | `uncertainty_ensemble_admission`; admit one exact complete all-members-successful GEN ensemble to one exact named UNC estimator and domain |
| Scientific owner | Grant Wilson, SCI-NOI scientific-policy owner |
| Authoritative source | Exact SCI-NOI profile, owner-decision, ODQ-111, manifest, commit, and source-register bindings above, including exact `SCI-NOI:uncertainty_member_admission@1` evaluations |
| Applicability and object | One exact GEN ensemble presented to one exact UNC method and target/domain |
| Required restrictions | Exact assignment law and admitted membership; every admitted member successfully completed; requested/resolved/completed/unique/admitted counts and design rank; complement/duplicate/dependence, fixed/relearned graph and source-imprint state; exact persistence/regeneration/streaming sufficiency and audit state; exact transformation/Wiener/FRUIT owner and generation state when applicable; exact estimator center/weights/member population/common domain; target, representation, response reference, effective information, estimator uncertainty/calibration/limitations/lifecycle/provenance; and any separately named covariance or inverse-scale domain/rank/null/regularization identity are compatible and complete |
| Decisive exclusions | Any admitted member is incomplete, failed or unavailable; survivor/partial-ensemble substitution; partial streaming state; silent persistence fallback; inadequate regeneration or sufficient-statistic identity; mixed fixed/relearned or parent/method generations; method-invalid design or insufficient cardinality/rank; owner-transformation mismatch or unauthorized relearning; unauthorized Wiener/FRUIT feedback, mutation, replay or member mixing; estimator recentering/`B-1`/member-subset substitution forbidden by ODQ-105B; inferred covariance zero/independence; invalid or implicitly regularized reciprocal; incompatible unit/WCS/response/support; unavailable estimator/target identity; or missing source-imprint disclosure |
| Exceptions | None for admitted-member failure, failed GEN ensemble or survivor/partial-ensemble substitution. A separately authorized covariance method may govern only its exact within-product/domain missingness |
| Source-imprint role | Preserves the approved meaning that randomization intends source suppression but does not by construction establish source-free maps; creates no physical-noise or calibrated-null meaning |
| Response and uncertainty roles | Exact response target/reference compatibility is `structural_gate`. Pre-existing uncertainty is `advisory`; passing authorizes only the exact stated UNC calculation and does not create every covariance representation or consumer use |
| Four named decision fields and action projection | The four fields and values are exactly those above. Only `requested + applicable + eligible + realized` projects to exact ensemble-to-estimator admission |
| Missing/conflicting behavior | Missing/conflicting required facts yield `applicability_unknown` and `decision_unavailable`; a decisive design/compatibility failure yields `ineligible`. Large count, balance or completed payload cannot rescue the decision |
| Lifecycle and consumer action | Binds exact GEN and UNC requested/effective/observation-resolved/applied/realized generations and immutable completed membership. Passing authorizes only the named estimator/domain: ODQ-105B's zero-centered design-weighted second moment on the common all-member domain; any separately authorized covariance method's exact representation/domain/rank/null/regularization claim; or ODQ-107's finite-positive reciprocal with unavailable-not-zero behavior and no PTC/MAP promotion. No UNC product exists until estimator realization and atomic publication succeed |
| Aggregation and propagation compatibility | One exact ensemble-to-one-UNC-method decision. No universal detector/pixel/consumer aggregate and no reverse modification of members or parent is implied |
| Action scope and nonimplication | Does not realize or publish an estimator, rescue a partial ensemble, create physical-noise variance, promote a pointwise field to covariance by shape, infer missing covariance as zero, or authorize another consumer use |
| Supersession | Any changed object/domain, source binding, required fact, restriction, exception, source-imprint/response/uncertainty role, missing/conflict behavior, propagation rule, lifecycle or consumer action requires a new immutable profile version and evaluation generation; prior decisions remain unchanged |

### `SCI-NOI:standardization_admission@1`

| Field | Binding |
| --- | --- |
| Registry key | `SCI-NOI:standardization_admission@1` |
| Named use and action | `standardization_admission`; permit construction of `S_cond=q_MAP/sqrt(V_hat_cond)` as the unit-`1` `empirical_scale_standardized_signal` on the exact compatible finite-positive intersection, with only the approved conditional-randomization-scale claim |
| Scientific owner | Grant Wilson, SCI-NOI scientific-policy owner |
| Authoritative source | Exact SCI-NOI profile, owner-decision, ODQ-111, manifest, commit, and source-register bindings above; exact approved MAP parent boundary; exact `V_hat_cond` UNC product/method generation; canonical square-root transformation; and exact STD method identity |
| Applicability and object | For the initial method, one exact immutable normalized real-observation MAP signal and its exact compatible ODQ-105B conditional second moment presented to `NOI-STD/MAP-CONDITIONAL-SECOND-MOMENT-SCALE@1`; JINC is outside this profile |
| Required restrictions | Exact MAP signal estimand/product/generation and same-frozen-operator relation; exact `V_hat_cond` method/generation; direct `sigma_cond=sqrt(V_hat_cond)` denominator finite and strictly positive in the MAP signal unit; exact valid-domain intersection; compatible parent/method, unit/beam, response, WCS/support/validity/lifecycle; explicit numerator/scale dependence and unavailable behavior; output unit `1`; and exact conditional-scale-standardized claim |
| Decisive exclusions | Incomplete/unavailable MAP parent; JINC or substituted numerator; variance/covariance/inverse-scale/precision/weight used directly as denominator; algebraic alternate route treated as an implicit second method; nonpositive/nonfinite/unavailable scale; interpolation/domain extension; incompatible estimator, response, unit/beam, WCS, support, validity, lifecycle or generation; or absent dependence/claim identity |
| Exceptions | None for direct denominator positivity, unit compatibility, exact parent/method identity or immutable parentage. Stronger claims require separate versioned policy and authority |
| Source-imprint role | Signal and scale source-imprint/dependence remain explicit; standardization does not remove source leakage |
| Response and uncertainty roles | Response identity/compatibility is `structural_gate`. The exact authorized `V_hat_cond` parent is `required_permission`; admission creates neither a new uncertainty estimate nor MAP/JINC covariance/response |
| Four named decision fields and action projection | The four fields and values are exactly those above. Only `requested + applicable + eligible + realized` projects to permission to construct the named standardized-signal product |
| Missing/conflicting behavior | Missing/conflicting required facts yield `applicability_unknown` and `decision_unavailable`; a local invalid scale yields unavailable on its exact domain, never numeric zero or infinity |
| Lifecycle and consumer action | Binds exact signal, UNC and STD requested/effective/applied/realized generations plus immutable parent joins. Passing permits only construction of the exact unit-`1` product with claim “MAP signal standardized by the stated conditional randomization second-moment scale” |
| Aggregation and propagation compatibility | Exact numerator/scale rows only. No interpolation, domain extension, aggregate pass or reverse change to either parent is implied |
| Action scope and nonimplication | Does not construct the uncertainty parent, create a new uncertainty estimate, claim Gaussian/Student/z/N-sigma significance, probability, detection, completeness, purity, catalog authority or authorize JINC standardization |
| Supersession | Any changed object/domain, source binding, required fact, restriction, exception, source-imprint/response/uncertainty role, missing/conflict behavior, propagation rule, lifecycle or consumer action requires a new immutable profile version and evaluation generation; prior decisions remain unchanged |

No profile automatically realizes the next GEN, UNC or STD operation, modifies
an immutable parent, or transfers scientific ownership to VAL.

## Aggregate Profile Rule

Every aggregate is a distinct scientific proposition and therefore requires
its own complete registry record. In addition to the general fields above, an
aggregate profile binds:

1. its own immutable registry identity and version;
2. the exact compatible atomic source-profile identity and version;
3. its scientific owner and authoritative source;
4. aggregate object type and applicability domain;
5. population and time support;
6. numerical operator, denominator and missing-decision treatment;
7. threshold and boundary polarity, if any;
8. advisory or binding role, uncertainty treatment, and failure scope; and
9. propagation authority and lifecycle-generation rule.

An aggregate cannot reuse the atomic profile identity or appear as another
atomic instance. For illustration only,
`SCI-PTC:detector_independent_exposure_fraction@1` could identify a future
PTC-owned detector aggregate whose atomic source is exactly
`SCI-VAL:independent_exposure@1`; this example is hypothetical and is not a
registered profile in v0.1. Base atomic compatibility remains homogeneous.
No heterogeneous transformation profile is registered here.

## Package-Qualified Names Unsupported, Deferred, Or Not Bound Here

The following names prevent broad labels from hiding scientifically different
questions. An `unsupported` disposition means the owner found no present
scientific proposition and VAL must not fabricate one. A `deferred` or
`unbound` name remains unavailable until its owner supplies a complete
immutable record satisfying the registry rule.

| Profile identity or template | Expected scientific owner | Status in this registry |
| --- | --- | --- |
| `SCI-PTC:coefficient_qc_population@1` | SCI-PTC | Explicitly unsupported under `WP5-OWNER-D008`. Informational diagnostics remain PTC-owned; a separately named complete profile is required only when a diagnostic gains decision authority |
| `SCI-PTC:empirical_or_simulation_population@1` | SCI-PTC | Explicitly unsupported under `WP5-OWNER-D010`. PTC v0.1 makes no scientific inference from an ensemble of alternative realizations |
| `<PACKAGE>:diagnostic_display` | Owning package or diagnostic consumer | Namespace template only; display permission conveys no stronger scientific use |

The earlier broad label `analysis_or_gridding_contribution` is not a v0.1
registry key. It is replaced by the narrower `map_upstream_admission`; no
automatic alias is permitted because the old wording could be mistaken for
actual numerical contribution.

## Registry Change Rule

A new source digest, profile version, domain, restriction, exception rule, or
compatibility declaration creates a new immutable registry record. It cannot
rewrite an earlier evaluated decision. A renamed profile is a new identity
unless its scientific owner publishes an explicit semantics-preserving alias;
VAL does not infer aliasing.
