# SCI-VAL v0.1 — Owner-Bound Profile Registry

Status: r0.3 continuing contract registry; mandatory canonical atomic profile
and five SCI-PTC named-use profiles registered and source-current; two broad
SCI-PTC identities explicitly unsupported; MAP unbound

Date: `2026-08-24`

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
| Authoritative source | `REVISION_DIRECTIVE_R0.3.md`, `VAL-R03-D001`, preserving the invariant approved in `REVISION_DIRECTIVE_R0.2.md`, `VAL-R02-D003`; formal clauses `SCI-VAL-REQ-019`, `SCI-VAL-REQ-043`, and `SCI-VAL-REQ-045`; current adjacent-source compatibility is bound by `SOURCE_BINDING_REGISTER.md` SHA-256 `ff5402b71c40f31daac1f7c820a705a5a23eb64688f70955fac76e10e2916430`, including frozen SCI-ALIGN r0.3, frozen SCI-RTC r0.12, and the approved WP-2 exposure-lineage boundary |
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
the source-current adjacent meanings in `SOURCE_BINDING_REGISTER.md` SHA-256
`ff5402b71c40f31daac1f7c820a705a5a23eb64688f70955fac76e10e2916430`,
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
| `SCI-MAP:map_upstream_admission` | SCI-MAP | Unbound; accepted MAP boundary exists, but no exact owner-approved profile is registered here |
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
