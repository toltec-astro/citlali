# SCI-FRUIT v0.1 — Candidate Parent Admission Matrix

Status: Stage A classification; **no route is numerically admitted**

The four rows are different scientific routes, not values of one generic map
type. “Candidate” means eligible for owner review, not scientifically or
numerically available.

| Candidate route | Exact authority state at launch | Product/estimand and grouping | Required FRUIT model-construction boundary | Response, uncertainty, support, and lifecycle obligations | Stage A admission state | Closure needed before admission |
| --- | --- | --- | --- | --- | --- | --- |
| **Ordinary MAP** | SCI-MAP v0.1/r0.7.1 frozen; exact freeze target `bd010e20e`; numerical route remains conditionally unavailable | Normalized ordinary MAP observation bundle or ordinary MAP coadd bundle; these are distinct grouping identities and are not interchangeable | Owner must define whether and how an immutable MAP signal becomes a selected feedback sky model, including transition role, calibration, and excluded/nonrecoverable modes | Exact MAP response family, null space, support/validity, coefficient provenance, covariance status, grouping, and generation must bind every application; coadd/observation operations do not commute by default | `UNAVAILABLE_PARENT_ROUTE` | Exact numerical PTC-to-MAP coefficient family; owner-admitted numerical `coverage_cut`; exact compatible response/support; FRUIT model/selection/projector decisions; Registry/evaluation route when required |
| **JINC** | SCI-JINC v0.1/r0.3 frozen; conditional scientific authority; ordinary numerical TolTEC route unavailable | Signed normalized JINC **observation-map** product in base v0.1; no inherited base-v0.1 JINC coadd route | Owner must decide whether the JINC estimand and its response can represent or inform the chosen feedback model; a JINC map is not automatically a deconvolved sky model | Exact JINC kernel/response/covariance identity, geometry/phase, support/tails, PTC coefficient route, TolTEC array parameters, adequacy profile/certificate, and observation grouping | `UNAVAILABLE_PARENT_ROUTE` | JINC-permitted numerical PTC coefficient family; array parameters; numerical adequacy profile/certificate; FRUIT model and forward-projector compatibility; any future coadd requires separate authority |
| **FLT-FIXED** | SCI-FLT-FIXED v0.1 candidate conditionally frozen by record at `7f9307ff...`; ordinary MAP/JINC parents and FLT profiles remain unavailable | Exact fixed-convolution transformed signal with its own parent/grouping/operator/state identity; diagnostics/support/validity are not signals | Owner must define whether the transformed signal estimates the chosen feedback model and whether projection requires an inverse, a response-aware forward model, or exclusion; no inverse is supplied by the fixed transform | Preserve exact fixed operator/state, normalization/units, parent identity, response/null space, covariance, support/edge/validity, observation/coadd order, and frozen lifecycle; transformation of a signal does not transform all metadata by analogy | `UNAVAILABLE_PARENT_ROUTE` | Available admitted ordinary MAP or JINC parent; exact registered FLT profile and Registry route; FRUIT scientific reason/model construction/projector; no silent deconvolution or lost-mode restoration |
| **FLT-MATCHED (provisional)** | Read-only holding study at exact commit `faff97565...`, tree `0dfa3cdf...`; not an approved package, author packet, Stage B launch, or authority | Proposed matched-template amplitude map for exact ordinary-MAP observation or coadd parents; not source detection/catalog and not posterior/Wiener sky reconstruction | Owner must first decide whether a template-amplitude field can ever form the chosen FRUIT model and how overlapping/local amplitudes, template response, selection, and synthesis would produce a projectable model | Exact template, weighting/covariance, influence support, approximation envelope, response, edge validity, learned-state lifecycle, NOI generation, and parent route; JINC/derived parents remain deferred in that study | `PROVISIONAL_AND_UNAVAILABLE` | Independent completion/approval of the matched-filter Stage A package and exact authority; its open learned-state/NOI and lifecycle decisions; numerical MAP parent; then separate FRUIT admission/model/projector decision |

## Cross-Route Prohibitions

- Do not substitute one route when another is unavailable.
- Do not treat a transformed or matched-filtered amplitude map as an ordinary
  MAP/JINC sky estimate.
- Do not infer common units, beam, response, covariance, support, selection, or
  terminal meaning from common WCS or array shape.
- Do not combine observation-local and coadd-local iterations without an exact
  owner-approved grouping/commutation rule.
- Do not let a map-only seed inherit exact-restart identity.
- Do not infer that a route is available from implementation support,
  configuration names, historical reductions, or a registered-looking file.

## Owner Gate

Route admission follows, rather than precedes, the primary estimand/update
decision in `SCI-FRUIT-ODQ-001`. The owner may admit zero, one, or multiple
routes, but each admitted route needs its own exact compatibility and failure
contract. A route left unresolved remains unavailable without fallback.
