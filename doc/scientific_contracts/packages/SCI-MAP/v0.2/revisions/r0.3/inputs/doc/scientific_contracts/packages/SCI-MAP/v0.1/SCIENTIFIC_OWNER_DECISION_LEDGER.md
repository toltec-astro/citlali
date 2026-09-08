# SCI-MAP v0.1 Scientific-Owner Decision Ledger

Document revision: `r0.7`

Scientific owner: Grant Wilson

Status: eight unresolved decisions. Six preserve questions carried by the
owner-approved author packet; SCI-MAP-OD-007 records the numeric-policy gap
exposed by support-domain review; SCI-MAP-OD-008 is resolved by the r0.5
one-hot projection disposition; SCI-MAP-OD-009 retains the common-grid
preparation and future reprojection/mosaic ownership gap.

On `2026-08-16`, scientific owner Grant Wilson approved `SCI-MAP-CI-001`:
`coverage_cut` is a dimensionless support-policy scalar. The disposition does
not determine its numerical domain or failure behavior. OD-007 is therefore
narrowed without renumbering to the numeric-domain, boundary-case,
recommended-range, authority, and failure questions.

| ID | Status | Exact decision requested | Why it matters | Conservative draft behavior pending decision | Evidence or authority needed |
|---|---|---|---|---|---|
| SCI-MAP-OD-001 | **OPEN** | What physical rationale is authoritative for the adopted v0.1 normalization-support and science-policy-support threshold rule? | The exact algorithm is approved, but a policy coefficient must not be presented as a derived physical law. | State the formula exactly, label it adopted policy, and make no physical optimality, completeness, noise, or response claim from it. | Scientific-owner statement naming the intended physical/operational objective and its valid domain. |
| SCI-MAP-OD-002 | **OPEN** | Who may change the support-threshold rule after v0.1, and what evidence is required for a successor? | Without change control, an implementation or validation campaign could silently redefine availability. | Freeze the v0.1 formula and population identity; treat any altered multiplier, quantile/index, population, or predicate as a contract change. | Owner-named authority plus preregistered deterministic, response, covariance, observational, and transition evidence appropriate to the claimed benefit. |
| SCI-MAP-OD-003 | **OPEN** | What response information must the original v0.1 MAP product provide for each response-dependent claim it makes, and which original product claims remain unavailable when response information is incomplete or unavailable? | The original product must label its actual response knowledge honestly without invalidating the numerical map or precluding later simulation and correction. | Report the actual response state, meaning, domain, and limits; fabricate neither zero nor identity response. Make no unsupported response-dependent claim. Permit later response or corrected-map products only with new versioned identity bound to the original MAP product and processing identity. | Owner-defined minimum response obligation for each MAP-authored claim; no exhaustive future-consumer registry is required. |
| SCI-MAP-OD-004 | **OPEN** | What minimum covariance representation must persist in a base/unfiltered MAP bundle, and what may remain resolvable through lineage? | A diagonal summary loses cross-pixel and cross-observation information, but absence of a complete model does not invalidate the map. | Report what uncertainty/covariance information exists, with meaning, domain, assumptions, limitations, omissions, and lineage. Do not promote normalization, hits, exposure, or diagnostic weight. Permit later estimates as new versioned products bound to the original. | Owner choice among persisted matrix/operator/structured summary/lineage forms, with round-trip and claim-specific requirements. |
| SCI-MAP-OD-005 | **OPEN** | Beyond the adopted v0.1 effective required-output rule, is physical observation-map publication an abstract contract obligation whenever coaddition is requested? | Logical complete-bundle construction and durable publication are different obligations. | Preserve the v0.1 rule that every product designated required by the effective plan, including required observation products during coaddition, must publish successfully. Do not generalize this into a future universal storage mandate. | Owner statement defining logical availability, required persisted artifacts, retention, and failure scope. |
| SCI-MAP-OD-006 | **OPEN** | Is Pointing/OOF ordinary-map arithmetic a registered use of the shared SCI-MAP v0.1 operator, with mode-specific interpretation remaining outside this package? | Registration determines capability metadata, frame routing, conformance profiles, and consumer claims without changing the arithmetic. | State only that reuse is conditional on a fully conforming input bundle; grant no mode-specific fitting, astrometric, response, or production authority. | Owner registration naming supported mode(s), exact input/output identity, AltAz WCS profile, and separate MODE ownership. |
| SCI-MAP-OD-007 | **OPEN** | What numerical domain is authoritative for the dimensionless scalar `c=coverage_cut`, including the required disposition of negative, zero, non-finite, and greater-than-one values, and what failure behavior applies when a numerical state is not authorized? | The threshold formula fixes `c` as dimensionless but supplies no numerical range or boundary-case disposition; inventing one would change support and scientific output membership. | Assert no general numerical domain. Require the exact numerical value to be explicitly admitted by the owner-authorized effective support policy; otherwise fail closed before constructing support-authorized output rows or mutating a required product. | Scientific-owner statement fixing the admissible numerical domain, boundary-case disposition, recommended range, effective-policy authority, failure scope, and required transition/validation evidence. |
| SCI-MAP-OD-008 | **RESOLVED** | Which sample-to-pixel projection classes are authorized for ordinary SCI-MAP v0.1, what normalization applies to MAP-owned `G_pi`, and how is boundary loss represented? | Projection normalization and boundaries determine the estimand, response, covariance, exposure, and edge behavior. | Authorize only `SCI-MAP:one_hot_containing_pixel@1` with lower-inclusive, upper-exclusive cells; `G_pi=1` only for the unique finite in-grid containing pixel and zero elsewhere; outer-upper-boundary and out-of-grid loss; no wrap or clamp; fractional projection deferred. | `SCI-MAP-R05-D003`, owner directive SHA-256 `210e8beafe26381a7d35cf38bacab9a9d959646055635a7c1179e0729a3cfa9a`, `2026-08-27`; carried unchanged by r0.6 directive SHA-256 `d57e90f8ed4407b0f727cd2ac981318e02101ddd9f73abac7e2772b66dac2c84`. |
| SCI-MAP-OD-009 | **OPEN** | Is upstream crop or pad to a canonical common grid an authorized preparation for v0.1 observation coaddition, which package owns it, and which future package owns reprojection or mosaicking beyond centered-integer placement? | The existing contract correctly rejects odd shape differences and different grids, but it does not authorize a producer to change an observation map merely to make it admissible or assign future grid-changing science. | Reject every odd-difference or otherwise incompatible observation bundle. SCI-MAP performs no crop, pad, fractional shift, reprojection, interpolation, or mosaic; no other package is named as authorized until the owner decides. | Scientific-owner statement defining whether canonical-grid preparation exists, its producer, response/covariance/validity/provenance rules, and the owner/scope of any future reprojection or mosaicking contract. |

## Resolved inconsistency disposition

| ID | Status | Date | Scientific-owner disposition | Normative effect |
|---|---|---|---|---|
| SCI-MAP-CI-001 | **RESOLVED** | 2026-08-16 | `coverage_cut` is a dimensionless support-policy scalar. This does not determine its admissible numerical range or the handling of negative, zero, non-finite, or greater-than-one values. | Shared threshold equations, REQ-031/032, and PRED-012 amended in place; OD-007 narrowed without renumbering. |
| SCI-MAP-OD-008 | **RESOLVED** | 2026-08-27 | Authorize `SCI-MAP:one_hot_containing_pixel@1` with exact half-open boundary behavior; defer fractional projection. | REQ-005, REQ-011, REQ-015, REQ-016, REQ-026, REQ-029--030 and PRED-004/009 amended in place; stable IDs preserved. |
| SCI-MAP-R06-D001 | **RESOLVED** | 2026-08-27 | Physical exposure is a sum over unique stable originals placed once by each original's own authorized AST coordinate; descendant influence is separate and cannot duplicate or relocate seconds. | REQ-007, REQ-016, REQ-029--030 and PRED-008 specify the own-coordinate construction and multi-descendant ECS fixture. |
| SCI-MAP-R06-D002 | **RESOLVED** | 2026-08-27 | Register source-current `SCI-MAP:map_upstream_admission@2` and VAL-governed aggregate `SCI-MAP:observation_coadd_admission@1`; retain `@1` as immutable history. | REQ-004 and REQ-039 bind exact four-axis records; no VAL policy authorship is inferred. |
| SCI-MAP-R06-D003 | **RESOLVED** | 2026-08-27 | Separate fixed-state linear response, PTC full-procedure finite differences, and re-resolved procedure response. | Superseded in terminology, not substance, by SCI-MAP-R07-D003 below. |
| SCI-MAP-R07-D001 | **RESOLVED** | 2026-08-28 | Coefficient availability is structural and does not require finiteness; MAP alone owns positive/zero/negative/non-finite/unrepresentable numerical classification. The PTC coefficient/QC pass is requested/applicable/eligible/realized under its exact PTC-owned profile and cannot grant or rescue MAP admission. | REQ-004/006/010/034 and PRED-010/011 separate producer truth, MAP admission, safe classification, and arithmetic. |
| SCI-MAP-R07-D002 | **RESOLVED** | 2026-08-28 | Use one exact operator identity \(A_{{\rm MAP},\Pi}\equiv A_{\rm out}\equiv D_{Q,{\rm out}}^{-1}J_{\rm out}G\Omega\) for signal and every named fixed linear companion. | REQ-016/019/051 and response/covariance equations bind exact domains, coefficient generation, projection, support, WCS, parents, and lifecycle. |
| SCI-MAP-R07-D003 | **RESOLVED** | 2026-08-28 | Name the three available response levels fixed-state, PTC full-procedure, and PTC+MAP re-resolved procedure response. Reserve whole-chain RTC-to-CAL-to-PTC-to-MAP for a separately authorized complete rerun; it is unavailable here. | REQ-008/017/046 and PRED-005--007/013 use exact family/domain names; the PTC state-change record is \(\Delta\mathcal S_{{\rm PTC}\text{-}{\rm FP}}\), not covariance notation. |
| SCI-MAP-R07-D004 | **RESOLVED** | 2026-08-28 | Original-footprint exposure uses each stable original's own layered AST ALIGN-grid coordinate in the target MAP WCS through `SCI-AST_TO_SCI-MAP_ORIGINAL_FOOTPRINT_COORDINATE v0.1/r0.1`; it is distinct from descendant RTC coordinates, causal influence, temporal support, effective integration time, and precision. | REQ-007/029/030/046, the exposure equations, and ECS fixtures bind exact original identity and coordinate failure behavior. |

## Blocking-scope classification

- **A — numerical single-observation map:** PTC coefficient family and
  `coverage_cut` numerical domain remain open. MAP admission, exact same-`n`
  join, one-hot projection, and original-footprint exposure role are closed by
  r0.7. No generally
  authorized ordinary numerical route is claimed.
- **B — response/uncertainty:** OD-003 and OD-004 remain open, together with
  upstream full-procedure response inputs/domain.
- **C — coadd:** the uniform observation coefficient, coadd admission profile,
  and unavailable-response policy are selected; OD-005 and OD-009 remain.
- **D — optional/future:** OD-006, fractional projection, correlated GLS, and
  mosaicking remain deferred.

## Upstream dependency gates (not owner decisions in this ledger)

MAP consumes realized PTC products, never CAL output directly. Stronger MAP
claims remain conditional on CAL facts carried through PTC, ALIGN/AST
coordinate/WCS/astrometric facts, PTC identity/availability/producer-local
validity/coefficient/covariance facts, MAP-owned admission policy, and NOI
empirical covariance/significance facts. VAL may represent or evaluate named
rules but does not author producer or MAP policy.

## Decision recording rule

When the owner answers an item, record the date, exact approved wording,
authority identity, affected requirement IDs, contract-version consequence,
and any required new validation. Do not replace the open question with an
implementation observation or an unversioned editorial change.
