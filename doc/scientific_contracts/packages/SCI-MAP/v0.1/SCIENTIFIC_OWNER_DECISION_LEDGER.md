# SCI-MAP v0.1 Scientific-Owner Decision Ledger

Document revision: `r0.2`

Scientific owner: Grant Wilson

Status: nine unresolved decisions. Six preserve questions carried by the
owner-approved author packet; SCI-MAP-OD-007 records the numeric-policy gap
exposed by support-domain review; SCI-MAP-OD-008 and SCI-MAP-OD-009 record
genuine projection and common-grid ownership gaps exposed by the first science
editing round. No entry below has been answered by the Stage B author.

`SCI-MAP-CI-001` separately records that dimensional consistency already
forces `coverage_cut` to be dimensionless. Its proposed amendment awaits owner
approval, so the exact r0.1 OD-007 wording remains unchanged below. After
approval, OD-007 must be narrowed without renumbering to the numeric domain,
boundary-case, recommended-range, authority, and failure questions.

| ID | Status | Exact decision requested | Why it matters | Conservative draft behavior pending decision | Evidence or authority needed |
|---|---|---|---|---|---|
| SCI-MAP-OD-001 | **OPEN** | What physical rationale is authoritative for the adopted v0.1 normalization-support and science-policy-support threshold rule? | The exact algorithm is approved, but a policy coefficient must not be presented as a derived physical law. | State the formula exactly, label it adopted policy, and make no physical optimality, completeness, noise, or response claim from it. | Scientific-owner statement naming the intended physical/operational objective and its valid domain. |
| SCI-MAP-OD-002 | **OPEN** | Who may change the support-threshold rule after v0.1, and what evidence is required for a successor? | Without change control, an implementation or validation campaign could silently redefine availability. | Freeze the v0.1 formula and population identity; treat any altered multiplier, quantile/index, population, or predicate as a contract change. | Owner-named authority plus preregistered deterministic, response, covariance, observational, and transition evidence appropriate to the claimed benefit. |
| SCI-MAP-OD-003 | **OPEN** | Must every scientifically usable v0.1 map have a realized response/kernel product, or may a bundle with a typed unavailable response be used by a restricted class of consumers? | The answer changes bundle completeness, final-validity conjuncts, consumer admission, and required publication. | Require an explicit response state; never fabricate a response; all response-dependent consumers fail closed. Do not assert that a response-unavailable bundle is generally usable or unusable. | Owner-defined restricted consumer list, required claims, response-independent rationale, and failure semantics. |
| SCI-MAP-OD-004 | **OPEN** | What minimum covariance representation must persist in a raw bundle, and what may remain resolvable through lineage? | A diagonal summary loses cross-pixel and cross-observation information, but the admitted packet does not choose a storage boundary. | Require the exact conditional covariance equation, representation status, assumptions, omissions, and lineage. Stronger consumers fail closed when their needed covariance is unavailable. | Owner choice among persisted matrix/operator/structured summary/lineage forms, with consumer and round-trip requirements. |
| SCI-MAP-OD-005 | **OPEN** | Beyond the adopted v0.1 effective required-output rule, is physical observation-map publication an abstract contract obligation whenever coaddition is requested? | Logical complete-bundle construction and durable publication are different obligations. | Preserve the v0.1 rule that every product designated required by the effective plan, including required observation products during coaddition, must publish successfully. Do not generalize this into a future universal storage mandate. | Owner statement defining logical availability, required persisted artifacts, retention, and failure scope. |
| SCI-MAP-OD-006 | **OPEN** | Is Pointing/OOF ordinary-map arithmetic a registered use of the shared SCI-MAP v0.1 operator, with mode-specific interpretation remaining outside this package? | Registration determines capability metadata, frame routing, conformance profiles, and consumer claims without changing the arithmetic. | State only that reuse is conditional on a fully conforming input bundle; grant no mode-specific fitting, astrometric, response, or production authority. | Owner registration naming supported mode(s), exact input/output identity, AltAz WCS profile, and separate MODE ownership. |
| SCI-MAP-OD-007 | **OPEN** | What numeric domain and unit status are authoritative for `c=coverage_cut`, including the required disposition of negative, zero, non-finite, and greater-than-one values, and what failure behavior applies when the state is not authorized? | The admitted threshold formula names `c` but supplies no range, unit, dimensionless declaration, or boundary-case disposition; inventing one would change support and scientific output membership. | Assert no general domain or units. Require the exact value and unit or dimensionless status to be explicitly admitted by the owner-authorized effective support policy; otherwise fail closed before constructing support-authorized output rows or mutating a required product. | Scientific-owner statement fixing the admissible domain, unit or dimensionless status, boundary-case disposition, effective-policy authority, failure scope, and required transition/validation evidence. |
| SCI-MAP-OD-008 | **OPEN** | Which sample-to-pixel projection classes are authorized for ordinary SCI-MAP v0.1, what normalization applies to `G_pi` (including whether and where `sum_p G_pi=1` is required), how is boundary loss represented, and which upstream authority owns those facts? | Projection normalization and boundaries determine the estimand, constant preservation, response, covariance, hits, exposure, and edge behavior; PTC D004 resolves `omega_i` coefficient semantics but not `G_pi`. | Require every product to declare its actual projection class, normalization, extent, and boundary convention. Permit the existing one-hot and fractional contract examples, but assert no unrecorded conservation property or additional projection class. | Scientific-owner statement identifying allowed classes, normalization and edge rules, owning producer, required metadata, and analytic/response/covariance/edge validation. |
| SCI-MAP-OD-009 | **OPEN** | Is upstream crop or pad to a canonical common grid an authorized preparation for v0.1 observation coaddition, which package owns it, and which future package owns reprojection or mosaicking beyond centered-integer placement? | The existing contract correctly rejects odd shape differences and different grids, but it does not authorize a producer to change an observation map merely to make it admissible or assign future grid-changing science. | Reject every odd-difference or otherwise incompatible observation bundle. SCI-MAP performs no crop, pad, fractional shift, reprojection, interpolation, or mosaic; no other package is named as authorized until the owner decides. | Scientific-owner statement defining whether canonical-grid preparation exists, its producer, response/covariance/validity/provenance rules, and the owner/scope of any future reprojection or mosaicking contract. |

## Upstream dependency gates (not owner decisions in this ledger)

Stronger MAP claims remain conditional on SCI-CAL calibration/unit/response
facts, ALIGN/AST coordinate/WCS/astrometric facts, PTC coefficient/covariance
facts, VAL eligibility/non-finite facts, and NOI empirical covariance and
significance facts. The draft does not infer or resolve those producers'
science.

## Decision recording rule

When the owner answers an item, record the date, exact approved wording,
authority identity, affected requirement IDs, contract-version consequence,
and any required new validation. Do not replace the open question with an
implementation observation or an unversioned editorial change.
