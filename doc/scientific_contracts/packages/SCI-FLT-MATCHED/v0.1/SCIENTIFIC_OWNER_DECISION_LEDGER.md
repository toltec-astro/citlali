# SCI-FLT-MATCHED v0.1 Scientific-Owner Decision Ledger

Status: Stage B draft; all entries below are **OPEN**. No authored alternative
or numerical route is selected. The ledger records later scientific-owner
questions; it is not a request for immediate clarification because the author
packet is sufficient to develop the bounded options.

Owner: Grant Wilson

Contract identity: `SCI-FLT-MATCHED v0.1`, **Optimal matched-template map
filtering** (the owner-assigned package title, not a realization-level
optimality claim)

## Disposition rule

A valid disposition must name the exact ledger ID, selected authored-option
identity, all conditional parameters listed for that route, applicable parent
class and named uses, effective contract generation, and any superseded
disposition. Silence, current/historical behavior, implementation convenience,
test success, or product availability cannot select an option.

Selecting an option updates scientific authority only after the resulting
contract bytes are reviewed and frozen. It does not establish implementation
conformity, validation, achieved performance, readiness, production, or Unity
status.

## Open owner questions

| Ledger ID | Exact scientific-owner question | Decision-ready alternatives/fields | Route unavailable pending disposition |
| --- | --- | --- | --- |
| `SCI-FLT-MATCHED-SODL-001` | Which `AO-001` weighting authority applies to observation parents, and which independently applies to coadd parents? | Select one of `AO-001-A` exact inverse-covariance GLS, `AO-001-B` structured covariance-derived, `AO-001-C` radially symmetrized average map-noise PSD weighting, or `AO-001-D` declared weaker positive-semidefinite weighting for each class. A class may be declared unavailable. | Every weighting realization for that class; any associated optimality or uncertainty meaning. |
| `SCI-FLT-MATCHED-SODL-002` | If `AO-001-A` is selected, what exact covariance population, conditioning state, domain/support, constraints, rank/null space, regularization authority, norm, `tau_inv`, `tau_id`, and parent-coefficient role define the GLS identity? | Bind the parent-covariance estimator/model identity and version, declaring or learning algorithm, immutable inputs/provenance, sufficient statistics or theoretical authority, population/conditioning, estimator tolerances, and inverse-construction identity; complete all other fields in `AO-001-A`; separately bind observation/coadd generations. | Exact-GLS claim, minimum-variance label, and `D^-1` marginal-variance use. |
| `SCI-FLT-MATCHED-SODL-003` | If `AO-001-B` is selected, what covariance family and structured inverse approximation are scientifically authorized? | Declare covariance-estimator/model identity/version, declaring or learning algorithm, immutable inputs/provenance, sufficient statistics or theoretical authority, population, model family/parameters, chart/units, structure, window/boundary, rank/null, regularization, coefficient role, residual norm, `epsilon_Q`, unresolved modes, and state-learning inputs. | Structured weighting and every result depending on its model or residual bound. |
| `SCI-FLT-MATCHED-SODL-004` | If `AO-001-C` is selected, what exact radially symmetrized average map-noise PSD definition is authorized? | Declare spectral-estimator identity/version, immutable inputs/provenance and sufficient statistics; estimation population; whether parent-learned or external; WCS chart/metric; coordinate and frequency units; transform phase and normalization; window; member/mode weights; averaging order; radial-bin edges and closure; conjugate/multiplicity rules; PSD units; effective-sample, anisotropy, dispersion, and leakage bounds; mode mask; regularization; parent-coefficient role; support/edge model; and state generation. | The radially averaged PSD weighting candidate and any claim about its population, stationarity, isotropy, covariance meaning, or performance. |
| `SCI-FLT-MATCHED-SODL-005` | If `AO-001-D` is selected, what exact weaker weighting and scientific purpose are authorized? | Declare the external-declaration or learn-once estimator identity/version, immutable inputs/provenance, sufficient statistics, population/state, linear operator, coordinate action, units, support/boundary/window, rank/null, regularization, coefficient role, Hermitian/PSD norms and tolerances, response, and uncertainty nonclaims. | Weaker-weighting realization and its product/state representation. |
| `SCI-FLT-MATCHED-SODL-006` | Which `AO-002` approximation envelope governs each authorized realization? | Select `AO-002-A` exact identity, `AO-002-B` strict uniform (`10^-3` operator/response and `2x10^-3` covariance bounds), or `AO-002-C` named-use stratified ceilings. Bind comparator; separate input, output, and template-anchor projections (`P_in`, `P_out`, `P_anc`) and their codomains; norms; dimensionally matched absolute zero scales; full test cover and regularity proof; state/support/boundary strata; unresolved-mode rule; and `tau_id` where applicable. | Any approximate-operator conformity statement and every approximation-dependent uncertainty or response use. |
| `SCI-FLT-MATCHED-SODL-007` | If `AO-002-C` is selected, what named-use profiles, core/tail measures, consumer projections, sampling cover, and any tighter tolerances apply? | Keep the authored ceilings; owner may tighten but not loosen them under this option identity. Define the 99-percent response-energy core and all tail/zero-response conventions before evaluation. | Each profile-specific approximate realization and dependent consumer. |
| `SCI-FLT-MATCHED-SODL-008` | Which `AO-003` conditional-covariance representation is authorized for each parent class and named consumer? | Select `AO-003-A` exact explicit, `AO-003-B` exact structured, `AO-003-C` authorized projected, `AO-003-D` exact lineage/on-demand, or `AO-003-E` unavailable. Bind the parent-covariance estimator/model identity and version, declaring or learning algorithm, immutable inputs/provenance, sufficient statistics or theoretical authority, population, conditioning, state generation, estimator tolerances, support/response, rank/null, regularization, calibration terms, omitted correlations, query/consumer set, fidelity tolerance, lifecycle, and failure policy. | Conditional-covariance publication, variance/standardization, draws/inversion, and every covariance-dependent consumer. |
| `SCI-FLT-MATCHED-SODL-009` | For any available `AO-003` route, what exact calibration and cross-covariance terms are authorized and which remain unavailable? | Bind CAL/BEAM and parent/template lineages, units, joint terms, omissions, and named consumers. Unknown terms remain unknown. | Calibration-aware covariance and any scalar total uncertainty combining U1--U7. |
| `SCI-FLT-MATCHED-SODL-010` | Which `AO-004` frozen-state persistence route applies to each realization and named use? | Select `AO-004-A` full materialization, `AO-004-B` structured compact state, or `AO-004-C` exact lineage reconstruction. Bind algorithm/version, inputs/population, environment facts, exact/discrete fields, `tau_state`, complete reconstruction cover, advertised query set, archive duration, NOI identity rule, successor rule, and failure policy. | State admission, audit/reanalysis, NOI parity evidence, and completion of any product whose policy requires reconstructable state. |
| `SCI-FLT-MATCHED-SODL-011` | Which `AO-005` response representation and FLT-side handoff interface applies? | Select `AO-005-A` full response, `AO-005-B` exact structured response/query contract, or `AO-005-C` exact lineage/on-demand response. Bind response domain/index/measure, `tau_R`, supported queries and consumers, resource/retention limits, calibration meanings, reconstruction cover, lifecycle, and failure policy. | Response-dependent consumers, reduced-response queries, and the FLT to FRUIT handoff. |
| `SCI-FLT-MATCHED-SODL-012` | Which `AO-006` immutable VAL profile granularity is authorized? | Select `AO-006-A` six role-separated profiles, `AO-006-B` three layers with six mandatory subverdicts, or `AO-006-C` one composite with a six-role verdict vector. Bind PA/SA/SP/CU/NU/FH producer facts, dependencies, named-use policies, actions, unavailable/failed routing, versioning, aggregation, and supersession. | Registered profile evaluation, public-signal publication policy, conditional-covariance and NOI use policy, and FLT to FRUIT handoff verdict. |
| `SCI-FLT-MATCHED-SODL-013` | Which qualified companions are required by each named use, rather than optional? | For each use, decide whether conditional covariance, frozen-NOI uncertainty, response materialization/reconstruction, and state/lineage are required; define failure propagation without changing their estimands. | The named use; optional companion failure otherwise does not invalidate a complete signal bundle. |
| `SCI-FLT-MATCHED-SODL-014` | May the qualified word `optimal` be published for any selected realization? | Authorize only after `AO-001-A` is selected and exact covariance population, inverse closure, support, conditioning, null/constraint, regularization, zero-mean, and validity premises are all part of the frozen contract. Otherwise require `matched-template amplitude estimator` without an optimality claim. | Any optimality or minimum-variance label. |
| `SCI-FLT-MATCHED-SODL-015` | Which template-amplitude conventions and CAL/BEAM lineages are admitted for point-source flux named uses? | Bind exact unit-amplitude template meaning, parent/template joint calibration, covariance/cross-terms, response measure, and validity. Other admitted templates remain shape-amplitude templates. | Literal point-source flux, matched-filter beam/solid angle, or calibration-covariance claim. |
| `SCI-FLT-MATCHED-SODL-016` | Are any intermediate or diagnostic objects to become qualified public science products? | For each proposed numerator, denominator, PSD, kernel, response slice, inverse scale, or standardized field, authorize an exact role, estimand/meaning, units, validity, lifecycle, named-use policy, and failure semantics. The minimal bundle requires none by default. | Publication or scientific consumption of each intermediate/diagnostic. |
| `SCI-FLT-MATCHED-SODL-017` | What immutable supersession and retention policy applies to signal bundles, companions, state generations, and profiles? | Define which successor event marks a predecessor `superseded`, retention/reconstruction duration, dependency versioning, and consumer behavior. Predecessor facts may never be rewritten. | Governance use of successor generations beyond coexistence as immutable distinct products. |

## Decisions that are per-realization declarations, not silent global defaults

Even after the ledger above is disposed, each realization must explicitly bind
the following; no package-wide default may be inferred:

- exact parent bundle, parent class, generation, grouping, WCS/frame, support,
  units/quantity, calibration provenance, and coadd membership;
- exact template-response product, amplitude scaling, phase, anchor, sampling,
  support, lineage, and template generation;
- requested/effective/resolved/realized/published state identities and the
  complete realization tuple;
- complete influence support, valid domain, boundary classes, fill-erosion
  proof, rank/null space, and regularization identity;
- uncertainty availability by U1--U7 term and all unavailable cross-terms;
- lifecycle state, atomic bundle membership, qualified-companion roles,
  provenance, named-use profile versions, and failure reason.

Missing a required per-realization declaration makes the affected route
unavailable; it never activates a default or fallback.

## Routes that remain categorically outside this ledger

No disposition here may authorize posterior/Wiener sky reconstruction,
ordinary convolution as an equivalent method, source/candidate/population/NOI-
member template learning, data-thresholded destriping, adaptive edge/background
methods, partial-support estimation, automatic selectors/fallbacks, historical
high-pass/delta behavior, FLT-owned coaddition, source analysis, SRC ownership,
or FRUIT algorithms/science. Each requires a separate scientific contract if
later commissioned.
