# SCI-FLT-MATCHED v0.1 Scientific-Owner Decision Ledger

Status: Stage B r0.2 draft. The r0.2 directive closes mathematical identities
and refactors the option roles; it selects no `AO-001` weighting, covariance
scope, state/response representation, named-use profile, or numerical route.

Owner: Grant Wilson

## r0.2 owner dispositions

| Disposition ID | State | Binding decision | Consequence |
| --- | --- | --- | --- |
| `SCI-FLT-MATCHED-R0.2-OD-001` | decided 2026-08-31 | Base v0.1 evaluates one anchor at every exact parent ordinary-MAP pixel center and preserves parent WCS/frame/grid/shape/index/order/pixel-center convention, apart from exact support restriction. No interpolation is admitted. | Subpixel, oversampled, or independently sampled output grids require separately named successors. |
| `SCI-FLT-MATCHED-R0.2-OD-002` | decided 2026-08-31 | The scientific operator is the exact normalized operator for selected `W_p`; ordinary finite precision belongs to one preregistered engineering numerical-conformance profile. | No r0.1 numerical constant is scientific authority. A deliberately different scientific operator requires a separate identity and owner-approved error budget. |
| `SCI-FLT-MATCHED-R0.2-OD-003` | decided 2026-08-31 | PA, SA, SP, CU, NU, RU, and FH meanings plus their dependency graph are normative; record layout is lossless representation only. | `AO-006` cannot encode different science by choosing separate, grouped, or vector storage. |

## Disposition rule

A valid remaining disposition names the ledger ID, exact r0.2 alternative or
scope identity, all scientific parameters, parent class and named use,
generation, and superseded decision. Representation-only and engineering
profile fields are declared separately and cannot select science.

Selection updates authority only after the resulting contract bytes are
reviewed and frozen. It establishes no implementation conformity, response or
covariance fidelity, observational validation, performance, readiness,
production, or Unity status.

## Questions and route status

| Ledger ID | State | Exact question / r0.2 alternatives | Route unavailable pending disposition |
| --- | --- | --- | --- |
| `SCI-FLT-MATCHED-SODL-001` | open | Which `AO-001` weight applies independently to observation and coadd parents: A exact constrained local inverse-covariance GLS; B structured covariance-derived exact weight; C radially symmetrized field-power spectral weight; D other declared weaker PSD weight; or unavailable? | Weighting realization for the class and every dependent optimality/uncertainty claim. |
| `SCI-FLT-MATCHED-SODL-002` | open | If `AO-001-A`, bind exact `C_parent` population/conditioning, `E_p`, `U_p/P_p`, positive-definite `C_p,E`, constraints, all four mode classifications, regularization authority, identifiable projected template, coefficient roles, Learn/Resolve provenance, and observation/coadd generations. | Constrained local GLS, minimum-variance wording, and `d_p^-1` variance. |
| `SCI-FLT-MATCHED-SODL-003` | open | If `AO-001-B`, bind covariance family/provenance, exact structured `W_p`, coordinate action/units, population, state, support/boundary/window, rank/subspaces, regularization, coefficient role, diagnostics, and all uncertainty nonclaims. | Structured covariance-derived weighting. |
| `SCI-FLT-MATCHED-SODL-004` | open | If `AO-001-C`, bind exact field source and deterministic-residual imprint; Learn/Resolve population; WCS metric; transform phase/normalization/units; fixed window; finite nonnegative weights and positive denominator; half-open radial bins and tie/final-boundary rule; conjugacy/multiplicity; averaging order; same-unit nonnegative `lambda`; regularized versus excluded-null modes; no-borrowing rule; support/edge; and generation. | Radially symmetrized field-power spectral weighting and every source/population statement. No noise, covariance, stationarity, isotropy, or optimality claim follows by implication. |
| `SCI-FLT-MATCHED-SODL-005` | open | If `AO-001-D`, bind the exact self-adjoint PSD weight, scientific purpose, declaration/Learn provenance, population/state, coordinate action/units, support/boundary/window, rank/subspaces, regularization, coefficient role, response, and uncertainty nonclaims. | Weaker-weight realization. |
| `SCI-FLT-MATCHED-SODL-006` | decided | The r0.2 directive selects exact scientific `n_p/d_p` as invariant and removes r0.1 threshold alternatives. `AO-002-A` is an engineering realization class; `AO-002-B` requires a new explicit owner authorization; `AO-002-C` is typed unavailability. | A numerical-conformance route remains unavailable until one engineering profile is preregistered; no science decision is missing for ordinary finite precision. |
| `SCI-FLT-MATCHED-SODL-007` | superseded | The former core/tail, 99-percent, `10^-2`, and related ceiling question is removed. Any future scientific approximation needs its own exact error-budget authorization under `AO-002-B`. | No r0.1 named-use approximation route exists. |
| `SCI-FLT-MATCHED-SODL-008` | open | For each parent class and named consumer, select covariance scientific scope: `AO-003-A` complete, `AO-003-B` named projected, or `AO-003-C` unavailable. For available scope, declare `AO-003-D` exact resident (explicit or structured) or `AO-003-E` exact lineage/on-demand representation without changing covariance identity. | Covariance publication/use, variance/standardization, draws/inversion, and dependent consumers. |
| `SCI-FLT-MATCHED-SODL-009` | open | For available covariance scope, which CAL/BEAM, parent/template, and cross-covariance terms are authorized, for what units/population/consumers, and which remain unavailable? | Calibration-aware covariance and any combined total uncertainty. |
| `SCI-FLT-MATCHED-SODL-010` | open | What exact state query vocabulary and retention are required by each named use? Declare A full, B compact exact, or C exact lineage representation only after that invariant is fixed. | State audit/reanalysis, NOI parity evidence, and any product requiring reconstructable state. |
| `SCI-FLT-MATCHED-SODL-011` | open | What fixed/FP/realized/reference response domains, query vocabulary, validity, and consumer scope are scientifically required? Then declare A full, B exact structured, or C exact lineage representation; do not infer future FRUIT queries. | Response companion publication/use and response-dependent FLT-to-FRUIT envelope uses. |
| `SCI-FLT-MATCHED-SODL-012` | decided | Seven role meanings and their dependency graph are normative. A/B/C are lossless separate/grouped/vector representations. Each profile must include a distinct RU response publication/use verdict and four SCI-VAL axes. | Actual named-use profile registration remains unavailable until its producer facts, policies, actions, versions, and representation are declared. |
| `SCI-FLT-MATCHED-SODL-013` | open | Which covariance, NOI, response, and state/lineage companions are required for each named use, and how does their failure propagate without changing estimands? | Each named use; optional companion failure otherwise does not invalidate the signal bundle. |
| `SCI-FLT-MATCHED-SODL-014` | open | May `optimal` be published for a selected realization? Only `AO-001-A` plus every local constrained-GLS premise is eligible. | Any realization-level optimality/minimum-variance label. |
| `SCI-FLT-MATCHED-SODL-015` | open | Which unit-amplitude conventions and CAL/BEAM lineages are admitted for point-source flux uses? | Literal point-source flux, matched-filter beam/solid angle, or calibration-covariance claims. |
| `SCI-FLT-MATCHED-SODL-016` | open | Does any intermediate become a qualified public science product? Each needs exact role, meaning, unit, validity, lifecycle, named-use policy, and failure semantics. | Scientific publication/use of each intermediate. |
| `SCI-FLT-MATCHED-SODL-017` | open | What immutable supersession, retention, reconstruction duration, dependency versioning, and consumer behavior apply to bundles, companions, state generations, and profiles? | Governance use of successor generations beyond immutable coexistence. |

## Per-realization declarations

Every realization still binds the exact parent and anchor lattice; template
scientific object and exact representation; Learn/Resolve/Apply identities;
five support roles; exact subspaces and regularization; actual and reference
response/covariance types; U1--U7 availability; complete lifecycle; boundary
records; named-use profile versions; and provenance. Missing facts create typed
unavailability, never defaults.

No disposition here can authorize posterior/Wiener reconstruction, convolution
equivalence, source/candidate/background fitting, adaptive edges, partial
support, automatic fallback, FLT-owned coaddition, SRC ownership, or FRUIT
science.
