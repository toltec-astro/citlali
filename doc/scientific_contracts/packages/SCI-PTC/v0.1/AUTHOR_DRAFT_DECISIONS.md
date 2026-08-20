# SCI-PTC v0.1 -- Stage B Author Decisions And Owner Ledger

Status: bounded scientific-owner freeze-candidate revision, document revision `r0.3`

Date: `2026-08-20`

This ledger records choices made while consolidating the owner-approved author
packet. It does not report implementation behavior or validation evidence.

## Author Draft Decisions

| ID | Decision | Consequence |
| --- | --- | --- |
| `PTC-AUTH-D001` | Use the calibrated detector-time model `signal + shared/template component + remaining noise` as the organizing model for every admitted base estimator family. | Estimator families remain specializations; correlation never establishes physical origin. |
| `PTC-AUTH-D002` | Treat the realized removed subspace, modeled subtraction, additive reference, gauge, and null space as the central scientific result alongside transformed TOD. | Same-unit output cannot be used as a response claim. |
| `PTC-AUTH-D003` | Express missing-data handling through estimator-specific zero influence and declared support, never generic zero filling. | Ordinary zero-filled PCA is outside the contract unless equivalence is proved. |
| `PTC-AUTH-D004` | Treat within-array array/network/local components as one declared hierarchical family with exact joint or sequential order. | No cross-array authority or order-independent assumption is introduced. |
| `PTC-AUTH-D005` | Implement D012 as conjunctive least-aggressive admission, not a weighted scalar optimization. | Every required predicate must pass; nonnested candidates use complete subspace and response. |
| `PTC-AUTH-D006` | Treat support-changing post-fit refinement as complete replay from the immutable CAL parent. | Cleaned output is never the detector-QC refit parent; sequential residual fitting is only an explicit hierarchical estimator stage. |
| `PTC-AUTH-D007` | Interpret resolved Q001 strictly: base-v0.1 conditioned-`r` analysis is inert/advisory. | It cannot change calibrated-`x` membership, subtraction, output, or coefficients. |
| `PTC-AUTH-D008` | Separate fitted loadings, coordinate parameters, diagnostics, and analysis/gridding coefficients at the type level. | Only a named analysis/gridding family may be MAP-facing; no coefficient is precision by default. |
| `PTC-AUTH-D009` | Separate fixed-state response companions, full PTC-procedure response, and whole-chain response studies. | PTC full-procedure response begins at the immutable CAL parent; upstream relearning remains cross-package. |
| `PTC-AUTH-D010` | Use typed availability rather than invented numeric defaults for missing estimator, threshold, covariance, response, or reference-map authority. | The affected product/claim fails closed while independent supported roles may remain available. |
| `PTC-AUTH-D011` | Preserve the 80 r0.1 requirement and 38 r0.1 prediction IDs, then append eight requirements and ten predictions for owner-approved r0.2 scope completion. | Existing identifiers are not renumbered; the r0.2 formal core contains 88 requirements and 48 predictions. |
| `PTC-AUTH-D012` | The r0.1 hybrid science view was useful for first review but is superseded for circulation by the owner-requested RTC-style split. | The standalone rationale explains the science and provides compact traceability; the engineering view alone carries the complete shared formal core. |
| `PTC-AUTH-D013` | Implement the owner-approved base-v0.1 PCA/SVD application rule as projection through the frozen realized subspace and metric, with linear coefficient recomputation. | Equality to `Y - Uhat` on the fitted parent no longer defines response-companion application. |
| `PTC-AUTH-D014` | Make temporal-left, detector-right, two-sided, detector/time-specific, and general vectorized actions explicit family state rather than selecting one universal orientation. | Every realization publishes its exact acting space and operator; frozen numerical subtraction is a distinct affine family with identity derivative. |
| `PTC-AUTH-D015` | Separate the PTC-local derivative/linear part, complete CAL-to-PTC chain response, and realized propagated companion by domain. | Source-domain parents use `K_CAL` exactly once; CAL-grid companions enter the local PTC operator directly. |
| `PTC-AUTH-D016` | Define full-procedure finite differences on the transformed-signal projection only and retain changed discrete/mixed state through a typed comparison. | Rank, mask, group, class, support, stop, and fallback state are never numerically subtracted. |
| `PTC-AUTH-D017` | Restore requested, effective-policy, observation-resolved, learned/fitted-evidence, resolved-selected, applied/realized, and published states. | Every candidate and predicate disposition remains reconstructible instead of collapsing into the selected rank. |
| `PTC-AUTH-D018` | Separate product realization from response/covariance availability and clean symbol roles, common-mode weights, diagnostic indices, and coefficient domains. | Disabled or not-requested products are not encoded as invalid responses; stage identity `q` is no longer reused as a diagnostic. |
| `PTC-AUTH-D019` | Add explicit source/residual, shifted/null-surrogate, typed astronomical-transfer, and estimator expectation/bias obligations already required by the approved scope. | r0.2 completes omitted formal coverage without broadening the scientific boundary. |
| `PTC-AUTH-D020` | Move program-adherence prose to a provenance appendix and add two compact vector schematics to the standalone rationale. | The first numbered section is the calibrated detector-time science while package-level program adherence remains explicit. |
| `PTC-AUTH-D021` | Separate the latent correlated component `U_*` from its fitted estimate `Uhat` and the realized removed subspace. | Family-specific expectation and bias claims no longer conflate the nuisance estimand with its realization. |
| `PTC-AUTH-D022` | Rename the source-domain-to-CAL-grid operator `K_up->CAL` and define it as the complete admitted upstream response carried by the CAL product. | The chain includes every applicable admitted source-to-detector, beam-convention, coordinate/scan, RTC, and CAL operation and cannot be mistaken for only the CAL-owned multiplier. |
| `PTC-AUTH-D023` | Make fit-excluded application availability an explicit scientific contract rather than an inferred extension of the fit population. | Append Definition 040, Assumption 028, Requirement 089, and Prediction 049; preserve every r0.2 normative ID. |
| `PTC-AUTH-D024` | Add the bounded science validation program, estimator-family orientation, surrogate purpose, resolved lifecycle wording, and named orphan-heading correction requested by the r0.2 review. | The rationale becomes a freeze-candidate content revision without adding thresholds, results, default methods, implementation evidence, or a broad stylistic rewrite. |

## Scientific Owner Decision Ledger

| ID | Owner | State | Decision or evidence required | Blocked claim/output | Resolution authority/date | Affected documents |
| --- | --- | --- | --- | --- | --- | --- |
| `PTC-OWNER-Q001` | Grant Wilson | decided | Base v0.1 uses diagnostic-only, inert/advisory conditioned `r`. | Cross-channel `x <- r` subtraction and `r`-controlled `x` decisions are unavailable. | Grant Wilson, `2026-08-19` | All shared modules and both views |
| `PTC-OWNER-Q002` | Grant Wilson | decided | Base-v0.1 PCA/SVD applies any admitted science input or fixed-state companion by projection through the frozen realized subspace and metric; each family declares exact acting space and coefficient recomputation. | Frozen numerical reconstruction subtraction is not the PCA/SVD response rule; it remains only a separately named affine family. | Grant Wilson, `2026-08-20` | Definitions 014, 029--032; Equations 2, 8--9, 15--16; Requirements 019, 029, 062, 083; Predictions 039--041; both views |
| `PTC-OD-001` | Scientific owner | open | Select numerical residual-contamination, typed astronomical-transfer, conditioning, support, stability, and QC predicates for a concrete product role. | Automatic candidate admission for that role. | Scientific owner; date open | Requirements 034--040 and 085; predictions 017--020 and 046 |
| `PTC-OD-002` | Scientific owner | open | Select a mandatory/default estimator family, if absence of an explicit request is ever to be accepted. | Automatic operation when estimator family is absent. | Scientific owner; date open | Requirements 008, 026--030 |
| `PTC-OD-003` | Scientific owner | open | Select concrete centering/scaling estimators and boundaries for each intended product role. | Automatic coordinate resolution when the request is incomplete. | Scientific owner; date open | Requirements 023--024 |
| `PTC-OD-004` | Scientific owner | open | Select registered post-fit diagnostic thresholds and finite refinement count for each product role. | Policy-changing detector classification; diagnostics remain reportable when otherwise defined. | Scientific owner; date open | Requirements 042--048 |
| `PTC-OD-005` | SCI-RTC or named conditioner | known but not supplied | Provide a PTC-compatible conditioned-`r` product with exact operator, unit, response, validity, optical-leakage state, and `x`-grid relation. | Optional diagnostic-$r$ branch. | Upstream owner; date open | Requirement 030; predictions 025--026 |
| `PTC-OD-006` | SCI-MAP | known but not supplied | Provide an exact named reference-map operator and center selector. | Optional estimated map-center point-source response diagnostic. | SCI-MAP owner; date open | Requirements 067--068; prediction 034 |
| `PTC-OD-007` | SCI-CAL | known but not supplied | Supply a numerically admitted CAL product with complete response and uncertainty state for the requested observation. | Numerical calibrated PTC product and complete composed response. | SCI-CAL owner; date open | Requirements 001--003, 061 |
| `PTC-OD-008` | Scientific owner | open | Select perturbation amplitude, side, source state, locations, validity domain, and typed state-comparison fields for a requested full-procedure response study. | A concrete full-procedure response family. | Scientific owner; date open | Requirements 063--065; prediction 042 |
| `PTC-OD-009` | NOI / scientific owner | deferred | Select or approve covariance and selection-uncertainty models for claims requiring more than conditional covariance. | Complete covariance, precision, and significance claims. | NOI/scientific owner; date open | Requirements 057--060 |
| `PTC-OD-010` | SCI-MAP / scientific owner | open | Select any MAP-facing analysis/gridding coefficient family and its normalization. | MAP consumption of a PTC-derived coefficient; transformed TOD remains independent. | SCI-MAP/scientific owner; date open | Requirements 052--055 |
| `PTC-OD-011` | Source-model owner | known but not supplied | Supply an exact source model/mask/residual or prior-pass parent with owner, unit, frame, registration, validity, response, support, and recurrence identity for a protected/fitted role. | Source-protection or external-parent fitting claim for that role. | Named source owner; date open | Requirements 018, 041, 081; prediction 044 |
| `PTC-OD-012` | Validation program | deferred | Supply implementation, representation-fidelity, observational-performance, and readiness evidence under later authorization. | Conformity, achieved transfer/performance, and production claims. | Later validation authority; date open | Requirement 080; prediction 038 |

## Draft Blocking Assessment

No open item blocks the structural scientific contract or either document
view. Open and known-but-not-supplied items block only automatic/default
instantiation or the specifically named numerical product or claim. No
conformity, validation, performance, freeze, or readiness claim is made.
