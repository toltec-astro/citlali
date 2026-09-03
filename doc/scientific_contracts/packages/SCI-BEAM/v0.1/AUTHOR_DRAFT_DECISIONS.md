# SCI-BEAM v0.1 author draft decisions

Status: implementation-blind Stage B record, document revision `r0.1`, 2026-08-16.

This record contains every author choice, new owner question, scientific inconsistency, unavailable claim, and stated consequence encountered during authorship. It does not modify the approved packet or close an upstream dependency.

## Author choices

| ID | Choice | First-principles reason | Consequence |
| --- | --- | --- | --- |
| SCI-BEAM-ADC-001 | Use one elliptical Gaussian beam convolved with the declared calibrator model. | It is the least v0.1 family satisfying the approved elliptical/finite-source boundary. | Multi-component shoulders/sidelobes are model inadequacy, not silently fitted components. |
| SCI-BEAM-ADC-002 | Give the calibrator a finite, zero first moment and normalize the convolved template to unity at that declared reference origin. | This makes `T(0)=1`, the translated modeled centroid, and the amplitude estimand exact without assuming an asymmetric convolved profile peaks at the origin. | `A` is excess at the translated reference origin in signal units, not necessarily the profile peak, intrinsic flux, or integrated beam power. |
| SCI-BEAM-ADC-003 | Order axes major >= minor and define theta counterclockwise from WCS +azimuth over `[0,pi)`. | Ordering removes label degeneracy; a tensor ellipse is invariant under a pi rotation. | Theta is unavailable at exact circularity. |
| SCI-BEAM-ADC-004 | Bound background to constant or plane about a recorded origin. | These are the smallest local nuisance families covering offset and gradient without authorizing flexible source absorption. | Higher orders require a successor decision; even a plane may be unavailable on weak support. |
| SCI-BEAM-ADC-005 | Use declared generalized least squares/likelihood as the normative objective. | It preserves covariance meaning and exposes approximations. | Silent weights/regularization are nonconformant; approximate errors are conditional. |
| SCI-BEAM-ADC-006 | Permit local inverse-Hessian covariance only under explicit regularity/identifiability conditions. | Curvature is not uncertainty when parameters are unidentified, constrained at a boundary, or the likelihood is misspecified. | Covariance can be partially or wholly unavailable despite optimizer completion. |
| SCI-BEAM-ADC-007 | Require nuisance propagation or named omission, with cross-terms when dependent. | Total uncertainty follows the joint quantity, not a list of independent variances. | The word “total” is unavailable if material terms are missing. |
| SCI-BEAM-ADC-008 | Limit soft priors to locator initialization and bounded gating; exclude them from the measurement objective. | This is the strongest implementation-blind way to preserve “soft” meaning and allow data to defeat a wrong prior. | Candidate route/influence must be recorded and blind fallback must remain available. |
| SCI-BEAM-ADC-009 | Compare prior-guided and blind candidates under a common measurement objective. | Different scoring operators would confound evidence with route. | A selected source is attributable to data under one declared model, subject to admitted support. |
| SCI-BEAM-ADC-010 | Define convergence using declared per-parameter metrics plus separate parameter-availability, objective, selected-candidate, support, and valid-detector stability. Orientation uses modulo-$\pi$ distance and is omitted numerically only when unavailable at both transitions. | Raw vector subtraction is invalid for periodic or unavailable parameters, and any one component can stop while the realized estimator is still changing. | Every stability component is provenance-bearing; no numeric tolerance/default is invented and maximum iteration is non-converged. |
| SCI-BEAM-ADC-011 | Allow partial availability of parameters and interpretations. | A payload can identify an effective amplitude while source/response nuisance prevents intrinsic beam meaning. | Terminal and per-quantity states must preserve causes. |
| SCI-BEAM-ADC-012 | Define calibration candidate as `K=F_source/A` only under matching conventions. | A ratio is meaningful only after numerator/denominator estimands and units align. | It remains unpromoted; missing covariance/dependence makes candidate uncertainty unavailable. |
| SCI-BEAM-ADC-013 | Treat required result/QC/identity/policy/uncertainty/prior/convergence/candidate/provenance as one atomic bundle. | Consumers cannot assess a scientific number without its identity and limitations. | Failure of any required member fails publication. |
| SCI-BEAM-ADC-014 | Separate hard algebraic/identifiability validity from thresholded review cues. | Physical/model-domain conditions differ from empirical production policies. | v0.1 does not declare production thresholds. |
| SCI-BEAM-ADC-015 | Use 46 requirements and 24 predictions with sequential stable IDs. | This gives complete traceability at the granularity of one independently checkable obligation/invariant. | Future changes must preserve IDs or explicitly version/supersede them. |
| SCI-BEAM-ADC-016 | Use one shared LaTeX core imported into both views. | It prevents independent engineering science. | Engineering prose is procedural only; equations/requirements/predictions are shared verbatim. |

## New scientific-owner questions

| ID | Precise question | Why open | Consequence until answered |
| --- | --- | --- | --- |
| SCI-BEAM-ODQ-001 | What is the exact one-to-one mapping and text of `BEAM-SCOPE-D001--D012` that may be exposed in the final crosswalk? | The approved packet names those IDs but does not include their individual dispositions; the manifest forbids opening the decision log/owner ledger. | Crosswalk cites exact owner-approved packet sections, not inferred D-ID mappings. This does not block the draft. |
| SCI-BEAM-ODQ-002 | Which numerical support, convergence, QC, and S/N policies are approved for any production profile? | The packet explicitly withholds production thresholds. | v0.1 keeps all such numbers required external policy inputs. |
| SCI-BEAM-ODQ-003 | What declared number of consecutive stable transitions is required for each effective iteration policy? | The scientific conjunction is defined, but no owner-approved count is supplied. | It must be explicit in realized policy; no default is supplied. |
| SCI-BEAM-ODQ-004 | Which singular-covariance retained-subspace or regularization procedures, if any, are authorized? | A procedure changes the objective and information content. | Singular covariance is unavailable unless a declared procedure is separately approved and recorded. |
| SCI-BEAM-ODQ-005 | What exact model-inadequacy diagnostics and owner-controlled dispositions are approved? | Residual structure must not be confused with optimizer failure, but no diagnostic family/threshold is authorized. | Model inadequacy is a typed cause/review state without a production cut. |
| SCI-BEAM-ODQ-006 | Which source/amplitude unit and normalization combinations permit a CAL candidate in each admitted data mode? | The algebraic ratio is defined, but upstream calibration/source conventions remain conditional. | Candidate state is unavailable unless compatibility is demonstrated for that observation. |
| SCI-BEAM-ODQ-007 | Which response completeness statement permits interpretation as intrinsic detector-plus-telescope beam rather than conditioned effective response? | RTC/PTC and SCI-MAP response authorities remain open. | Intrinsic-beam interpretation stays unavailable when response is incomplete. |
| SCI-BEAM-ODQ-008 | Should a successor authorize richer beam/background families, and with what selection and identifiability rule? | v0.1 deliberately chooses the minimal authorized base family. | Multi-component structure is recorded as possible model inadequacy, not fitted ad hoc. |

## Scientific inconsistencies or tensions

| ID | Observation | Disposition | Consequence |
| --- | --- | --- | --- |
| SCI-BEAM-INC-001 | The task invokes `BEAM-SCOPE-D001--D012`, while their individual text is outside the allowed packet. | Do not inspect or reconstruct them; use the approved dispositions actually restated in the Scope Brief and sanitized conventions. | Owner-D-ID traceability is explicitly unavailable pending ODQ-001. |
| SCI-BEAM-INC-002 | The active map unit is `mJy/beam`, yet the packet states that the label alone does not establish absolute calibration or complete response. | Preserve unit and authority as separate state. | An amplitude can be in `mJy/beam` while calibration promotion or intrinsic-beam interpretation remains unavailable. |
| SCI-BEAM-INC-003 | A soft prior may define bounded gating, yet a strong observed source must be able to defeat a wrong prior. | Always retain a declared blind route and compare admitted candidates under one measurement objective. | A gate alone cannot veto the blind candidate. |
| SCI-BEAM-INC-004 | An ellipse requires orientation, but a circular profile has none. | Canonicalize ordered axes and mark theta unavailable at exact circularity. | No arbitrary zero-angle scientific claim. |
| SCI-BEAM-INC-005 | SCI-BEAM may form a calibration candidate but does not own calibration promotion or sensitivity. | Define only the typed ratio and uncertainty conditions. | Candidate outputs carry explicit non-promotion and sensitivity remains unavailable. |

## Unavailable claims and dependencies

| ID | Unavailable claim | Cause/owner | Consequence |
| --- | --- | --- | --- |
| SCI-BEAM-UNV-001 | Current implementation conformance | Implementation was prohibited input and no conformance evidence was authorized. | This draft cannot claim current behavior or repairs. |
| SCI-BEAM-UNV-002 | Representation/response fidelity | ALIGN/AST, RTC/PTC, VAL, SCI-MAP, and SCI-CAL inputs are conditional. | Fits describe only the admitted bundle; intrinsic interpretation is conditional. |
| SCI-BEAM-UNV-003 | Observational bias, coverage, and uncertainty calibration | No validation or observational-performance evidence is admitted. | Predictions remain falsifiable tests, not passed results. |
| SCI-BEAM-UNV-004 | Production readiness | Numerical thresholds, operations, and production evidence are not approved inputs. | Compile and QA do not freeze or accept v0.1. |
| SCI-BEAM-UNV-005 | Absolute astrometry, pointing correction, timing, or detector-coordinate truth | ALIGN/AST owns these quantities. | BEAM reports only a relative centroid in the admitted frame. |
| SCI-BEAM-UNV-006 | Absolute calibration promotion | SCI-CAL owns signal meaning and promotion. | `K` is a candidate only. |
| SCI-BEAM-UNV-007 | BEAM-owned sensitivity | Exact noise/time/atmosphere/calibration/bandwidth convention is absent and downstream-owned. | Sensitivity is not a v0.1 BEAM result. |
| SCI-BEAM-UNV-008 | Intrinsic detector-plus-telescope beam under incomplete response | RTC/PTC and SCI-MAP response status is incomplete or unavailable. | Report conditioned effective profile only. |
| SCI-BEAM-UNV-009 | Total covariance with missing nuisance or dependence terms | Upstream uncertainty/cross-covariance not supplied. | Report component/conditional covariance and name omissions. |
| SCI-BEAM-UNV-010 | Universal beam-width, fit-radius, S/N, QC, or convergence thresholds | Primary-reference boundary and Scope Brief forbid invention. | Thresholds are explicit versioned policy inputs. |
| SCI-BEAM-UNV-011 | Exact UID truth from TolAPT soft prior | Prior identity is array/network/slot-local, not detector truth. | Use stable artifact binding; ambiguous relations reject. |
| SCI-BEAM-UNV-012 | Cross-observation learning/restart semantics | FRUIT owns general recurrence and restart. | Internal loop ends within one immutable observation result. |
| SCI-BEAM-UNV-013 | Source selection or catalog flux estimation | TolProj/TolTECA photometry boundary owns them. | BEAM records immutable supplied source model only. |
| SCI-BEAM-UNV-014 | Downstream matched/reference APT correctness | TolAPT owns construction and later priors. | BEAM provides compatibility-bearing inputs without redefining downstream products. |

No item above is closed by clean compilation, mechanical checks, PDF rendering, or visual inspection.
