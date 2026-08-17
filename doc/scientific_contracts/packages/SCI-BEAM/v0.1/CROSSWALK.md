# SCI-BEAM v0.1 r0.3 Rationale-to-Contract Crosswalk

Status: frozen scientific authority; exact implementation-independent traceability

`SR` denotes the science-team rationale and `FC` the Formal
Scientific/Engineering Contract. Exact equations, states, requirements, and
predictions are normative in the shared formal core. The rationale explains
the same science without reproducing the complete inventory.

## Requirements

| Requirement | Scientist-facing source | Engineering interpretation |
| --- | --- | --- |
| SCI-BEAM-REQ-001 | SR Executive; Sec. 8 | Bind the complete immutable result and policy identity. |
| SCI-BEAM-REQ-002 | SR Executive; Sec. 1 | Preserve raw `Delta f/f` meaning and conditioning. |
| SCI-BEAM-REQ-003 | SR Sec. 1 | Fit standardized raw-signal detector maps only. |
| SCI-BEAM-REQ-004 | SR Secs. 1, 3 | Preserve complete parent-map response/provenance. |
| SCI-BEAM-REQ-005 | SR Secs. 3, 9 | Establish the orthonormal WCS/Jacobian metric. |
| SCI-BEAM-REQ-006 | SR Sec. 2 | Bind the finite-source model and reference origin. |
| SCI-BEAM-REQ-007 | SR Sec. 2 | Require the fixed nominal-beam TOA source flux in mJy per nominal beam. |
| SCI-BEAM-REQ-008 | SR Secs. 1, 3 | Exclude invalid payloads causally. |
| SCI-BEAM-REQ-009 | SR Sec. 3 | Realize the complete map-domain forward model. |
| SCI-BEAM-REQ-010 | SR Sec. 3 | Fit all three 2-D tensor degrees jointly. |
| SCI-BEAM-REQ-011 | SR Sec. 3 | Enforce ordered axes, periodic angle, circular limit. |
| SCI-BEAM-REQ-012 | SR Sec. 2 | Use reference-normalized finite-source template. |
| SCI-BEAM-REQ-013 | SR Secs. 2, 5 | Keep fitted amplitude in observed-plane `Delta f/f`. |
| SCI-BEAM-REQ-014 | SR Secs. 3, 7 | Bound and record the background family. |
| SCI-BEAM-REQ-015 | SR Secs. 1, 3, 7 | Treat support and map extent as estimator state. |
| SCI-BEAM-REQ-016 | SR Sec. 3 | Use declared map covariance/objective with no silent regularization. |
| SCI-BEAM-REQ-017 | SR Secs. 1, 3, 8 | Publish the residual map on admitted support. |
| SCI-BEAM-REQ-018 | SR Sec. 3 | Assess joint identifiability; unavailable is explicit. |
| SCI-BEAM-REQ-019 | SR Secs. 3, 9 | Resolve and perturbation-check the complete map-model Jacobian. |
| SCI-BEAM-REQ-020 | SR Secs. 3, 9 | Retain map-fit covariance and material fitted-parameter cross terms. |
| SCI-BEAM-REQ-021 | SR Sec. 3 | Label approximations and use invariant circular covariance. |
| SCI-BEAM-REQ-022 | SR Secs. 1, 7 | Limit fit meaning to effective core unless stronger evidence exists. |
| SCI-BEAM-REQ-023 | SR Secs. 1, 7, 8 | Preserve empirical map and adequacy/wing companions. |
| SCI-BEAM-REQ-024 | SR Sec. 3 | Soft prior guides location but is not measurement truth. |
| SCI-BEAM-REQ-025 | SR Sec. 3 | Keep blind fallback and common-objective selection. |
| SCI-BEAM-REQ-026 | SR Secs. 3, 8 | Preserve prior compatibility, route, and influence. |
| SCI-BEAM-REQ-027 | SR Sec. 3 | Record the complete observation-local estimator trace. |
| SCI-BEAM-REQ-028 | SR Sec. 3 | Convergence requires stable parameters, states, objective, candidate, support, and detector set. |
| SCI-BEAM-REQ-029 | SR Executive; Sec. 8 | Publish independent per-quantity state for every detector. |
| SCI-BEAM-REQ-030 | SR Secs. 3, 8 | Enforce parameter/model validity without collapsing other fields. |
| SCI-BEAM-REQ-031 | SR Secs. 1, 7, 8 | Keep validity, adequacy, health, calibration, sensitivity, and kernel use separate. |
| SCI-BEAM-REQ-032 | SR Secs. 7, 9 | Numerical thresholds require evidence-based versioned policy. |
| SCI-BEAM-REQ-033 | SR Secs. 1, 7, 9 | Quantify model inadequacy and hidden response. |
| SCI-BEAM-REQ-034 | SR Sec. 3 | Report FWHM and conditional Gaussian-core area only. |
| SCI-BEAM-REQ-035 | SR Sec. 7 | Derive broadening tensor with PSD/uncertainty disposition. |
| SCI-BEAM-REQ-036 | SR Sec. 4 | Declare centroid-to-detector sign/frame transformation and derive rotation from parent-sample contribution support. |
| SCI-BEAM-REQ-037 | SR Sec. 4 | Record origin gauge and do not claim physical boresight/pivot. |
| SCI-BEAM-REQ-038 | SR Sec. 4 | Require the same immutable APT artifact and AST convention for pointing transfer. |
| SCI-BEAM-REQ-039 | SR Sec. 5 | BEAM publishes TOA nominal-beam `flxscale` from the required mJy-per-nominal-beam source flux. |
| SCI-BEAM-REQ-040 | SR Secs. 3, 5, 8, 9 | Preserve calibration lineage and separate derived-product covariance/correlation. |
| SCI-BEAM-REQ-041 | SR Sec. 6 | BEAM publishes strictly positive scan-domain NEFD-like `sens` using `|flxscale|`. |
| SCI-BEAM-REQ-042 | SR Secs. 3, 6, 9 | Keep exact `sens` policy open and retain scan statistics, scatter, and derived-product uncertainty. |
| SCI-BEAM-REQ-043 | SR Sec. 8 | `responsivity` is deprecated and noncanonical. |
| SCI-BEAM-REQ-044 | SR Sec. 8 | Publish mandatory APT with per-row, global-lineage, and dense-companion content. |
| SCI-BEAM-REQ-045 | SR Sec. 7 | Stacked PSFs are optional diagnostics, never implicit kernels. |
| SCI-BEAM-REQ-046 | SR Sec. 9 | Separate evidence layers and require the named future studies. |

## Predictions and edge cases

| Prediction | Scientist-facing source | Formal falsifiable interpretation |
| --- | --- | --- |
| SCI-BEAM-PRED-001 | SR Executive; Sec. 1 | Equal payload with different raw-signal convention is incompatible. |
| SCI-BEAM-PRED-002 | SR Sec. 1 | Calibrated-map or timestream substitution changes the estimand. |
| SCI-BEAM-PRED-003 | SR Sec. 3 | Circular injection recovers amplitude/centroid/common width, not angle. |
| SCI-BEAM-PRED-004 | SR Sec. 3 | Rotated ellipse requires the complete tensor. |
| SCI-BEAM-PRED-005 | SR Sec. 3 | Tensor canonicalization respects axis and period invariance. |
| SCI-BEAM-PRED-006 | SR Secs. 3, 9 | WCS/Jacobian recovery exposes omitted metric terms. |
| SCI-BEAM-PRED-007 | SR Sec. 2 | Finite-source model has the declared point limit and reference amplitude. |
| SCI-BEAM-PRED-008 | SR Secs. 2, 5 | A source flux with the wrong unit, nominal beam, or reference origin is incompatible. |
| SCI-BEAM-PRED-009 | SR Sec. 5 | Correction/transmission forms agree under BEAM source-atmosphere ownership. |
| SCI-BEAM-PRED-010 | SR Secs. 5, 8 | Fit can remain valid when calibration is unavailable. |
| SCI-BEAM-PRED-011 | SR Secs. 6, 8 | Either `flxscale` sign gives the same positive sensitivity magnitude; failed support leaves sensitivity unavailable. |
| SCI-BEAM-PRED-012 | SR Secs. 1, 7 | Precise core does not establish wing/complete-PSF state. |
| SCI-BEAM-PRED-013 | SR Secs. 3, 7 | Background/wing degeneracy appears as support-dependent inadequacy. |
| SCI-BEAM-PRED-014 | SR Secs. 3, 9 | Derivative methods agree; map-fit and derived-product Jacobians remain distinct and cross-stage dependence is retained. |
| SCI-BEAM-PRED-015 | SR Secs. 3, 4, 9 | Tensor/covariance rotate covariantly and circular angle disappears. |
| SCI-BEAM-PRED-016 | SR Sec. 7 | Compatible injected broadening is PSD; indefinite results are not clipped. |
| SCI-BEAM-PRED-017 | SR Secs. 4, 9 | Declared centroid sign and parent-sample-weighted derotation succeed where sign, pixel-weight, or average-angle substitutions fail. |
| SCI-BEAM-PRED-018 | SR Sec. 4 | Common origin translation leaves relative geometry invariant. |
| SCI-BEAM-PRED-019 | SR Secs. 4, 9 | Pivot perturbation and same immutable APT-artifact transfer have the declared behavior. |
| SCI-BEAM-PRED-020 | SR Sec. 3 | Soft-prior routing cannot move the prior-free optimum or veto blind evidence. |
| SCI-BEAM-PRED-021 | SR Sec. 3 | Optimizer completion does not override unstable convergence components. |
| SCI-BEAM-PRED-022 | SR Sec. 8 | Removing `responsivity` changes no canonical result. |
| SCI-BEAM-PRED-023 | SR Executive; Sec. 8 | Independent states coexist and every detector remains represented. |
| SCI-BEAM-PRED-024 | SR Secs. 7, 9 | Hidden response/model mismatch can affect science despite finite core parameters. |

Count: 46 requirements and 24 predictions, each represented exactly once.

## Open decision groups

| Stable group ID | Rationale register | Formal contract | Atomic external ledger |
| --- | --- | --- | --- |
| `SCI-BEAM-OD-001` | Sensitivity estimator and support | Open decision groups | `SCI-BEAM-ODQ-101--103` |
| `SCI-BEAM-OD-002` | Adequacy, wings, and kernels | Open decision groups | `SCI-BEAM-ODQ-104--106`, `SCI-BEAM-ODQ-109` |
| `SCI-BEAM-OD-003` | Physical pivot registration | Open decision groups | `SCI-BEAM-ODQ-107--108` |
