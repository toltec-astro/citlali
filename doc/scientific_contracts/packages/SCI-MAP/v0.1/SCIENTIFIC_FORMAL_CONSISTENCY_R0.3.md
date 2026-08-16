# SCI-MAP v0.1 Rationale r0.3 Formal-Contract Consistency Report

Status: complete for the r0.3 correction and freeze pass

Date: `2026-08-16`

The formal scientific/engineering contract remains normative. The r0.3
rationale preserves the v0.1 estimator and all 52 requirement IDs and 25
prediction IDs while incorporating the bounded owner-approved CI-001
dimensional correction in the shared authority.

| Consistency axis | r0.3 result |
| --- | --- |
| Dimensional consistency | Consistent: `coverage_cut` is a dimensionless support-policy scalar. SCI-MAP-CI-001 is resolved; the shared equations, REQ-031/032, and PRED-012 carry the same classification. |
| Remaining `coverage_cut` authority | Consistent: OD-007 remains open only for numerical domain, boundary cases, recommended range, policy authority, and failure behavior. No range is inferred. |
| Upstream ownership and inherited status | Consistent: SCI-CAL owns calibrated scale/unit/beam/validity/uncertainty; ALIGN/AST owns coordinates/frame/validity; PTC owns processed samples and coefficient/covariance meaning; VAL owns eligibility; NOI owns empirical significance. |
| Map and coadd estimands | Consistent: both are positive-coefficient normalized averages on declared valid domains. Bundle admission is atomic; pixel-level support determines contribution locations. |
| Projection and coefficient meaning | Consistent: the worked example explicitly uses independent unit-variance samples and true inverse-variance coefficients. Projection conservation and boundary normalization remain OD-008. |
| Normalization, precision, and covariance | Consistent: normalization and formal precision remain separate; the symmetric split example has covariance `1/2` and unity correlation. |
| WCS and common-grid placement | Consistent with ADR 0009: FITS WCS supports ordinary pixel-to-sky use at declared serialization fidelity; the sidecar retains exact admission/conformance/provenance/coadd identity. |
| Claim-layer separation | Consistent: no implementation, representation-fidelity, observational-performance, or production-readiness claim was introduced. |

The external scientist crosswalk, canonical crosswalk, owner ledger, generated
decision register, and all three rendered views are mechanically checked as a
single bundle. OD-001--009 remain open; CI-001 is not an open decision.
