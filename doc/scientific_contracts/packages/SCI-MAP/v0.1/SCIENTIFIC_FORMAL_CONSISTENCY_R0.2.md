# SCI-MAP v0.1 Rationale r0.2 Formal-Contract Consistency Report

Status: complete for the r0.2 authorship revision

Date: `2026-08-16`

The formal scientific/engineering contract remains normative. The science
rationale changes audience, teaching order, and ownership clarity without
changing the v0.1 estimator or its 52 requirements and 25 predictions.

| Consistency axis | r0.2 result |
| --- | --- |
| Dimensional consistency | One formal inconsistency found: the threshold equations force `coverage_cut` to be dimensionless while r0.1 leaves its unit status open. `SCI-MAP-CI-001` records the exact proposed amendment; normative clauses remain unchanged pending owner approval. |
| Upstream ownership and inherited status | Consistent: SCI-CAL owns calibrated scale/unit/beam/validity/uncertainty; ALIGN/AST owns coordinates/frame/validity; PTC owns processed samples and coefficient/covariance meaning; VAL owns eligibility; NOI owns empirical significance. MAP transforms without strengthening claims. |
| Map and coadd estimands | Consistent: both are positive-coefficient normalized averages on their declared valid domains. Neither is unconvolved sky truth, and no unsupported storage value becomes a measured zero. |
| Projection and coefficient meaning | Coefficient meaning is resolved by approved PTC D004 as scalar analysis/gridding status. Projection conservation and boundary normalization remain a real gap, now OD-008. |
| Normalization versus precision | Consistent: `Q` is gridding normalization by default. The worked fractional example preserves the formal result `Q=1`, variance `1/2`, and marginal inverse variance `2`. |
| Response and covariance | Consistent: response is `R=AH`; fixed-state covariance is `A N A^T`; response/tracer/unavailable status and consequential correlations remain visible. |
| Support and validity | Consistent: all eight formal facts are retained and translated without aliasing. The global-population dependence of the provisional threshold policy is explicit. |
| WCS and common-grid placement | Consistent with ADR 0009 and its 2026-08-05 owner amendment. Typed/sidecar WCS remains lossless authority; FITS is the ordinary science-tool representation within the approved tolerance. Centered-integer placement and odd-shape rejection are unchanged. Future grid preparation/reprojection ownership remains OD-009. |
| Claim-layer separation | Consistent: algebraic contract, implementation conformance, representation/response fidelity, observational validation, and production readiness remain independent. No achieved claim was introduced. |

## Open consistency dependencies

- SCI-MAP-CI-001 requires owner approval before the formal
  `coverage_cut` clauses can be corrected.
- OD-001--OD-007 remain open as recorded; after SCI-MAP-CI-001 approval,
  OD-007 should be narrowed to numerical domain and failure behavior.
- OD-008 is required before a general projection-conservation or boundary-
  normalization property can be asserted.
- OD-009 is required before upstream crop/pad preparation or a future
  reprojection/mosaicking owner can be claimed.

No unresolved item was converted into scientific fact.

