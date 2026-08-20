# SCI-PTC v0.1 — Cross-Package Follow-Up

Status: Stage A routing record; no adjacent package is amended or launched

| Target authority | Follow-up required | Current disposition |
| --- | --- | --- |
| SCI-CAL Q02 | Record CAL-before-PTC ordering for the primary v0.1 route and use the point-source-equivalent mJy-per-fixed-nominal-beam convention with explicit downstream response | Deferred to SCI-CAL owner disposition; PTC cannot freeze CAL |
| SCI-RTC | Define or explicitly decline a PTC-compatible conditioned-`r` diagnostic product with channel-specific conditioning, exact `x`-grid relation, response, validity, and uncertainty | Required before the optional diagnostic-only `r` branch can operate; raw-`r` identity alone is insufficient; Q001 forbids `r` control of `x` in base v0.1 |
| SCI-RTC/PTC response composition | Preserve one named detector-time response companion through the exact realized RTC/CAL/PTC signal operators; distinguish fixed-state, package-full-procedure, and whole-chain response | PTC full-procedure response begins at its immutable CAL parent; whole-chain injection requires separately named RTC/CAL/PTC ownership |
| Raw Beammap processing | Decide which package owns common-mode cleaning in raw `Delta f/f`, including its null space, coefficients, response, and downstream Beammap meaning | Explicitly outside SCI-PTC v0.1; no silent reuse of the calibrated contract |
| SCI-MAP | Distinguish the PTC-dependent route, any separately authorized direct CAL-to-MAP route, and the exact reference-map functional used only by the optional PTC map-center diagnostic | Deferred; imported reference functional grants no MAP authority to PTC |
| SCI-VAL | Define shared cause/knowledge-state types, preservation and provenance rules, profile vocabulary, and reusable evaluation machinery for owner-supplied policies; do not define PTC-local or downstream scientific admission rules | PTC owns its local composite supports; each downstream named-use owner owns its rule |
| NOI | Bind empirical noise/covariance products to exact PTC grouping, selection, removed subspace, response, and coefficient family | Deferred to NOI |
| BEAM | Treat a PTC map-center diagnostic as a bounded response functional, never complete beam or extended-source authority | Deferred to BEAM/response qualification |

This file records dependencies only. It authorizes no implementation, repair,
validation, or adjacent-package scientific change.
