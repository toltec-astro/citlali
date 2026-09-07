# SCI-MAP v0.2/r0.1 prediction crosswalk

Status: candidate prospective-evidence map. These are discriminating cases,
not application tests or completed evidence.

| Prediction | Governing requirement(s) | Prospective distinction |
| --- | --- | --- |
| SCI-MAP-PRED-001 | REQ-012, REQ-014, REQ-018 | Uniform positive coefficients preserve a constant only on authorized finite-positive rows. |
| SCI-MAP-PRED-002 | REQ-014, REQ-019--021 | Unequal positive coefficients preserve a constant while normalization and conditional variance may vary. |
| SCI-MAP-PRED-003 | REQ-011--013 | One-pixel numerator, normalization, and quotient match the exact hand calculation; unsupported value is absent. |
| SCI-MAP-PRED-004 | REQ-005, REQ-015, REQ-026 | One-hot half-open placement owns exactly one in-grid cell and loses outer-boundary points. |
| SCI-MAP-PRED-005 | REQ-008, REQ-016--017 | Delta response is the exact support-row column for the declared fixed family, distinct from procedure response. |
| SCI-MAP-PRED-006 | REQ-008, REQ-017, REQ-052 | Point-template response respects exact family and functional identity; whole-chain response is not inferred. |
| SCI-MAP-PRED-007 | REQ-008, REQ-014, REQ-017 | Extended, gradient, and Fourier inputs test response modes that constant preservation cannot establish. |
| SCI-MAP-PRED-008 | REQ-007, REQ-014, REQ-025, REQ-029--030 | Variable support can coexist with constant preservation and distinct exposure/covariance/response. |
| SCI-MAP-PRED-009 | REQ-005, REQ-014--015 | Finite-edge loss neither wraps nor clamps and can change compact-source response. |
| SCI-MAP-PRED-010 | REQ-004, REQ-006, REQ-010, REQ-027, REQ-035 | Registry presence without selection/publication/exact handoff remains unavailable; exact uniform one is independently MAP-classified and is not precision. |
| SCI-MAP-PRED-011 | REQ-004, REQ-010, REQ-027, REQ-034 | Excluded non-finite payloads never enter arithmetic; exact typed causes and field states remain. |
| SCI-MAP-PRED-012 | REQ-031--032, REQ-046--047 | Empty population, exact-zero cut, both inclusive thresholds, ordinary/expert cut, and invalid/missing cut remain distinct and fail or support exactly as approved. |
| SCI-MAP-PRED-013 | REQ-008, REQ-016--017, REQ-033, REQ-050 | MAP-local operator remains known when upstream response is unavailable; no hidden subset or fabricated whole-chain response. |
| SCI-MAP-PRED-014 | REQ-036--037, REQ-047--048 | One incompatible/missing required observation fact rejects atomically before coadd mutation. |
| SCI-MAP-PRED-015 | REQ-036, REQ-039--040, REQ-048 | One observation reproduces its signal, identically from in-memory or persisted conforming bundles, with honest response state. |
| SCI-MAP-PRED-016 | REQ-028, REQ-039 | Two compatible observations form the equal-observation arithmetic mean and count-like normalization. |
| SCI-MAP-PRED-017 | REQ-025, REQ-028, REQ-037, REQ-040 | Unsupported rows and rejected observations contribute nowhere and create no measured-zero substitute. |
| SCI-MAP-PRED-018 | REQ-038, REQ-043, REQ-045--047 | Preconstruction target grid and even centered offsets succeed; incompatible placement fails; later crop has new identity. |
| SCI-MAP-PRED-019 | REQ-019--022, REQ-041--042 | Complete coadd covariance includes cross terms; missing blocks remain unknown rather than zero/independent. |
| SCI-MAP-PRED-020 | REQ-016, REQ-023, REQ-046, REQ-051 | A fixed noise realization uses exactly the signal operator and rows; re-estimation creates a named difference. |
| SCI-MAP-PRED-021 | REQ-051--052 | Sequential and parallel routes preserve exact discrete facts and preregistered bounded floating behavior. |
| SCI-MAP-PRED-022 | REQ-013, REQ-021, REQ-024, REQ-050 | Merely formal standardized signal is never promoted to empirical significance. |
| SCI-MAP-PRED-023 | REQ-033, REQ-049--050 | A downstream local decision cannot rewrite base MAP validity or parentage. |
| SCI-MAP-PRED-024 | REQ-002, REQ-042, REQ-044, REQ-052 | Ordinary MAP predicates cannot be applied to JINC or another method by analogy; permitted mode reuse is only the exact operator boundary. |
| SCI-MAP-PRED-025 | REQ-012, REQ-033--034, REQ-047--048 | Zero/non-finite/unrepresentable aggregate, index, or required-publication failure never becomes a completed valid product. |

Completeness: 25 unique contiguous prediction IDs appear once in this table
and once in `src/common/edge_cases.tex`.
