# SCI-MAP v0.2/r0.4 prediction crosswalk

Status: candidate prospective-evidence map. These are discriminating cases,
not application tests or completed evidence.

| Prediction | Governing requirement(s) | Prospective distinction |
| --- | --- | --- |
| SCI-MAP-PRED-001 | REQ-012, REQ-014, REQ-018 | Uniform positive coefficients preserve a constant only on authorized finite-positive rows. |
| SCI-MAP-PRED-002 | REQ-014, REQ-019--021 | Unequal positive coefficients preserve a constant while normalization and conditional variance may vary. |
| SCI-MAP-PRED-003 | REQ-011--013 | One-pixel numerator, normalization, and quotient match the exact hand calculation; unsupported value is absent. |
| SCI-MAP-PRED-004 | REQ-005, REQ-015, REQ-026 | One-hot half-open placement owns exactly one in-grid cell and loses outer-boundary points. |
| SCI-MAP-PRED-005 | REQ-008, REQ-016--017 | Fixed-family delta response is the exact support-row column; every numerical full-procedure difference additionally requires the complete authorized common domain and both exact row maps. |
| SCI-MAP-PRED-006 | REQ-008, REQ-017, REQ-052 | Point-template response respects exact family, functional, perturbation, gauge, support, and comparison-space identity; whole-chain response is not inferred. |
| SCI-MAP-PRED-007 | REQ-008, REQ-014, REQ-017 | Extended, gradient, and Fourier inputs test response modes that constant preservation cannot establish. |
| SCI-MAP-PRED-008 | REQ-007, REQ-014, REQ-025, REQ-029--030 | Variable support can coexist with constant preservation and distinct exposure/covariance/response. |
| SCI-MAP-PRED-009 | REQ-005, REQ-014--015 | Finite-edge loss neither wraps nor clamps and can change compact-source response. |
| SCI-MAP-PRED-010 | REQ-004, REQ-006, REQ-010, REQ-027, REQ-035 | Registry presence without selection/publication/exact handoff remains unavailable; exact uniform one is independently MAP-classified and is not precision. |
| SCI-MAP-PRED-011 | REQ-004, REQ-010, REQ-027, REQ-034 | Excluded non-finite payloads never enter arithmetic; exact typed causes and field states remain. |
| SCI-MAP-PRED-012 | REQ-009, REQ-012, REQ-031--032, REQ-046--047 | Empty population with exact `Q_star=0` boundary convention, small-N transitions, inclusive thresholds, above-one override, invalid input, reached lifecycle, and owner-directed no-support `not_produced` outcome remain distinct. |
| SCI-MAP-PRED-013 | REQ-008, REQ-016--017, REQ-033, REQ-040, REQ-050 | MAP-local operator remains known when upstream response is unavailable; each response role names one exact family, every numerical difference needs `SCI-MAP-FULL-PROCEDURE-COMPARISON-SPACE`, and current candidate profile evaluation remains unavailable. |
| SCI-MAP-PRED-014 | REQ-036--041, REQ-047--049 | Required companion incompatibility rejects before mutation only at the selected role/family scope; base-coadd response incompatibility preserves membership; same-family fixed-state response composition requires the exact theorem. |
| SCI-MAP-PRED-015 | REQ-012, REQ-036, REQ-039--040, REQ-048--049 | Under a future active profile, one-observation base coadd reproduces signal and a response-bearing role requires one exact compatible family; empty support instead yields no product or no-row coadd member. |
| SCI-MAP-PRED-016 | REQ-028, REQ-036--040, REQ-049 | Under a future active profile, two compatible observations form the equal-observation mean; a response product uses one exact same family and fixed state, while coadd-procedure and whole-chain response remain unavailable. |
| SCI-MAP-PRED-017 | REQ-012, REQ-025, REQ-028, REQ-037, REQ-040, REQ-047--049 | Unsupported rows contribute nowhere; a permitted unavailable companion does not remove a realized base member; wholly empty support yields `not_produced`, no base product, and no coadd member. |
| SCI-MAP-PRED-018 | REQ-038, REQ-043, REQ-045--047 | Preconstruction target grid and even centered offsets succeed; incompatible placement fails; later crop has new identity; WCS comparison covers every preregistered center and outer-boundary vertex/corner. |
| SCI-MAP-PRED-019 | REQ-019--022, REQ-036, REQ-041--042, REQ-049 | A complete covariance-qualified coadd includes every named within/cross block through `bold Sigma^obs`; a missing block blocks only the stronger role and never becomes zero/independence or a member subset. |
| SCI-MAP-PRED-020 | REQ-016, REQ-023, REQ-046, REQ-051 | A fixed noise realization uses exactly the signal operator and rows; re-estimation creates a named difference. |
| SCI-MAP-PRED-021 | REQ-051--052 | Sequential and parallel routes preserve exact discrete facts and preregistered bounded floating behavior. |
| SCI-MAP-PRED-022 | REQ-013, REQ-019--024, REQ-050 | Q, formal conditional precision, complete conditional covariance, the exact NOI conditional marginal second moment, NOI standardized signal, and any future calibrated significance remain distinct; formal or standardized signal is never promoted. |
| SCI-MAP-PRED-023 | REQ-033, REQ-049--050 | A downstream local decision cannot rewrite base MAP validity or parentage. |
| SCI-MAP-PRED-024 | REQ-002, REQ-042, REQ-044, REQ-052 | Ordinary MAP predicates cannot be applied to JINC or another method by analogy; permitted mode reuse is only the exact operator boundary. |
| SCI-MAP-PRED-025 | REQ-009, REQ-012, REQ-031--035, REQ-038--039, REQ-043, REQ-045--048 | [Fixtures A--D](records/PRED025_QUANTITY_SPECIFIC_ZERO_FIXTURES.md) distinguish an admissible zero signal numerator and normalized value at `Q_p > 0`, constant-zero input, valid index zero and zero centered offset, forbidden division at `Q_p = 0`, the retained `N = 0, Q_star = 0` convention, noncontributing zero coefficients, and empty applied output support with `not_produced` and exact cause `no_support_authorized_output_rows`. |

Author-side completeness target: 25 unique contiguous prediction IDs appear
once in this table and once in `src/common/edge_cases.tex`. Mechanical
verification is recorded separately and does not establish conformance.
