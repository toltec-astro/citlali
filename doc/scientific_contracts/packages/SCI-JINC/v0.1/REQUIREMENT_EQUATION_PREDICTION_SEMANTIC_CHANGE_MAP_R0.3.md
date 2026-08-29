# SCI-JINC v0.1 r0.3 Requirement/Equation/Prediction Semantic-Change Map

Status: implementation-blind Stage B author-draft change map; final Stage B
acceptance and freeze remain pending

Prepared: `2026-08-29`

No requirement or prediction identifier is added, removed, or renumbered.
Requirements remain `SCI-JINC-REQ-001`--`044`; predictions remain
`SCI-JINC-PRED-001`--`036`.

| r0.3 decision | Shared equations/definitions/assumptions | Requirements whose semantics are clarified | Predictions whose semantics are clarified |
| --- | --- | --- | --- |
| Owner Disposition A and positive-axis center tie | Notation for `n_sub` and `phi_hat`; definition "Point phase"; equations `rounded-center`, `phase-bin`, `phase-representative`, and `discrete-radius`; an unnumbered exact even-lattice center consequence is added after `discrete-radius`; `SCI-JINC-ASM-006` | `SCI-JINC-REQ-007`, `010`--`012`, `044` | `SCI-JINC-PRED-017` now states the exact even-`n_sub` center radius and analytic coefficient; no `PRED-037` is added |
| Numerical-certificate claim boundary | Numerical-adequacy definition; notation for `P_NA` and `E_p`; denominator-separation equations unchanged; `SCI-JINC-ASM-007` | `SCI-JINC-REQ-023`, `042` | `SCI-JINC-PRED-011` limits a passing pair to finite-precision discrete-oracle fidelity |
| Canonical admission term with immutable identity retained | Definition "JINC upstream-occurrence admission" | `SCI-JINC-REQ-035`--`036` | `SCI-JINC-PRED-032`, `034` use the canonical semantic term |
| Exact-source-lock lifecycle model A | Definition "Exact-source lock" | `SCI-JINC-REQ-037`, `043`--`044` | `SCI-JINC-PRED-033` rejects ambient current-Registry substitution and requires a versioned successor for changed bytes |
| Coefficient-squared temporal-accounting terminology | Notation for `T_p^(kappa^2)`; Equation `kappa-time` is unchanged | `SCI-JINC-REQ-026`--`027` | No prediction semantics change |

The added even-lattice center consequence is unnumbered so the existing
equation sequence is stable. In particular, Equation 24 remains
`kappa-time` with its r0.2 formula unchanged.

All other shared equations, assumptions, requirements, and predictions retain
their r0.2 scientific semantics. In particular, the signed estimator,
five-role ODQ-107 bundle, conditional response/covariance mathematics, typed
unavailability of the numerical route, and prohibitions on inherited TolTEC
values remain unchanged.

This map describes document semantics only. It is not implementation mapping,
conformance evidence, validation, achieved performance, readiness, or
production authorization.
