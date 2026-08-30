# SCI-NOI v0.1 — FLT, Wiener, And FRUIT Scope Record

Status: ODQ-110A/B/C owner-approved; exact packet bytes await final owner
approval

## Proposed Base-v0.1 Boundary

| Route | Proposed disposition | Exact consequence |
| --- | --- | --- |
| Externally owned deterministic transformation | Conditionally permitted only when the appropriate upstream/downstream scientific process supplies an exact content-bound transformation authority and parity interface; NOI does not choose or define it | NOI applies exactly the transformation defining the immutable scientific product to every admitted compatible randomization; resulting uncertainty is scoped only to that exact transformed product |
| Owner-defined Wiener transformation frozen before realization application | Governed by ODQ-110A even when initially data-derived; conditionally permitted only through exact owner authority/parity | NOI applies the exact frozen Wiener transformation to every admitted compatible randomization; uncertainty is only for the exact Wiener-transformed product |
| Wiener transformation learned/selected/updated using an NOI product | Separate owner-defined successor generation; unavailable pending a complete inference/feedback contract | Prior UNC is immutable declared input, not independent validation; new transformation, transformed science product, GEN, and UNC generations remain distinct |
| Wiener transformation relearned per realization | Separate ODQ-104 relearned method; unavailable pending complete owner-authored member graph | Cannot mix with fixed-Wiener members without a separately authorized mixture estimand |
| Exact fixed FRUIT residual or terminal transformation | Conditional fixed-state method under ODQ-110A; unavailable pending exact FRUIT owner authority and parity interface | NOI applies the exact FRUIT-defined transformation to every admitted compatible randomization; uncertainty is conditional on frozen FRUIT state and only for the exact residual/terminal product |
| NOI-informed later FRUIT iteration | Separate immutable FRUIT/science/GEN/UNC successor generation; unavailable pending complete FRUIT inference/feedback contract | Prior NOI product is dependent input, not independent validation; no mutation or retroactive validation |
| Partial or complete per-realization FRUIT replay | Separate ODQ-104 relearned method; unavailable pending complete FRUIT-owned member graph | Fixed residual/terminal and replayed members cannot mix without a separately authorized mixture estimand |

These rows remain separate owner decisions: external transformation ownership
is `SCI-NOI-ODQ-110A`, Wiener scope is `SCI-NOI-ODQ-110B`, and fixed or relearned FRUIT
is `SCI-NOI-ODQ-110C`. Sharing this artifact does not combine their approvals.

No FLT or FRUIT source, audit, validation, or implementation material enters the
Stage B author packet. Under owner-approved ODQ-110A, the transformation-owning
scientific process—not NOI—owns the scientific purpose, algorithm, operator,
parameters/learned state, operation order, domain, support and edge rules,
normalization, units, response/transfer meaning, validity, lifecycle, and
failure semantics. NOI owns only the ensemble/uncertainty method identity,
exact binding, conforming application to admitted randomizations, and truthful
scope and limitations.

A transformed-product uncertainty route is not numerically admitted until its
owning process supplies an exact content-bound authority. Every admitted
randomization must receive exactly the transformation that defines the
immutable scientific product: exact owner/version/generation, operator and
state, parameters, insertion point/order, parent, domain, units, normalization,
support, boundary/edge/missing-data behavior, response, lifecycle, and failure
policy. No commutation, relocation, substitution, simplification, inference,
or silent omission is allowed. Relearning the transformation per realization
creates a separate ODQ-104 method.

The resulting uncertainty applies only to that exact transformed scientific
product. Exact application does not establish source cancellation, physical-
noise equivalence, covariance completeness, calibrated significance, or the
scientific validity of the externally owned transformation.

Under owner-approved ODQ-110C, FRUIT retains authority over its source model,
subtraction/add-back, learned state, recurrence/iteration, stopping, restart,
selection, response, support, validity, lifecycle, and failure. A fixed FRUIT
residual or terminal transformation yields only uncertainty conditional on that
exact frozen state. It does not include learning, convergence, selection,
restart, residual-bias, or source-leakage variation unless a separate method
targets it explicitly.

## Successor-Generation Rule

Under owner-approved ODQ-110B, a Wiener transformation learned, selected, or
updated by its owning scientific process using a prior NOI product follows the
immutable graph

```text
UNC_k -> TRANSFORM_OWNER:LearnSelect_(k+1) -> T_(k+1)
science_parent -> T_(k+1) -> transformed_science_(k+1)
GEN_base_(k+1) -> T_(k+1) -> GEN_transformed_(k+1) -> UNC_(k+1)
```

The transformation-owning process defines the learning target, inputs, fitting
population, dependence, regularization, learned state, response, support,
stopping/selection, failure, and lifecycle. NOI does not author the procedure.
`UNC_k` remains the exact earlier product. Neither `T_(k+1)` nor `UNC_(k+1)`
mutates or validates it retroactively. `UNC_k` cannot be treated as independent
evidence validating the successor transformation or uncertainty. Reusing an
assignment design does not merge generations or establish independence.

A data-derived Wiener transformation learned once and frozen before realization
application is fixed-state for NOI and follows ODQ-110A. A separately learned
operator for each realization is instead a distinct ODQ-104 method. Fixed and
relearned Wiener members do not share one UNC ensemble without a separately
authorized mixture estimand.

If an NOI product informs later FRUIT continuation, the corresponding graph is

```text
UNC_k -> FRUIT_OWNER:LearnIterate_(k+1) -> F_(k+1)
science_parent -> F_(k+1) -> fruit_science_(k+1)
GEN_base_(k+1) -> F_(k+1) -> GEN_fruit_(k+1) -> UNC_(k+1)
```

The prior product is immutable dependent input and not independent validation.
Partial or complete FRUIT replay for individual realizations is a separate
ODQ-104 method with exact member-specific learning state. Fixed and replayed
members cannot be pooled or substituted without a separately authorized
mixture estimand.

This record establishes no filter/FRUIT implementation conformity, response or
covariance fidelity, empirical calibration, performance, readiness, or
production authorization.
