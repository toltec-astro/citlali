# SCI-NOI v0.1 — FLT, Wiener, And FRUIT Scope Record

Status: ODQ-110A/B owner-approved; ODQ-110C remains proposed and open; exact
packet bytes await final owner approval

## Proposed Base-v0.1 Boundary

| Route | Proposed disposition | Exact consequence |
| --- | --- | --- |
| Externally owned deterministic transformation | Conditionally permitted only when the appropriate upstream/downstream scientific process supplies an exact content-bound transformation authority and parity interface; NOI does not choose or define it | NOI applies exactly the transformation defining the immutable scientific product to every admitted compatible randomization; resulting uncertainty is scoped only to that exact transformed product |
| Owner-defined Wiener transformation frozen before realization application | Governed by ODQ-110A even when initially data-derived; conditionally permitted only through exact owner authority/parity | NOI applies the exact frozen Wiener transformation to every admitted compatible randomization; uncertainty is only for the exact Wiener-transformed product |
| Wiener transformation learned/selected/updated using an NOI product | Separate owner-defined successor generation; unavailable pending a complete inference/feedback contract | Prior UNC is immutable declared input, not independent validation; new transformation, transformed science product, GEN, and UNC generations remain distinct |
| Wiener transformation relearned per realization | Separate ODQ-104 relearned method; unavailable pending complete owner-authored member graph | Cannot mix with fixed-Wiener members without a separately authorized mixture estimand |
| Fixed source-residual GEN | Unavailable | Requires an exact source/FRUIT residual-parent boundary and truthful residual/source-leakage state |
| Full or partial relearned FRUIT GEN | Unavailable | Requires an exact FRUIT boundary naming subtraction/add-back, learned state, recurrence, stopping, restart, selection, response, and failure for every member |

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

This record establishes no filter/FRUIT implementation conformity, response or
covariance fidelity, empirical calibration, performance, readiness, or
production authorization.
