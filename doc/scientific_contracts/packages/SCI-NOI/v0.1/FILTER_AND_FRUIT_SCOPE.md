# SCI-NOI v0.1 — FLT, Wiener, And FRUIT Scope Record

Status: ODQ-110A owner-approved; ODQ-110B/C remain proposed and open; exact
packet bytes await final owner approval

## Proposed Base-v0.1 Boundary

| Route | Proposed disposition | Exact consequence |
| --- | --- | --- |
| Externally owned deterministic transformation | Conditionally permitted only when the appropriate upstream/downstream scientific process supplies an exact content-bound transformation authority and parity interface; NOI does not choose or define it | NOI applies exactly the transformation defining the immutable scientific product to every admitted compatible randomization; resulting uncertainty is scoped only to that exact transformed product |
| Re-estimated Wiener filter | Unavailable | Requires a complete inference/feedback contract naming the noise-model target, fitting population, regularization, operator state, response, covariance, dependence, failure, and successor lifecycle |
| Fixed source-residual GEN | Unavailable | Requires an exact source/FRUIT residual-parent boundary and truthful residual/source-leakage state |
| Full or partial relearned FRUIT GEN | Unavailable | Requires an exact FRUIT boundary naming subtraction/add-back, learned state, recurrence, stopping, restart, selection, response, and failure for every member |

These rows remain separate owner decisions: deterministic FLT is
`SCI-NOI-ODQ-110A`, Wiener is `SCI-NOI-ODQ-110B`, and fixed or relearned FRUIT
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

If an uncertainty product selects or constructs a later filter, the immutable
graph is

```text
UNC_k -> FLT_(k+1) -> GEN_(k+1) -> UNC_(k+1).
```

`UNC_k` remains the exact earlier product. Neither `FLT_(k+1)` nor
`UNC_(k+1)` mutates or validates it retroactively. The uncertainty used to
construct a Wiener operator cannot be treated as independent evidence
validating that same operator in the same generation.

This record establishes no filter/FRUIT implementation conformity, response or
covariance fidelity, empirical calibration, performance, readiness, or
production authorization.
