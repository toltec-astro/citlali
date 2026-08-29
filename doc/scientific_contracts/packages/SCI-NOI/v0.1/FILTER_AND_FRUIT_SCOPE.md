# SCI-NOI v0.1 — FLT, Wiener, And FRUIT Scope Record

Status: proposed sanitized Stage A disposition; exact bytes await owner approval

## Proposed Base-v0.1 Boundary

| Route | Proposed disposition | Exact consequence |
| --- | --- | --- |
| Deterministic held-fixed FLT | Conditionally permitted only through a future content-bound `SCI-FLT_TO_SCI-NOI` boundary | Signal and every compatible realization use the same exact operator, support, edge treatment, response, covariance transformation, parent, and lifecycle |
| Re-estimated Wiener filter | Unavailable | Requires a complete inference/feedback contract naming the noise-model target, fitting population, regularization, operator state, response, covariance, dependence, failure, and successor lifecycle |
| Fixed source-residual GEN | Unavailable | Requires an exact source/FRUIT residual-parent boundary and truthful residual/source-leakage state |
| Full or partial relearned FRUIT GEN | Unavailable | Requires an exact FRUIT boundary naming subtraction/add-back, learned state, recurrence, stopping, restart, selection, response, and failure for every member |

These rows remain separate owner decisions: deterministic FLT is
`SCI-NOI-ODQ-110A`, Wiener is `SCI-NOI-ODQ-110B`, and fixed or relearned FRUIT
is `SCI-NOI-ODQ-110C`. Sharing this artifact does not combine their approvals.

No FLT or FRUIT source, audit, validation, or implementation material enters the
Stage B author packet. A deterministic FLT route is not numerically admitted
until its own exact boundary is content-bound and owner-approved.

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
