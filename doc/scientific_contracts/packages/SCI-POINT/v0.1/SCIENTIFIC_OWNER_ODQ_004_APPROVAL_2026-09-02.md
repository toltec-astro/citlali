# SCI-POINT ODQ-004 Scientific-Owner Approval

Record identity: `SCI-POINT-ODQ-004-APPROVAL-2026-09-02`

Scientific owner: Grant Wilson

Decision date: `2026-09-02`

Status: approved Stage A scientific direction

## Approved Decision

SCI-POINT v0.1 adopts the established six-parameter elliptical-Gaussian
Pointing fit as the compatibility estimator
`POINT-FIT/ELLIPTICAL-GAUSSIAN-COMPATIBILITY@1`.

Its fitted parameters are:

- amplitude;
- two centroid coordinates;
- two fitted widths; and
- orientation angle.

This is adoption of the mature working estimator, not authorization to
redesign it. Stage B shall describe its existing scientific method completely,
including the zero-background treatment, parameterization and conventions,
map weights or covariance use, fit support, initialization/search, constraints,
degeneracies, failure behavior, parent-response interpretation, and formal
uncertainty meaning. It shall not copy implementation text or silently alter
the estimand.

No additional profile family is part of base v0.1. A future scientifically
motivated profile or estimator must have a separate versioned method identity
and must not rewrite the compatibility result.

## Non-Effects

This approval does not settle the exact center/search/support/constraint
policy in ODQ-005, fit acceptance in ODQ-006, covariance publication in
ODQ-007, or any later owner decision. It does not approve the complete Stage A
packet, authorize Stage B, change an algorithm, or establish implementation
conformity, validation, achieved performance, readiness, or production state.
