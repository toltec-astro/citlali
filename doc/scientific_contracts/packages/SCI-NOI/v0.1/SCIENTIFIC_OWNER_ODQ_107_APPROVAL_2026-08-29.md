# SCI-NOI v0.1 — Scientific-Owner ODQ-107 Approval

Decision identity: `SCI-NOI-ODQ-107`

Scientific owner: Grant Wilson

Decision date: `2026-08-29`

Status: approved empirical inverse and weight-product policy

## Exact Owner Decision

The owner approved the proposed ODQ-107 disposition: authorize a pointwise
reciprocal of the approved conditional second moment under its own exact
inverse-scale identity, preserve separately typed inverse-variance, precision,
and consumer-effective-weight roles, and forbid promotion of any such product
to validity, support, exposure, or a PTC/MAP coefficient by numerical
resemblance.

## Approved Initial Inverse-Scale Product

For an exact available ODQ-105B product, define

```text
D_inv = {p in D_common : V_hat_cond(p) is finite and strictly positive},
W_hat_cond(p) = 1 / V_hat_cond(p).
```

The method identity is
`NOI-UNC/INVERSE-CONDITIONAL-SECOND-MOMENT-SCALE`. The product role is an
`inverse_conditional_second_moment_scale` with inverse squared signal units.
It is not an inverse variance, precision matrix, probability, support mask,
sample-validity decision, exposure, or calibration merely because it is
positive and weight-like.

At zero, negative, nonfinite, unavailable, or outside-parent-domain input, the
derived product is unavailable. It shall not substitute a numerical zero.
Any floor, cap, clipping, epsilon, shrinkage, or other regularization defines a
separately identified method with the requested/effective/realized rule and
the altered scientific meaning; none is implicit here.

## Separately Typed Inverse Products

A `marginal_inverse_variance` may be produced only from a separately
authorized, finite, strictly positive marginal-variance product on its exact
valid domain. The initial conditional second moment does not become a marginal
variance parent by shape or name.

A `precision` product requires an authorized covariance and an exact inverse
or generalized inverse on a declared domain/subspace, with rank, null space,
unresolved modes, conditioning, and regularization semantics. Reciprocal
covariance diagonal entries are not a precision matrix by default.

A `consumer_effective_weight` is separately identified and bound to one exact
consumer estimator, projection, response, and domain. It is not portable to a
different estimator or consumer by numerical coincidence.

## Ownership And Cross-Boundary Rule

No NOI empirical inverse or weight is a PTC/MAP analysis or gridding
coefficient, sample validity, support, exposure, or instruction to mutate an
immutable parent. Any future use across that boundary requires explicit
scientific authority naming the exact source product, consumer operation, and
meaning. A consumer may use a representation-specific numerical zero to omit
an unavailable location only when its own authorized application contract
preserves the unavailable state; that zero is not an estimated inverse value.

## Non-Implications

This approval does not create numerical availability before the parent
second-moment and reciprocal-domain gates pass. It does not authorize a
covariance inverse, universal consumer weight, PTC/MAP coefficient, STD scale,
implementation conformity, validation, calibration, performance, readiness,
or production use.
