# SCI-FLT-FIXED v0.1 Numerical-Conformance Policy Draft

Document identity: `SCI-FLT-FIXED-NUMERICAL-CONFORMANCE v0.1/draft-r0.3`

Status: future finite-precision evidence-policy draft; not preregistered; no candidate finding

Scientific owner: Grant Wilson

## 1. Purpose and boundary

This policy defines what a future finite-precision comparison record must bind
before a candidate result is observed. Exact operator coefficients, selector,
row domain, and scientific identity remain controlled by the normative core.
The policy authors no filter science and makes no implementation-conformity,
validation, numerical-adequacy, or performance claim.

## 2. Preregistration identity

A future immutable policy instance must bind:

- policy identity, version, content digest, scientific-policy owner, reviewer,
  approval state, and effective scope;
- exact normative-core revision and candidate scientific identity;
- exact input, operator, selector, row ordering, response, and covariance test
  generations;
- all comparison rules before candidate output is inspected; and
- an explicit prohibition on changing a bound after observing a failure.

This draft is not that approved instance.

## 3. Independent oracle

The policy instance must identify an independent exact or high-precision
oracle, its arithmetic representation, coefficient ingestion, summation order,
rounding behavior, software and environment identity, and why it is
independent of the candidate. Shared code paths or copied candidate output are
not an independent oracle.

## 4. Comparison regimes

The policy instance must define together:

- absolute error behavior;
- relative error behavior and its denominator;
- near-zero classification and comparison behavior;
- exact-zero expectations that remain exact;
- signed-cancellation and zero-sum cases;
- scale ranges and unit handling;
- conditioning and operation-count dependence; and
- whether any bound is row-dependent, with its preregistered derivation.

No single relative tolerance may silently govern exact zero or near-zero
cases.

## 5. Signal and simultaneous row decision

All rows in one comparison unit must be evaluated under one simultaneous
decision rule. The record must expose per-row absolute and relative residuals,
near-zero class, applicable bound, pass or fail result, and aggregate action.
Dropping a failed row, changing `J_full`, or reporting only a passing subset is
not allowed.

## 6. Covariance comparison

The policy instance must bind parent stochastic authority, output
representation, domain, ordering, rank, null space, symmetry treatment,
positive-semidefinite expectations where applicable, diagonal and off-diagonal
comparison behavior, and any structured-operator evaluation. Marginal and
complete representations must not share a stronger claim than either supports.

## 7. Sequential and parallel agreement

The instance must define candidate sequential and parallel configurations,
allowed determinism class, thread or task counts, reduction-order treatment,
and comparison with both the oracle and each other. A parallel speed result is
outside this policy unless separately evaluated as performance evidence.

## 8. Exceptional arithmetic

Overflow, underflow, subnormal handling, signed zero, NaN, infinity, and other
non-finite values must each have a preregistered expected state and action.
Unexpected non-finite output is a failure, not an automatically excluded row.

## 9. Lifecycle and provenance

The evidence lifecycle is `draft -> reviewed -> preregistered -> executed ->
decision_recorded -> superseded`. Each transition binds actor, time, exact
bytes, environment, inputs, outputs, causes, and immutable predecessor. The
executed record must prove that comparison bytes match the preregistered bytes.

## 10. Nonclaims

This draft supplies no implementation result, conformance decision,
validation, calibration, achieved response or covariance, numerical adequacy,
performance, readiness, scientific freeze, production, or Unity claim.
