# SCI-FLT-INF-ODQ-006 scientific-owner approval

Decision identity: `SCI-FLT-INF-ODQ-006`

Date: `2026-08-31`

Scientific owner: Grant Wilson

Status: approved and closed; quantitative conformance-envelope alternatives
author-delegated for later owner selection

## Approved reference operator

For each admitted location `x`, let `m_x` be the exact admitted parent samples
on the eventual ODQ-007 support, let `t_x` be the exact immutable ODQ-005
template-response product on that support, and let `Q_x` be the exact realized
noise/covariance or weaker spectral-weighting operator supplied by the
eventual owner-selected ODQ-004 option. Conditional on those exact objects, the
authoritative reference estimator is

```text
N(x) = <t_x, Q_x m_x>
D(x) = <t_x, Q_x t_x>
A_hat(x) = N(x) / D(x)
```

or a mathematically identical representation under the declared discrete
inner product, units, indexing, WCS, boundary, and support conventions.

When `Q_x` is the admitted inverse covariance on the declared domain, this is
the optimal generalized-least-squares matched-template amplitude estimator
under the complete model assumptions. If ODQ-004 selects only a weaker
spectral-weighting object, the same normalized estimator may retain matching-
template amplitude unbiasedness under its stated fixed-state assumptions, but
its optimality and uncertainty claims must be weakened exactly as that option
requires. ODQ-006 does not select or manufacture the ODQ-004 object.

## Selected realization policy

The exact reference operator is scientific authority. An exact numerical
evaluation is conformant. A numerical approximation is also permitted only
when both contract views:

1. identify the exact reference operator and approximation identity;
2. state the approximation parameters and applicability domain;
3. give a scientifically meaningful bound on its effects on normalization,
   matching-template amplitude response, support/null behavior, and any
   uncertainty claim;
4. define an owner-approved conformance envelope and how its applicability is
   established for the realized product; and
5. require the realized approximation identity, bound/envelope result, and
   completion status to be retained with the product.

The implementation-blind author shall develop the smallest bounded set of
quantitative conformance-envelope alternatives in both the Scientific
Rationale and Contract and the Engineering Conformance Specification, using
the same stable option identities and consequences. The scientific owner must
select or otherwise dispose of those alternatives before scientific freeze or
any approximate numerical route is authorized.

FFT evaluation, interpolation, iterative evaluation, and finite numerical
truncation are implementation techniques rather than separate science only
when they reproduce the declared discrete reference operator within the
selected conformance envelope, including exact transform normalization,
boundary, support, phase, and unit conventions.

## Regularization and method identity

A spectral floor, pseudoinverse cutoff, eigenmode omission, clipping rule,
tail cap, or other regularization that defines `Q_x`, its null space, or its
admitted modes is scientific weighting state and must be declared under the
ODQ-004 option and exact method identity. It is not a numerical repair.

Any approximation or regularization that changes the operator, estimand,
amplitude response, support, null space, or uncertainty beyond the selected
conformance envelope is a separately identified and versioned scientific
method. If no such method is authorized, the requested base method is
unavailable; it does not silently realize the altered operator.

## Null, convergence, and failure behavior

`N(x)` and `D(x)` must be finite, and `D(x)` must be strictly positive on the
admitted support under the exact selected weighting semantics. An empty or
invalid support, a template in the weighting null space, singular or
unresolved normalization, nonfinite value, or nonpositive `D(x)` produces a
typed null/unavailable location or required-product failure under the later
support/lifecycle decisions. It never establishes the scientific amplitude
`A_hat(x)=0`.

Reaching an iteration limit, update-count limit, or tail cap is not successful
completion unless the selected conformance envelope is established. Failure
to establish the bound fails the approximation route closed. Floors, clipping,
or fallback values may not convert that failure into a successful result.

## Consequences

- `SCI-FLT-INF-ODQ-006` is approved and closed at the reference-operator and
  realization-policy level.
- Quantitative conformance-envelope alternatives are author-delegated and
  require later owner disposition before freeze or approximate execution.
- Exact evaluation is the limiting conformant realization and needs no
  approximation waiver.
- Approximation is an implementation freedom only inside the selected
  scientific envelope; outside it, method identity changes.
- Zero/nonfinite/nonpositive normalization is null or unavailable, not zero
  amplitude.
- ODQ-007 edge, missing, nonfinite, and learned-support policy is the next
  owner gate.

## Nonclaims

This approval does not select the ODQ-004 noise/covariance option, a numerical
conformance tolerance, approximation algorithm, FFT convention, support or
edge method, regularization value, null-mode threshold, interpolation scheme,
response product, uncertainty, NOI lifecycle, public product bundle,
implementation conformity, validation, performance, readiness, production,
freeze, or Unity action. It changes no SCI-FLT-FIXED or frozen SCI-NOI byte.
