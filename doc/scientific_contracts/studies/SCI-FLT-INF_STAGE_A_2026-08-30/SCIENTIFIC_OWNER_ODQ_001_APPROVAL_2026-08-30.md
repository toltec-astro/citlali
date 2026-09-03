# SCI-FLT-INF-ODQ-001 scientific-owner approval

Decision identity: `SCI-FLT-INF-ODQ-001`

Date: `2026-08-30`

Scientific owner: Grant Wilson

Status: approved; closes ODQ-001 only

## Approved scientific identity

The existing Citlali path historically called the `Wiener filter` is an
**optimal matched-template amplitude estimator**. It uses the supplied kernel
as the expected source/template response and an exact declared noise model to
estimate the amplitude of that template as a function of map position.

For the ordinary point-source case, the supplied kernel represents the
point-source response. The resulting estimand is therefore a matched
point-source amplitude field. More generally, when another scientifically
defined kernel is supplied, the estimand is the amplitude field of that exact
specified template or shape.

For a parent matching the supplied template with amplitude `A`, the exact
estimator normalization must return an unbiased estimate of `A`, subject to
the declared noise model, support, edge, missing/nonfinite, validity, response,
and other method assumptions. The future contract must state the precise
optimality criterion and the conditions under which it holds; the word
`optimal` is not an unconditional achieved-performance claim.

## Explicit exclusions and distinctions

This method is not a posterior or Wiener reconstruction of the underlying sky.
It has no intended sky-signal prior, posterior-sky estimand, or posterior
covariance. `Wiener filter` may remain only where historical or implementation
compatibility terminology is necessary; scientific products and contracts
shall use `optimal matched-template amplitude estimator` or an equivalently
precise owner-approved term.

The estimator remains scientifically distinct from ordinary source-shaped
convolution. Convolution with a kernel alone is a deterministic transformation
and does not become the noise-weighted, normalized matched estimator merely
because it uses the same kernel.

A genuine prior-bearing Wiener/posterior sky reconstruction, if later desired,
is a separate method requiring its own recovery, Stage A decisions, scientific
contract, response, covariance, products, and lifecycle.

## Consequences

- `SCI-FLT-INF-ODQ-001` is closed with the matched-template amplitude-field
  estimand.
- Candidate family `INF-A` becomes the owner-selected scientific family for
  recovery of the historical full path.
- Candidate family `INF-B` remains separate, unselected, and unavailable.
- The point-source case is one scientifically defined template specialization,
  not the universal definition of the method.
- Exact parent, grouping, noise/covariance authority, template normalization,
  discretized operator, optimality criterion, edge/support, response,
  uncertainty, NOI parity, product, and failure decisions remain open in
  ODQ-002 onward.

## Nonclaims

This decision does not approve a package name, combined contract, author
packet, Stage B launch, numerical implementation, implementation conformity,
representation fidelity, covariance model, estimator discretization,
regularization, template instance, calibration, edge/support realization,
uncertainty, significance, achieved unbiasedness/optimality, validation,
performance, readiness, production, freeze, or Unity action. It changes no
SCI-FLT-FIXED or frozen SCI-NOI byte.
