# SCI-FLT-MATCHED v0.1 — Scope Brief

Status: sanitized Stage A candidate; exact-byte owner approval and Stage B
launch pending

Scientific owner: Grant Wilson

Version/date: `v0.1`, `2026-08-31`

## Program Adherence And Prior-Work Recovery

This package follows the [program charter](../../../README.md),
[pilot review](../../../PILOT_PROCESS_REVIEW_2026-08-16.md), and
[downstream roadmap](../../../DOWNSTREAM_CONTRACT_ROADMAP_2026-08-26.md).
Package-specific recovery is complete in [`PRIOR_WORK.md`](PRIOR_WORK.md).
The future implementation-blind author may inspect only the exact objects in
[`AUTHOR_PACKET_MANIFEST.md`](AUTHOR_PACKET_MANIFEST.md).

## Assignment

Produce a scientist-readable Scientific Rationale and Contract, a matching
Engineering Conformance Specification, one shared normative core, stable
requirements and falsifiable predictions, exact traceability, and rendered
drafts for `SCI-FLT-MATCHED`: **Optimal matched-template map filtering**.

The method estimates the local amplitude of one exact declared template in an
immutable admitted ordinary-MAP observation or coadd parent. It publishes a
matched-filtered map preserving applicable map-domain structure and semantics.
It does not detect, select, fit, deblend, catalog, or interpret sources.

## Normative Reference Estimator

For parent samples `m_x`, template `t_x`, and realized weighting operator
`Q_x`, the scientific reference is

```text
N(x)     = <t_x, Q_x m_x>
D(x)     = <t_x, Q_x t_x>
A_hat(x) = N(x) / D(x).
```

The contract shall define every domain, inner product, coordinate/indexing,
unit, template scaling, phase, support, boundary, null-space, regularization,
and validity fact. A matching signal `m=A t_x+n` shall return an unbiased
estimate of `A` under the exact stated zero-mean noise and validity
assumptions. The phrase “optimal” shall be attached only to a precise
criterion and complete assumptions. Only exact authorized inverse-covariance
GLS premises allow `D(x)^-1` to be identified with marginal conditional
variance; otherwise `D` is normalization.

## Fixed Owner Boundaries

The author shall preserve all content in the owner-decision and boundary
objects. In particular:

- observation and coadd applications are distinct, with no commutation or
  filter-owned coaddition;
- each application has one immutable declared template-response product;
- complete influence support is required; affected locations are unavailable,
  not zero or partially estimated;
- the output quantity is template amplitude with
  `unit(A_hat)=unit(m)/unit(t)`;
- fixed-state response is explicit and uncertainty availability is truthful;
- state is externally declared or learned once from the exact parent and then
  frozen; identical state applies to compatible NOI members;
- no automatic method selector or fallback exists;
- products use the approved tiered atomic lifecycle;
- the FLT→FRUIT interface is first class, while FRUIT science remains separate;
  and
- genuine posterior/Wiener reconstruction and ordinary convolution are
  different methods.

## Required Authored Options

Both contract views shall contain the same options, stable IDs, assumptions,
scientific consequences, observables/bounds, failure rules, and validation
consequences for every family in
[`REQUIRED_AUTHORED_OPTION_SETS.md`](REQUIRED_AUTHORED_OPTION_SETS.md).
Options are analysis for later owner disposition, not permission to choose a
default or authorize a numerical route.

## Required Outputs And Lifecycle

Define one role-complete atomic signal bundle and independently atomic
qualified companions. At minimum bind filtered quantity, exact parent,
template, response identity, valid support, WCS/frame/grouping, units and
calibration, method/state identity, uncertainty availability, lifecycle,
failure, and provenance. Additional exact state required by a later authorized
FRUIT method must remain available or reconstructable through lineage.

Disabled, unavailable, failed, complete, and superseded are distinct. Failure
of any required signal-bundle member fails that bundle. Optional companion
failure does not retroactively invalidate a complete signal bundle unless an
exact named-use policy requires the companion.

## Exclusions And Stop Rule

Do not inspect or infer current or historical implementation, configuration,
schemas, tests, audits, repairs, validation, reductions, Unity, defaults,
performance, or production behavior. Do not introduce source-analysis or
FRUIT science. If the exact packet cannot answer a required scientific
question, return one precise question and stop rather than inventing a fact.

