# SCI-NOI v0.1 — Scientific-Owner ODQ-110B Approval

Decision identity: `SCI-NOI-ODQ-110B`

Scientific owner: Grant Wilson

Decision date: `2026-08-29`

Status: approved Wiener ownership, fixed-state, and successor-generation policy

## Exact Owner Decision

An exact Wiener transformation already defined by its appropriate scientific
process may be used under ODQ-110A: NOI does not choose or define the Wiener
operator and must apply exactly that owner-defined transformation to every
admitted compatible randomization when estimating uncertainty for the exact
Wiener-transformed scientific product.

A Wiener operator learned, selected, or updated using an NOI product, or
relearned separately for realizations, is not folded into that fixed-state
method. It requires a separately identified, versioned inference/feedback or
relearned method owned by the scientific process that defines the Wiener
transformation.

## Fixed-State Wiener Transformation

A Wiener operator may be data-derived and still be fixed-state for NOI when
the owning process learns or selects it once, publishes its exact content-bound
identity and transformed scientific product, and freezes it before NOI applies
it to the randomizations. The complete ODQ-110A owner/version/generation,
operator/state/parameter/order, parent/domain/support/edge/missing-data,
normalization/unit/response, lifecycle, and failure parity requirements apply.

No special NOI authority follows from the label “Wiener.” The uncertainty
applies only to the exact scientific product transformed by that exact frozen
operator. Without its owner-supplied authority and parity interface, the route
is unavailable.

## Successor-Generation And Feedback Rule

If an NOI uncertainty product is used by the owning process to learn, select,
or update a Wiener transformation, the prior uncertainty remains immutable and
the transformation begins a new scientific-product and NOI generation:

```text
UNC_k -> TRANSFORM_OWNER:LearnSelect_(k+1) -> T_(k+1)
science_parent -> T_(k+1) -> transformed_science_(k+1)
GEN_base_(k+1) -> T_(k+1) -> GEN_transformed_(k+1) -> UNC_(k+1)
```

The exact inputs, target, fitting population, dependence, regularization,
learned state, response, support, stopping/selection rule, failure, and
lifecycle of `T_(k+1)` belong to the transformation-owning process. NOI binds
the resulting exact transformation and its generation; it does not author the
learning procedure.

`UNC_k` is an input to the successor transformation when so declared. It is
not independent evidence validating `T_(k+1)` or `UNC_(k+1)`, and neither the
successor transformation nor its uncertainty mutates or retroactively
validates `UNC_k`. Reusing an assignment design or realization key does not
merge product generations or establish independence.

## Per-Realization Relearning

Learning or selecting a distinct Wiener operator for each realization is a
separate ODQ-104 relearned method. Its scientific owner must define the exact
member-specific learning graph and changed state. Such members cannot be mixed
with fixed-Wiener members in one UNC estimate without a separately authorized
mixture estimand.

## Availability And Claim Boundary

No numerical fixed, feedback, or per-realization Wiener route is admitted by
this decision alone. Fixed routes remain unavailable pending exact ODQ-110A
authority and parity. Feedback and per-realization routes remain unavailable
pending a complete owner-authored inference/relearning contract and exact NOI
method boundary.

This approval establishes no filter validity, independence, covariance
completeness, source cancellation, calibration, significance, implementation
conformity, performance, readiness, or production authorization.
