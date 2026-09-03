# SCI-FLT-MATCHED v0.1 — Operator, State, And Product Taxonomy

Status: sanitized author input

## Method Identity

Every realization binds

```text
(method_id, estimand, parent_id, parent_grouping, domain, WCS/frame,
 template_id, state_generation, weighting_object, operator_realization,
 approximation, response, support/null, units/calibration, covariance,
 NOI_lifecycle, product_bundle, validity, failure_policy, provenance).
```

Different scientifically consequential fields imply different realizations.

## Reference Operator

For exact parent field `m`, unit-amplitude response template `t_x`, and
realized weighting `Q_x`, define

```text
L_x u    = <t_x, Q_x u_x> / <t_x, Q_x t_x>
A_hat(x) = L_x m
R_t(x,y) = L_x t_y.
```

On admitted complete support, a matching template satisfies `R_t(y,y)=1`.
Off-diagonal response may be asymmetric, anisotropic, nonstationary,
position-dependent, or nonlocal. A single response kernel is sufficient only
when every needed invariance is established.

## State Generations

```text
state_g   = Declare_or_Learn_Once(parent_g, external_inputs_g)
product_g = Apply(parent_g; state_g).
```

The template is a fixed declared input, not learned from the target. Learning
population, inputs, state, dependence, failures, and generation are explicit.
NOI-informed updates create successors. Per-member relearning belongs to a
separate future NOI-GEN method.

## Product Roles

The required atomic signal bundle contains the filtered amplitude field plus
the exact facts needed to establish its identity, response reference, valid
support, units/calibration, method/state, uncertainty availability, lifecycle,
and provenance. It cannot be complete if a required member fails.

Qualified independently atomic companions may include an authorized
conditional covariance representation, an authorized frozen-NOI uncertainty
product, response materialization, or exact state/lineage material required by
a named use. Their absence or failure is explicit and does not silently alter
the signal estimand.

No numerator/denominator diagnostic, kernel, PSD, response slice, inverse
scale, standardized field, or intermediate becomes a public science product
without an exact role, meaning, unit, validity, lifecycle, and policy.

