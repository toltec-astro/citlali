# SCI-POINT Objective, Weighting, And Formal-Error Decision

Identity: `SCI-POINT_METHOD_AUTHORITY_DECISION v0.1/r0.3`

## Numerical Objective

The generic conditional form is

`theta_hat = Resolve_{theta in Theta_effective} D(m, g_theta; W_fit)`.

`D`, `W_fit`, admitted rows, normalization, constraints, solution comparison,
and termination are not supplied by this packet. They belong to the unavailable
`POINT-COMPATIBILITY-METHOD v0.1`. No route may instantiate this expression by
guessing least squares, inverse variance, reliability weighting, or covariance
weighting.

Fit metric/reliability weight, parent stochastic covariance, inverse
covariance, formal-error model, and parent uncertainty product are distinct
roles. Names or inverse-square units do not collapse them.

## Formal Error

The marginal-error calculation belongs to the separately unavailable
`POINT-FORMAL-ERROR-METHOD v0.1`. A finite optimizer covariance, inverse
Hessian, or curvature is not admitted unless that exact method record
authorizes it with rank, scaling, constraint, degeneracy, and consistency
rules.

The compatibility-method absence blocks numerical fits. Formal-error-method
absence alone blocks formal uncertainty and dependent uses, not fitted values
already authorized by an available compatibility method.
