# SCI-POINT ODQ-007 Scientific-Owner Approval

Record identity: `SCI-POINT-ODQ-007-APPROVAL-2026-09-02`

Scientific owner: Grant Wilson

Decision date: `2026-09-02`

Status: approved Stage A scientific direction

## Approved Decision

SCI-POINT v0.1 requires the established marginal formal parameter errors as
the compatibility uncertainty representation when they are available. The
product must state their estimation method, assumptions, parameter domain,
conditioning, and limitations. They are formal fit uncertainties under the
declared estimator and parent state; they are not automatically empirical
repeatability, astrometric/correction uncertainty, calibration uncertainty,
or detection significance.

A full joint parameter covariance may be unavailable in base v0.1. Its absence:

- does not invalidate an otherwise honest fit result;
- is not zero covariance;
- is not diagonal covariance; and
- does not authorize treating the marginal formal errors as independent.

A named use that requires joint covariance is unavailable unless that use's
owner defines and records another scientifically authorized treatment. A
downstream producer may not silently invent missing correlation information.

Later joint-covariance, astrometric, empirical-repeatability, or NOI
uncertainty estimates may attach to the immutable POINT result as separately
versioned companion products with exact parent and method identity. They do
not rewrite the original POINT uncertainty claim.

## Non-Effects

This approval does not establish uncertainty calibration or coverage, answer
ODQ-008 or ODQ-009, approve the complete Stage A packet, authorize Stage B,
change an algorithm, or establish implementation conformity, validation,
achieved performance, readiness, or production state.
