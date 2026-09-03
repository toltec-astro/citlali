# SCI-NOI v0.1 — Scientific-Owner ODQ-105A Approval

Decision identity: `SCI-NOI-ODQ-105A`

Scientific owner: Grant Wilson

Decision date: `2026-08-29`

Status: approved fail-closed realization-completion rule

## Exact Owner Decision

> **NOI-GEN fails closed on any failure of an admitted realization.** Candidate
> assignments rejected during finite-design construction are not failures and
> never become ensemble members. Once an assignment is admitted, every
> requested realization must complete successfully through the declared frozen
> operator. Failure of any admitted member invalidates the generated ensemble
> for NOI-UNC use; surviving members must not be silently retained as a partial
> ensemble. The failure cause should be reported sufficiently to diagnose the
> run, without requiring exhaustive implementation provenance.

## Sanitized Disposition

Finite-design construction and realization execution have a hard lifecycle
boundary. A candidate assignment rejected before admission is a design-search
outcome, not an admitted member and not a realization failure. Rejected
candidates cannot enter member counts, ensemble identity, or NOI-UNC input.

Once the design admits an assignment, that assignment is an exact requested
ensemble member. Every admitted member shall complete successfully through the
entire declared frozen operator and realize its required atomic product. If any
admitted member is incomplete, failed, or unavailable, GEN assigns the ensemble
a failed terminal state and the entire ensemble is unavailable for every
NOI-UNC use.

Completed members from a failed ensemble shall not be silently reinterpreted
as a smaller or partial ensemble. If retained for diagnosis, they remain bound
to the failed ensemble identity and carry no NOI-UNC admission authority. A new
attempt requires a new exact generation/design identity; it cannot mutate the
failed ensemble.

Disabled GEN remains an explicit zero-member/no-work state. Enabled GEN requires
a positive resolved admitted design. Inability to resolve the required admitted
design under the declared bounded construction is a design-resolution failure;
individual candidate rejection alone is not.

## Failure Reporting Boundary

GEN shall report the failed ensemble/member identity, failed operator stage or
scientifically meaningful location, terminal state, cause category, and enough
diagnostic context to investigate the run. This does not require exhaustive
implementation provenance, tracing every internal operation, or publishing
scientifically irrelevant implementation details.

## Non-Implications

This approval does not select finite-design mechanics, authorize retry or
replacement after admission under the same ensemble identity, make a numerical
route available, or resolve the initial UNC estimator, covariance, STD,
persistence, filtering, FRUIT, VAL Registry, conformity, validation,
performance, readiness, or production questions.
