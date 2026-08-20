# SCI-RTC v0.1/r0.8 Change Log

Date: `2026-08-20`

Scope: apply the binding scientific-owner Decision 9 in
`SCIENTIFIC_OWNER_DECISION_R0.8.md`. No normative definition, assumption,
equation tag, requirement, prediction, or existing owner-ledger ID was added,
removed, or renumbered. One resolved owner-ledger entry was appended.

## Scientific and normative corrections

- Replaced the instantaneous-step representation with stable additive-baseline
  plateaus separated by finite physical-time transition support.
- Required transition sample support to be derived from the applicable timing
  vector, so the contract is invariant to sampling cadence and scan strategy.
- Defined transition support as explicitly flagged and unmodeled, excluded it
  from both plateau estimators, and separated it from downstream propagated
  influence.
- Admitted a plan-selected additive translation between sufficiently supported
  stable `x` plateaus, including explicit estimator, support, quality,
  uncertainty, reference, direction, and failure behavior.
- Required no invented offset under insufficient support while preserving the
  event boundary and explicit per-plateau disposition.
- Explicitly excluded gain, responsivity, and general response-change fitting
  from the v0.1 level-shift model.
- Allowed compact event/treatment state and population summaries in ordinary
  output while leaving detailed event fits to verbose/diagnostic products.
- Updated cross-cadence, correction, transition-validity, response-change, and
  insufficient-support falsifiers.

No numerical estimator, threshold, implementation conformity, validation
result, science-impact result, readiness claim, or production claim was
created.
