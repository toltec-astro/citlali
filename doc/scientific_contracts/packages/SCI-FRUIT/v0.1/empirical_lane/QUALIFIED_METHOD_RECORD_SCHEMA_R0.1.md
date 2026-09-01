# SCI-FRUIT v0.1 — Qualified-Method Record Schema r0.1

Status: **future record schema; no method, claim, evidence, or qualification
decision exists**

## Purpose

This schema defines the terminal scientific object that an approved empirical
lane must return before any method-specific Stage B packet can be considered.
It prevents prototype code, a favorable plot, or an aggregate score from
standing in for an exact qualification decision.

## Required Identity

```text
QUALIFIED_METHOD_RECORD = (
  METHOD_ID,
  CLAIM_ID,
  EVIDENCE_ID,
  QUALIFICATION_DECISION,
  recurrence_and_operator_order,
  causal_state_and_checkpoint_contract,
  response_uncertainty_support_validity_and_failure_claims,
  historical_compatibility_assessment,
  limitations_and_out_of_domain_behavior,
  forbidden_claims,
  owner_binding
)
```

The component identities retain their accepted definitions. A changed method,
claim, population split, execution generation, software/environment, paired
result, or uncertainty/failure record creates a different record.

## Mandatory Evidence Summary

The record includes:

1. absolute truth-recovery and nuisance/null results;
2. paired candidate-versus-historical contrasts on frozen common support;
3. support gained/lost and availability accounting;
4. protected non-inferiority and prioritized material-improvement decisions;
5. cluster/dependence-aware uncertainty and multiplicity treatment;
6. lower-tail, important-stratum, failure, unavailable, catastrophic, rescue,
   and regression outcomes with exact denominators;
7. actual-terminal, oracle-only evaluation, hard-cap, censored, oscillation,
   drift, and time-to-quality results;
8. computational performance reported separately unless a trade was frozen;
9. all protocol deviations, access events, exclusions, and contamination
   incidents; and
10. complete lineage, software/environment, restart/replay, and retention
    evidence.

## Allowed Dispositions

An owner decision may record broad qualification, exact profile/domain-
restricted qualification, materially justified specialization, no qualifying
replacement, or invalid/unavailable evidence. Qualification never silently
creates an operational fallback.

## Stage B Sanitization Boundary

Only an owner-approved record may be considered for `EL-GS`. The later author
packet may contain the accepted scientific method and claim boundary, but it
must exclude implementation source, prototype mechanics, search/tuning
history, failed candidates, raw qualification outcomes, and validation claims.
The sanitized packet is a new exact object requiring separate owner approval.
