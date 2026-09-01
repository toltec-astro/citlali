# SCI-FRUIT v0.1 — Method, Claim, Evidence, And Decision Identity Taxonomy

Status: **Stage A candidate taxonomy; no identity is instantiated or qualified**

## Separate Typed Identities

The former overloaded tuple `K=(M,P,S,Q,D,H,Pi,E)` is retired. It mixed the
scientific method, the claim being tested, the evidence generation, and the
owner's disposition. SCI-FRUIT instead uses:

```text
METHOD_ID = (
  parent_and_reduction_route,
  recurrence,
  feedback_state_schema,
  parameter_or_adaptation_policy,
  stopping_and_terminal_policy
)

CLAIM_ID = (
  science_profile,
  applicability_domain,
  exact_historical_control,
  frozen_qualification_protocol
)

EVIDENCE_ID = (
  population_split,
  execution_generation,
  software_and_environment,
  paired_results,
  uncertainty_and_failure_record
)

QUALIFICATION_DECISION = (
  METHOD_ID,
  CLAIM_ID,
  EVIDENCE_ID,
  disposition,
  owner_and_date
)
```

## Identity Rules

- `parent_and_reduction_route` is an exact ordinary-MAP, JINC, FLT-FIXED, or
  future owner-admitted FLT-MATCHED route, including grouping and every
  scientifically consequential parent option. Evidence is not pooled across
  parent routes by implication.
- A changed recurrence, feedback-state schema, parameter/adaptation policy, or
  stopping/terminal policy creates a new `METHOD_ID`.
- A changed science profile, applicability domain, historical control, or
  qualification protocol creates a new `CLAIM_ID`.
- Evidence generation is not part of the method. Independent
  `EVIDENCE_ID` generations may address the same frozen `CLAIM_ID` only under
  a prospectively declared evidence-combination rule.
- Changed population split, execution generation, software/environment,
  paired results, or uncertainty/failure record creates a new `EVIDENCE_ID`.
- A qualification decision binds exactly one method, claim, evidence record,
  disposition, owner, and date. It is not the method itself and cannot silently
  enlarge the claim domain.

## Lineage And Restart

Per-iteration update contributions may receive stable lineage identities and
may be retained when required for diagnostics or causal restart. Stable
identity does not require permanent retention. A contribution is not an
independently calibrated or scientifically interpretable sky product unless a
separate authority explicitly establishes its estimand, response, support,
uncertainty, and validity.

No symbol in this taxonomy uses bare `Q` for a science profile; `Q` already has
authoritative meanings in MAP/JINC contexts.

## Non-Effect

This taxonomy does not select or instantiate a parent route, recurrence,
feedback schema, profile, population, protocol, method, threshold, or evidence
generation.
