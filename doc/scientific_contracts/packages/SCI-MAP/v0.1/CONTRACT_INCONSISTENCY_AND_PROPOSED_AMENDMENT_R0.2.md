# SCI-MAP v0.1 Contract Inconsistency And Proposed Amendment

Record ID: `SCI-MAP-CI-001`

Date: `2026-08-16`

Status: **OPEN — owner approval required before normative incorporation**

## Inconsistency

The unchanged v0.1 formal authority defines

```text
T_norm = Q_star * c / 10
T_sci  = Q_star * c
c      = coverage_cut
```

and compares `Q_p` directly with both thresholds. `Q_p`, `Q_star`, `T_norm`,
and `T_sci` therefore have the same coefficient unit. Dimensional consistency
requires `c=coverage_cut` to be dimensionless.

The r0.1 shared authority nevertheless says that the unit or dimensionless
status of `c` is unresolved in the coverage-cut admission equation,
`SCI-MAP-REQ-031`, `SCI-MAP-REQ-032`, `SCI-MAP-PRED-012`, and
`SCI-MAP-OD-007`. Those statements are inconsistent with the governing
threshold equations.

## Proposed owner amendment

Approve the following exact scientific correction:

> `coverage_cut` is a dimensionless support-policy parameter. Dimensional
> status is fixed by the threshold equations and is not an open owner choice.
> The admissible numerical domain, boundary-case dispositions, production-
> recommended range, policy authority, and failure scope remain open under
> `SCI-MAP-OD-007`.

After approval, make only these bounded normative changes:

1. replace the open `unit_status(c)` admission with an explicit dimensionless
   declaration and exact-value policy admission;
2. revise `SCI-MAP-REQ-031` and `SCI-MAP-REQ-032` to record `c` as
   dimensionless while retaining fail-closed numerical-domain behavior;
3. revise `SCI-MAP-PRED-012` to test exact value/domain admission, not unit
   admission;
4. narrow `SCI-MAP-OD-007` to numerical domain, boundary cases,
   production-recommended range, policy authority, and failure behavior; and
5. update the formal crosswalk and generated decision register without
   changing any requirement or decision ID.

## Explicit non-changes before approval

This r0.2 authorship pass does **not** alter the shared equations, canonical
requirements, prediction, generated formal decision register wording, or
engineering contract. It records the inconsistency, teaches the dimensional
consequence in the non-normative science rationale, and awaits owner action.

No range is inferred for negative, zero, non-finite, or greater-than-one
values. No validation or implementation claim follows from this proposal.

