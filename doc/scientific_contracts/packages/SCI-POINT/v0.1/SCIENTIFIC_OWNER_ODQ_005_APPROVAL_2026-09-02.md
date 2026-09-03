# SCI-POINT ODQ-005 Scientific-Owner Approval

Record identity: `SCI-POINT-ODQ-005-APPROVAL-2026-09-02`

Scientific owner: Grant Wilson

Decision date: `2026-09-02`

Status: approved Stage A scientific direction

## Approved Decision

The base-v0.1 compatibility method preserves the established configurable
Pointing-fit machinery:

- declared expected source center and configurable central search domain;
- weighted-peak initialization;
- the established global-search fallback;
- a bounded fit domain; and
- amplitude, fitted-width, and orientation-angle constraints.

These controls are scientifically consequential method state. The requested,
effective, and realized values or named states must be separately represented
and bound to the fit result. Use of the global-search fallback must be reported
as realized state. A numeric sentinel such as zero must resolve to an explicit
named effective state; it may not remain a scientifically invisible default.

The contract does not freeze one universal numerical value for every
observation. It preserves the existing configurable method and requires its
actual state to be explicit. An unavailable or incoherent required state may
not be silently replaced by a different search, domain, bound, or fallback.

No new search, initialization, support, or constraint algorithm is authorized
by this decision.

## Non-Effects

This approval does not define the ODQ-006 scientific acceptance or partial-
success policy, the ODQ-007 covariance publication policy, or any later owner
decision. It does not approve the complete Stage A packet, authorize Stage B,
change numerical behavior, or establish implementation conformity,
validation, achieved performance, readiness, or production state.
