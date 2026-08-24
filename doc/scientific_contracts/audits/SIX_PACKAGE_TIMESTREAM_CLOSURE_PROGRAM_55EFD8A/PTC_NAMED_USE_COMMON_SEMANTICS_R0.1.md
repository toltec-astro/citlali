# SCI-PTC Named-Use Common Semantics Fragment v0.1/r0.1

Status: owner-approved documentation and Registry-factoring authority

Date: `2026-08-24`

Scientific owner: Grant Wilson, SCI-PTC

Artifact identity:
`SCI-PTC-COMMON-NAMED-USE-SEMANTICS-v0.1/r0.1`

This artifact is not a VAL profile and its identity is not a profile key. It
cannot be requested or evaluated independently, grants no permission, and
produces no eligibility result.

## Exact authority

This fragment records `WP5-OWNER-D003` and `WP5-OWNER-D011` in
`WP5_VAL_SCIENTIFIC_OWNER_DECISION_PACKET.md` at Git commit `44662a36b`, file
SHA-256 `9bc101e8447173836380e00ea58185fc2e67cbcbac5077ff1578ca5dc27139fd`.
It is consistent with frozen SCI-PTC v0.1/r0.5
`SCI-PTC-REQ-012`, `SCI-PTC-REQ-079`, and `SCI-PTC-REQ-098`, whose freeze
record has SHA-256
`8357961a49272adc40e27a8aa9e760e0d01ff2419ae2c88a62c0f93c9f959e66`.

This is a post-freeze named-use policy artifact. It neither edits nor silently
reopens the frozen PTC mathematical contract.

## Common semantics

For every registered PTC named use `U`:

1. `U` is a complete and distinct scientific proposition. Permission under
   another use `V` conveys no permission under `U`:

   \[
   E_V \not\Rightarrow E_U \qquad (U\ne V).
   \]

2. PTC preserves upstream facts, causes, and classifications and does not
   upgrade them. In particular, CAL `engineering-only` remains a producer
   fact; its admission consequence is declared by the exact named use rather
   than imposed globally by CAL, PTC, or VAL.
3. Only facts that `U` explicitly declares scientifically relevant to its
   decision may affect that decision. The mere existence, availability, or
   unknown state of other metadata has no admission consequence.
4. VAL evaluates the complete owner-supplied proposition under its frozen
   T/F/U/C and four-axis semantics. VAL supplies no missing predicate,
   threshold, exception, or permission.

## Explicit noncontents

This common fragment contains no direct-origin exclusion, fit population,
loading-estimator input, group or rank guard, output-retention rule, response
or uncertainty requirement, exception, or missing/conflict disposition. Those
remain in the complete named-use records where scientifically applicable.

It creates no runtime common-policy object, inheritance mechanism, sidecar,
payload, serialization requirement, duplicated provenance, or separate
engineering route.

## Version and compatibility

Every profile using this fragment binds this exact identity and byte digest.
Any content change creates a new fragment version and new dependent profile
identity or version. No change acts retroactively on an earlier evaluation.
