# SCI-VAL v0.1 — Ownership Decision r0.2

Status: scientific-owner approved

Decision date: `2026-08-20`

Scientific owner: Grant Wilson

## Decision

The scientific owner approved the following correction after reviewing the
first Stage A revision:

1. A producer owns the truth, typed causes, and Boolean composition of the
   facts and producer-local supports it publishes.
2. A producer shall not combine facts it does not own or decide admission for
   a downstream scientific use it does not own.
3. The owner of a named scientific use owns the policy that maps admitted
   producer facts and producer-local supports into admission for that use.
   Thus PTC owns its fit/application/output supports and MAP owns its map
   contribution/support/final-validity decisions.
4. SCI-VAL owns shared types, knowledge-state logic, immutable identity and
   provenance, cause preservation, and deterministic evaluation mechanics. It
   may execute an exact supplied policy but does not originate or own that
   policy.
5. Causes accumulate by set union without erasure. For one exact use,
   applicable restrictions are conjunctive in permission (equivalently,
   exclusions are disjunctive): one permission cannot rescue an occurrence
   excluded by another applicable restriction. Any override, supersession, or
   exception must be explicit in the same use-owner policy and must preserve
   the underlying causes.
6. The eight canonical profile names are shared vocabulary, not VAL-owned
   scientific policies.
7. A statement such as “diagnostic display permitted” is a use-specific
   disposition, not an additional cause.

## Effect On Earlier Stage A Recommendations

- `VAL-OWNER-Q001` is resolved with the narrower authority above.
- `VAL-OWNER-Q002` is resolved by retaining the eight names as vocabulary and
  assigning each realized policy to its scientific-use owner.
- `VAL-OWNER-Q003` is resolved with the proposed four-axis and
  `decision_unavailable` truth-domain behavior.
- `VAL-OWNER-Q004` is resolved with exact representative
  synthesis/replacement disqualifying `independent_exposure` only; other uses
  remain owner-policy governed.
- `VAL-OWNER-Q005` records the newly explicit composition-owner decision.

This decision authorizes implementation-blind Stage B contract derivation. It
does not modify PTC, MAP, RTC, or another adjacent contract; establish
implementation conformity; validate an implementation; freeze SCI-VAL; or
authorize production use.
