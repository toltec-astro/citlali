# SCI-MAP v0.1 Sanitized PTC-Handoff Authority Extract

Date: `2026-08-26`

Status: proposed implementation-blind author reference; owner approval required

This extract contains only approved scientific boundary facts needed for the
bounded MAP revision. It contains no implementation observation, audit
finding, repair, test result, validation evidence, achieved performance, Unity
result, or production claim.

## Frozen PTC Facts Admitted By Reference

- `SCI-PTC-001-D005` and `SCI-PTC-REQ-069`: the in-memory
  PTC-transformed timestream is the authoritative PTC-to-MAP intermediate on
  the PTC-dependent route and is not an independent sky estimator.
- `SCI-PTC-REQ-076--077`: an explicitly disabled PTC request terminates on
  the authorized RTC export route without CAL, PTC, or MAP products. Ordinary
  map production requires a realized PTC-transformed timestream; disabled PTC
  and invalid rank produce no map and do not select a direct CAL-to-MAP
  fallback.
- `SCI-PTC-REQ-052` and `SCI-PTC-REQ-054--055`: loadings,
  centering/scaling parameters, diagnostics, and analysis/gridding
  coefficients are distinct. Only an explicitly named analysis/gridding
  family may face MAP. That family declares its indices, statistic/factors,
  domains, unit, normalization, support, lifecycle, consumers, numerical use,
  and prohibited interpretations. Re-estimation creates a new realized
  product state.
- `SCI-PTC-REQ-066`, `068`, and `088`: response availability is a typed axis
  distinct from product realization. A map-center or compact-source response
  does not establish a general response or MAP authority. Unavailable response
  does not imply that an otherwise supported realized PTC signal does not
  exist.
- `SCI-PTC-REQ-071`, `073`, and `079`: material state affecting a declared
  consumer remains linked through the realized provenance chain; fallbacks
  are explicit and typed; consuming a product does not transfer ownership of
  MAP estimation or named-use admission policy.

## Current VAL Boundary Admitted By Reference

The current SCI-VAL source-binding register binds frozen ALIGN, AST, RTC, CAL,
and PTC meaning. It intentionally leaves MAP deferred and unbound. It contains
no MAP predicate, threshold, or exception and does not make
`SCI-MAP:map_upstream_admission` evaluable.

VAL provides shared vocabulary, immutable source/profile binding, registry,
and evaluation machinery. The owner of a named scientific use authors the
profile. Therefore MAP must first define its own exact versioned admission
profile; VAL may then register and evaluate it without changing the policy.

## Approved Horizontal Owner Facts

- A paired x/r datum is one physical occurrence, but
  `independent_exposure` is evaluated for the explicitly named component or
  product aspect. Replaced x is not an independent x measurement and does not
  rewrite r origin or universally invalidate the pair.
- PTC availability is necessary but MAP's own use-specific admission is also
  required. Causes remain distinguishable; a downstream use consequence is
  not inferred from a cause name alone.
- Response and covariance absence limits only the claims that require the
  missing information. It does not automatically invalidate a map or prohibit
  later analysis.
- Later response, covariance, or corrected-map results are new versioned
  derivatives attached to exact parents and domains. Earlier MAP claims remain
  immutable.
- ALIGN/AST own realized sample-coordinate facts. MAP owns its target grid and
  projection operation.

## Authoring Constraints

Preserve every existing normative identifier. Amend only the obligations
whose premise is superseded by the facts above, append only genuinely new
obligations, and keep all unresolved local projection and threshold questions
explicitly unresolved. Do not claim implementation conformity, validation,
performance, freeze, or readiness.
