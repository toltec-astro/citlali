# SCI-FLT-FIXED v0.1 Numerical-Domain And Partial-Function Amendment

Record identity: `SCI-FLT-FIXED-NUMERICAL-DOMAIN-AMENDMENT v0.1/freeze-candidate`

Status: implementation-blind conditional scientific-owner freeze-candidate amendment; owner signature required

Scientific owner: Grant Wilson

Stage B date: `2026-08-31`

The shared normative core controls. This amendment records the final typed
domain repair without adding another scientific method.

## Typed domains

```text
S_parent_fact
  = exact parent row-identity and fact domain;

D_m
  = {q in S_parent_fact:
       an available finite real signal payload exists at q};

m : D_m -> R.
```

A stored missing, unavailable, or non-finite payload is a typed fact in
`S_parent_fact`; it is not an element of `m`. Parent-shaped storage is a
representation, not the scientific vector domain.

For an ordinary convolution,

```text
S_out
  = {p:
       for every r in K_req,
         p-r is in S_parent_fact,
         p-r is admitted for the exact FLT use,
         p-r is in D_m,
         and every required predicate passes};

y : S_out -> R.
```

FLT establishes membership in `D_m` before evaluating `m_q`. The selector is
resolved from immutable typed parent facts; no payload amplitude selects or
tunes the plan.

## Nonclaims

This amendment supplies no implementation, numerical-adequacy, validation,
readiness, production, or Unity finding.
