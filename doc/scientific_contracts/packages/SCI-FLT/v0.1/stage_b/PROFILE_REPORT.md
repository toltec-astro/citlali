# SCI-FLT-FIXED v0.1 r0.4 Profile Architecture Report

Report identity: `SCI-FLT-FIXED-PROFILE-ARCHITECTURE v0.1/draft-r0.4`

Status: PASS; exact draft identities, dispositions, and actor boundaries reproduced; r0.3-origin profiles remain unregistered and not Registry-evaluated

Policy-record SHA-256: `d4002f9a830339717ab2253a12409dc6c66110b1804375987e29c56babf4d017`

## Profiles

- `SCI-FLT-FIXED:input_bundle_admission@1`: domain `one exact request, one complete immutable parent bundle, and one exact resolved FLT-owned plan`; consumer action `admit the exact bundle to parent-row admission, or fail closed with typed causes; perform no payload arithmetic`
- `SCI-FLT-FIXED:input_parent_row_admission@1`: domain `one exact parent row under one eligible bundle, frozen plan, immutable parent domain, and named FLT use`; consumer action `return one immutable parent-row decision with typed causes to FLT; do not construct J_full or S_out and perform no payload arithmetic`
- `SCI-FLT-FIXED:output_publication@1`: domain `one eligible applied SCI-FLT-FIXED route, including either a complete publication candidate or an exact applied-no-support branch`; consumer action `return disposition and prescribed action to the FLT publisher; the profile or VAL evaluation performs no publication, and the publisher alone realizes or declines the FLT product`

## Actor boundary

Parent-row decisions feed FLT-owned construction of `J_full` and `S_out`. VAL may create a decision artifact. The FLT publisher alone performs or declines publication and owns realization and FLT-local validity.

## Exact r0.4 dispositions

- Empty nonzero `S_out`: `applied_no_scientific_output_support` -> `not_produced` (`no_full_footprint_output_rows`).
- Base versus qualified companions: honest unavailable companions are permitted only for the base request; each qualified request requires its named exact companion.
- Identity and zero: complete candidate and publisher action precede `realized_identity` or `realized_zero`.
- Publication failure: eligible transformation failure is `realization_failed` and emits no complete product.
- Late NOI request: publication-time not-requested is provenance; the child owns its lifecycle and cannot mutate FLT.

## Nonclaims

This report supplies no Registry approval or evaluation, numerical route, implementation conformity, validation, readiness, production authorization, or Unity claim.
