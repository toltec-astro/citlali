# SCI-FLT-FIXED v0.1 Freeze-Candidate Profile Architecture Report

Report identity: `SCI-FLT-FIXED-PROFILE-ARCHITECTURE v0.1/freeze-candidate`

Status: PASS; exact draft identities, dispositions, and actor boundaries reproduced; r0.3-origin profiles remain unregistered and not Registry-evaluated

Policy-record SHA-256: `c8767f7bbe044ae521f1b8473c2bde1eff2802d156bbc5edecfdd1a15cd3ae4c`

## Profiles

- `SCI-FLT-FIXED:input_bundle_admission@1`: domain `one exact request, one complete immutable parent bundle, and one exact resolved FLT-owned plan`; consumer action `admit the exact bundle to parent-row admission, or fail closed with typed causes; perform no payload arithmetic`
- `SCI-FLT-FIXED:input_parent_row_admission@1`: domain `one exact row in S_parent_fact under one eligible bundle, frozen plan, immutable parent facts, and named FLT use`; consumer action `return one immutable parent-row decision with typed causes to FLT; do not construct J_full or S_out and perform no payload arithmetic`
- `SCI-FLT-FIXED:output_publication@1`: domain `one exact complete_publication_disposition_candidate with product_candidate or no_output_support_candidate variant`; consumer action `return disposition and prescribed action to the FLT publisher; the profile or VAL evaluation performs no publication, and the publisher alone realizes or declines the FLT product`

## Actor boundary

Parent-row decisions establish `S_parent_fact` and `D_m` facts for FLT-owned construction of `J_full` and `S_out`. VAL may create a decision artifact. The FLT publisher alone performs or declines publication and owns realization and FLT-local validity.

## Exact freeze-candidate dispositions

- Empty nonzero `S_out`: complete `no_output_support_candidate`; requested, applicable, ineligible, `not_produced` (`no_full_footprint_output_rows`).
- Base versus qualified companions: honest unavailable companions are permitted only for the base request; each qualified request requires its named exact companion.
- Identity and zero: `product_candidate` and publisher action precede `realized_identity` or `realized_zero`.
- Publication failure: eligible transformation failure is `realization_failed` and emits no complete product.
- Late NOI request: publication-time not-requested is provenance; the child owns its lifecycle and cannot mutate FLT.

## Nonclaims

This report supplies no Registry approval or evaluation, numerical route, implementation conformity, validation, readiness, production authorization, or Unity claim.
