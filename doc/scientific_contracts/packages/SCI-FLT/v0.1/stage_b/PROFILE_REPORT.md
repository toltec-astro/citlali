# SCI-FLT-FIXED v0.1 r0.3 Profile Architecture Report

Report identity: `SCI-FLT-FIXED-PROFILE-ARCHITECTURE v0.1/draft-r0.3`

Status: PASS; corrected draft identities and actor boundaries reproduced; not registered or Registry-evaluated

Policy-record SHA-256: `e9422403061f3c0b5b107d88ecc323eec181ba83a726a34e53a36540fb474e66`

## Profiles

- `SCI-FLT-FIXED:input_bundle_admission@1`: domain `one exact request, one complete immutable parent bundle, and one exact resolved FLT-owned plan`; consumer action `admit the exact bundle to parent-row admission, or fail closed with typed causes; perform no payload arithmetic`
- `SCI-FLT-FIXED:input_parent_row_admission@1`: domain `one exact parent row under one eligible bundle, frozen plan, immutable parent domain, and named FLT use`; consumer action `return one immutable parent-row decision with typed causes to FLT; do not construct J_full or S_out and perform no payload arithmetic`
- `SCI-FLT-FIXED:output_publication@1`: domain `one eligible complete SCI-FLT-FIXED publication candidate after application`; consumer action `return disposition and prescribed action to the FLT publisher; the profile or VAL evaluation performs no publication, and the publisher alone realizes or declines the FLT product`

## Actor boundary

Parent-row decisions feed FLT-owned construction of `J_full` and `S_out`. VAL may create a decision artifact. The FLT publisher alone performs or declines publication and owns realization and FLT-local validity.

## Nonclaims

This report supplies no Registry approval or evaluation, numerical route, implementation conformity, validation, readiness, production authorization, or Unity claim.
