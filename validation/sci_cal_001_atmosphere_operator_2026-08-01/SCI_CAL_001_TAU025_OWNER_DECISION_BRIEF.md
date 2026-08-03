# SCI-CAL-001 tau225 engineering-extension owner decision brief

Decision requested: approve, revise, or reject the proposed future AM study
described in `SCI_CAL_001_TAU025_ENGINEERING_EXTENSION_PROTOCOL.md`
(SHA-256 `2552a8fee94bc64719504528d3a763c402bfedd9c7ec1380c2d4f5d1775b6967`).

`CAL-ATM-D006` already fixes the policy, not the execution:

- `0 <= tau225 <= .15` is a future science-qualification target.
- `.15 < tau225 <= .25` is engineering availability only.
- Above `.25`, non-finite tau, or missing identity is unsupported: no silent
  extrapolation or calibrated-science claim.

The extension protocol proposes direct AM 12.2 truth at construction nodes
`.15/.20/.25`, independent held-out opacities `.1625/.175/.1875/.2125/.225/.2375`,
construction elevations `25,35,45,55,65,75,80`, and independent elevations
`29,41,53,67,79`. It requires one continuous future candidate across `.15`,
not a science/engineering selector switch, and binds the existing exact
TolTECA v1 ECSV passband convention. The `.15` left/exact/right `nextafter`
triplet is only a no-AM algebraic/operator-evaluator diagnostic after that
candidate is defined; it is not a direct-AM target or an AM scale search.

Owner choices required before any AM request:

1. Approve or revise the exact copied-AM profile matrix; generic q95 remains
   ineligible and cannot fill the `.158313198574890929` to `.25` gap.
2. Approve or revise the proposed held-out node lattice and exact execution
   cost/cache request. That request must bind and enforce the existing
   approved `WARN-001` bounded warning-bearing numerical-evidence policy
   verbatim; new warning classes or deviations fail closed.
3. Approve, revise, or reject the proposed **5% maximum held-out fractional
   extinction-correction-error** engineering screen. It is not the science
   `<=1%` criterion, a photometric-accuracy claim, or an adopted threshold.

Any eventual quality state must be assigned once per coherent observation or
declared processing segment from its maximum eligible tau225, with compact
identity/coverage/provenance. A mixed unit exceeding `.15` is engineering
qualified unless a later contract explicitly partitions it.

Current status: protocol preparation complete; no AM execution, operator,
application change, output format, adoption, repair, re-audit, Unity action,
or production change has been authorized or performed.
