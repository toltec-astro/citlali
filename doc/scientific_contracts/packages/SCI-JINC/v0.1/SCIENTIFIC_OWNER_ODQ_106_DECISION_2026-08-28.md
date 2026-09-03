# SCI-JINC-ODQ-106 — Scientific-Owner Disposition

Status: owner approved; bounded Stage A disposition

Scientific owner: Grant Wilson

Decision date: `2026-08-28`

## Approved Scientific Disposition

For one observation, SCI-JINC v0.1 may produce at most one bundle for each
stable TolTEC array admitted and requested under the exact JINC realization.
The possible produced-bundle cardinality is therefore zero through three over
the exact stable array set `a1100`, `a1400`, and `a2000`; this is not a
requirement to synthesize all three.

A missing, unavailable, or unrequested array produces no placeholder bundle,
empty-array product, or synthetic failure product. Its absence does not make a
different produced JINC bundle invalid. Any applicable requested, effective,
resolved, realized, availability, cause and failure facts remain represented
at their existing plan/bundle provenance granularity; this decision creates no
additional per-contribution provenance.

Each produced bundle has an independent scientific identity bound to the exact
observation, stable array, JINC plan and realization, destination map geometry
and lifecycle generation. Contributions may accumulate incrementally within
that identity, consistently with `SCI-JINC-ODQ-105`. Contributions belonging
to different array identities or different destination map identities must not
be merged. Cross-array, network-combined or shared-destination products remain
unavailable unless separately authorized.

## Stage Consequence

`SCI-JINC-ODQ-106` is closed for base-v0.1 grouping, cardinality and
destination identity. The decision changes sanitized Stage A author-control
bytes and remains subject to renewed exact-byte approval under
`SCI-JINC-STAGE-A-Q002`. It does not launch Stage B, prescribe implementation
containers or concurrency, add placeholder products or provenance machinery,
modify implementation, or establish conformity, validation, achieved
performance, readiness or production status.
