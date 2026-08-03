# SCI-MAP-002 JINC parameter and coefficient-admission owner decision — 2026-08-03

Status: owner approved contract clarification; no implementation work
authorized

Package: `SCI-MAP-002`

Decision ID: `SCI-MAP-002-D003-ADMISSION-001`

Authority: project owner

## Decision

JINC parameter and coefficient admission is fail closed. For every selected
array, the shape parameters `a`, `b`, and `c`, `r_max`, pixel size, and array
scale must be finite and strictly positive. Required stable array identity
must be present. Requested and effective values, including effective
`subpixel_n`, must be serialized.

Finite negative JINC coefficients are valid physical lobes and must retain
their sign. A non-finite evaluated coefficient is invalid before deposition:
it must not be clipped to zero or silently omitted. Missing, nonphysical, or
non-finite parameter/coefficient state fails the selected JINC map product as
a required failure; it is not converted into a detector-local omission.

Future validation must cover missing array identity, each parameter-domain
boundary, non-finite values, signed finite lobes, and non-finite coefficients.
This decision does not alter parameters, start a numerical campaign, or
authorize code changes, Unity evidence, repair, re-audit, or production-status
change.
