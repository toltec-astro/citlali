# SCI-MAP-002 JINC formal-support validity-mask owner decision — 2026-08-03

Status: owner approved contract clarification; no implementation work
authorized

Package: `SCI-MAP-002`

Decision ID: `SCI-MAP-002-D003-MASK-001`

Authority: project owner

## Decision

The existing `coverage_bool` product name is retained, but its defined
identity is the authoritative JINC formal-support validity mask. A pixel is
valid only when the finalized signal and formal weight are finite, formal
weight is strictly positive, and every required JINC admission and
unit-invariant conditioning check passed.

Coverage is a distinct coefficient-squared-time product and is not a validity
proxy. An empirical working-weight policy may further exclude a formally valid
pixel for a named downstream product, but it may never promote a formally
invalid pixel. The formal/empirical distinction must remain explicit in
product metadata and provenance.

This corrects the observed zero/non-finite-weight mask defect without adding a
new output format. Future validation must cover zero, negative, NaN, Inf,
below/equal/above conditioning boundaries, empirical downgrade, and attempted
empirical promotion. No code change, Unity evidence, repair, re-audit, or
production-status change is authorized.
