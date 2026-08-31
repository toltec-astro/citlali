# SCI-FLT-FIXED v0.1 Immutable FLT/NOI Relation Design

Record identity: `SCI-FLT-FIXED-NOI-RELATION-DESIGN v0.1/draft-r0.4`

Status: implementation-blind Stage B closure draft; scientific-owner review required

Scientific owner: Grant Wilson

Stage B date: `2026-08-31`

## Immutable FLT role

`FLT-NOI-COMPATIBILITY` is part of the immutable FLT bundle. It contains only:

- exact FLT product identity;
- exact operator and row-domain identity;
- exact FLT-to-NOI boundary/profile compatibility;
- fixed-state transformation semantics;
- request state known at FLT publication; and
- typed compatibility or unavailability.

It contains no future NOI product identity.

A `not_requested_at_FLT_publication` value records historical provenance only.
It does not prohibit a later independently requested SCI-NOI child.

## Later relation

A later SCI-NOI product references the immutable FLT parent. An optional
reverse `SCI-FLT-FIXED_TO_SCI-NOI-RELATION` is separately versioned, is not an
FLT atomic role, and cannot change FLT completeness, realization, validity,
identity, or bytes.

The later child owns its request, applicability, eligibility, realization,
generation, and failure. It must still satisfy exact FLT boundary and profile
compatibility, and no child-owned state is copied back into FLT.

## Nonclaims

This design supplies no numerical transformed-uncertainty route and makes no
implementation, validation, achieved-uncertainty, readiness, or production
claim.
