# SCI-JINC v0.1 Source/Registry Lifecycle Disposition r0.3

Status: incorporated into an implementation-blind Stage B author draft;
final Stage B acceptance and freeze remain pending

Prepared: `2026-08-29`

## Selected Lifecycle Model

Lifecycle model A, exact-source lock, is retained.

SCI-JINC v0.1 is compatible only with the exact complete bound objects named
by the author packet and r0.3 source-closure report, including:

- `SCI-PTC_TO_SCI-JINC v0.1/r0.3`;
- `SCI-AST_TO_SCI-JINC v0.1/r0.2`;
- immutable Registry identity `SCI-JINC:jinc_map_contribution@1`;
- the exact SCI-VAL JINC binding record;
- the exact JINC-specific source-binding Registry snapshot; and
- the exact JINC-specific profile Registry snapshot.

No ambient "current Registry," nearby successor, inferred alias, or partial
name match may substitute for those exact bytes. A change to any bound object
requires a versioned SCI-JINC successor.

The lock is intentionally over the exact complete JINC-specific snapshot
files supplied by the closed source packet. This draft does not invent
row-level object identities or a compatible-succession mechanism absent from
that authority.

The shared exact-source-lock definition, requirements `SCI-JINC-REQ-037`,
`SCI-JINC-REQ-043`, and `SCI-JINC-REQ-044`, and prediction
`SCI-JINC-PRED-033` carry this lifecycle rule without changing identifiers.

SCI-VAL binding records are process controls and add no coefficient family,
TolTEC parameter values, numerical-adequacy profile, or scientific content.
This disposition makes no implementation, validation, readiness, or
production claim.
