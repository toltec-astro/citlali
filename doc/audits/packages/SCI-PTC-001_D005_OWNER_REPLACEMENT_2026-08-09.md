# SCI-PTC-001 D005 owner replacement — 2026-08-09

Record ID: `SCI-PTC-001-D005-OWNER-REPLACEMENT-2026-08-09`

Status: owner-approved reject-and-replace disposition; documentation and
coordination only

## Exact authority and preserved identities

The project owner approves this replacement for `SCI-PTC-001` at governing
application SHA `46ad23888a40f5102cdfd50c06e49a549bdf8a20`.

The completed audit remains immutable at final commit
`01ee247461d6c19bc4db81ccac4fec21af162c88`, parent/core
`66e8d6f98c3e22da74de4eea84e568a0b4cc6310`, and tree
`e6685c920ff37f1d4e51d27ecf23b73ac16087b5`. Its independent-core and final
report SHA-256 digests remain, respectively,
`82c0835f51ea9b1fa8a37489f289be89a8018a0b2700e84b1e25c2e4d2a013c2`
and `c46a15c142d0938baf9576d84a19332e0d46b34852b4d59c0029ba00ac62d7e6`.

This record preserves the bytes and meaning of D001--D004. D006 remains
unresolved.

## Rejected proposal and replacement basis

The owner rejects and replaces the original uniform immutable-bundle and
exhaustive-replay proposal for D005. The application computes `ptcdata` in
memory, optionally persists it, and passes that same transformed state to
mapmaking. Provenance burden therefore follows scientific authority and
declared consumption rather than mere persistence.

The original proposal and immutable audit remain historical evidence. This
record supersedes their D005 coordination recommendation and the corresponding
uniform replay/completeness closure wording for F007 and F008.

## Approved D005 interface and product contract

1. The PTC-transformed timestream is the authoritative intermediate state at
   the PTC-to-mapmaking interface, not an independent sky estimator. Its
   in-memory contract defines signal, validity, stable detector identity, time
   support and rate, coefficient semantics, units, and the material realized
   PTC state consumed by mapmaking.
2. Final maps and downstream science products record the material RTC/PTC
   processing state that affected them.
3. Any persisted PTC timestream explicitly declares its role as either a
   `diagnostic_artifact` or a `requested_derived_analysis_product`.
   Persistence alone does not confer science-product status.
4. A diagnostic artifact records enough observation/scan, detector,
   time/rate, PTC-mode, configuration/code, validity, and honest completeness
   identity to prevent ambiguity. It does not require archival-grade replay or
   exhaustive internal-state serialization.
5. A requested derived analysis product additionally binds parent identity and
   the material realized PTC choices required by its declared consumer.
   Exhaustive selector, mask, random, or intermediate-state serialization is
   required only when a declared consumer or explicit reproducibility claim
   depends on it.
6. Required-output failure propagation and diagnostic best-effort behavior
   follow explicit output policy. No partial artifact may be represented as
   complete. A diagnostic failure does not invalidate an otherwise valid map
   unless that artifact was declared required.

## Finding disposition and non-authorization

F007 and F008 remain open until the bounded interface, role, material-state,
parent, output-policy, and honest-completeness contracts are implemented,
tested at an exact successor, and independently re-audited. Neither finding
requires uniform archival-grade replay or exhaustive internal-state
serialization absent a declared consumer or reproducibility claim.

Package axes remain `proposed`, `nonconformant`, `in_progress`, and
`existing_use_only`, with verdict `amend`. This decision does not authorize
PTC repair, optional transfer characterization, validation execution,
application/test/configuration changes, Unity, reductions, external contact,
re-audit, downstream launch, production change, merge, or push.
