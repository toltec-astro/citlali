# SCI-PTC-001 D004 owner amendment — 2026-08-09

Record ID: `SCI-PTC-001-D004-OWNER-AMENDMENT-2026-08-09`

Status: owner-approved; documentation and coordination only

## Exact authority and preserved identities

The project owner approves this amendment for `SCI-PTC-001` at governing
application SHA `46ad23888a40f5102cdfd50c06e49a549bdf8a20`.

The completed audit remains immutable at final commit
`01ee247461d6c19bc4db81ccac4fec21af162c88`, parent/core
`66e8d6f98c3e22da74de4eea84e568a0b4cc6310`, and tree
`e6685c920ff37f1d4e51d27ecf23b73ac16087b5`. Its independent-core and final
report SHA-256 digests remain, respectively,
`82c0835f51ea9b1fa8a37489f289be89a8018a0b2700e84b1e25c2e4d2a013c2`
and `c46a15c142d0938baf9576d84a19332e0d46b34852b4d59c0029ba00ac62d7e6`.

This record does not change the frozen audit artifacts or the bytes or meaning
of the approved D001--D003 authorities. D005 and D006 remain unresolved.

## Approved D004 coefficient and covariance contract

Existing PTC detector-weight families remain valid scalar analysis and
gridding coefficients. For every coefficient family, the product contract
must publish:

- the coefficient-family identity;
- units;
- normalization scope;
- lifecycle; and
- every applied factor.

These coefficients must not be labeled or consumed as formal precision,
inverse variance, significance, or independent-noise authority unless the
complete stronger conditions for the exact declared interpretation are
independently proved.

Full covariance construction or retention is not mandatory. When covariance
is not supported, its status must be explicitly `unavailable`, and consumers
must fail closed for claims that require covariance or the stronger precision,
inverse-variance, significance, or independent-noise interpretation.

Existing coefficient-weighted mapmaking remains permitted under these scalar
coefficient semantics and the package's existing-use restrictions. This
approval does not promote the coefficients to a stronger statistical class or
expand production authority.

## F006 disposition and non-authorization

`SCI-PTC-001-F006` remains a P1 contract gap until the approved identities,
units, normalization scopes, lifecycles, applied factors, and covariance
availability are implemented, tested at an exact successor, and independently
re-audited. Its bounded closure does not require constructing or retaining a
full covariance product.

Package axes remain `proposed`, `nonconformant`, `in_progress`, and
`existing_use_only`, with verdict `amend`. This decision does not authorize
PTC repair, optional transfer characterization, application/test/configuration
changes, Unity, reductions, external contact, re-audit, downstream launch,
production change, merge, or push.
