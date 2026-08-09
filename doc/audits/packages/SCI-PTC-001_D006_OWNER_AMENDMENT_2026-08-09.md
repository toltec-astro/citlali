# SCI-PTC-001 D006 owner amendment — 2026-08-09

Record ID: `SCI-PTC-001-D006-OWNER-AMENDMENT-2026-08-09`

Status: owner-approved; PTC contract decisions complete; documentation and
coordination only

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

This record preserves the bytes and meaning of D001--D005. With D006, all six
PTC owner decisions are approved and the package contract is complete.

## Approved D006 eligibility, fallback, and reproducibility contract

1. All PTC fitted-state arithmetic admits only eligible finite samples.
   Flags, validity, eligibility, and finiteness are distinct states; a finite
   value is not thereby eligible.
2. Shifted or null surrogates shift the signal and its associated validity and
   eligibility mask together.
3. Insufficient eligible support produces an explicit `unavailable` or
   `rejected` state. Zero or any fallback must never be represented as a valid
   estimate of the unavailable quantity.
4. Every fallback is typed with its cause and decision stage.
5. When randomness affects a scientific output, persist the seed,
   algorithm/version, and relevant input identity needed for deterministic
   reproduction. Storing every realized shift is not required unless a
   declared consumer depends on it.
6. Selection uncertainty may be explicitly `unavailable`. No calculated-
   uncertainty claim is permitted when that status is unavailable.

## Finding disposition and non-authorization

D006 resolves the owner policy dependency for F001 and the applicable
missing-data, fallback, surrogate, and reproducibility portions of the PTC
contract. F001 and every other applicable implementation, dependency, and
evidence finding remain open until implemented and independently re-audited.

The package contract axis becomes `approved`. Implementation remains
`nonconformant`, validation remains `in_progress`, production remains
`existing_use_only`, and verdict remains `amend`. This decision does not
authorize PTC repair, optional transfer characterization, validation
execution, application/test/configuration changes, Unity, reductions,
external contact, re-audit, downstream launch, production change, merge, or
push.
