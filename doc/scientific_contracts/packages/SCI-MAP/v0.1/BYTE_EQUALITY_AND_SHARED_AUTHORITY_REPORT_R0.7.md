# SCI-MAP v0.1 r0.7 Byte Equality And Shared-Authority Report

Status: document-source integrity check only; no scientific freeze,
implementation conformity, validation, or readiness claim

## PTC-to-MAP boundary

The following two files compare byte-identically (`cmp` exit status 0):

- `doc/scientific_contracts/packages/SCI-PTC/v0.1/SCI-PTC_TO_SCI-MAP_BOUNDARY.md`
- `doc/scientific_contracts/packages/SCI-MAP/v0.1/SCI-PTC_TO_SCI-MAP_BOUNDARY.md`

Both have SHA-256
`db0eae0aeeb63a61ce1fdbc71914a8cb424e94cc6ae34e64f1b0ccbfe714e52d`
and canonical identity `SCI-PTC_TO_SCI-MAP v0.1/r0.1`.

## Shared MAP authority

Rationale, formal contract, and ECS do not maintain independent copies. Each
contains exactly one import of the same file:

`src/SCI-MAP-v0.1_SHARED_AUTHORITY_r0.7.tex`

The wrapper SHA-256 is
`08181fce5348103ac22ab13602c6726f3bb487e3537a4722966a11c149e87d3f`.
The digest over wrapper bytes followed by the six modules in manifest order is
`275cd4fa296b690011dd54fa326724573a8d854e7047734bdd8bc075e3f170d5`.
Because all views import that one path, the bytes are identical by
construction; there is no separately editable view-local authority copy.

## Owner-decision authority

All views derive decision semantics from
`SCIENTIFIC_OWNER_DECISION_LEDGER.md`, SHA-256
`bfcc18eced116309356d9d597a4423881ef67df058df07be30ec80532452dd9b`.
The formal and ECS rendered register is mechanically generated from it as
`src/SCI-MAP-v0.1_OWNER_DECISION_REGISTER_r0.1.tex`, SHA-256
`740478258f16ef67095798d53a02f7679a20d7a8a8be0c6e6644c4475bc5615d`.
The verifier checks exact decision IDs, statuses, prompts, and conservative
dispositions before accepting either rendered view.
