# SCI-JINC v0.1 — Scientific-Owner ODQ-101 Decision

Date: `2026-08-28`

Scientific owner: Grant Wilson

Status: approved Stage A successor input; numerical JINC route remains typed
unavailable until an exact registered family is selected and realized

## Decision

`SCI-JINC-ODQ-101` is resolved for the PTC/JINC coefficient ownership,
selection, permission and failure architecture:

1. PTC owns one versioned registry of positive analysis/gridding coefficient
   families.
2. Each exact family/version declares its permitted named consumers:
   `SCI-MAP`, `SCI-JINC`, or both.
3. When a family permits `SCI-JINC`, JINC consumes the same positive
   PTC-produced coefficient `omega_i` and its separately typed availability/
   QC, identity, normalization, support, provenance and covariance meaning.
4. SCI-JINC alone applies its signed spatial coefficient `kappa_ip` and owns
   `w_ip=kappa_ip omega_i`, signed normalization, conditioning, support,
   response, covariance and JINC product semantics.
5. Permission for MAP does not imply permission for JINC, or conversely. No
   ordinary MAP projection, normalization, support, exposure, coadd, response,
   covariance or validity rule is inherited by analogy.
6. The user selects from the exact allowed family list. An explicit versioned
   mode policy may supply a default. Requested, effective, observation-
   resolved and realized family identities remain distinct.
7. Missing selection with no authorized default, an unregistered family,
   missing named-consumer permission, or an unavailable/mismatched payload
   makes the affected numerical route unavailable. No hidden fallback is
   allowed.

This decision establishes the architecture and failure semantics. It does not
register a family, select one for a request, prove inverse-variance meaning, or
make a numerical JINC route available.

## Controlled Predecessor Source

The canonical post-freeze predecessor record is:

- branch: `codex/scientific-contract-library`;
- commit: `54475956f6aefb839d43b2f0fb019a142cb64310`;
- object:
  `doc/scientific_contracts/packages/SCI-MAP/v0.1/POST_FREEZE_SCIENTIFIC_OWNER_DECISIONS_2026-08-28.md`;
- SHA-256:
  `4d2b857b7ec9efe489fe065f464df4ecd23b57a4c1320cda6a10a56592825d1c`.

Its `PTC-OD-010` section establishes the versioned registry, explicit user or
mode-policy selection, distinct lifecycle identities, producer-owned family
definition and payload, required family metadata, consumer-side numerical
classification and fail-closed no-fallback route. The SCI-JINC owner decision
above extends the predecessor's MAP-facing wording to explicit named-consumer
permissions in one PTC-owned registry.

The source's MAP-local decisions remain outside JINC scientific authorship.
Frozen SCI-MAP v0.1/r0.7.1 and SCI-PTC v0.1/r0.5 remain unchanged; this is a
controlled successor input, not a retroactive edit.

## Stage A Consequence

The approved Stage A baseline at
`6639bff3d94b92ace8faf3e407ccaefd5a38ea1f` remains the predecessor. This
bounded delta updates the JINC Scope Brief, `SCI-PTC_TO_SCI-JINC` boundary,
owner ledger, necessary author-facing controls and packet hashes only.

Because allowed author-input bytes change, the prior exact-byte approval does
not approve the successor packet. Stage B remains blocked until the remaining
owner questions are dispositioned and the scientific owner approves the exact
successor manifest bytes. No normative Stage B content is authorized here.
