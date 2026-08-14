# ADR 0011: Canonical Observation-Specific APT v1

- Status: Accepted
- Date: 2026-08-14
- Decision owner: project owner
- Implementation state: bounded candidate; not integrated or activated
- Normative artifact contract: [`../CANONICAL_APT_OBSERVATION_V1.md`](../CANONICAL_APT_OBSERVATION_V1.md)

## Context

APT-PROD-001 established a producer-owned canonical Beammap baseline APT with
an embedded raw-channel relation and artifact-local UIDs. It deliberately did
not make a UID or table position a persistent detector identity.

The downstream observation application defect was therefore not missing
serialization media. It was fragile correspondence by row position: a matcher
ordinal or an equal integer in two tables could be mistaken for detector
identity. Citlali needed a machine-callable way to verify an immutable
baseline, accept legitimate observation and match facts, apply only authorized
observation-specific values, and persist an ordinary canonical APT-family
product without changing Beammap science or inventing a detector namespace.

Phase-A logical work demonstrated complete typed target and generalized
relation contracts. An initial Phase-B design considered separately published
target and relation ECSV files and a bundle. Reassessment against the original
defect found those public intermediates unnecessary: their complete content is
required for integrity and reproducibility, but only the resulting
observation-specific APT is a scientific product that v1 needs to publish.

## Decision

Citlali defines the unactivated successor contracts
`apt-prod-002-observation-target-manifest-v1`,
`apt-prod-002-match-dispositions-v1`, and
`apt-prod-002-observation-matched-apt-v1` with these rules:

1. Persist exactly one canonical observation APT-family `.apt.ecsv` and its
   adjacent `.apt.ecsv.sha256` completion receipt.
2. Keep target and relation as complete canonical typed logical records with
   their own semantic/envelope identities, embedded in the final ECSV's
   normative metadata. They are not separately published v1 artifacts.
3. Bind every relation endpoint by parent schema, opaque occurrence, envelope
   digest, and artifact-local key. Bare row position or equal local-key
   spelling never proves correspondence.
4. Issue a new opaque Citlali output occurrence and new output-artifact-local
   UIDs. No persistent detector identifier is created.
5. Require complete target and seed dispositions. Every target is matched or
   unmatched, every baseline seed is matched or unused, and pair sets are
   reciprocal. Unmatched/unused records have no fabricated endpoint.
6. Represent one-to-zero, one-to-one, one-to-many, and many-to-one pair sets
   without defining matcher policy. Downstream matcher ordinals must be
   translated to occurrence-scoped references before application.
7. Keep source, target-application, seed-source, and output-presentation
   sequences explicit and complete, but never treat their ordering as detector
   identity or an implied relation.
8. Register only `kids_fr`, `kids_f_out`, `kids_Qr`, and optional target-level
   `kids_flag` as observation-specific KMP facts. Unknown KMP diagnostics may
   remain in the fully hash/count-bound source but acquire no canonical
   meaning and fail closed when requested for a semantic role.
9. Preserve target fields exactly. Copy verified baseline fields only through
   a selected relation pair, using a typed null when the target is unmatched.
   Record typed before/after values, value source, pair/row source, authority,
   and provenance for every output-field transformation. V1 authorizes no
   issuer-declared mutation.
10. Keep the verified baseline bytes, membership/order, fields, units,
    authority, provenance, and science values immutable. Final output
    membership is exactly target membership: matched baseline-sourced fields
    copy from their named seed and unmatched baseline-sourced fields are typed
    null. Baseline `kids_flag` is excluded from the derived catalog so it
    cannot collide with or supply the target KMP flag.
11. Keep canonical APT-family scientific products in ECSV. Use strict JSON
    only for the versioned `describe-baseline-v1`,
    `issue-observation-apt-v1`, and `validate-observation-apt-v1` machine
    protocol.
12. Make Citlali authoritative for schemas, canonical encoding, validation,
    final occurrence/event issuance, digests, receipt creation, reread, and
    no-replace receipt-last publication. TolProj remains authoritative for
    legitimate observation-specific values, relation selection, matcher and
    network evidence, and transformation provenance; this does not transfer
    matcher policy.
13. Require byte-identical canonical reread/reserialization before successful
    publication. Publish the artifact first and receipt last; never replace an
    existing artifact or receipt, and never report a receipt-absent artifact as
    complete.

## Consequences

- Stable correspondence depends on occurrence-scoped row references rather
  than presentation order or a fabricated cross-product detector ID.
- The final ECSV remains an ordinary APT-family table while carrying all
  target/relation integrity and lineage needed to reproduce and audit its
  application.
- Logical target and relation identities can be validated independently even
  though they have no separate transport or publication transition.
- The machine request supplies values and provenance, not Citlali's private
  canonicalization algorithm, fixed schemas, digests, output catalog, output
  local keys, or final occurrence.
- A caller can recover authoritative state with
  `validate-observation-apt-v1` after an ambiguous process acknowledgement.
- The registry remains unactivated. Conformance is not validation-profile
  admission, accepted-run evidence, downstream ingestion, or production
  authorization.

## Rejected Alternatives

The following are not part of v1:

- publishing independent target and relation ECSV files plus a public bundle;
- encoding any APT scientific artifact as JSON;
- creating a generic arbitrary-column framework or opaque KMP semantic bag;
- granting unknown `kids_*` diagnostics meaning because they are present;
- inventing a persistent detector identifier from UID, channel, table order,
  source path, content, or time;
- copying private Citlali digest/serializer algorithms into TolProj or asking
  TolProj to import an unpackaged validator script; or
- changing matcher, fitting, mapmaking, calibration, detector selection, or
  Beammap numerical policy to solve a correspondence defect.

These ideas may be separately reviewed future architecture, but they are not
required to close the original defect and gain no authority from this ADR.

## Accepted Limitations

The no-replace receipt-last publisher is not an `fsync`/crash-durable
transaction before the receipt becomes visible. A crash can leave an
incomplete artifact without its completion receipt.

The final receipt can be published successfully before writing the JSON
success response to standard output fails. That is a possible false-negative
acknowledgement; the validate operation is the authoritative recovery route.

The v1 strict JSON-line protocol has no project-owner-specified absolute
standard-input size quota.

## Non-Goals And Activation

This decision does not activate any registry entry, validation profile,
accepted run, ingestion path, calibration route, or production consumer. It
does not modify TolProj, TolAPT, `toltec_beammap`, TolTECA, CAL, ALIGN,
fitting/mapmaking, RTC/PTC, or historical APTs. Target and relation occurrences
are logical provenance under the Citlali contract, not newly published
scientific products.

Activation, downstream application, adding a KMP field, changing the
transformation registry, changing ECSV framing, changing occurrence scope,
adding matcher policy, or introducing a persistent detector namespace all
require separate owner authority. A material contract change requires a
successor artifact-contract version and ADR; historical artifacts retain their
original meaning.
