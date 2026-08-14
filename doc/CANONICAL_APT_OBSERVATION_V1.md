# Canonical Observation APT v1

Status: accepted bounded candidate; unactivated

Durable decision: [ADR 0011](adr/0011-canonical-observation-apt-contract.md)

Executable authority:

- [`../validation/product_contracts.json`](../validation/product_contracts.json)
- [`../include/citlali/core/pipeline/canonical_apt_observation_v1.h`](../include/citlali/core/pipeline/canonical_apt_observation_v1.h)
- [`../include/citlali/core/pipeline/canonical_artifact_publication.h`](../include/citlali/core/pipeline/canonical_artifact_publication.h)
- [`../include/citlali/core/cli/canonical_apt_contract_protocol_v1.h`](../include/citlali/core/cli/canonical_apt_contract_protocol_v1.h)

This document describes the accepted APT-PROD-002 v1 contract. The executable
registry and codecs remain authoritative for exact member names, canonical
framing, validation, and failure behavior. Nothing in this document activates
the contract for a validation profile, accepted run, ingestion path, or
production use.

## Product Boundary

The original defect was correspondence by table position. A row ordinal in an
observation KMP table and a row ordinal in a Beammap APT are presentation
facts, not proof that the rows describe the same detector. V1 replaces that
assumption with explicit occurrence-scoped row references and a complete
validated relation.

The desired data flow is deliberately narrow:

```text
verified immutable canonical Beammap baseline APT
  + selected observation raw/KMP facts
  + complete occurrence-scoped match relation
  -> one observation-specific canonical APT-family ECSV
  + one adjacent completion receipt
```

Only the final matched observation APT is a published scientific artifact.
Its suffix is `.apt.ecsv`; its completion marker is the adjacent
`.apt.ecsv.sha256` receipt. The target manifest and match relation retain
complete typed schemas, semantic identities, envelope identities, and
validators, but they are embedded logical records and integrity-covered
provenance inside that ECSV. They have no independent v1 suffix, transport,
receipt, or publication transition.

The earlier proposal for independently published `.target.ecsv` and
`.relation.ecsv` artifacts, followed by a public bundle, is superseded. JSON
is only the strict machine request/response representation. It is never an APT
data format; canonical APT-family scientific products remain ECSV.

## Authority Split

Citlali owns:

- the baseline-descriptor, target, relation, output, encoding, digest,
  receipt, and validation contracts;
- reconstruction of the verified baseline descriptor from immutable baseline
  ECSV and receipt bytes;
- materialization of full row references from occurrence-scoped facts;
- the derived output field catalog and authorized transformations;
- the final output's opaque occurrence, event reference, software revision,
  semantic/envelope/transport identities, receipt, canonical reread, and
  no-replace publication.

TolProj is the legitimate issuer of the observation-specific values and
matching provenance supplied under that Citlali-owned contract. It supplies
the actual selected raw/KMP/network/channel facts, target and relation logical
occurrences and envelope context, match pairs and dispositions, matcher and
network evidence, per-field source selections, transformation provenance,
configuration reference, and event time. This does not transfer matcher
policy, schema authority, digest authority, final occurrence issuance, or
publication authority to TolProj.

A protocol caller cannot assert the baseline descriptor, fixed field schema,
derived output catalog, canonical digests, output-local UIDs, or final output
occurrence. When a caller pins an expected baseline, artifact, or transport,
Citlali compares that pin with identities independently recomputed from bytes.

## Identity Model

An artifact identity is the tuple of schema, opaque occurrence, semantic
SHA-256, and envelope SHA-256. A row reference adds the parent artifact's
envelope SHA-256 and its artifact-local key. Every endpoint used for matching
therefore names a verified occurrence and local row, not a bare ordinal.

The four key scopes remain separate:

1. The verified baseline occurrence and artifact-local `uid` identify a seed
   row only inside that immutable baseline occurrence.
2. The target logical occurrence and target-local `row_key` identify one
   selected observation row only inside that target occurrence.
3. The relation logical occurrence has one local-key namespace shared by pair,
   target-disposition, and seed-disposition records.
4. The final APT receives a new Citlali-issued occurrence and new
   output-artifact-local `uid` values.

No `uid`, `row_key`, `pair_key`, disposition key, source key, input key,
channel number, KMP row index, table order, content digest, path, or timestamp
is a persistent detector identifier. Equal integer spellings across artifacts
have no cross-artifact meaning without the full occurrence-scoped reference.
Matcher ordinals must be translated back to those references before the
protocol applies them.

`target_source_sequence`, `target_application_sequence`,
`seed_source_sequence`, and `output_presentation_sequence` are separate,
explicit, complete permutations of their respective local-key sets. They
record source, application, and presentation order. Their ordering must never
be used as detector identity or as an implied match.

## Embedded Target Manifest

The target manifest binds one observation tuple and one or more selected input
bindings. Each input has exactly one raw and one KMP source. Each source is
bound completely by its artifact-local source key, contract-assigned role,
network, interface, positive channel count, diagnostic locator, exact content
SHA-256, byte count, and source-header observation tuple. Source key, role,
SHA-256, byte count, header tuple, network, interface, and channel count enter
target semantic identity. The presentation-only `diagnostic_locator` is
deliberately excluded from that semantic digest and instead bound, with source
key and role, in the target envelope. Issuance treats the locator as an access
handle, rereads the regular file, and verifies its declared SHA-256 and count.

The raw source-header tuple must equal the target observation. The exact KMP
source-header tuple remains immutable fallback/source provenance and may
legitimately differ; neither tuple is used to reconstruct a tune identity.

Each selected target row carries a target-local key, input and KMP-source
references, exact KMP row index, raw network/channel relation, array, and the
closed set of registered KMP values. The target logical digest covers the
complete source bindings and typed rows. Changing an otherwise ignored source
diagnostic changes the bound source SHA-256 (and the byte count if its length
changes) without granting that diagnostic canonical field semantics.

### Restrictive KMP v1 Registry

Exactly these observation-specific KMP fields are registered:

| Canonical field | Source column | Type | Unit | Cardinality and authority |
| --- | --- | --- | --- | --- |
| `kids_fr` | `fr` | finite `float64` | `Hz` | required; copied-declared from `kids:model-params-v1` |
| `kids_f_out` | `f_out` | finite `float64` | `Hz` | required; copied-declared from `kids:model-params-v1` |
| `kids_Qr` | `Qr` | finite `float64` | `N/A` | required; copied-declared from `kids:model-params-v1` |
| `kids_flag` | `flag` | exact signed `int64` | `N/A` | optional at target-artifact level; copied-declared from `kids:fit-report-v1` |

The closed use-role registry authorizes `kids_fr` and `kids_Qr` as matching
inputs, `kids_f_out` as the target application value, and all declared fields
as typed output and authority claims. No KMP field is authorized as identity
or caller-supplied transformation evidence. In particular, presence of
`kids_flag` does not make it matching evidence.

An otherwise valid KMP source may contain other diagnostics, including other
`kids_*` columns. Their source bytes remain bound as provenance, but the
columns are not copied, typed, normalized, assigned units, included in output
fields or transformation evidence, or granted identity or authority. A
request that names an unregistered field for identity, matching, application,
transformation input/output, canonical output, or authoritative provenance
fails closed. V1 has no generic diagnostic bag or arbitrary-column payload.
Adding a field requires a separately accepted, field-specific successor
contract and fixed cross-language vectors.

## Embedded Complete Relation

The generalized relation binds the verified baseline parent and target parent
and records its own opaque TolProj-issued logical occurrence. It contains:

- matcher-run occurrence, implementation revision, configuration, method, and
  backend evidence;
- exactly one network-evidence record per target network, including the
  registered matching-frequency and quality-factor authorities;
- zero or more match pairs, each with one occurrence-scoped target endpoint,
  one occurrence-scoped baseline-seed endpoint, exact separation, and the
  supplied match-quality fact;
- exactly one disposition for every target, with state `matched` or
  `unmatched` and its complete pair-key set; and
- exactly one disposition for every baseline seed, with state `matched` or
  `unused` and its complete pair-key set.

Every pair must be named reciprocally by both endpoint dispositions. Matched
dispositions have a nonempty complete pair set; unmatched and unused
dispositions have none. They retain disposition keys and reasons but do not
fabricate a missing endpoint. One-to-zero, one-to-one, one-to-many, and
many-to-one cardinalities are representable and validated. This contract
records a relation selected by TolProj; it does not define or implement the
matching algorithm.

## Final Matched APT

There is exactly one output row per target row. Its fixed structural columns,
in order, are:

| Position | Column | ECSV datatype | Unit | V1 value |
| --- | --- | --- | --- | --- |
| 1 | `uid` | `int64` | `N/A` | nonnegative output-artifact-local row key |
| 2 | `target_row_key` | `int64` | `N/A` | exact target-parent local row reference |
| 3 | `target_input_key` | `int64` | `N/A` | exact target-parent input reference |
| 4 | `tone_freq` | `float64` | `Hz` | exact target `kids_f_out` application value |
| 5 | `array` | `int64` | `N/A` | target array value; nonidentity |
| 6 | `nw` | `int64` | `N/A` | target raw network key |
| 7 | `kids_tone` | `int64` | `N/A` | zero-based target raw channel key |
| 8 | `relation_pair_keys` | `string` | `N/A` | quoted canonical `bracketed-int64-set-v1`, including `[]` when unmatched |

All eight fixed columns are nonnullable. Their local integers and ordering do
not create persistent identity.

The exact derived registered-field columns follow in canonical field-name
order. The catalog contains every declared target KMP field and every verified
baseline registered field except names reserved by the target registry. The
four reserved names are `kids_fr`, `kids_f_out`, `kids_Qr`, and `kids_flag`.
Consequently, optional baseline `kids_flag` can never overwrite or supply the
target KMP flag.

Each target field uses `preserve-target` and keeps the exact typed value and
bound source-row provenance. Each admitted baseline field uses
`copy-baseline-when-matched-null-when-unmatched`: a matched output names the
exact selected relation pair and baseline seed row; an unmatched output uses a
typed canonical null with no fabricated pair or seed. A target with multiple
pairs may select different authorized source pairs for different baseline
fields. The complete pair set remains on the row regardless of per-field
selection.

Every output field on every row has explicit transformation evidence:
operation, typed before and after values, value source, optional relation pair,
optional source row, authority reference, and TolProj provenance reference.
Issuer-declared mutation is unauthorized for every v1 field. Structural/raw
facts cannot be mutated through transformation evidence. The output metadata
also binds the verified baseline reference, complete target and relation
parent identities, derived field contracts, final issuance envelope, and
output presentation sequence.

The verified baseline artifact remains byte-for-byte immutable, including its
own detector membership, row order, quantities, units, authorities, and
provenance. Final output membership is instead exactly target membership, and
its explicit presentation sequence follows the validated target application
sequence. For a matched target, each selected baseline-sourced field copies
the exact value, unit, field authority, and seed-row provenance from the named
baseline pair; for an unmatched target those derived baseline fields are typed
null. APT-PROD-002 changes correspondence and bounded observation-value
application; it does not alter the baseline, Beammap science or detector
selection/order, calibration, fitting, or matcher policy.

## Canonical ECSV And Identities

The persisted artifact uses exact ECSV 1.0 framing, canonical YAML metadata
order, comma-separated rows, valid UTF-8, and LF-only termination. It reuses
the canonical baseline APT's exact scalar, CSV quoting, and lexical machinery.
Semantic framing is `citlali-labelled-type-length-v1`; labels, types, and
payloads carry exact UTF-8 byte lengths. Signed and unsigned integers use
canonical decimal text in their declared range. Binary64 frames use exactly 16
lowercase IEEE-754 hexadecimal digits, preserve negative zero and denormals,
and map every NaN payload/sign to `7ff8000000000000`. Typed null uses
`null-<declared-type>` with payload `null`.

Metadata `float64` values use the quoted binary64 bit tokens. Table scalars use
the declared canonical ECSV lexical form: locale-independent round-trip float
text, exact decimal integers, lowercase metadata Booleans, an unquoted empty
cell as the only null spelling, and quoted nonempty single-line UTF-8 strings.
Allowed nullable `nan-token` baseline fields serialize NaN canonically;
infinity is not admitted by this output contract. Alternate lexical spellings
that parse to the same value are not canonical bytes.

The normative metadata contains the complete target and relation logical
records, all four identity scopes, parent references, field declarations,
dispositions, matcher/network evidence, sequences, transformations, and
lineage. Typed row columns carry the ordinary APT-family output values and
occurrence-scoped local references. Parsing recomputes target, relation, and
output semantic/envelope identities, reconstructs the typed document, and
requires byte-identical canonical reserialization.

The identity layers are deliberately distinct:

- target, relation, and final output each have a canonical semantic SHA-256;
- each semantic identity is bound to its own occurrence and provenance by an
  envelope SHA-256;
- the final ECSV has a byte-transport SHA-256/count bound to the final
  envelope; and
- the adjacent receipt has its own SHA-256/count when embedded in a verified
  baseline reference.

Content-equivalent output may be issued as a different occurrence. Table
presentation cannot substitute for semantic identity, and semantic equality
cannot substitute for byte-transport or publication-completion evidence.

## Machine Protocol

Invoke the versioned interface as:

```text
citlali --canonical-apt-contract-v1
```

The option must be the only argument. The process consumes exactly one
LF-terminated strict JSON object from standard input and emits exactly one JSON
response line. Duplicate or unknown members, extra input lines, CR/CRLF,
trailing input, wrong scalar types, noncanonical integer strings, invalid
binary64 bit tokens, and unsupported operations fail closed. JSON remains
protocol representation only.

Every request has exactly `protocol`, `request_id`, `operation`, and `payload`.
`protocol` is `citlali-canonical-apt-protocol-v1`. The operations are:

| Operation | Function |
| --- | --- |
| `describe-baseline-v1` | Reread a baseline `.ecsv` and receipt and return the complete verified typed descriptor plus immutable baseline reference. |
| `issue-observation-apt-v1` | Verify baseline and source bytes, validate and materialize target/relation facts and field-source selections, create the final occurrence, canonicalize/reread, then publish one `.apt.ecsv` and receipt. |
| `validate-observation-apt-v1` | Reread a final `.apt.ecsv` and receipt against the verified baseline, optionally compare expected identities, and return the complete target, relation, output, artifact, and transport descriptors. |

Exit status `0` is success, `1` is contract or publication rejection, and `2`
is protocol/framing or internal failure. Error responses are strict objects
with category, code, and message. The protocol is self-contained for its
public operations; callers do not import the repository-private Python
validator or copy Citlali canonicalization algorithms.

## Publication Completion

Issuance serializes the ECSV, creates its envelope-bound receipt, rereads and
validates both, and prepares the success response before publication. It then
uses same-directory staging and no-replace publication. The artifact becomes
visible before the receipt; the receipt is the last successful transition.
The publisher rereads/revalidates the visible destination and removes only its
own staged or incomplete entries on failure. An existing artifact or receipt
is never replaced. Receipt absence therefore means incomplete, never
successful, publication.

The receipt is a completion marker and transport binding. It does not replace
the embedded semantic records or prove historical directory-entry ordering to
a later validator.

## Fixed Contract And Transport Vectors

The following SHA-256 values pin the accepted executable registry objects:

| Contract object | SHA-256 |
| --- | --- |
| `apt-prod-001-canonical-baseline-apt-v1` | `eb343ced3d4c8f303095b53f3fdca087bb478bd53d675b12958b47df244173b9` |
| `apt-prod-002-observation-target-manifest-v1` | `139b76cf556384d34d1b1923694a008dc7b21f1f8022584ec49ff3f8bf2bb72c` |
| `apt-prod-002-match-dispositions-v1` | `acade470dbbb1ffd9327ada8db8a3df69e26ba02e7393864f1ad90de00d22785` |
| `apt-prod-002-observation-matched-apt-v1` | `3e51715484a17be7ebc8677fb51d3e2d54cd11602025c8bf6005c3e7f151d286` |

The fixed cross-language fixture pins these identities. They are test-vector
values, not the expected digest of every issued artifact:

| Fixture identity | Value |
| --- | --- |
| Canonical scalar-framing SHA-256 | `a97e7c29a17da562d44108968d120c393428577f9f218154bc5147e8f32029ec` |
| Target semantic / envelope | `8ad86d382b31eed82deab3118bbd5efe1fc5ce41389eac561ad2aef7e24cb30b` / `3dca742ac86f93666762e33557ab91b4d061be178b936d17d379077233bd6fc5` |
| Relation semantic / envelope | `7555c3f35ef57db23d32ef833d635c06cd06690ca1543cb635272328f29c93a4` / `25cd94197f41b3ec6132adfde525eeb1baae9b2dbeae7d47af38966817f5e8dc` |
| Output semantic / envelope | `cac3fabbb34907013b7558c5db855c3c861e370bb05ff0ff15051dd9f4e44dba` / `96fe37adc1b743dbcd7d907bb0f63b4859ff44102cc1be8556914f7978212dce` |
| Final ECSV transport | `a4016feb82b2d7b007ea6ae3dbbfbbf18022f25f467e50bb0fd324552bff6ded` over 125,302 bytes |
| Adjacent receipt | `fa48cca9fc8218712ac0be2e3e86bd9ed2dbd3877d2af436677c880c9a90e1e8` over 298 bytes |

C++ and Python tests must reproduce these vectors exactly. Changing a pinned
contract object, canonical framing, logical digest, final ECSV transport, or
receipt requires an explicitly reviewed successor decision rather than a
silent update.

## Accepted Limitations

These limitations are recorded, not repaired or waived by this contract:

- publication is no-replace and receipt-safe, but it is not an
  `fsync`/crash-durable transaction before the receipt transition; a crash may
  leave a receipt-absent incomplete artifact;
- standard-output delivery can fail after the receipt has been published,
  yielding a false-negative acknowledgement even though the product is
  complete; `validate-observation-apt-v1` is the authoritative recovery path;
  and
- the strict protocol has no project-owner-specified absolute standard-input
  size quota.

## Activation And Non-Goals

All three APT-PROD-002 registry entries are `unactivated`. They are absent from
validation profiles, accepted runs, ingestion, CAL, ALIGN, and production
routes. This contract does not activate or modify TolProj, TolAPT,
`toltec_beammap`, TolTECA, fitting, mapmaking, RTC/PTC, Beammap detector
selection, or any downstream consumer.

It also does not create a persistent detector namespace, reconstruct tune
identity, reinterpret historical APTs, admit historical artifacts as current
inputs, define matching policy, authorize arbitrary KMP diagnostics, or create
public target/relation artifacts for implementation convenience. Historical
APTs remain historical/test-only under their original contracts.

Production/profile admission, a new KMP field, a changed occurrence scope,
different serialization, a matcher-policy rule, a persistent detector ID, or
any change to transformation/publication semantics requires separate owner
authority and, where material, a successor contract and ADR.
