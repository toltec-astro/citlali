# Citlali Canonical Baseline APT v1

> Historical contract. APT-PROD-003 compact v2 supersedes all new issuance
> and ordinary admission. V1 is accepted only by an explicitly selected
> migration or comparison operation and must never be called a fresh Beammap
> baseline. See [`CANONICAL_APT_V2.md`](CANONICAL_APT_V2.md).

## Status And Authority

This document is the normative human-readable contract for the candidate
Citlali-produced canonical Beammap baseline APT. The artifact schema is
`citlali-canonical-apt-v1`, the profile is
`citlali-beammap-baseline-apt-v1`, and the output role is
`beammap-baseline-apt`. [ADR 0010](adr/0010-canonical-baseline-apt-v1.md)
records the durable decision.

The producer implementation exists only on the bounded APT-PROD-001 candidate
branch. This document does not activate the artifact as a current production
input, add it to an active validation profile, migrate a historical APT, or
authorize any downstream consumer. Final gates, a coherent candidate commit,
owner-controlled integration, and a separate downstream admission decision
remain required.

The executable authorities are the C++ model and codec in
`include/citlali/core/pipeline/canonical_apt_v1.h` and
`include/citlali/core/pipeline/canonical_apt_ecsv.h`, the bounded Beammap
producer adapter, and the unactivated product contract in
`validation/product_contracts.json`. A disagreement is a defect; it is not
permission to infer a missing identity or scientific value.

In that executable record, `contract_schema_version` is
`citlali-canonical-apt-product-contract-v1` and describes the contract-object
shape. Its separate `schema_version` is `citlali-canonical-apt-v1` and pins the
schema embedded in ECSV metadata. The artifact-contract ID is
`apt-prod-001-canonical-baseline-apt-v1` and its `activation_state` is
`unactivated`.

## Scope And Non-Goals

Canonical APT v1 describes one Citlali Beammap output occurrence. It carries:

- an exact artifact-local detector-row key;
- a complete row-to-raw-channel relation and raw input manifest;
- the current Beammap baseline quantities with explicit type, unit,
  nullability, non-finite, and authority declarations;
- an opaque occurrence envelope distinct from scientific content identity;
- order-independent semantic and envelope SHA-256 identities;
- an exact byte-transport SHA-256; and
- an adjacent, envelope-bound completion receipt published last.

It deliberately does not define a persistent measured-detector namespace,
reconstruct tune identity, promote unresolved design or polarization fields to
identity, change detector selection/order or scientific values, establish CAL
physical-science closure, ingest historical APTs, or alter matcher, calibration,
fit, map, RTC, PTC, or detector-selection policy.

## Identity Model

### Artifact-Local UID

`uid` is an exact signed `int64` value constrained to
`0 <= uid <= 9007199254740991` (`2^53 - 1`). Values are unique within one APT
artifact, may be sparse, and need not be dense or ordered. The upper bound is
an explicit v1 compatibility constraint for exact interchange with consumers
whose integer-exact range is limited to the binary64 safe-integer domain.

`uid` is only an artifact-local row key. It is never a persistent detector
identity, does not establish equality across APT occurrences, and must not be
used alone for cross-artifact detector joins. A persistent measured-detector
identity is omitted until an authoritative namespace and lifecycle are
independently proven.

### Raw-Channel Relation

The durable relation inside this artifact is:

```text
uid -> (nw, kids_tone)
```

`nw` is the raw network and `kids_tone` is the zero-based channel within that
network. The relation is part of semantic content, not a private sidecar.
Repeated channel numbers are valid across different networks; a repeated
`(nw, kids_tone)` pair is not. The raw manifest and rows form a complete
bijection, so every declared channel appears exactly once and every row refers
to a declared channel.

### Occurrence And Event

`occurrence` and `event_reference` are nonempty, single-line, valid UTF-8
issuer-supplied opaque references. They are not UUID requirements and are not
parsed for scientific meaning. They must not be derived from content, path,
wall-clock time, UID, or any claimed detector identity. The default issuer uses
operating-system entropy and the prefixes `apt-occurrence:entropy/` and
`apt-event:entropy/`; tests and controlled callers may inject other opaque
values.

The same semantic content may be published in distinct occurrences. Changing
only occurrence or event reference leaves the semantic digest unchanged and
changes the envelope digest.

The envelope also requires exact `output_role: beammap-baseline-apt`, exact
`producer: citlali`, nonempty single-line `software_revision` and
`configuration_reference`, and a valid-calendar UTC `event_time_utc` ending in
`Z`.

## Structural Columns

The five structural columns appear first and are protected names. They are all
nonnullable.

| Name | Exact type | Unit | Authority | Contract role |
| --- | --- | --- | --- | --- |
| `uid` | `int64` | `N/A` | canonical issuer | Artifact-local row key; unique, sparse allowed, never persistent identity |
| `tone_freq` | `float64` | `Hz` | raw readout | Finite ToneFreq copied bit-for-bit; nonidentity attribute |
| `array` | `int64` | `N/A` | network map | TolTEC array enum derived from `nw`; nonidentity attribute |
| `nw` | `int64` | `N/A` | raw manifest | Network component of the raw-channel relation |
| `kids_tone` | `int64` | `N/A` | raw manifest | Zero-based channel component of the raw-channel relation |

The v1 network-to-array mapping is exact: networks 0--6 map to array 0
(`a1100`), 7--10 to array 1 (`a1400`), and 11--12 to array 2 (`a2000`).

## Observation, Scientific Context, And Raw Manifest

The observation tuple consists of nonnegative exact `int64` values named
`observation`, `subobservation`, and `scan`. The producer obtains the complete
tuple from the current raw/telescope/output observation state and requires
agreement at that boundary. A KIDs fit report corroborates only network and
observation identity; it is not an authority for subobservation or scan.
Raw inventory and issuance are observation-owned and rebuilt for each output
occurrence; stale state from a preceding observation is not an allowed source.

The scientific context contains nonempty `project_id`, `source_name`, an exact
valid-calendar UTC `observation_time_utc` ending in `Z`, and the fixed
`coordinate_frame` value `altaz`. These values are semantic content.

The raw manifest contains at least one input. Each entry declares:

- a unique network in the TolTEC enum 0--12;
- the exact canonical interface `toltecN` for network `N`; and
- an exact `int64` `channel_count` in `1..9007199254740992`.

Exactly one manifest/KIDs input is admitted per network in v1. Duplicate or
split inputs for one network fail closed; no accepted producer fixture proves a
legitimate split-network case. Each network has channels
`0..channel_count-1`, the total channel count equals the row count, ToneFreq is
finite for every row, the total cannot exceed `9007199254740992`, and row
`array`, `nw`, `kids_tone`, and ToneFreq must not drift from the retained raw
inventory. Each raw KIDs item must agree on observation, network, canonical
interface, channel count, and ToneFreq-vector length. No tune identifier is
reconstructed.

## Registered Field Contract

The field registry version is
`citlali-canonical-apt-field-registry-v1`. Every registered declaration is
part of semantic content and includes exact name, datatype, unit, nullability,
authority, authority reference, non-finite policy, registry, description, and
`identity_role: nonidentity`.

`nonidentity` means a field does not identify a persistent detector/entity. It
does not mean the field is excluded from content identity: every declaration
and every value, including `fg`, `pg`, `ori`, and `loc`, contributes to the
semantic digest.

The standalone v1 artifact contract admits only the exact 27 required fields
and 20 optional extensions below. The general C++ caller-supplied strict-
extension seam is not an artifact self-declaration mechanism and does not
authorize custom `bool`, `string`, or other columns in this contract. A new
field or registry requires a separately accepted successor artifact contract.

### Required Baseline Fields

All entries use registry `citlali-canonical-apt-baseline-fields-v1` and
`identity_role: nonidentity`.

Descriptions reproduce the registry strings exactly. The explicit suffix
`; value >= 0` on selected rows is an additional value-domain annotation, not
part of that registry description.

| Name | Type | Unit | Nullable | Authority and reference | Non-finite | Exact description / value domain |
| --- | --- | --- | --- | --- | --- | --- |
| `a_fwhm` | `float64` | `arcsec` | yes | producer; `citlali:beammap-fit-v1` | `nan-token` | fitted azimuthal FWHM |
| `a_fwhm_err` | `float64` | `arcsec` | yes | producer; `citlali:beammap-fit-v1` | `nan-token` | fitted azimuthal FWHM error |
| `amp` | `float64` | `xs` | yes | producer; `citlali:beammap-fit-v1` | `nan-token` | fitted amplitude |
| `amp_err` | `float64` | `xs` | yes | producer; `citlali:beammap-fit-v1` | `nan-token` | fitted amplitude error |
| `angle` | `float64` | `rad` | yes | producer; `citlali:beammap-fit-v1` | `nan-token` | fitted rotation angle |
| `angle_err` | `float64` | `rad` | yes | producer; `citlali:beammap-fit-v1` | `nan-token` | fitted angle uncertainty |
| `b_fwhm` | `float64` | `arcsec` | yes | producer; `citlali:beammap-fit-v1` | `nan-token` | fitted altitude FWHM |
| `b_fwhm_err` | `float64` | `arcsec` | yes | producer; `citlali:beammap-fit-v1` | `nan-token` | fitted altitude FWHM error |
| `converge_iter` | `int64` | `N/A` | no | producer; `citlali:beammap-fit-v1` | `reject` | beammap convergence iteration; value >= 0 |
| `derot_elev` | `float64` | `rad` | yes | producer; `citlali:beammap-geometry-v1` | `nan-token` | derotation elevation angle |
| `fg` | `int64` | `N/A` | yes | unavailable; `authority-unresolved-v1` | `reject` | frequency group; authority unresolved and nonidentity |
| `flag` | `int64` | `N/A` | no | producer; `citlali:beammap-quality-v1` | `reject` | bad detector flag; closed values `{0,1}` |
| `flxscale` | `float64` | `mJy/beam/xs` | yes | producer; `citlali:beammap-calibration-v1` | `nan-token` | flux conversion scale |
| `loc` | `int64` | `N/A` | yes | unavailable; `authority-unresolved-v1` | `reject` | location; authority unresolved and nonidentity |
| `ori` | `int64` | `N/A` | yes | unavailable; `authority-unresolved-v1` | `reject` | orientation; authority unresolved and nonidentity |
| `pg` | `int64` | `N/A` | yes | unavailable; `authority-unresolved-v1` | `reject` | polarization group; authority unresolved and nonidentity |
| `responsivity` | `float64` | `N/A` | yes | unavailable; `authority-unresolved-v1` | `nan-token` | responsivity; physical authority unresolved |
| `sens` | `float64` | `mJy/beam x s^0.5` | yes | producer; `citlali:beammap-calibration-v1` | `nan-token` | sensitivity |
| `sig2noise` | `float64` | `N/A` | yes | producer; `citlali:beammap-fit-v1` | `nan-token` | fitted signal to noise |
| `x_t` | `float64` | `arcsec` | yes | producer; `citlali:beammap-geometry-v1` | `nan-token` | fitted azimuthal offset |
| `x_t_derot` | `float64` | `arcsec` | yes | producer; `citlali:beammap-geometry-v1` | `nan-token` | derotated azimuthal offset |
| `x_t_err` | `float64` | `arcsec` | yes | producer; `citlali:beammap-fit-v1` | `nan-token` | fitted azimuthal offset error |
| `x_t_raw` | `float64` | `arcsec` | yes | producer; `citlali:beammap-geometry-v1` | `nan-token` | raw azimuthal offset |
| `y_t` | `float64` | `arcsec` | yes | producer; `citlali:beammap-geometry-v1` | `nan-token` | fitted altitude offset |
| `y_t_derot` | `float64` | `arcsec` | yes | producer; `citlali:beammap-geometry-v1` | `nan-token` | derotated altitude offset |
| `y_t_err` | `float64` | `arcsec` | yes | producer; `citlali:beammap-fit-v1` | `nan-token` | fitted altitude offset error |
| `y_t_raw` | `float64` | `arcsec` | yes | producer; `citlali:beammap-geometry-v1` | `nan-token` | raw altitude offset |

`fg`, `pg`, `ori`, and `loc` remain required so current producer values are not
silently removed. Their `unavailable` authority and nullability make the
unresolved truth explicit. A present numeric value is preserved as semantic
content, but it is not promoted into detector identity and must not be
reconstructed when unavailable. `responsivity` has the same unresolved
authority discipline for its physical meaning.

### Optional Extensions

All entries use registry `citlali-canonical-apt-extension-registry-v1` and
`identity_role: nonidentity`. If an optional column is registered for an
artifact, every row contains exactly one value; null is allowed only where the
declaration says so.

Descriptions reproduce the registry strings exactly. The explicit suffix
`; value >= 0` on selected rows is an additional value-domain annotation, not
part of that registry description.

| Name | Type | Unit | Nullable | Authority and reference | Non-finite | Exact description / value domain |
| --- | --- | --- | --- | --- | --- | --- |
| `cal_amp` | `float64` | `xs` | yes | producer; `citlali:beammap-empirical-calibration-v1` | `nan-token` | beammap calibration amplitude |
| `cal_amp_method` | `int64` | `N/A` | no | producer; `citlali:beammap-empirical-calibration-v1` | `reject` | calibration amplitude method code `{0,1}` |
| `cal_amp_over_fit_amp` | `float64` | `N/A` | yes | producer; `citlali:beammap-empirical-calibration-v1` | `nan-token` | calibration amplitude divided by fit amplitude |
| `final_prior_d2` | `float64` | `N/A` | yes | producer; `citlali:beammap-soft-prior-v1` | `nan-token` | nearest soft-prior Mahalanobis distance squared |
| `final_prior_slot_index` | `int64` | `N/A` | yes | producer; `citlali:beammap-soft-prior-v1` | `reject` | nearest soft-prior slot; nonidentity |
| `flag2` | `int64` | `N/A` | no | producer; `citlali:beammap-quality-v1` | `reject` | bitwise Beammap quality flag; allowed mask `0xff` |
| `kids_flag` | `int64` | `N/A` | no | copied-declared; `kids:fit-report-v1` | `reject` | imported KIDs model-fit flag; exact integral values, nonidentity |
| `map_peak_amp` | `float64` | `xs` | yes | producer; `citlali:beammap-empirical-calibration-v1` | `nan-token` | baseline-subtracted local map peak |
| `map_peak_amp_over_fit_amp` | `float64` | `N/A` | yes | producer; `citlali:beammap-empirical-calibration-v1` | `nan-token` | map peak divided by fit amplitude |
| `rfi_masked_samples` | `int64` | `samples` | no | producer; `citlali:beammap-mask-diagnostics-v1` | `reject` | number of samples masked by `rfi_mask`; value >= 0 |
| `rfi_masked_scans` | `int64` | `scans` | no | producer; `citlali:beammap-mask-diagnostics-v1` | `reject` | number of scans masked by `rfi_mask`; value >= 0 |
| `scan_band_mask_rejected` | `int64` | `N/A` | no | producer; `citlali:beammap-mask-diagnostics-v1` | `reject` | scan-band mask rejection code `{0,1}` |
| `scan_band_masked_edge` | `int64` | `N/A` | no | producer; `citlali:beammap-mask-diagnostics-v1` | `reject` | scan-band edge code `{0,1,2,3}` |
| `scan_band_masked_rows` | `int64` | `rows` | no | producer; `citlali:beammap-mask-diagnostics-v1` | `reject` | number of detector-map edge rows masked; value >= 0 |
| `scan_band_masked_samples` | `int64` | `samples` | no | producer; `citlali:beammap-mask-diagnostics-v1` | `reject` | number of samples masked by `scan_band_mask`; value >= 0 |
| `template_amp` | `float64` | `xs` | yes | producer; `citlali:beammap-empirical-calibration-v1` | `nan-token` | empirical template matched amplitude |
| `template_amp_over_fit_amp` | `float64` | `N/A` | yes | producer; `citlali:beammap-empirical-calibration-v1` | `nan-token` | template amplitude divided by fit amplitude |
| `template_npix` | `int64` | `pix` | no | producer; `citlali:beammap-empirical-calibration-v1` | `reject` | empirical template fitted pixel count; value >= 0 |
| `template_offset` | `float64` | `xs` | yes | producer; `citlali:beammap-empirical-calibration-v1` | `nan-token` | empirical template fitted offset |
| `template_resid_rms` | `float64` | `xs` | yes | producer; `citlali:beammap-empirical-calibration-v1` | `nan-token` | empirical template residual RMS |

`kids_flag` is optional at the artifact level and nonnullable once present. It
copies the exact signed `int64` KIDs fit-report value, including legitimate
nonbinary values. The KIDs source's legacy `flag` is renamed to `kids_flag` at
this boundary; a source column already named `kids_flag` or `flag2` is rejected
to prevent authority collision. It is distinct from baseline `flag` (`0` or
`1`) and Beammap `flag2` (mask `0x00..0xff`). Simulation has no KIDs fit report
and therefore omits `kids_flag`.

## Canonicalization And Digest Scopes

Artifact SHA-256 references use lowercase hexadecimal prefixed with `sha256:`.
Raw utility-hash vector outputs below are deliberately shown without that
reference prefix where stated. The canonical framing is
`citlali-labelled-type-length-v1`:

```text
F<label-byte-count>:<label>T<type-byte-count>:<type>V<payload-byte-count>:<payload>;
```

Lengths are exact byte counts. Text is valid UTF-8 and is not Unicode-
normalized. Canonical text rejects NUL, DEL, C0 controls other than tab, C1
controls, U+0085, U+2028, U+2029, U+FDD0..U+FDEF, and code points ending in
FFFE or FFFF; envelope, context, registry, and row strings are single-line.
Exact integers use ordinary base-10 text. Finite float values use
the 16-lowercase-hex-digit IEEE-754 binary64 bit pattern with type
`float64-ieee754`; all NaNs canonicalize to `nan`, and infinities to `+inf` or
`-inf`. A typed null uses type `null-<declared-type>` and payload `null`.

### Semantic SHA-256

Scope `citlali-canonical-apt-semantic-sha256-v1` covers the schema, profile,
field registry version, all five structural contracts, every exact registered
field declaration, observation tuple, scientific context, raw inputs, expanded
raw channels, and every row value. Registered fields are sorted by name, raw
inputs by `(network, interface)`, expanded channels by `(network, channel)`,
and rows by `uid`. Therefore row, manifest-entry, or declaration presentation
order does not change semantic identity.

### Envelope SHA-256

Scope `citlali-canonical-apt-envelope-sha256-v1` binds the semantic SHA-256 to
the occurrence, event reference, output role, producer, software revision,
configuration reference, and event UTC time. It is separate from content
identity so identical content may have distinct occurrences.

### Byte-Transport SHA-256

Scope `citlali-canonical-apt-byte-transport-sha256-v1` hashes the exact ECSV
bytes and is recorded with the envelope SHA-256 and exact unsigned byte count.
It is intentionally distinct from the order-independent semantic digest:
changing row presentation order can preserve semantic and envelope identity
while changing bytes and transport identity.

## Canonical ECSV Wire Contract

The artifact is ECSV 1.0 with delimiter `,` and schema `astropy-2.0`. It must
be nonempty, valid UTF-8, end in LF, contain LF line endings only, and contain
no CR byte. Canonical reserialization must reproduce the exact bytes.

Metadata and column-declaration text scalars are canonical YAML double-quoted
strings, metadata booleans are lowercase `true`/`false`, and metadata integers
are exact decimal. String escaping is deterministic; alternate YAML spellings
that parse to the same value are not canonical bytes.

Columns are the five structural columns followed by registered fields sorted
by name. Raw-manifest entries serialize by `(network, interface)`, while row
presentation order is retained in the byte stream. Physical units are emitted
in ECSV column declarations; `N/A` does not emit a physical-unit declaration.
Integer cells are exact decimal,
floating cells use locale-independent round-trip text (`nan`, `inf`, and
`-inf` where allowed), an unquoted empty cell is the only null spelling
(`unquoted-empty-v1`), and every nonempty string cell is quoted single-line
UTF-8 (`quoted-utf8-single-line-v1`). Empty non-null strings are forbidden.
The fixed v1 catalog permits canonical NaN only on nullable `nan-token`
`float64` fields and permits no infinity; core `tone_freq` is finite.

The metadata root `canonical_apt_v1` contains these exact members in canonical
order:

- `schema_version`, `profile`, `field_registry`, `framing_encoding`;
- `semantic_scope`, `semantic_sha256`, `envelope_scope`, `envelope_sha256`,
  and `byte_transport_scope`;
- `occurrence`, `event_reference`, `output_role`, `producer`,
  `software_revision`, `configuration_reference`, and `event_time_utc`;
- `scientific_context` containing `project_id`, `source_name`,
  `observation_time_utc`, and `coordinate_frame`;
- `observation` containing `observation`, `subobservation`, and `scan`;
- `raw_manifest` entries containing `network`, `interface`, and
  `channel_count`;
- `registered_fields`, each carrying every declaration property plus exact
  `identity_role: nonidentity`; and
- `null_cell: unquoted-empty-v1` and
  `string_cell: quoted-utf8-single-line-v1`.

Protected structural and metadata names cannot be registered as extensions.
The exact protected set is:

```text
uid, tone_freq, array, nw, kids_tone,
schema_version, profile, field_registry, framing_encoding,
semantic_scope, semantic_sha256, envelope_scope, envelope_sha256,
byte_transport_scope, occurrence, event_reference, output_role, producer,
software_revision, configuration_reference, event_time_utc,
observation, subobservation, scan, raw_manifest, network, channel,
interface, channel_count, scientific_context, project_id, source_name,
observation_time_utc, coordinate_frame
```

A duplicate declaration, protected-name collision, absent required field,
unknown optional field, or declaration that differs from the accepted catalog
in any property fails validation.

## Fixed Interoperability Vectors

These fixed values make independent implementations fail closed on framing,
integer, binary64, null, semantic, envelope, and byte-order differences:

| Vector | Exact value |
| --- | --- |
| SHA-256 of bytes `abc` | `sha256:ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad` |
| Maximum UID frame | `F3:uidT5:int64V16:9007199254740991;` |
| SHA-256 of maximum UID frame | `5e86d924a3acd47ae21e8fcb5c21bb40da9f37a9a16755dcdb6cf112c166250b` |
| SHA-256 of the fixed binary64 frame bundle | `4a566f76572a46c00bd06d035851e1cd80dbfbe640f90d1643bfc38197732ded` |
| Typed-null frame | `F7:missingT10:null-int64V4:null;` |
| SHA-256 of typed-null frame | `667dbdc49e83c7d94a4d5ea215328fbd76440a4a80e89cef2b984b7c19c0c872` |
| Complete fixture semantic SHA-256 | `sha256:a7911ac3b08ffdb9f3c6aaab36c33bb5abb47fac2bb729c2d09d79e68228f6db` |
| Complete fixture envelope SHA-256 | `sha256:cb1e83e3f1a236f51ae80d8ab3f4f79106f3dde20153699513d15c883673b67b` |
| Complete fixture byte SHA-256 | `sha256:4adc27eac0f9934b885b916a9d2b537ea70f37d6a14fe6e1db891c6c51dee9be` |
| Complete fixture byte count | `18759` |

The fixed binary64 bundle hashed above is the exact concatenation:

```text
F3:oneT15:float64-ieee754V16:3ff0000000000000;F13:negative_zeroT15:float64-ieee754V16:8000000000000000;F10:denorm_minT15:float64-ieee754V16:0000000000000001;F17:positive_infinityT15:float64-ieee754V4:+inf;F9:quiet_nanT15:float64-ieee754V3:nan;
```

The complete fixture is defined by the focused C++ test vector. A non-C++
validator must recompute these values independently rather than treating
embedded declarations as sufficient evidence.

## Publication Receipt And Completion

The adjacent receipt path is `<artifact>.ecsv.sha256`. Its exact five LF-
terminated lines are:

```text
citlali-canonical-apt-publication-receipt-v1
scope=citlali-canonical-apt-byte-transport-sha256-v1
envelope_sha256=sha256:<64 lowercase hex>
byte_sha256=sha256:<64 lowercase hex>
byte_count=<uint64>
```

The producer serializes to a private staged file, rereads and parses it,
recomputes semantic, envelope, and byte identities, stages and validates the
receipt, and refuses to replace an existing artifact or receipt. It publishes
the artifact first through a no-replace operation, rereads the final artifact
against the staged receipt, and publishes the receipt last. A visible valid
receipt is the pair's completion transition. Owned partial output and staging
state are cleaned up on failure; a pre-existing or raced destination is never
overwritten.

The receipt is a producer-owned, envelope-bound publication marker. It is not
a detector/content identity, private relation sidecar, or substitute for the
embedded schema, semantic digest, raw manifest, or row relation. A post-hoc
validator can validate a complete pair but cannot prove the historical order
in which its two directory entries became visible; writer behavior and focused
failure-injection tests establish receipt-last publication.

## Validation And Admission

A conforming validator fails closed unless it independently:

1. validates exact ECSV lexical form and canonical reserialization;
2. admits only the fixed v1 profile, schema, scopes, frame, registries, and
   built-in 27+20 field catalog;
3. validates exact types, units, nullability, authority declarations,
   non-finite policy, value domains, protected-name collisions, and row shape;
4. validates UID range/uniqueness and the complete raw-manifest bijection;
5. recomputes semantic, envelope, and byte-transport hashes;
6. validates the adjacent receipt and its envelope, byte hash, and byte count;
   and
7. rejects a missing receipt as an incomplete publication.

The unactivated standalone product contract validates an artifact already
named by its caller. It does not discover products from arbitrary paths, amend
an active production profile, or make the artifact a downstream input.

The explicit validation form is:

```bash
$HOME/tolteca/bin/python tools/baseline/validate_product_contract.py \
  /path/to/beammap_apt.ecsv \
  --artifact-contract apt-prod-001-canonical-baseline-apt-v1
```

A zero exit status establishes conformance of the named visible artifact/
receipt pair to the unactivated contract only; it is not production-profile or
downstream admission.

## Deferred Authority And Activation Limits

- Historical APTs and fixtures remain historical or test-only. They are not
  current canonical producer inputs and are not silently repaired.
- The candidate writer preserves current Beammap row order, detector set, and
  scientific values while adding typed validation and publication. Any drift
  is a stop condition, not an allowed schema migration.
- `kids_flag` is copied only when the authoritative KIDs fit report is present;
  simulation omits it.
- Persistent detector identity, tune identity, and split-network input support
  remain unavailable. Contrary authoritative evidence requires an owner stop
  and successor decision.
- `fg`, `pg`, `ori`, `loc`, and `responsivity` retain explicit unresolved
  authority; the producer neither invents nor promotes them.
- APT binding correctness is separate from CAL physical-science closure.
- No candidate production profile, downstream ingestion, historical
  migration, TolTECA/TolProj/TolAPT/toltec_beammap change, or CAL/ALIGN
  conclusion is authorized by this contract.
