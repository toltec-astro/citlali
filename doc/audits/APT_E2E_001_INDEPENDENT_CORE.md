# APT-E2E-001 Independent Architectural Core

Package: `APT-E2E-001`
Title: End-to-End APT Scientific, Identity, and Provenance Contract Audit
Status: frozen pre-candidate architectural core
Derivation cutoff: 2026-08-13T19:05:24Z
Git base: `46ad23888a40f5102cdfd50c06e49a549bdf8a20`
Audit branch: `codex/apt-e2e-001-audit-20260813`

## 1. Status, ceiling, and independence

This document is the independent core required before any inspection of the
stopped TolProj identity proposal. It is not the final APT audit, an accepted
schema implementation, or permission to resume APT, ALIGN, matching, reduction,
or production work.

The core is derived only from:

1. the component roles in the owner directive;
2. the current Citlali Beammap APT producer, its directly invoked data model and
   serializers, and the corresponding Citlali consumer boundary;
3. current repository-owned documentation and contracts at the exact revisions
   in section 15; and
4. logical proof obligations imposed by the documented cross-repository
   operations.

The artifact ceiling for this phase is exactly this Markdown file. No source,
runtime, schema, fixture, test, reduction, or second audit document is part of
the phase.

### 1.1 Quarantine

The following remained quarantined throughout derivation:

- the stopped TolProj worktree and its uncommitted identity/sidecar proposal;
- the prior Citlali APT-audit worktree, branch, ref, and task artifacts;
- ALIGN B2 and all ALIGN artifacts; and
- Unity, production data, and network services.

No quarantined path, proposal, diff, blob, digest, test, or artifact was opened,
read, copied, compared, or hashed. The candidate has therefore contributed no
concept, field, name, layout, conclusion, or wording to this core. Its first
inspection event has not occurred.

After this file is hashed and committed, work stops for coordinator approval.
Only then may the candidate be opened read-only and evaluated against this
already committed core. Candidate agreement cannot retroactively make the
candidate authoritative; disagreement must be recorded rather than silently
changing this baseline.

### 1.2 Evidence labels

This document uses four epistemic labels:

- **Observed**: directly established by an identified current source or
  executable contract.
- **Documented**: stated by a current repository-owned document but not yet
  verified through the owning implementation.
- **Derived requirement**: logically necessary for the stated operations and
  counterexamples; it is not a claim that current code satisfies it.
- **Open**: deliberately unresolved until the post-core source audit.

## 2. Governing invariant and authority

The invariant frozen by this core is:

> There is one canonical APT contract. Producers emit it, transformers preserve
> it and record their transformations, matchers represent realized mappings
> explicitly, and consumers validate it before scientific use.

The current authority dependency is producer-first:

1. Citlali owns the Beammap producer, the current in-process APT representation,
   the final APT writer, and the science ingestion behavior.
2. No separate shared APT schema or data-model repository is invoked by the
   inspected producer path. The concrete current definition is distributed
   among `Calib`, the Beammap writer/extensions, ECSV utilities, and the product
   registry rather than expressed as one closed versioned schema.
3. TolProj owns cohort/seed selection, tone matching/transformation, calibrated
   observation-specific copies, and project handoff. It does not independently
   define canonical APT identity.
4. TolAPT owns its independent measured-to-design matching domain, immutable
   run products, and matched/hero producer contracts. A design assignment is not
   a measured-detector identity.
5. `toltec_beammap` owns downstream Beammap analysis, APT diagnostics/updates,
   and calibration behavior while inheriting Citlali's APT structure.
6. TolTECA owns observation input selection and numbered-config merge semantics;
   configuration precedence is not APT row identity or lineage.

Current documentation overloads the phrase "matched APT": TolProj says it
creates observation-matched APTs, while TolAPT claims matched-APT construction
authority. The core does not resolve this by choosing one phrase. Every mapping
must instead name its domain, policy owner, source and output artifact roles,
row-key namespaces, cardinality, and provenance.

## 3. Current Citlali production and ingestion paths

### 3.1 Baseline Beammap production

For detector or automatic map grouping, current Citlali constructs an internal
APT from the raw KIDs inventory instead of loading an external APT. The observed
path is:

```text
RawObs data items
  -> numeric interface ordering
  -> one detector-count/network/array segment per KIDs file
  -> one concatenated Calib table
  -> uid = 0..n_dets-1 in that concatenated order
  -> first-sweep tone frequencies assigned by network segment and row
  -> KIDs fit-report columns overlaid by row
  -> Beammap fits, calibration, flags, offsets, and diagnostics written by row
  -> one all-double ECSV matrix plus table metadata
```

Observed details:

- `RawObs` sorts TolTEC data items by numeric interface and exposes that order
  through `kidsdata()`.
- The detector inventory reads only detector count, interface-derived network,
  and the network-to-array map. It concatenates these segments.
- The internal producer initializes the base table, assigns `uid` with
  `Eigen::VectorXd::LinSpaced(n_dets, 0, n_dets - 1)`, runs `Calib::setup()`,
  and labels the source filepath only as "internally generated for beammap."
- Tone frequencies are loaded from each raw network, the first sweep is chosen,
  and values are copied into the corresponding contiguous detector segment.
- `kids_tone` is the row offset within a network; it resets when the next
  network is encountered.
- KIDs fit-report columns are copied by row. A fit-report column named `flag`
  becomes `kids_flag`; other same-named columns can replace current APT values,
  and new names extend the output header.
- Beammap fitting overwrites amplitude, position, FWHM, angle, and uncertainty
  columns. It derives raw/reference-adjusted/derotated positions, sensitivities,
  convergence, flags, calibration-amplitude diagnostics, mask diagnostics, and
  prior diagnostics.
- The writer materializes all current header columns in a `MatrixXd` and writes
  the matrix atomically as ECSV.

The construction is deterministic for a fixed ordered raw input set, but its
detector key and readout application are positional. Reordering input files or
channels can relabel the same physical channels. Reordering an external APT
inside a network without identically reordering the raw detector columns can
silently apply the wrong row to a channel.

### 3.2 Science and external-APT ingestion

Current Citlali receives an APT calibration item as a filepath. It requires one
array-properties-table item but receives no declared APT digest or target
binding with it.

Observed validation and transformation are:

1. Read ECSV numeric columns into `std::map<std::string, Eigen::VectorXd>`.
2. Cast integer, `int16`, `int64`, Boolean, and float columns to double.
3. Require the current base header names to be present and nonempty.
4. Require table metadata `Radesys == altaz`.
5. Keep rows whose `nw` occurs in the raw observation interfaces, preserving
   source row order; copy only the current base header columns.
6. Require matching column lengths; require `nw` and `array` values to be
   finite/integral and each group to occupy one contiguous row range.
7. Require only total raw detector count to equal the retained APT row count.
8. Overwrite `tone_freq` from raw data by network-segment position.

The loader reads `Header.Toltec.RoachIndex` values but does not use them in the
binding. It does not validate `uid` as finite, integral, nonnegative, or unique.
It does not validate a schema version, content digest, source artifact,
observation, tune, raw-file manifest, network/channel mapping, transformation
lineage, or software version. A wrong or forged APT with compatible columns,
frame, group contiguity, and count can therefore pass this boundary. This is a
current fail-open identity/binding condition even though several structural
errors correctly fail closed.

### 3.3 Persisted downstream identity

Citlali writes detector-resolved products with APT columns and UID-derived
fields, but the linkage remains weaker than the documentation implies:

- FITS and TOD metadata carry an APT basename, not a content or occurrence
  identity.
- TOD APT variables are NetCDF doubles.
- PTC diagnostics round APT UID values to NetCDF integers.
- several downstream paths round a finite UID; some fall back to detector row
  when UID is absent/nonfinite.
- UID lookup in the learning path returns the first row whose rounded value
  matches, so duplicate UIDs are ambiguous.

These observations do not prove that duplicate UIDs occur in accepted data.
They prove that current validation and lookup do not exclude or disambiguate
them.

## 4. Current field surface

All fields below are stored in `Calib` as binary64 `Eigen::VectorXd` values and
are assembled into a binary64 output matrix. "Integer semantic" means the value
is intended to be integral despite that storage type. This table describes the
31-field base header, not a claim that each field is a valid identity.

| Field | Unit | Current production/source meaning | Mutability and identity disposition |
| --- | --- | --- | --- |
| `uid` | N/A | Generated Beammap row ordinal or imported upstream value; described only as "unique id" | Copied through Beammap; current legacy join token, not proven artifact-independent or persistent identity |
| `tone_freq` | Hz | First raw tone-frequency sweep assigned by network and row | Observation/readout attribute; overwritten from raw data; floating value is not identity |
| `array` | N/A | Array ID associated with network | Integer semantic; structural grouping, not dense array index or row identity |
| `nw` | N/A | Readout network ID | Integer semantic; necessary readout scope, insufficient alone for detector identity |
| `fg` | N/A | Frequency-group value | Placeholder/default in internal production; upstream authority and trust vary; not identity by itself |
| `pg` | N/A | Polarization-group value | Placeholder/default in internal production; not measured matching evidence in TolAPT docs |
| `ori` | N/A | Design/orientation attribute | Placeholder or KIDs/upstream value; design attribute, not measured identity |
| `loc` | N/A | Design/location attribute | Placeholder or KIDs/upstream value; design-space attribute, not measured identity |
| `responsivity` | N/A | Detector response quantity | Calibration value; mutable and nonidentity |
| `flxscale` | mJy/beam/xs | Detector flux conversion | Calibration value transformed later; must change artifact integrity, not detector identity |
| `sens` | mJy/beam s^0.5 | Beammap sensitivity | Fitted/derived quantity; mutable and nonidentity |
| `derot_elev` | rad | Elevation used for detector-position derotation | Observation/fitting provenance; mutable and nonidentity |
| `amp` | xs | Fitted Beammap amplitude | Overwritten by fit; nonidentity |
| `amp_err` | xs | Fitted amplitude uncertainty | Overwritten by fit; nonidentity |
| `x_t` | arcsec | Current fitted detector azimuthal offset | Overwritten and possibly reference-adjusted/derotated; nonidentity |
| `x_t_err` | arcsec | Fitted azimuthal-offset uncertainty | Overwritten by fit; nonidentity |
| `y_t` | arcsec | Current fitted detector altitude offset | Overwritten and possibly reference-adjusted/derotated; nonidentity |
| `y_t_err` | arcsec | Fitted altitude-offset uncertainty | Overwritten by fit; nonidentity |
| `a_fwhm` | arcsec | Fitted azimuthal/major FWHM | Overwritten by fit; nonidentity |
| `a_fwhm_err` | arcsec | Fitted `a_fwhm` uncertainty | Overwritten by fit; nonidentity |
| `b_fwhm` | arcsec | Fitted altitude/minor FWHM | Overwritten by fit; nonidentity |
| `b_fwhm_err` | arcsec | Fitted `b_fwhm` uncertainty | Overwritten by fit; nonidentity |
| `angle` | rad | Fitted beam rotation angle | Overwritten by fit; nonidentity |
| `angle_err` | rad | Fitted angle uncertainty | Overwritten by fit; nonidentity |
| `converge_iter` | N/A | Beammap convergence iteration | Integer semantic; fit lifecycle/provenance, not identity |
| `flag` | N/A | Citlali detector-quality flag | Integer/Boolean semantic; validity state, not identity |
| `sig2noise` | N/A | Current fitted signal-to-noise diagnostic | Mutable diagnostic, not identity |
| `x_t_raw` | arcsec | Position before reference subtraction/derotation | Derived provenance value, not identity |
| `y_t_raw` | arcsec | Position before reference subtraction/derotation | Derived provenance value, not identity |
| `x_t_derot` | arcsec | Position after derotation transform | Derived value, not identity |
| `y_t_derot` | arcsec | Position after derotation transform | Derived value, not identity |

The current final header is open-ended rather than closed. Citlali appends:

- `kids_tone` (network-scoped row offset);
- runtime KIDs fit-report columns, with only `flag` specially renamed;
- `flag2` and Beammap mask, prior, and fit diagnostics;
- ten empirical-template calibration columns; and
- any other fit-report header not already present.

The executable Beammap product contract requires only a subset of these names
and checks no column data type, key uniqueness, artifact identity, or digest.
The final field set can therefore depend on runtime fit-report content. A
canonical successor must close and version the normative field surface,
extension rules, units, missing states, trust/authority, and migration rules.

### 4.1 Current artifact metadata

The Beammap writer currently records useful metadata such as observation
number, source, project, creation date, observation date/MJD, reference frame,
per-array flux/tau, column units/descriptions, phase strategy, reference
handling, mask parameters, and fitting/weighting policy.

It does not record a canonical APT schema/version, artifact occurrence
reference, canonical content digest, originating or seed APT identity/digest,
raw readout manifest, explicit row-to-channel mapping, transformation event,
software revision, or supersession. Existing metadata is therefore valuable
scientific context but not a complete identity/provenance contract.

## 5. Exact `uid` disposition before candidate inspection

The following is the independent source-derived result.

| Question | Current answer |
| --- | --- |
| Construction | Internally generated Beammap APTs use `0..n_dets-1` in concatenated raw KIDs interface/channel order. Imported APTs retain upstream values for surviving network rows. |
| Type | Every in-process APT column is `Eigen::VectorXd`; integer ECSV UIDs are cast to double. The final APT writer receives a `MatrixXd`. This is the precise meaning of "double-valued" established here. |
| Precision | Small nonnegative integers are exactly representable in binary64, but no UID range or integrality check exists. Casting arbitrary `int64` values above binary64's exact-integer range can lose information. |
| Namespace | No namespace is encoded or declared. Generated values are occurrence-local row ordinals and recur across artifacts/observations. |
| Uniqueness | The internal sequence is unique within its generated table. External UID finiteness, integrality, range, and uniqueness are not validated. No duplicate occurrence was inspected or asserted. |
| Lifecycle | Generated anew from each Beammap raw inventory. Current documentation explicitly does not establish a stronger lifetime than the upstream APT. |
| Current uses | Row-count anchor, cross-product join label, diagnostics, map contributions, learning state, and persisted detector metadata, generally after rounding/casting. |
| Failed proof obligation | It does not prove artifact identity, row-order invariance, target raw-channel binding, cross-observation persistence, or design identity. |

Therefore the current field is neither accepted nor discarded in advance. It
can continue to describe a bounded legacy join only where its artifact scope
and positional binding are known. The post-core audit must determine whether
any externally supplied UID has a stronger authoritative lifecycle and whether
an existing field can satisfy one or more roles in section 7. No repair may
silently reinterpret a row ordinal as persistent detector identity.

## 6. Documented cross-repository operations

These are documented behavior, not yet implementation-complete transition
findings.

| Boundary | Documented operation | Identity/provenance obligation derived from it |
| --- | --- | --- |
| Citlali Beammap -> TolProj | Curated Beammap APT enters a shared library and a selected seed can serve all science observations plus left/right pointings in a cohort | One source artifact can produce many target artifacts; every target occurrence and source-row disposition must be explicit |
| TolProj seed -> matched observation APT | Emits `apt_<obsnum>_matched.ecsv` for each target | Filename/obsnum cannot stand in for content integrity or row mapping; tone/readout binding must be scoped |
| TolProj matched -> calibrated APT | Leaves the matched APT unchanged and writes calibrated per-observation copies after combining pointing recovery and flux reference inputs | Many-input transformation provenance, changed-field declaration, source/output digests, and row correspondence are required |
| TolProj -> Citlali science | Observation configuration supplies a selected APT filepath | Citlali must independently verify supported schema, digest, and target readout binding before scientific use |
| Citlali Beammap -> TolAPT | TolAPT treats measured APT as immutable and writes enriched copies, explicit match relations, candidates, and unmatched sets | Measured row identity must remain distinct from design identity and from run-local indices |
| TolAPT measured -> design | Final assignment is one-to-one; candidates are one-to-many diagnostics; unmatched measured/design and inactive design rows are explicit | Realized mapping/status/cardinality and both artifact-scoped row references are normative |
| TolAPT soft prior -> Citlali | One row per `(array, network, slot)` is a broad source-initialization prior | A local soft slot is neither exact detector identity nor a canonical APT row |
| Citlali Beammap -> `toltec_beammap` | Downstream process includes APT diagnostics/update and calibration | Source/output artifacts, row preservation/cardinality, changed fields, and scientific provenance remain to be proven in implementation |

TolAPT's documented consistency join currently traverses
`measured.enriched[meas_idx].uid -> matched APT det_id -> matched APT uid`, and
invalid matched rows may contain placeholder UIDs. This proves that row index,
`det_id`, and `uid` have distinct operational roles. It does not prove the
construction, namespace, or lifetime of any of them.

## 7. Minimal canonical identity and provenance model

The terms below are semantic roles, not prescribed serialized field names or a
preselected field count. An existing field may satisfy a role only after its
construction, type, namespace, uniqueness, lifecycle, and validation are
proved. Several roles may be encoded compactly when doing so remains
unambiguous.

### 7.1 Artifact occurrence and integrity

Each issued APT needs an unambiguous **artifact occurrence reference** and an
algorithm/version/scope-labelled **content digest** over all normative content.
The occurrence reference can be an explicit value or a derivable pair such as
`(transformation event, output role)`; an additional opaque identifier is not
required if that pair is globally unambiguous.

Occurrence and digest are different facts. Two independently issued artifacts
can have identical canonical content/digests yet different origins,
transformations, target bindings, or supersession histories. Conversely, a
changed normative value must produce a changed content digest.

The final audit must select canonicalization and digest scope. The digest must
cover the schema-defined table and every normative metadata fact, whether those
facts are embedded or stored in an atomically bound companion. A component-
private, unbound TolProj sidecar cannot be the end-to-end authority.

### 7.2 Artifact-scoped row reference

Every row, including invalid, unmatched, unused, or superseded rows, needs a
key unique within its artifact. The complete row reference is the artifact
occurrence plus that row key. Row position, source order, presentation order,
floating-point values, fit results, or a filename must not substitute.

### 7.3 Observation/readout binding

An observation-specific APT needs a target-data reference sufficient to
distinguish the observation and the realized readout/tune/network/tone or
channel domain. Repeated tone numbers in different networks, tunes, or
observations must remain distinct. The binding can reference a normative raw
readout manifest instead of repeating every component in every row.

The exact smallest tuple is open until the raw-data and TolProj implementation
audit establishes which stable identities exist. Bare tone frequency, tone
ordinal, network ordinal, or observation number alone is insufficient.

### 7.4 Persistent measured detector or resonator identity

This role is optional until an authoritative lifecycle is demonstrated. It
must never be synthesized from row number, a floating value, a fit, a design
assignment, or presentation order. Without it, continuity across artifacts is
an explicit lineage/mapping assertion, not equality of a persistent identity.

### 7.5 Design-space reference

A design assignment references a design artifact and a row in that artifact.
It is separate from measured/readout identity. Reassigning a measurement to a
different design row changes and supersedes the mapping; it does not erase or
replace the measurement identity.

### 7.6 Transformation and realized mapping

Each transformation/matching event records:

- event identity, timestamp, operation/mapping domain, and owning component;
- verified source/seed/input and output artifact occurrences and digests;
- target observation/readout manifest where applicable;
- software revision/version and resolved configuration/policy identity;
- explicit permitted, copied, added, dropped, overwritten, and reordered
  fields;
- source and output row references, plus readout and design references when
  applicable;
- outcome/status, reason, ambiguity evidence required by the existing policy,
  and supersession; and
- complete cardinality.

A relation must support sets on both sides so `1->0`, `1->1`, `1->many`, and
`many->1` are representable. Unused seed rows, unmatched tones, duplicate or
invalid inputs, ambiguous candidates, and superseded assignments must be
explicit rather than disappearing or acquiring fabricated identity.

This contract records the realized result of TolProj or TolAPT policy. It does
not choose new pairs, thresholds, gates, tie rules, reassignment rules, or
matching algorithms.

### 7.7 Values, validity, and order

Calibration quantities, fitted parameters, flags, units, missing states, and
quality decisions are attributes, not identity. Changing them creates a new
artifact/digest and records a transformation while preserving or explicitly
mapping the relevant row/readout identity.

Source order, application order, and presentation order may be recorded as
separate sequences of row references where operationally necessary. They may
all differ. None is persistent detector identity. If current numerical code
requires a particular application order, that order is an explicit validated
binding/provenance fact, not a hidden identifier.

## 8. Required separations

| Concern | Frozen separation |
| --- | --- |
| 1. Canonical APT schema and semantics | One versioned field/type/unit/validity/extension contract at the Citlali producer boundary or a directly invoked shared authority |
| 2. Artifact and row identity | Artifact occurrence, content integrity, and artifact-scoped row key are distinct roles |
| 3. Observation/readout binding | Target raw data and readout mapping are independent of detector/design identity and verified against actual consumer inputs |
| 4. Cross-artifact matching lineage | Explicit mapping events join source, seed, output, readout, and design references with complete cardinality |
| 5. TolProj matching policy | Existing scientific policy remains unchanged; the contract records its realized selections and all unmatched/unused/ambiguous outcomes |
| 6. TolAPT design-space matching policy | Existing policy remains unchanged; design assignment cannot become measurement identity |
| 7. Calibration/fitted quantities | Mutable values and flags are versioned attributes with units/validity, never identity material |
| 8. Ordering/presentation provenance | Source, application, and presentation orders are named separately and never silently identify a detector |
| 9. Production admissibility/compatibility | Consumer verification is fail closed; legacy assurance is explicit and no unavailable identity is invented |

## 9. Boundary proof obligations

### 9.1 Canonical producer

Before publication, the producer must prove:

- supported schema and closed extension rules;
- typed units, missing states, and validity semantics;
- unique artifact-scoped row keys;
- artifact occurrence and recomputable integrity;
- explicit readout/observation source when the artifact is observational;
- no hidden row-order identity; and
- complete creation/software/configuration provenance.

### 9.2 Transformer or matcher

Before transformation, it validates the input contract and integrity. It emits
a new artifact rather than mutating the source, preserves identity only where
the relationship warrants it, records every mapping outcome and cardinality,
declares all field/order changes, and recomputes output integrity.

### 9.3 Observation-specific producer

It binds every realized target readout row to the exact observation/readout
manifest. A source/seed row that is not used is explicitly `unused`; an
observation tone with no source match is explicitly `unmatched`; ambiguity is
represented without using the selected pair as identity.

### 9.4 Scientific consumer

Before calibration, mapmaking, or scientific use, the consumer independently:

- recognizes a supported schema/compatibility profile;
- recomputes the declared digest under its algorithm/version/scope;
- verifies artifact and row-key integrity;
- derives the expected observation/readout reference from the actual input
  data and verifies the declared target binding;
- verifies mapping coverage, statuses, and referential integrity; and
- rejects stale, forged, mismatched, ambiguous-invalid, or unverifiable input
  according to the production-admissibility contract.

A path, basename, row count, network count, declared digest without
recomputation, or producer assertion is not sufficient proof.

## 10. Falsification matrix

| Case | Contract-preserving result |
| --- | --- |
| Arbitrary row reordering | Artifact-scoped row references and mappings retain meaning. A byte/order digest may change, but semantic verification and target binding cannot rely on position. |
| Identical content in two distinct APTs | Content digests may match; occurrence/provenance references remain distinct. No history or target binding is collapsed. |
| Duplicate or missing row identifiers | Strict publication/ingestion fails. Invalid rows still require non-placeholder row keys. |
| Repeated tone numbers in different networks/observations | Scoped readout references remain distinct; bare tone equality proves nothing. |
| Unmatched observation tones | Target readout rows are explicitly unmatched; no seed/persistent identity is fabricated. |
| Unused seed rows | Source rows are explicitly unused (`1->0`), not silently dropped. |
| Ambiguous/equal-quality matches | Existing policy's selected/unmatched result and ambiguity evidence are recorded; selection is not converted into identity. |
| One detector in several tunes/observations | Each readout occurrence remains distinct. Shared persistence is asserted only if an authoritative measured/resonator lifecycle exists. |
| Changed fitted/calibration values | New artifact occurrence/digest and transformation; proven detector/readout continuity is preserved or mapped explicitly. |
| Design-space reassignment | Measurement/readout identity stays fixed; a new design mapping supersedes the prior mapping without erasing it. |
| Stale or incorrect digest | Consumer recomputation fails before scientific use. |
| Forged/mismatched observation APT | Consumer-derived target readout identity disagrees and ingestion fails before calibration or science. |
| Old APT lacking new fields | Recognized only under an explicit legacy profile; unknown facts remain unknown and production eligibility reflects the reduced proof. |
| Round trip through TolProj, TolAPT, `toltec_beammap` | Every derived artifact has verified sources, permitted-change evidence, complete mappings, referential integrity, and recomputed integrity. |
| Source, application, and presentation order differ | Each may be recorded separately as row-reference sequences; selected mappings remain auditable and no ordinal becomes persistent identity. |

## 11. Backward-compatibility core

Historical APTs must be recognized by an explicit schema/compatibility profile.
Migration may normalize representations or add facts that are deterministically
derivable and recorded with provenance. It must not mint a persistent detector
identity, source mapping, observation binding, or artifact history that the old
artifact cannot prove.

At minimum, legacy disposition needs separate assurance states such as:

- structurally readable only;
- structurally valid with a bounded artifact-local row join;
- externally bound to a verified observation/readout manifest; or
- fully canonical under the successor contract.

The names and exact admissibility tiers remain a final-audit decision. An old
file may remain usable for a nonproduction or reduced-guarantee workflow while
being prohibited from fail-closed production science. Compatibility must never
be silent acceptance under the strongest profile.

## 12. Smallest producer-first repair dependency

This is a dependency statement, not authorization to implement:

1. Close and version the canonical producer/schema at the Citlali Beammap
   boundary, including row/readout/artifact semantics and integrity.
2. Make TolProj preserve the canonical contract and emit explicit realized
   observation mapping/transformation provenance without changing matching
   science.
3. Make Citlali science ingestion independently verify integrity and target
   binding and fail closed before scientific use.
4. Conform TolAPT's independent measured/design products and lineage.
5. Conform `toltec_beammap` update/calibration products and round trips.
6. Establish end-to-end evidence across all paths.
7. Resume ALIGN B2 only after the authoritative observation-specific APT
   identity contract and its minimum accepted implementation/evidence exist.

No downstream private sidecar can substitute for step 1. Normative metadata
may ultimately be embedded or atomically bound, but its authority, integrity,
and consumer verification must be end to end.

## 13. Explicit non-goals

This core does not:

- redesign TolProj tone matching or TolAPT measured-to-design matching;
- introduce or authorize Hungarian/global assignment, new tie policy, new
  thresholds, reassignment, or changed selected pairs;
- select a two-field identity replacement or any predetermined serialization;
- claim a persistent measured detector/resonator identity exists;
- equate a digest with an artifact occurrence;
- treat row order, tone frequency, coordinates, fitted/calibration values,
  flags, or presentation position as persistent identity;
- approve a TolProj-private sidecar as the canonical production contract;
- decide final legacy production admissibility; or
- authorize application/runtime/schema edits, tests, reductions, integration,
  ref movement, push, merge, acceptance, or production activation.

## 14. Open questions reserved for the post-core audit

The following must be answered by source tracing after checkpoint approval:

- exact TolProj UID/detector/tone construction, copy/add/drop/overwrite/reorder
  behavior, mapping cardinality/status, and observation binding;
- exact meaning and namespace of TolProj `det_id` and any placeholders;
- exact TolAPT design ID and measured UID construction/lifecycle and actual
  index dependence;
- exact `toltec_beammap` field mutations, row cardinality, and output lineage;
- whether any authoritative readout manifest or persistent resonator identity
  already exists outside the inspected producer path;
- exact historical APT populations, duplicate/missing UID evidence, and the
  strongest defensible compatibility guarantees;
- the minimum canonical byte representation and whether an order-insensitive
  payload-equivalence digest is operationally necessary in addition to the
  normative artifact digest;
- embedded versus atomically bound normative metadata placement; and
- the minimum accepted implementation/evidence needed to unblock ALIGN B2.

## 15. Exact independent source manifest

All Git blobs below were read from the named local revision without fetch or
network access. Blob IDs are Git object IDs. The owner directive itself is a
non-file source received through source thread
`019fb85b-e6b1-7021-a49b-dc7500b82e84` on 2026-08-13.

### 15.1 Citlali authority and mandatory repository context

Revision: `46ad23888a40f5102cdfd50c06e49a549bdf8a20`

| Path | Blob |
| --- | --- |
| `AGENTS.md` | `d8e7e5a8c94f81416abf8f7bab11118b5939db4c` |
| `doc/ARCHITECTURE.md` | `4b0d43a32f56f37939963eb5799b52abf814e14a` |
| `doc/SCIENTIFIC_CONVENTIONS.md` | `88612c1ef1007c3baeba0478117636f9037d1d46` |
| `doc/REFACTOR_STATUS.md` | `7cae44314d4ab38ce8f6d77b91e6a3cea1592eaa` |
| `doc/RETAINED_DEBT.md` | `402b8eb514348f50503da232b3116593509d5e2c` |
| `doc/BEAMMAP_CONFIG_AUTHORITY.md` | `1fc4697b9e95fa8f0532fd5bd5f8be39a9a0e5cc` |
| `doc/PHASE5_PREPARATION_AND_INTEGRATION_PLAN_2026-07-16.md` | `a53414e4dfdd5055c47d32699b6ee18e5aa4efcf` |
| `doc/TOLTECA_BUILD_INTEGRATION_REQUIREMENTS_2026-07-23.md` | `10fe80b057e9adf3c6579a94c884a9b63d538764` |
| `doc/TOLTECA_BUILD_INTEGRATION_REVIEW_2026-07-26.md` | `da30df50fefa9d1ee1a2bededef88a9d86350d14` |
| `doc/PHASE4_1_TOLTECA_CONFIG_STRUCTURE_PLAN_2026-07-16.md` | `5760ed91782e0b2df417183340b98231d3332d45` |
| `doc/PHASE4_2_TECHNIQUE_PERFORMANCE_REVIEW_PLAN_2026-07-16.md` | `d49d8cacf0a314039925720a9f149982f6a0e72b` |
| `handoff/EXTERNAL_REFACTOR_ARCHITECTURE_REVIEW_2026-07-10.md` | `1bb8da0e2dee6825f266c496ff7c3df9302f6c03` |
| `doc/STRUCTURAL_REFACTOR_PLAN_2026-06-29.md` | `d5c2f460059ad856ec6f37b1516545f16c05b28c` |
| `doc/adr/README.md` | `e33f05fee8eb0602e1124e24669b154b31eb89ae` |
| `validation/product_contracts.json` | `f335052e42a9331e0ded901790457fa9fe244dcd` |

The dated plans/handoff establish repository context and stop/authority rules;
they do not override the living architecture, scientific conventions, product
contract, or current source.

### 15.2 Citlali producer, model, serialization, and consumer sources

Revision: `46ad23888a40f5102cdfd50c06e49a549bdf8a20`

| Path | Blob |
| --- | --- |
| `include/citlali/core/engine/io.h` | `10caf2c72590df37d5afaa9edd287c3d03a21bf1` |
| `include/citlali/core/engine/detail/rawobs_collection_impl.h` | `8e0e6f0d02754063fe643ccba559985d9a95f4e6` |
| `include/citlali/core/engine/calib.h` | `ef0a367b0191662a3817caf4cbf3ee0d0ba8e2c4` |
| `src/citlali/core/engine/calib.cpp` | `c48adfef12ac243e6457d75be24a2fc6abc471c6` |
| `include/citlali/core/utils/ecsv_io.h` | `fca63b60738d194f807665f91b15bf1a6dc8cab2` |
| `include/citlali/core/pipeline/array_properties_table_source.h` | `2ffed89d4950cb7c5fe958aa854d0aeaaaa23278` |
| `include/citlali/core/pipeline/array_properties_table.h` | `690b40c5cf7117eca2a5826df8448696eb19981f` |
| `include/citlali/core/pipeline/observation_calibration_config.h` | `32be18a706f8b08fcbc653dc00cf5b7a92e08dca` |
| `include/citlali/core/pipeline/rawobs_detector_inventory.h` | `78310a7c8abd779bc84f4e14374d802008f00359` |
| `include/citlali/core/pipeline/rawobs_tone_frequency_inventory.h` | `c43f9641e51d80f6150144bf4eb64f5ffa10ca3a` |
| `include/citlali/core/engine/detail/todproc_raw_input_impl.h` | `b160fe50de02dc50889236d88ae062bbc43b5ca0` |
| `include/citlali/core/pipeline/observation_input_checks.h` | `f2668a9af85d9fab9ddbeffa339cbaf547ec51fc` |
| `include/citlali/core/pipeline/reduction_observation_inputs.h` | `9be312bdf0708f3427da9f48e72b389883a4168c` |
| `include/citlali/core/engine/detail/kidsproc_load_rawobs_impl.h` | `99f9c85dff65f9a014ed584e58b74148d7302ddf` |
| `include/citlali/core/engine/detail/kidsproc_direct_rtc_impl.h` | `38a1067bedcba6fc388addd99f2db639136562fb` |
| `include/citlali/core/engine/detail/kidsproc_metadata_reduce_impl.h` | `b62324b97115183273425695b39466ad2203301b` |
| `include/citlali/core/engine/detail/beammap_setup_state_impl.h` | `8a094fb9ccbde7ef9f434d65888e61718ca41f06` |
| `include/citlali/core/engine/detail/beammap_setup_metadata_impl.h` | `42a6dbba643c7d1f4909c583fa5ad11b80a87191` |
| `include/citlali/core/engine/detail/beammap_setup_diagnostics_impl.h` | `6fbd3811c35df0719d5b447e58c6dcef0599f375` |
| `include/citlali/core/engine/detail/beammap_empirical_template_schema.h` | `668849ae87fcdf4852c2e7f9ec22f00ee20cc4c0` |
| `include/citlali/core/engine/detail/beammap_empirical_template_columns_impl.h` | `5de1b110eea46e5c5ef9a44bfaca5004e7f94c33` |
| `include/citlali/core/engine/detail/beammap_pipeline_entry_impl.h` | `fae782a0880966d417e0e0f0f9b6d65b2e73fee0` |
| `include/citlali/core/engine/detail/beammap_final_apt_impl.h` | `8ba4aa3f853be4d344af6e43f3a64413db5fb26d` |
| `include/citlali/core/engine/detail/beammap_process_apt_impl.h` | `d6d0a1cdf796c1ff4dd0ab9d54982a2b09e2ecf0` |
| `include/citlali/core/engine/detail/beammap_apt_derotation_impl.h` | `ad1984a3ab7beeba023a839563e534c36c7d1d65` |
| `include/citlali/core/engine/detail/beammap_apt_table_output_helpers.h` | `01b78fe1b3af01440f624af88eb107938dbf1e00` |
| `include/citlali/core/engine/detail/beammap_apt_table_output_impl.h` | `08615dc1337f6a26d0d825eb3758ec3f7fd0cd48` |
| `include/citlali/core/engine/detail/beammap_detector_table_output_impl.h` | `55690c9e8fa99955adc947f722a711d42f582ca1` |
| `include/citlali/core/engine/detail/beammap_fit_qc_columns.h` | `d3878aadcc4ff861012d576d2f7d57dc60c2d7ab` |
| `include/citlali/core/engine/detail/beammap_fit_qc_units_descriptions.h` | `94adc4f15d9598a5d23d8c4fd2e15aee92d09109` |
| `include/citlali/core/pipeline/tod_data_static_metadata_vars.h` | `44d3e2ef0bc0cea7c25a5863f14b733ee7798d85` |
| `include/citlali/core/pipeline/phdu_telescope_values.h` | `589a0c95551870a078f9db0e1bb379385be6c64b` |
| `include/citlali/core/pipeline/phdu_observation_auxiliary_keys.h` | `a2f0ca6c29ccbafa554a6bf0b54aacb77dbb9df9` |
| `include/citlali/core/engine/detail/tod_file_output_impl.h` | `b02647a671277102b8ea2b31ead79f2ce37dedd1` |
| `include/citlali/core/pipeline/ptcdiag_detector_metadata_outputs.h` | `74a5aa934b8f37b3cfc797db08396f841f840c66` |
| `include/citlali/core/mapmaking/naive_mm.h` | `5111876a371056d2f3dafa3aef39b0c6263cae00` |
| `include/citlali/core/pipeline/learning_apt_helpers.h` | `c69d0a7bc676a07cd53d16eba75a90feef67cc10` |

### 15.3 TolProj documentation authority

Revision: `0fcd33ff9d805246a54d15d67751b762410f2e86`

| Path | Blob |
| --- | --- |
| `AGENTS.md` | `1029c78010ea79a863616b5bb83f884ace934e65` |
| `README.md` | `c60795093c85d28b974ceed238b90f97760a6ab0` |
| `docs/STATUS.md` | `30982b42c12d088d09df0df23eb8a9f533329314` |
| `docs/WORKFLOW_V0_2.md` | `78098f23a2e531f06834793389fd9f86e8fa4c81` |
| `docs/CITLALI_REFACTOR_CONFIG.md` | `630a303929bdd5e238c2c835207469fc92a401c1` |
| `docs/CITLALI_VALIDATION_SUITE.md` | `fe3f02c31aa208c6ec1ecc788953c553e61a6f32` |

Only documentation in the clean canonical TolProj worktree was used. No
TolProj implementation body, stopped worktree, proposal, fixture, or test was
inspected.

### 15.4 TolAPT documentation authority

Revision: `3a07cc551faf903da3e1d49d7d3a6b20381afc3d`

| Path | Blob |
| --- | --- |
| `AGENTS.md` | `9d86cef13948e86a82e1e0149286a91d0eb796e1` |
| `README.md` | `11c879ab46b50734b2e17f7a1de26f320f5aa0ef` |
| `docs/STATUS.md` | `5afa763fc0b56bc7efedc185bca8da1af0d5a8a3` |
| `docs/output_contract.md` | `aa8bea3a08364689d4628c7c1dd639c8dc99cb56` |
| `docs/application_reliability_contract_beammap_priors_2026-06-18.md` | `7ea689b08ca7b71761d4f4d8f0e963b3912b1574` |

The dated application contract is included only as the named current
Beammap-prior producer contract. A separate dated application note (blob
`e9668389630dc17d86b641093723e78df92cf115`) was inspected, classified as
scoped evidence under repository rules, and excluded from derivation.

### 15.5 `toltec_beammap` documentation authority

Revision: `958a2a15f43189846a24556a63ef908da789c7b8`

| Path | Blob |
| --- | --- |
| `AGENTS.md` | `405316e7c8ea729f394e53424fc87ab1c5ecb286` |
| `README.md` | `b1edbe9735476da8db07e6a97bbb2bf74d7113f1` |
| `docs/STATUS.md` | `cb239ba702edfe39a2924f78b0a0c02b806b012f` |

The README's aging `Notes` section was not used for architectural claims. The
three unrelated untracked scripts recorded at READY remained unopened.

### 15.6 TolTECA documentation authority

Revision: locally stored `origin/main` at
`2791e6a1e6349ad1d3ac549a648f41cbc51b98c7`

| Path | Blob |
| --- | --- |
| `tolteca/data/examples/workdir_README_template.md` | `a62d09ac440ae8fb1b1fad0c425a878262c9e7ba` |

This source establishes workdir/config precedence and compatibility context. It
does not define an APT schema, digest, or observation binding.

### 15.7 Routing material and non-authorities

The versioned `toltec-context` skill and its routing references were read to
locate repositories and grade authority. They are routing guidance, not the
canonical APT evidence, and no claim above depends on them. Repository
`pyproject.toml` files were manifest-inspected for package identity only and
were not used to derive the model. No external shared APT schema repository was
discovered in the directly invoked Citlali path.

## 16. Freeze and next stop

At this core's freeze:

- the only intended repository delta is this file;
- no test or implementation work has run;
- the final report path `doc/audits/APT_E2E_001_AUDIT_REPORT.md` remains absent;
- no candidate inspection event exists; and
- ALIGN B2 and the stopped TolProj task remain on hold.

The next permissible action is the mechanical hash/one-file commit and
checkpoint report. After that checkpoint, no candidate may be inspected until
the coordinator explicitly approves the next phase.
