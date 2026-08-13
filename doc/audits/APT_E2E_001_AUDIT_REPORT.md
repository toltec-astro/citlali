# APT-E2E-001 End-to-End Audit Report

Package: `APT-E2E-001`
Title: End-to-End APT Scientific, Identity, and Provenance Contract Audit
Status: documentation-only final audit; implementation not authorized
Audit base: `46ad23888a40f5102cdfd50c06e49a549bdf8a20`
Independent core: `2ec2e42732a1c54575dff52ec060b3fdc90d8cf6`
Candidate exposure began: `2026-08-13T19:46:54Z`

## 1. Verdict and stop state

The current ecosystem does not implement one verifiable APT contract. Citlali
Beammap is the authoritative baseline producer, but its concrete schema is
distributed across an all-double in-memory map, an open-ended writer, metadata
helpers, and a name-only product contract. TolProj, TolAPT, `toltec_beammap`,
TolTECA, and Citlali science each compensate independently with row ordinals,
floating-point casts, paths, filenames, heuristic joins, or positional
application. None can prove end to end that the APT applied to detector column
`i` is the declared artifact, the intended observation/readout row, or the
result of the declared transformations.

The architectural repair is producer-first:

1. Citlali must issue a typed, versioned, integrity-covered baseline APT with
   artifact occurrence, artifact-local row, and raw readout binding semantics.
2. TolProj may add observation-specific target binding and realized tone-match
   provenance, but it may not define or reconstruct producer-owned identity.
3. TolTECA must carry the resolved occurrence/digest/binding, and Citlali
   science must independently recompute and verify them before calibration.
4. TolAPT and `toltec_beammap` then conform their independent products without
   changing their matching or calibration science.

The smallest compatible identity disposition is to retain `uid` in the first
canonical profile only as an exact, unique **artifact-local row key**. It is not
a persistent detector/resonator identity. A persistent measured identity
remains absent until an owner, namespace, construction, and lifecycle are
demonstrated. The baseline readout binding is separately scoped by a normative
raw readout manifest and a network/channel key; design identity and all mapping
events remain separate again.

The stopped TolProj proposal is not accepted. Its observation/readout facts,
explicit realized relations, unused/unmatched dispositions, and separation of
source/application/presentation order identify legitimate TolProj obligations.
Its TolProj-private canonicalization, seed identity derived from `uid`, private
JSON identity sidecar, and `project.yaml` anchor are rejected as production
authority. Its tests and examples may remain read-only audit/fixture evidence
for a later conformance implementation.

No implementation task is launched by this report. The stopped TolProj
worktree remains unchanged and uncommitted. ALIGN B2 remains held. Current
canonical claims and production activation remain prohibited.

## 2. Audit controls, independence, and repository state

### 2.1 Independent freeze

The architectural core was derived before candidate exposure and committed as
`doc/audits/APT_E2E_001_INDEPENDENT_CORE.md` at
`2ec2e42732a1c54575dff52ec060b3fdc90d8cf6`, parent
`46ad23888a40f5102cdfd50c06e49a549bdf8a20`, tree
`e05b1aa140075b2fe00232dc2061f14cfdec04ae`, committed at
`2026-08-13T19:14:02Z`. Its file SHA-256 is
`c11fcc08676d62ecddbce5b7a8c1c8395eb258d9c74fe944a7a408451ab94a7b`
and its standard parent-to-commit patch SHA-256 is
`8df677effe5c63e3c12538400dda9044da68c7ebd6c1251a70b90c2164c07abf`.
It remained byte-for-byte unchanged during the final audit.

The clean producer and downstream implementations were traced only after that
freeze. Candidate exposure occurred only after the complete canonical model,
Citlali producer comparison, smallest producer repair, and clean downstream
requirements were fixed. The first candidate read began at
`2026-08-13T19:46:54Z`.

### 2.2 Inspected repositories and refs

No fetch or network operation was used. “Ahead/behind” is relative to the
locally stored upstream ref.

| Repository / inspected worktree | Branch or inspected ref | Exact SHA | Upstream / divergence | Dirty state at inspection |
| --- | --- | --- | --- | --- |
| Citlali audit, `/Users/gwilson/.codex/worktrees/b4c7/citlali-refactor` | `codex/apt-e2e-001-audit-20260813`; application source read from parent/base | audit core `2ec2e42732a1c54575dff52ec060b3fdc90d8cf6`; source base `46ad23888a40f5102cdfd50c06e49a549bdf8a20` | audit branch has no upstream; locally stored `origin/codex/refactor-mainline` = base | clean before this report |
| clean TolProj, `/Users/gwilson/GitHub/tolproj` | `main` | `0fcd33ff9d805246a54d15d67751b762410f2e86` | `origin/main` = `74395c824860ca41410dde5cf2e0272e5535fc19`, ahead 1 / behind 0 | clean |
| stopped TolProj evidence, `/Users/gwilson/.codex/worktrees/f1ab/tolproj` | `codex/repair-tolproj-apt-identity-contract` | `0fcd33ff9d805246a54d15d67751b762410f2e86` | no upstream | four tracked unstaged modifications; preserved exactly |
| TolAPT, `/Users/gwilson/GitHub/tolapt` | `codex/rework-foundation` | `3a07cc551faf903da3e1d49d7d3a6b20381afc3d` | `origin/codex/rework-foundation` = `3a07cc551faf903da3e1d49d7d3a6b20381afc3d`, ahead 0 / behind 0 | clean |
| `toltec_beammap`, `/Users/gwilson/GitHub/toltec_beammap` | `main` | `958a2a15f43189846a24556a63ef908da789c7b8` | `origin/main` = `444806e2c52c5ce42129c7abe06d24f429a6cca8`, ahead 1 / behind 0 | three unrelated untracked scripts; contents never opened |
| TolTECA checkout, `/Users/gwilson/GitHub/tolteca` | `v3.x` | `8d05ecde7c116d52b7a80a84d21e0ade367f163a` | `origin/v3.x` = `8d05ecde7c116d52b7a80a84d21e0ade367f163a`, ahead 0 / behind 0 | clean |
| TolTECA reduction authority read as Git objects | locally stored `origin/main` | `2791e6a1e6349ad1d3ac549a648f41cbc51b98c7` | local remote-tracking ref; no fetch | no checkout or write |

The `toltec_beammap` untracked paths were
`scripts/audit_flux_ratio_calculation.py`,
`scripts/fit_a1400_correction.py`, and
`scripts/plot_flux_ratio_vs_elevation.py`. They were treated as user evidence
outside this audit and were never read. Registered unrelated/prunable
worktrees were inventoried at READY but not used as sources. The prohibited
prior Citlali APT-audit ref/worktree/artifacts were not opened, read, diffed,
or hashed. No separate shared APT schema repository was found in the directly
invoked Citlali producer chain.

### 2.3 Evidence and activity ceiling

This audit used commit-addressed source and documentation plus read-only local
Git commands. Existing tests were inspected as evidence but **no test, build,
reduction, or scientific data operation was run**. There was no Unity access,
network service, production data generation, app/runtime/schema/config change,
push, merge, acceptance, promotion, or production activation.

## 3. Authority map

| Contract activity | Current authority | Representative current defining/acting locations (exact manifests in core §15 and report §17) | Required disposition |
| --- | --- | --- | --- |
| Canonical APT field semantics, types, units, missing states, extensions | Citlali Beammap producer boundary, or a shared model directly invoked by it | Citlali `engine/calib.h`, `rawobs_detector_inventory.h`, `beammap_*apt*`, `ecsv_io.h`, `product_contracts.json` | consolidate into one versioned producer-owned contract |
| Baseline artifact issuance and integrity | Citlali Beammap | `include/citlali/core/engine/detail/beammap_apt_table_output_impl.h`; `include/citlali/core/engine/detail/beammap_setup_metadata_impl.h`; `include/citlali/core/pipeline/beammap_provenance*.h`; `include/citlali/core/pipeline/product_index_file.h` | issue occurrence, semantic/envelope digests, raw manifest, row/readout keys, software/config event |
| Observation target/KMP/readout selection | TolProj | `match_apts.py`, `make_matched_apt.py`, raw/KMP discovery | add only observation-specific target facts under canonical schema |
| TolProj tone-matching policy | TolProj | `legacy_scripts/make_matched_apt.py` | preserve current selected pairs, shift, gate, backend, and tie behavior; record results, do not redesign |
| Measured-to-design matching policy | TolAPT | supported `src/tolapt/matching.py` and run products | preserve current policy; emit explicit mapping lineage using canonical measured references |
| Common mapping/relation status schema | unassigned pending Grant appointment; default producer-first location is the Citlali/shared canonical APT authority | no current end-to-end schema; clean components use private status/layouts | canonical authority defines shared relation envelope; TolProj, TolAPT, and each transformer remain the value issuers for their own events |
| Design artifact/row namespace | unassigned pending Grant appointment to the owning design source | TolAPT currently derives design `det_id` from input `uid` | TolAPT references it and issues measured↔design mapping values but does not become its ultimate authority |
| Beammap fitting/update/calibration diagnostics | `toltec_beammap` | `apt_modifier.py`, `toltec_beammap.py`, `pipeline/process.py` | issue conforming derived APTs and independent reports; do not redefine identity |
| Input selection/config precedence | TolTECA | `tolteca/reduce/engines/citlali.py` at local `origin/main`; numbered config docs | carry selected occurrence/digest/binding; never synthesize canonical APT identity |
| Scientific admission and raw-binding verification | Citlali science consumer | `engine/io.h`, `calib.cpp`, raw-input/tone/application paths | independently recompute and fail closed before RTC calibration or geometry |
| CAL numerical/unit/response science | unassigned pending Grant appointment; `SCI-CAL-001` is named in Citlali living docs | current calibration applies APT `flxscale(i)` and `array(i)` positionally | the computing component issues its own calibration values/events; APT binding repair is prerequisite, not CAL scientific closure |
| ALIGN algorithms and B2 gate | ALIGN owns temporal/sample alignment; B2 is coordinator-owned gate | Citlali alignment helpers do not consume APT identity | keep held; APT repair must not claim to close temporal ALIGN science |

The phrase “matched APT” is overloaded. The contract must always name the
mapping domain: TolProj raw/observation tone-to-seed mapping or TolAPT
measured-to-design mapping. Each event names its policy owner, input/output
artifact roles, row namespaces, cardinality, and software/config provenance.

## 4. Current Citlali APT contract

### 4.1 Concrete representation and production path

`Calib::apt` is `std::map<std::string, Eigen::VectorXd>`. The ECSV reader casts
integer, `int16`, `int64`, Boolean, and float columns to binary64. The writer
materializes every current field into one `Eigen::MatrixXd`. Consequently all
base fields, including semantically integral identifiers and flags, are double
in process and in the writer matrix.

For an internally generated Beammap APT, Citlali concatenates the numerically
ordered raw KIDs interfaces, records detector count/network/array segments,
initializes nearly all fields to one (`x_t`, `y_t`, and `flag` are special), and
sets `uid = 0..n_dets-1`. It assigns first-sweep tone frequencies by network
segment. `kids_tone` is currently synthesized as the row offset reset when
`nw` increases. KIDs fit-report columns are then overlaid by row; only a
colliding `flag` is renamed to `kids_flag`, while every other collision may
overwrite a base field. Beammap fit/calibration/quality stages mutate values
and append diagnostics before one ECSV is written.

### 4.2 Base field inventory

Every current base column has physical storage type binary64. “Integral” below
means intended semantics, not enforced ECSV/in-memory type. Unless explicitly
noted, uniqueness is neither claimed nor validated.

| Field | Unit | Producer / current meaning | Mutability and uniqueness | Material consumers / contract result |
| --- | --- | --- | --- | --- |
| `uid` | N/A | internal concatenated row ordinal; imported value otherwise; described only as “unique id” | internal sequence unique locally; external values unvalidated; copied and rounded/cast downstream | retain only as exact unique artifact-local row key in v1; never persistent identity |
| `tone_freq` | Hz | first raw tone-frequency sweep by network/row | overwritten on external ingestion and TolProj output; mutable float | readout/calibration attribute, not identity |
| `array` | N/A | array ID derived from network map | integral grouping; expected constant within contiguous group | calibration/extinction/grouping; not row identity |
| `nw` | N/A | readout network/interface number | integral; current consumer requires contiguous network groups | necessary scope but insufficient identity |
| `fg` | N/A | frequency group | internal default is one unless fit report overrides; authority varies | design/matching attribute; nonidentity |
| `pg` | N/A | polarization group | internal default is one unless overridden; authority varies | carried provenance/design attribute; nonidentity |
| `ori` | N/A | design/orientation attribute | internal default is one unless overridden | nonidentity; authoritative source unresolved |
| `loc` | N/A | design/location attribute | internal default is one unless overridden | nonidentity; authoritative source unresolved |
| `responsivity` | N/A | detector response quantity | mutable calibration quantity | scientific calibration input; nonidentity |
| `flxscale` | mJy/beam/xs | detector flux conversion | computed/overwritten by Citlali, TolProj, and `toltec_beammap` paths | every change requires new artifact/event/digest; nonidentity |
| `sens` | mJy/beam s^0.5 | detector sensitivity | derived and rescaled; invalid rows can be zeroed | mutable calibration quantity; nonidentity |
| `derot_elev` | rad | elevation used for position derotation | observation/fitting value | transformation provenance; nonidentity |
| `amp` | xs | fitted Gaussian amplitude | Beammap fit output; can be fit-report collision target | mutable fit result; nonidentity |
| `amp_err` | xs | fitted amplitude error | Beammap fit output | mutable uncertainty; nonidentity |
| `x_t` | arcsec | current fitted azimuthal offset | fit, reference subtraction, and optional derotation overwrite it | geometry attribute; nonidentity |
| `x_t_err` | arcsec | fitted azimuthal-offset error | fit output | mutable uncertainty; nonidentity |
| `y_t` | arcsec | current fitted altitude offset | fit, reference subtraction, and optional derotation overwrite it | geometry attribute; nonidentity |
| `y_t_err` | arcsec | fitted altitude-offset error | fit output | mutable uncertainty; nonidentity |
| `a_fwhm` | arcsec | fitted azimuthal/major FWHM | fit/update output | mutable fit result; nonidentity |
| `a_fwhm_err` | arcsec | fitted `a_fwhm` error | fit output | mutable uncertainty; nonidentity |
| `b_fwhm` | arcsec | fitted altitude/minor FWHM | fit/update output | mutable fit result; nonidentity |
| `b_fwhm_err` | arcsec | fitted `b_fwhm` error | fit output | mutable uncertainty; nonidentity |
| `angle` | rad | fitted beam rotation angle | fit/update output | mutable fit result; `toltec_beammap` currently writes degrees here, a unit defect |
| `angle_err` | rad | fitted angle error | fit output | mutable uncertainty; nonidentity |
| `converge_iter` | N/A | Beammap convergence iteration | integral diagnostic | fit lifecycle, not identity |
| `flag` | N/A | bad-detector/validity state | mutable; unmatched TolProj rows filled `1`; several downstream gates use it | explicit validity/status, not identity |
| `sig2noise` | N/A | fit signal-to-noise | derived and used for quality flagging | mutable diagnostic, not identity |
| `x_t_raw` | arcsec | position before reference/derotation | derived copy | transformation provenance, not identity |
| `y_t_raw` | arcsec | position before reference/derotation | derived copy | transformation provenance, not identity |
| `x_t_derot` | arcsec | derotated position | derived output | transformation value, not identity |
| `y_t_derot` | arcsec | derotated position | derived output | transformation value, not identity |

### 4.3 Current known extension fields

| Field or class | Current meaning / unit | Authority and disposition |
| --- | --- | --- |
| `kids_tone` | network-local row offset, N/A | must become an exact network/channel key derived from the raw inventory and scoped by the raw manifest; not persistent identity |
| `flag2` | bitwise Beammap quality flags | Citlali validity diagnostic; typed integral field in canonical schema |
| `rfi_masked_samples`, `rfi_masked_scans` | counts of masked samples/scans | Citlali diagnostic, nonidentity |
| `scan_band_masked_samples`, `scan_band_masked_rows`, `scan_band_masked_edge`, `scan_band_mask_rejected` | scan-band masking diagnostics | Citlali diagnostic/status, nonidentity |
| `final_prior_slot_index`, `final_prior_d2` | soft-prior slot/distance diagnostics | local prior evidence; explicitly not detector identity |
| `cal_amp`, `cal_amp_method`, `template_amp`, `template_offset`, `template_resid_rms`, `template_npix`, `template_amp_over_fit_amp`, `cal_amp_over_fit_amp`, `map_peak_amp`, `map_peak_amp_over_fit_amp` | empirical-template/Gaussian calibration diagnostics | Citlali-owned values with declared units/status; mutable, nonidentity |
| KIDs fit-report columns | runtime-dependent values copied by row; `flag` alone renamed `kids_flag` | current open extension/collision surface must be replaced with an allowlist, typed extension registry, and protected-column rejection |

### 4.4 Current metadata

The writer records observation number, source, project, creation and observation
times, MJD, reference frame, per-array flux/tau, column descriptions, phase
strategy, reference/derotation settings, masking configuration, weighting, and
fit policy. It does not record a schema version, occurrence reference,
canonical digest, raw manifest, row/readout relation, source/seed identity,
transformation event, software revision, or supersession. FITS/TOD products
later record only an APT basename.

### 4.5 Canonical v1 field and admission classification

This is the complete baseline field contract recommended by the audit. `R`
means required for every canonical v1 APT; `C` means conditionally required by
the named scientific mode; `O` means a registered optional extension. Every
typed row field is artifact-scoped unless another namespace is stated. Missing
numeric results use the schema's typed invalid/null representation plus
validity/status; sentinel values never identify a row.

| Field(s) | Profile / exact type | Namespace, validity, uniqueness | Value and mutation authority | Strict science-ingestion duty |
| --- | --- | --- | --- | --- |
| `uid` | R / exact unsigned or nonnegative signed integer within the selected serialized range | unique artifact-local row key; never null; no cross-artifact persistence claim | every APT issuer assigns a unique target/output key or preserves it only for a proven 1:1 same-row transformation | preserve exact type; reject missing, duplicate, nonintegral, negative, or out-of-range values |
| `nw` | R / integer | raw-manifest network/interface namespace; never null | baseline Citlali raw inventory; TolProj observation target may legitimately replace it from the target manifest | derive allowed interfaces from actual raw input and verify every row |
| `kids_tone` | R / integer | channel index unique only within `(raw manifest, nw)`; never null | baseline Citlali raw inventory or observation-target issuer from authoritative KMP/raw channel order | verify `(nw,kids_tone)` uniqueness/completeness and build explicit raw-column permutation |
| `array` | R / integer enum | canonical array namespace `{0,1,2}` unless schema extends it; never null; need not be row-unique | producer derives from authoritative network map; transformers preserve unless target manifest proves replacement | verify value and consistency with network/readout manifest |
| `tone_freq` | R / float64 Hz | row/readout attribute; finite when row is applicable; nonunique | observation/readout issuer; consumer may compare/refresh from actual raw input only as a recorded transform | verify unit/type and declared raw agreement/tolerance; never use equality as row identity |
| `fg`, `pg`, `ori`, `loc` | C / integer or declared enum | design/polarization namespaces remain unassigned pending Grant; schema-defined unknown state allowed; nonunique | owning design/polarization source unassigned; Citlali may copy only with declared authority; matching does not manufacture truth | require `fg` for current setup/polarization paths and other fields only for the enabled mode; reject unregistered semantics, do not infer identity |
| `responsivity` | C / float64 in the schema-declared physical unit | detector calibration value; finite/positive when the consuming calibration mode requires it; nonunique | computing calibration component | validate for enabled RTC/despike mode; no identity use |
| `flxscale` | R for calibrated science / float64 `mJy/beam/xs` | detector calibration value; finite/positive for usable rows, typed invalid for unusable rows; nonunique | Citlali baseline, TolProj calibration, or other declared calibration transform | require before flux application; verify transformation provenance and apply by explicit row binding |
| `sens` | C / float64 `mJy/beam s^0.5` | derived detector sensitivity; finite/positive for applicable usable rows | Citlali or declared calibration transform | require only for consuming sensitivity/weighting mode; validate status and provenance |
| `derot_elev` | C / float64 rad | observation transform attribute; finite for derotation | Citlali Beammap or declared geometry transform | require for the selected geometry convention; verify units |
| `amp`, `amp_err` | C / float64 `xs` | fit result and uncertainty; validity tied to fit status | fit/calibration producer | require only when selected calibration/diagnostic mode consumes them; never identity |
| `x_t`, `x_t_err`, `y_t`, `y_t_err` | R for map geometry / float64 arcsec | current detector geometry and uncertainties; valid positions finite for usable rows | Citlali or declared geometry transform | require before map geometry; verify frame/transform lineage and row binding |
| `a_fwhm`, `a_fwhm_err`, `b_fwhm`, `b_fwhm_err` | R for current setup / float64 arcsec | fit results; usable widths finite and positive; invalid permitted with flag/status | fitting producer | validate before beam-area/group calculations; invalid/unflagged combinations fail |
| `angle`, `angle_err` | R for current setup / float64 rad | fit result and uncertainty; finite when applicable | fitting producer | enforce radians and status; reject unit-conflicting updates |
| `converge_iter` | C / nonnegative integer | fit lifecycle diagnostic; nonunique | fitting producer | require only for fit audit/quality mode; preserve exact type |
| `flag` | R / integer or closed bit/enum contract | row scientific validity; never null; nonunique | each declared quality transform may change it and must record reasons | validate closed values and require mapping coverage for every row, including invalid rows |
| `sig2noise` | C / float64 dimensionless | fit diagnostic; finite when applicable | fitting producer | require for enabled quality policy; no identity use |
| `x_t_raw`, `y_t_raw`, `x_t_derot`, `y_t_derot` | C / float64 arcsec | declared geometry-stage values; frame/transform metadata required | Citlali or declared geometry transform | require only when the selected reference/derotation profile claims them; verify transform provenance |
| `flag2` | O/C / closed integer bitmask | Citlali Beammap quality namespace; nonunique | Citlali quality stage | validate bit contract if present/required by profile |
| mask diagnostics named in §4.3 | O / integer counts/codes | Citlali diagnostic namespaces | Citlali masking stage | preserve if registered; do not require for ordinary science ingestion |
| `final_prior_slot_index`, `final_prior_d2` | O / integer and float64 | soft-prior diagnostic namespace; explicit unknown state; nonidentity | Citlali prior stage | accept only as registered diagnostics; never bind detector rows through slot |
| empirical-template fields named in §4.3 | O/C / typed per current declared units; `cal_amp_method` closed enum | Citlali calibration diagnostic namespace | Citlali calibration stage | require only for the selected calibration-amplitude profile; validate method/value consistency |
| registered KIDs fit-report fields | O / individually versioned types/units | KIDs diagnostic namespace; no name collision with protected fields | authoritative KIDs fit-report producer; Citlali copies under registry | accept only registered extensions and preserve their authority metadata |

Any runtime fit-report field not in the versioned registry is a **noncanonical
extension**: it may be retained only in an explicitly permissive evidence
profile and cannot enter the canonical digest or strict production APT until it
is registered. Protected identity/readout fields may never be overwritten by a
fit-report collision.

The observation-specific APT required by Citlali science contains all `R`
fields for the enabled reduction profile, every `C` field that its selected
calibration/geometry/polarization modes consume, and this normative metadata:

| Metadata element | Profile / authority | Required strict check |
| --- | --- | --- |
| schema/profile and extension registry version | R / Citlali schema | supported exact version and closed field contract |
| semantic content digest + scope/version | R / issuer under Citlali algorithm | recompute over table and semantic metadata |
| envelope/event digest + scope/version | R / issuer | recompute over occurrence, lineage, software/config/time, sources/outputs, declared content and relation refs |
| artifact occurrence and output role | R / issuer | unambiguous occurrence; distinguish identical-content issuances |
| target raw manifest ref/digest and observation/tune/interface scope | R / Citlali baseline or TolProj observation issuer | derive expected facts from actual raw inputs and require equality |
| row-key/readout-key declarations and mapping/relation ref/digest | R / canonical schema + mapping issuer | verify uniqueness, completeness, statuses, cardinality, and referential integrity |
| source/seed occurrence and digest refs | R for derived APT / transformer | recompute/verify available sources and exact lineage; unknown is not silently upgraded |
| operation/mapping domain, field-change and order declarations | R for derived APT / transformer | verify only permitted copied/added/dropped/overwritten/reordered fields |
| producer/software revision, resolved configuration/policy identity, UTC event time | R / issuer | parse and integrity-check; never substitute for content/readout validation |
| supersession / compatibility assurance | R when applicable / issuer under canonical policy | enforce current admissibility and do not infer missing history |
| frame/scientific context (`Radesys`, reference/derotation, source/project/obs time, units/descriptions) | R/C / Citlali and computing stage | require the subset used by the enabled science mode and validate exact units/frame |

## 5. Exact `uid` finding

“Double-valued” is a representation finding, with additional downstream
consequences; it is not evidence that one UID contains two semantic values.

| Component | Exact mechanism | Consequence |
| --- | --- | --- |
| Citlali | all numeric ECSV types are cast to `Eigen::VectorXd`; internal UID is a `VectorXd::LinSpaced` row sequence | identifier type/namespace are erased; arbitrary `int64` values above `2^53` can lose exactness |
| TolProj clean current | packaged/current seed UIDs are float; any `np.int64` output columns are cast to float; UID is copied from seed only and ignored by the matcher | matched UIDs are unverified; unmatched generic fill creates repeated `0.0` placeholders |
| TolAPT | measured UID is unconditionally cast to binary64; several later paths parse via `float`, round integral-looking values, or format to 12 significant digits | lexical distinctions, values above `2^53`, and some decimal distinctions can collapse |
| `toltec_beammap` | casts UID to Python/NumPy integer and builds dictionaries | duplicate UIDs warn or silently overwrite; last row/value can win |
| Citlali science products | rounds UID to integer in several paths; some paths fall back to row; learning returns first rounded equality | duplicates/nonfinite values are ambiguous and row order can become hidden identity |

Current internally generated UID is unique only within that table and is
regenerated from concatenated input order. External UID finiteness,
integrality, range, uniqueness, namespace, and lifecycle are not validated.
The failed proof obligation is therefore not merely precision: no component
proves a bijection between the UID-bearing row and the actual raw detector
column, nor an immutable cross-artifact lifecycle.

## 6. Minimal canonical identity and provenance model

The model uses semantic roles, not a predetermined two-field replacement.

| Required element | Contract/value authority | Minimal semantics |
| --- | --- | --- |
| schema/profile | Citlali producer or directly invoked shared schema | versioned fields, exact types, units, null/validity, protected/extension rules, migration profile |
| artifact occurrence | every issuer under canonical semantics | distinct issuance/event output reference; two byte-identical independently issued artifacts remain distinct |
| semantic content digest | canonical schema defines SHA-256 algorithm/version/scope/canonicalization; issuer computes, consumer recomputes | covers the typed normative row set and schema-declared semantic metadata, but excludes occurrence/event/self-digest fields so byte-identical semantic content may recur in distinct issuances; digest is not occurrence identity |
| artifact-local row key | Citlali schema; every APT issuer assigns or validly preserves it | exact, nonmissing, unique within occurrence for every valid/invalid row; recommended v1 representation is existing `uid` with narrowed semantics; an observation output keys its target rows and references a seed row only through the mapping relation |
| baseline raw readout binding | Citlali Beammap | normative raw manifest plus row relation keyed at least by manifest-scoped network/interface and channel index; exact smallest manifest tuple remains an owner gate |
| observation-specific target binding | TolProj issues; Citlali independently derives/verifies | target observation/readout manifest and complete target-row relation; bare obsnum/network/tone/frequency/path is insufficient |
| persistent measured detector/resonator identity | no current authority | absent unless a real owner, namespace, construction, uniqueness, and lifecycle are demonstrated |
| design reference | owning design source; TolAPT owns only realized mapping policy | design artifact occurrence/digest plus design row key; reassignment supersedes mapping without changing measurement identity |
| mapping/transformation event | common relation semantics; each policy/transformer owns its realized event | sources/outputs/digests, source/output row refs, statuses/reasons/ambiguity/supersession, complete 1:0/1:1/1:N/N:1 cardinality, software/config/time |
| calibration/fitted values and flags | canonical field semantics from Citlali; numerical result from computing component | mutable attributes; change occurrence/digest and declare mutations, never identity |
| source/application/presentation order | stage that creates each sequence | explicit row-reference sequences where needed; none is detector identity |
| production admission | Citlali science consumer | recognize profile, recompute digest, verify row keys, derive actual raw manifest, verify complete binding/mapping, fail before science |

### 6.1 Serialization and metadata placement

The first canonical profile should remain one ECSV APT, not introduce a new
table format. Row-level facts required for application belong in typed table
columns: artifact-local row key, network/channel readout key, validity/status,
and any direct output-row source reference allowed by the schema. Artifact
occurrence, schema/profile, digest declaration, raw-manifest reference/digest,
producer event, software/config/time, protected-extension declaration, and
mapping summary belong in normative ECSV metadata. The schema must label two
integrity scopes: the semantic content digest above, and an envelope/event
digest that covers occurrence, lineage, source/output references, software,
configuration, time, declared content digest, and relation references while
excluding only its own value. An optional ECSV byte hash can additionally
protect transport. This separation is what permits identical semantic content
to have the same content digest but distinct, verifiable issuance histories.

A complete set-valued relation may be too large or non-tabular for the APT
header. It may be a canonical relation artifact only if the producer-owned
contract defines its schema, the APT and relation bind each other by digest,
they are published atomically, and every consumer verifies them. Such a
relation is an end-to-end canonical artifact, not a TolProj-private sidecar.
A private sidecar remains legitimate only as audit evidence or explicit
legacy-migration evidence; it cannot establish production identity.

The recommended semantic and envelope/event digests are SHA-256 over labelled,
versioned, length-delimited, typed canonical representations. Row members are
canonicalized by exact row key; integral fields remain exact integers and
floating values use an exact canonical representation. Physical ECSV row order
is not a detector identity. Declared order sequences, if normative, are
integrity-covered in their declared scope. The ECSV byte hash is transport
integrity, not semantic identity. Because current serialization round-trip
exactness has not been proven, the producer must re-read the temporary output
and recompute every declared digest before atomic publication.

## 7. Producer comparison and smallest Citlali repair

| Required proof | Current Beammap result | Smallest producer repair |
| --- | --- | --- |
| closed typed schema | 31 all-double base fields plus open runtime extensions | one typed versioned profile; exact integer/float/status types; units/nullability; protected and registered extension rules |
| occurrence distinct from content | filename/path/obsnum only | issue an unambiguous creation-event/output occurrence reference |
| recomputable integrity | none | producer-defined semantic and envelope/event SHA-256 scopes plus optional byte transport SHA; post-write re-read verification |
| exact row key | UID equals current row ordinal; external values unvalidated | retain `uid` only as finite exact nonnegative integral unique artifact-local key; reject duplicates/missing/range loss |
| raw channel relation | row order and synthesized `kids_tone` | derive `(raw-manifest, network/interface, actual channel index)` from authoritative inventory; validate bijection |
| stable field authority | fit-report collisions may overwrite any base field except `flag` | protect `uid`, `nw`, `array`, `kids_tone`, `tone_freq` at minimum; type/allowlist other collisions; classify `fg/pg/ori/loc` authority |
| mapping/provenance | observation/config metadata but no raw relation/software/source digest | normative creation event, raw manifest/digest, software/config/time, field mutation and order declarations |
| executable validation | minimum rows and column names | schema/dtype/unit/key/digest/manifest/extension validator and product contract |
| historical external seed | loader trusts shape/order and discards extensions | canonical only when source and mapping are verified; otherwise explicit reduced-assurance legacy-derived output without fabricated history |

This lane does not create a persistent detector ID, change Beammap fitting or
calibration numerics, alter matcher policy, require a package registry, or
duplicate artifact metadata in every row.

## 8. End-to-end transition matrix

| Boundary | Current input -> output/cardinality | Current field/order/provenance behavior | Current failure mode | Required preservation/repair |
| --- | --- | --- | --- | --- |
| raw KIDs -> Citlali Beammap baseline | network files/channels -> one APT row each | concatenates interfaces/channels; positional UID and `kids_tone`; fit/calibration overwrites; open extensions | reorder relabels; collision can overwrite structure; no digest/raw relation | canonical producer repair in section 7 |
| baseline -> TolProj library/selection | byte copy; one seed may serve N observations | quality checks only `flag`/array; manifest records paths, obsnum, dates | path/obsnum/status treated as identity; stale/tampered file undetected | validate schema/digest; preserve producer occurrence; selection record references exact artifact |
| target raw/KMP -> TolProj target APT | one KMP channel -> one target row | KMP fields prefixed `kids_`; network-sorted rows; local positional `kids_tone`/`det_id`; multi-tune chooses max with warning | target key is order; last KMP supplies metadata; no complete target manifest proof | issue observation/readout manifest and typed target row refs; do not alter tune-selection policy in identity lane |
| seed + target -> TolProj matched APT | each target row -> exactly one output; selected seed row 0:1 per output; same seed can serve N outputs | current 200 kHz, shift, good-first then flagged policy; copies target fields and non-`kids_` seed fields; drops unused seed; unmatched seed fields become zeros; `tone_freq` becomes target `kids_f_out`; int64 becomes float | ambiguity/alternates and unused seed vanish; filler UID duplicates; source/application/presentation order conflated | new occurrence/digest and unique output-local UID for every target row; seed UID remains only a source-row reference in the explicit target->seed/unmatched relation; record unused seed, policy evidence, and typed missing; exact current pairs unchanged |
| TolProj matched -> calibrated APT | bracket/reference inputs -> one copy per science obs; rows intended 1:1 | preserves rows/order/columns; overwrites only `flxscale`; metadata stores method/factors and absolute input path | no verified input/output or bracket/reference/software digests | verify input and mapping; new occurrence/digest; explicit only-`flxscale` mutation; complete input lineage |
| TolProj/TolTECA -> Citlali science | path + selector -> loaded filtered 31-column table | TolProj handoff carries path/obsnum; TolTECA legacy `_fix_apt` can reset UID to row number/fill defaults; `_make_apt` emits `uid=-1`; Citlali filters networks, drops extensions, overwrites tone frequency, applies rows positionally | forged same-shape APT passes; canonical evidence can be destroyed before ingestion | TolTECA carry exact occurrence/digest/binding and retire synthetic repair from strict mode; Citlali typed fail-closed admission and explicit application permutation |
| Citlali science application | retained APT row `i` -> raw detector column `i` | RTC uses `flxscale(i)`/array correction; geometry uses `x_t(i),y_t(i),flag(i)` | count/contiguity prove shape, not row-channel identity | derive raw manifest, validate relation, apply explicit permutation, fail before calibration/geometry |
| baseline -> TolAPT measured input | source APT -> filtered narrow measured table | flagged and nonfinite rows dropped; generated `mNNNNNN` follows post-filter order; UID cast float; most source fields dropped | reorder changes IDs/indices; exclusions lack complete reason lineage; digest only recorded, not reverified | lossless canonical row ref; verify digest/binding; explicit filtered statuses; indices local diagnostics only |
| TolAPT measured -> dewarp/alignment/grouping | intended 1:1 copies | overwrites x/y, adds model/group fields, retains local IDs | no output digest or source-row validation | preserve canonical row refs; issue each derived occurrence/event/digest; declare changes |
| TolAPT measured/design -> candidates/matches | measured 1:0..N candidates -> final 1:1 subset plus unmatched | current Hungarian assignment, block constraints, top-five/ambiguity diagnostics; indices drive mapping | mapping proves index partition, not artifact/row identity; unmatched reasons incomplete | retain current pairs/policy exactly; persist artifact-scoped mapping edges/status/ambiguity/supersession |
| TolAPT matches -> hero APT | selected design anchors -> subset of raw source APT; positions overwritten | join normalizes UID through float; forces flag=0; adds hero columns | source manifest path trusted without rechecking digest; hero no digest | explicit source-row lineage, new occurrence/digest, overwritten-field declaration; no UID normalization |
| TolAPT -> beammap prior | many design/health rows -> one aggregate row per slot/network | intentionally a soft initialization/gating product | slot could be mistaken for detector identity | retain as independent non-APT product; never canonical detector mapping |
| baseline -> `toltec_beammap` update | source rows intended 1:1 -> `_psf` copy | filename selects source; UID/det_id/kids_tone heuristic; duplicates last-win; >=90% overlap; flags and fit fields change; source order preserved | ambiguous mappings; extra/missing rows incompletely represented; no digest; `angle` radians replaced with degrees | consume canonical refs; explicit update relation/status; fail duplicates; declare mutations; correct angle unit in separate reviewed conformance fix |
| `toltec_beammap` flux refresh | derived APT -> same filepath | changes flux metadata, `flxscale`, `sens` in place | source occurrence destroyed; no lineage/digest | immutable new occurrence with exact source and change manifest |
| `toltec_beammap` robustness audit | APT/Fit-QC -> CSV/ECSV/report sidecars | UID joins and cross-reduction summaries; does not modify APT | duplicate UID can make join ambiguous | retain as independent analysis evidence; reference canonical artifacts/rows, never identity authority |
| CAL | APT row -> flux/unit/response application | current Citlali applies `flxscale(i)` positionally | detector-binding uncertainty contaminates CAL evidence | APT contract removes binding uncertainty; CAL numerical/unit science remains separately unclosed |
| ALIGN | raw networks/telescope/HWPR -> temporally aligned streams | sample/packet time alignment, no APT UID use | later calibrated columns still inherit positional APT uncertainty | leave ALIGN algorithms unchanged; keep B2 held until owner gate and APT critical path are accepted |

### 8.1 Representative row and enclosing-artifact traces

These traces use symbolic exact references because no production fixture was
opened or generated. `A0/r17` denotes one Citlali baseline occurrence and its
artifact-local row; `Mobs/r4` denotes the observation-specific output row.
They describe what current code does and the proof that the canonical repair
must add.

#### Path A — Citlali Beammap → TolProj → Citlali science

| Hop for representative row | Current row/artifact behavior | Required canonical evidence |
| --- | --- | --- |
| raw channel → `A0/r17` | `r17.uid=17` because it is concatenated row 17; `nw`, `array`, first-sweep `tone_freq`, and network-local `kids_tone` are assigned by segment/order; fit/calibration values overwrite the row; ECSV `A0` has no digest/occurrence/raw relation | Citlali issues `A0` occurrence, semantic/envelope digests, raw manifest `R0`, exact `r17`, and bijection `r17 ↔ (R0,nw,channel)`; one raw channel → one row |
| `A0/r17` → TolProj seed row | library byte-copies the table and later prepends a source-row `det_id`; path/obsnum select the seed | verify and preserve `A0` occurrence/digests and exact `r17`; selection event names source without changing its identity |
| target KMP channel → TolProj target row | target artifact is network-sorted; `det_id` and `kids_tone` come from order; current matching selects zero or one seed row under unchanged shift/gate/good-first policy | issue target manifest `RT`; target row key `rt4`; record `rt4 → A0/r17` or explicit unmatched plus ambiguity evidence; unused seed rows are 1→0 records |
| seed/target → `Mobs/r4` | one output row per target row; seed `uid` copied on match or filled `0.0`; target readout fields win; seed science fields attach; `tone_freq` overwritten; unused seed rows disappear | issue new `Mobs` occurrence/digests; unique output-local `r4`; relation references `A0/r17`; declare every copied/added/dropped/overwritten field and separate source/application/presentation orders |
| matched → calibrated `Cobs/r4` | current copy preserves row/order and overwrites only `flxscale`, with path/factor metadata | verify `Mobs`; issue new `Cobs`; preserve `r4` for proven 1:1 transform; record brackets/reference/software/config digests and only-`flxscale` mutation |
| handoff → Citlali detector column | TolProj/TolTECA pass path+obs selector; Citlali filters networks, drops extensions, overwrites tone frequency, and applies retained row position to raw column position | TolTECA carries `Cobs` occurrence/digests/`RT`; Citlali recomputes them, derives actual `RT`, verifies relation/statuses, builds explicit `r4 → detector-column` permutation, and fails before calibration/geometry on mismatch |

Current recording for this path is obsnum/path, selected seed path, some KMP
metadata, calibration factors, and configuration/software kit evidence. Tune is
chosen but not bound as canonical identity; network/tone are values rather than
verified scoped keys; source/output digests, complete mapping, and consumer
recomputation are absent. The downstream consumer trusts the declared path and
shape.

#### Path B — Citlali Beammap → TolAPT → later-use product

| Hop for representative row | Current row/artifact behavior | Required canonical evidence |
| --- | --- | --- |
| `A0/r17` → canonical measured row | measured reader first drops flagged rows, generates order label such as `m000012`, casts UID to float, then drops nonfinite rows and most columns | verify `A0`; carry exact `A0/r17`; if excluded, persist `filtered_flagged`/`filtered_invalid` status rather than erase it; local `m...` remains presentation-only |
| measured row → dewarp/geometry/FG row | intended 1:1 copies overwrite x/y or grouping attributes; local indices and IDs remain the practical link; no output digests | each stage issues occurrence/digest/event, preserves `A0/r17` reference, declares overwritten/added fields, and records application/presentation order |
| measured row ↔ design candidate(s) | transient measured row has 0..N candidates in array/network/FG blocks; final current policy selects at most one design row; unmatched/design status tables are separate | mapping event names measured occurrence/row and design occurrence/row, candidate/ambiguity evidence, selected/unmatched status and supersession; no policy change |
| selected row → hero or consistency product | consistency traverses float-normalized UIDs and external matched APTs; hero subsets the raw source row, overwrites x/y, forces `flag=0`, and adds hero columns | verify every source digest; source-row lineage points to `A0/r17`; hero is a new occurrence/digest with explicit position/flag mutation; cross-observation consensus remains empirical mapping evidence |
| later Citlali/TolProj offer | current hero is source-shaped but has no canonical output digest; beammap prior is aggregate slot product | any offered APT conforms to Citlali canonical schema and target binding; soft prior remains a separately typed non-APT product |

Current TolAPT run manifests record raw paths, sizes, mtimes, SHA-256, and
resolved configuration, but later readers do not reverify the hash and most
outputs lack output/software-chain digests. Observation is optional/inferred;
network exists as a matching block; tune/readout binding and consumer
recomputation are absent.

#### Path C — Citlali Beammap → `toltec_beammap` → derived/calibration product

| Hop for representative row | Current row/artifact behavior | Required canonical evidence |
| --- | --- | --- |
| `A0/r17` + quick fit → `_psf` row | source selected by filename; quick row may join by UID, det_id-as-UID, or array-scoped kids_tone with ≥90% overlap; duplicate dictionaries last-win; source row order/cardinality normally preserved | verify `A0`; quick-fit artifact references exact raw/readout row; explicit update edge to `A1/r17`; duplicate/ambiguous mappings fail; every unmatched/extra row has status |
| update values | flags, FWHM/PSF diagnostics, and angle change; current angle writes degrees into radian field | issue `A1` occurrence/digests; declare field mutations and retain valid row relation; enforce canonical radians; source/application/presentation orders recorded separately |
| calibrator refresh | reads `A1`, changes flux metadata, `flxscale`, and `sens`, and overwrites the same path | issue immutable `A2` occurrence/digests; preserve `r17` for verified 1:1 transform; record model/service/config/software inputs and exact changed fields |
| audit/calibration outputs | power/calibration and robustness tables/reports are separate products; robustness joins by UID across reductions | bind every independent product to exact source/output occurrences/rows and define its own schema; it does not become canonical APT identity |

Current `toltec_beammap` recording is primarily paths, basenames, metadata,
mtime, and scientific values. Observation/source metadata is read but no target
tune/readout manifest, source/output digest, software revision chain, or
consumer recomputation exists.

## 9. Current defects and ambiguities

1. `uid` is type-erased, occurrence-local, unnamespaced, and not bound to raw
   channels; placeholder/duplicate/fallback behavior makes joins ambiguous.
2. Raw interface/channel order, seed row number, `det_id`, `kids_tone`,
   transient indices, application position, and presentation position are
   repeatedly overloaded as identity.
3. Citlali accepts external APTs by columns/frame/count/contiguity and then
   applies them positionally. A forged or wrong observation APT with the same
   shape can pass.
4. Citlali drops unknown columns on ingestion, preventing end-to-end
   preservation of a future lineage extension unless the reader is repaired.
5. Citlali fit-report collisions can overwrite base structural/scientific
   columns silently.
6. TolProj matching ignores UID, keys seed rows by position, discards unused
   rows and alternate candidates, and fills unmatched seed-derived values with
   synthetic zeros, including duplicate UID placeholders.
7. TolProj selection/calibration/handoff use paths, obsnums, mtimes/status, and
   mutable `project.yaml`, not a verified artifact chain.
8. TolAPT measured IDs and indices are order-derived; UID conversion through
   float is lossy; stored source hashes are not reverified; output artifacts
   generally lack digests and supersession.
9. `toltec_beammap` accepts ambiguous UID/det_id/kids_tone heuristics,
   dictionary last-wins behavior, and mtime reuse. It writes degrees into the
   radian `angle` field and refreshes calibration in place.
10. TolTECA legacy compatibility can rebuild UID from row order, insert
    defaults for absent scientific fields, discard metadata, or synthesize an
    all-`-1` UID APT. These operations cannot appear in a strict canonical
    path.
11. No current consumer recomputes a canonical APT digest and verifies the
    declared observation/readout binding against actual input data.
12. CAL and ALIGN have distinct scientific scopes. An APT repair is a
    prerequisite to trusted calibrated aligned data but does not close either
    task's independent scientific obligations.

## 10. Falsification and counterexample disposition

| Required case | Current result | Canonical acceptance result |
| --- | --- | --- |
| arbitrary row reorder | changes generated IDs/indices and can break positional application | semantic row/mapping refs invariant; physical/order change never changes detector meaning |
| identical content, distinct artifacts | paths/runs may differ; no explicit occurrence/content separation | same semantic digest allowed; distinct occurrences and histories retained |
| duplicate/missing ID | inconsistently warn, last-win, pass, or fail only on design side | publication/admission fails for canonical row keys; invalid rows still get unique row keys/status |
| repeated tone in different networks/observations | some network blocking; other joins use array/tone or local IDs | raw-manifest-scoped network/channel refs remain distinct |
| unmatched observation tone | TolProj flag/zero placeholders; TolAPT separate table without complete reason | explicit target row and `unmatched` status/reason; no seed identity fabricated |
| unused seed row | disappears in TolProj; TolAPT active/inactive handling split | explicit `unused`/excluded source disposition in mapping relation |
| ambiguous/equal-quality match | TolProj alternatives disappear; TolAPT records review evidence but selects one | retain exact current policy result plus ambiguity evidence; do not redefine selection as identity |
| detector in multiple tunes/observations | common-UID consensus is external/empirical | distinct measurement occurrences; persistent equality only from an authoritative lifecycle |
| changed fitted/calibration values | often same paths/IDs; one in-place update | new occurrence/digest/event with stable proven row/readout mapping |
| design reassignment | diagnostic conflict/consensus only | new design mapping supersedes old; measured/readout reference unchanged |
| stale/incorrect digest | TolAPT/TolProj stored hashes are not end-to-end admission checks | consumer recomputation fails before science |
| forged observation APT | same shape/order can pass Citlali | actual raw manifest mismatch fails before state mutation/calibration |
| old APT missing new fields | ad hoc fill/cast/rebuild | explicit legacy profile; no invented identity/history; admissibility reduced or rejected |
| round trip through all components | columns/IDs/metadata can be dropped, cast, reordered, or overwritten | source/output occurrences, complete mapping, exact permitted changes, and recomputed digests survive |
| source/application/presentation order differ | typically conflated or implicit | three named row-reference sequences where needed; no ordinal is identity |

No conceptual case authorizes a new Hungarian/global assignment, tie policy,
threshold, reassignment policy, or altered selected pair.

## 11. Backward compatibility and production admissibility

| Profile | Defensible guarantee | Permitted use |
| --- | --- | --- |
| `legacy-structural` | recognized ECSV and minimum current fields only | inspection/migration evidence; no production identity claim |
| `legacy-local-row` | exact unique integral UID supports a bounded join within that exact artifact | local diagnostics or migration; no cross-artifact persistence |
| `legacy-externally-bound` | an authorized migration event independently proves raw manifest and row permutation from retained evidence | explicitly reduced-guarantee application only if Grant approves the mode |
| canonical v1 baseline | producer schema/digest/occurrence/row/readout creation evidence valid | canonical source and downstream input |
| canonical v1 observation-specific | baseline lineage plus verified target manifest and complete realized mapping valid | eligible for strict Citlali admission |

Migration may normalize units/types or add facts deterministically derivable
from preserved evidence, and must issue a new occurrence/event/digest. It may
not invent a persistent detector ID, original artifact history, source mapping,
or observation binding. Legacy UID values above the exact binary64 integer
range, duplicates, missing values, or unknown casts cannot be silently
repaired. A path, filename, obsnum, mtime, or copied content is not proof.

Until a production profile and evidence gate are accepted, historical APTs are
`existing_use_only` or nonproduction according to owner decision. Reduced
assurance must be visible in every derived product. No downstream component
may upgrade assurance merely by adding a sidecar.

## 12. Canonical producer-first implementation decomposition

### Phase 1 — Citlali canonical baseline producer/schema

Owner: Citlali. Define the canonical v1 schema/envelope/digest, artifact-local
row/readout contract, protected extensions, post-write verification, product
validator, and producer evidence. No downstream or matching change.

### Phase 2 — TolProj conformance and observation-specific mapping

Owner: TolProj. Consume and verify the producer contract; bind the actual
target readout; issue a new occurrence and complete realized relation;
preserve the current matcher policy and exact selected pairs; use typed missing
states; make flux calibration a new verified transformation.

### Phase 3 — TolTECA handoff and Citlali strict ingestion

Owners: TolTECA for resolved selection/handoff; Citlali for admission. Carry
occurrence/digest/readout/mapping refs; remove UID rebuilding/default synthesis
from strict mode; derive the expected manifest from actual raw inputs;
recompute, permute explicitly, and fail before RTC/map geometry. Emit applied
identity/binding in scientific products.

### Phase 4 — critical-path end-to-end integration and ALIGN evidence gate

Owners: Citlali, TolProj, and TolTECA. Exercise the canonical producer,
observation transformer, resolved handoff, and strict consumer together. A
valid canonical observation APT must reach scientific application with one
verified applied digest/binding; stale, tampered, duplicate, reordered without
mapping, and forged/wrong-observation cases must fail before science. This is
the bounded evidence package Grant may use for an ALIGN B2 resumption decision.

### Conditional ALIGN B2 owner decision

ALIGN remains held through Phase 4. Grant may decide whether to resume B2 only
after accepting the critical-path evidence and confirming that the owner-
defined ALIGN input surface does not consume pending TolAPT or
`toltec_beammap` products. If it does consume them, the corresponding Phase-5
or Phase-6 conformance becomes an additional gate. APT work must not modify
temporal alignment algorithms or claim their scientific closure.

The remaining phases complete the broader ecosystem and do not move ahead of
the critical gate merely because they are numbered later.

### Phase 5 — TolAPT conformance

Owner: TolAPT. Preserve exact canonical measured refs, verify inputs, digest
outputs, make filtered/unmatched/invalid/superseded statuses explicit, and use
source-row lineage for hero products. Current measured/design matching policy,
Hungarian assignment, thresholds, ties, and selected pairs remain unchanged.

### Phase 6 — `toltec_beammap` conformance

Owner: `toltec_beammap`. Replace heuristic/duplicate joins with canonical row
relations, issue immutable update/calibration artifacts, declare all mutations,
correct the separately reviewed angle-unit defect, and bind independent audit
products to exact source occurrences.

### Phase 7 — TolAPT/`toltec_beammap` ecosystem integration and CAL boundary

Owners: Citlali, TolProj, TolTECA, TolAPT, `toltec_beammap`, and the owner-
defined CAL lane. Prove unchanged calibrated numerics for a valid explicit
permutation, fail mismatched input before calibration, and distinguish APT
binding evidence from physical flux/unit/response closure. Exercise all
remaining counterexamples and round trips without production data activation.

## 13. Critical path and prohibited production use

The minimum implementation that can unblock an owner decision on ALIGN B2 is:

1. accepted Citlali canonical producer/schema and reproducible baseline
   issuance;
2. TolProj preservation plus an explicit observation target/readout relation
   with frozen current selected pairs;
3. TolTECA resolved handoff of the exact occurrence/digest/binding;
4. Citlali fail-closed recomputation and actual-raw binding before scientific
   application; and
5. end-to-end success for a valid APT and pre-science failure for stale,
   tampered, reordered-without-mapping, duplicate, and forged/wrong-observation
   cases, with applied provenance in outputs.

Grant must explicitly accept that evidence and the owner-defined ALIGN gate.
TolAPT and `toltec_beammap` ecosystem completion may follow if ALIGN does not
consume their derived products, but their outputs remain prohibited from
claiming canonical round-trip or production eligibility until their respective
conformance phases pass. CAL physical calibration remains unclosed regardless
of APT identity success. Historical/reduced-profile APTs, candidate-sidecar
products, in-place calibrated artifacts, and any APT accepted only by shape/
path/order remain prohibited from new canonical production use.

## 14. Stopped TolProj proposal disposition

The candidate was compared only after sections 3–13 were derived from clean
sources. It consists of four unstaged tracked modifications at clean TolProj
HEAD `0fcd33ff9d805246a54d15d67751b762410f2e86`:

| Path | Worktree SHA-256 | Diff size vs HEAD | Disposition |
| --- | --- | --- | --- |
| `tolproj/utils.py` | `248ceccd2fd5ed91698f94e3123bfe745ca3d70b7521a7a0842e67344f13b2c6` | +619 / -0 | private canonicalization/artifact/component/seed identity: reject as authority; exact-type and strict-parser ideas defer to shared canonical implementation |
| `tolproj/legacy_scripts/make_matched_apt.py` | `3c3751745abc819b1239fffae28d965d913b297035e5588dca8fa7910a607866` | +713 / -77 | target facts and realized mapping fields are legitimate TolProj concerns; seed UID identity and `tolproj_*` canonical schema are obsolete/reject; implementation deferred and unaccepted |
| `tolproj/steps/match_apts.py` | `aac6fe9730b91eafc3dd5e90445ee7302a2cbaa72601855f2edc511b02f98fde` | +1500 / -34 | atomic verify/publish and mapping/order concepts are later-phase patterns; TolProj-private JSON identity sidecar and mutable project anchor are rejected as canonical production solution |
| `tests/test_make_matched_apt.py` | `395b772ee984b7f11c01d3a799d016d385502094a22be6a1351a508806c65597` | +1109 / -144 | retain only as audit/fixture/migration evidence; rewrite against the accepted producer contract before execution/acceptance |

The SHA-256 of stdout from
`git diff --binary HEAD -- tests/test_make_matched_apt.py tolproj/legacy_scripts/make_matched_apt.py tolproj/steps/match_apts.py tolproj/utils.py`
is
`4ce3c7e71ce5ad8ff2facf446748ec7c6a44af66c1c80c234a7e97fec17b057a`;
the diff is 3,941 insertions and 255 deletions. No untracked candidate file or
staged change exists. Before and after minimal inspection, every file hash,
the patch hash, branch, HEAD, and dirty status were identical.

Material classification:

| Candidate part | Classification | Reason |
| --- | --- | --- |
| `tolproj.canonical_identity.v1` and related component/artifact schemas | unnecessary/obsolete and reject | canonical semantics/canonicalization belong at the Citlali producer/shared authority, not TolProj |
| seed component keyed by seed `uid` plus issuer obsnum | reject | current UID lifecycle is unproved and producer occurrence/row refs must be consumed, not reconstructed |
| exact ECSV byte SHA and strict scalar handling | later-phase implementation pattern | useful transport/admission technique only after canonical semantic scope is producer-defined |
| KMP artifact facts and observation/network/channel scope | legitimate TolProj-owned observation-specific provenance | TolProj actually selects these target inputs; facts must conform to the shared raw-manifest contract |
| target-to-seed matched/unmatched edges, unused seed records, shift/separation/tolerance/method | legitimate TolProj-owned realized provenance | records current operation without authorizing policy changes; exact layout awaits canonical relation schema |
| source/application/presentation bindings | legitimate concept, later-phase work | required by the core; serialized names/digests remain unaccepted |
| `apt_<obs>_matched.identity.json` private sidecar | reject for production; retain only as audit/migration evidence | consumer does not own/verify this schema and it cannot replace producer semantics |
| `project.yaml` APT anchor | reject as identity authority; later locator role possible | mutable live plan may locate a verified artifact but cannot prove immutable lineage |
| atomic publication and stale/forged checks | later-phase work | desirable mechanism once it publishes producer-governed APT/relation artifacts |
| permutation, duplicate, ambiguity, large-integer, stale/forged, and order-separation test cases | retain as fixture/migration evidence | valuable falsification cases, but candidate expectations and matching equivalence were not executed or accepted |

No candidate implementation is incorporated into the canonical contract, and
no part is authorized for test, commit, transport, or production use.

### 14.1 Obsolete provisional work

The provisional TolProj-owned identity namespace/canonicalizer, UID-derived
seed identity, private identity sidecar, and `project.yaml` identity anchor are
superseded by the producer-first contract and should not be completed as an
independent production design. TolTECA row-ordinal UID rebuilding, default
scientific-field synthesis, and all-`-1` UID generation are likewise obsolete
for strict canonical mode. Downstream reconstruction of any schema, row
identity, raw binding, or digest that Citlali Beammap should have issued is
prohibited. Only the candidate's read-only fixtures/counterexamples and the
TolProj-owned observation/mapping provenance concepts survive for later
re-expression under the accepted canonical schema.

## 15. Audit-manager handoff

### 15.1 First recommended implementation task for Grant approval

Proposed package: `APT-PROD-001`
Title: Citlali Canonical Baseline APT v1 Producer Contract
Repository owner: Citlali
Effort: Ultra
Dependency: Grant resolves the gates in section 16 that affect serialization
and admissibility.

Exact scope ceiling:

- one Citlali-only implementation worktree at the accepted mainline base;
- canonical APT v1 model/schema, typed field/metadata envelope, semantic
  SHA-256 canonicalization, occurrence/event reference, artifact-local `uid`
  validation, raw manifest and `(network, channel)` relation, protected
  extension policy, Beammap writer/post-write re-read, executable product
  validator, focused producer tests, and producer documentation;
- reuse existing SHA-256, canonical encoding, version, atomic-output, and
  product-registry infrastructure where it satisfies the accepted contract;
- no external science-ingestion enforcement in this task except a read-only
  compatible parser/helper seam needed by producer self-validation.

Explicit exclusions:

- no TolProj, TolAPT, `toltec_beammap`, TolTECA, CAL, or ALIGN edit;
- no matcher, pair, shift, tolerance, tie, assignment, or reassignment change;
- no calibration/fitting numerical change, persistent detector-ID invention,
  historical migration, production reduction, or network/Unity activity;
- no TolProj sidecar adoption or downstream implementation launch.

Acceptance evidence:

1. complete machine-readable field/type/unit/nullability/authority/extension
   contract and normative metadata schema;
2. deterministic semantic digest with exact integer and float test vectors,
   row-order semantics documented and tested, and temporary ECSV re-read
   recomputation before publication;
3. unique artifact-local UID and bijective raw-manifest/network/channel mapping
   for every row, including invalid rows;
4. arbitrary physical row reorder, identical-content/distinct-occurrence,
   duplicate/missing/nonintegral/out-of-range key, repeated tone across
   networks, collision, stale digest, and serialization round-trip cases;
5. a current representative synthetic/focused Beammap producer fixture showing
   unchanged scientific fit/calibration values and a fully canonical APT;
6. product validator rejects missing/wrong schema, unit, key, digest, manifest,
   and protected-field collision; and
7. exact source/test manifest, commit, report, and explicit statement that no
   downstream or production acceptance is implied.

Stop conditions:

- the authoritative raw manifest/network/channel tuple cannot be derived from
  current raw inputs without owner/scientist direction;
- canonicalization/digest scope or occurrence semantics remain disputed;
- retaining `uid` as artifact-local key would require claiming unsupported
  persistence or silently repairing historical values;
- scientific fit/calibration values or current selected detector sets change;
- work expands into ingestion, matching, CAL, ALIGN, legacy migration, or any
  downstream repository; or
- required evidence would need Unity/production data or new authority.

At the stop, return the bounded implementation and evidence to Grant. Do not
automatically launch Phase 2.

### 15.2 Proposed follow-up tasks and dependency order

| Order | Proposed task | Repository owner(s) | Effort | Scope ceiling | Acceptance/stop summary |
| --- | --- | --- | --- | --- | --- |
| 1 | `APT-PROD-001` baseline producer | Citlali | Ultra | section 15.1 only | canonical output + falsification evidence; stop before downstream |
| 2 | `APT-TOLPROJ-001` observation conformance | TolProj | Ultra | verify baseline, target manifest, realized relation, calibrated-copy lineage; exact pairs frozen | counterexamples + pair-by-pair fixture equality; stop on policy/numeric change |
| 3 | `APT-INGEST-001` handoff/admission | TolTECA + Citlali in separately owned commits | Ultra | exact handoff, typed reader, raw-derived binding, explicit permutation, fail-closed outputs | valid path succeeds; stale/forged/wrong obs fails before science; stop on CAL/ALIGN expansion |
| 4 | `APT-CRIT-INT-001` producer→TolProj→TolTECA→Citlali critical integration | Citlali + TolProj + TolTECA | Ultra | focused critical-path evidence only | valid application plus pre-science falsification failures; no ecosystem/CAL closure |
| 5 | conditional ALIGN B2 resumption decision | Grant / ALIGN owner | owner decision | no automatic implementation | after Phase 4; add later consumers only if owner-defined ALIGN surface requires them |
| 6 | `APT-TOLAPT-001` independent conformance | TolAPT | High/Ultra | lossless measured refs, verified artifacts, mapping/status/output digest, hero lineage | current matching outputs fixed; no design-policy change |
| 7 | `APT-BEAMMAP-001` downstream conformance | `toltec_beammap` | High | canonical join/update/calibration, angle-unit repair as separately reviewed change, immutable outputs | round-trip and exact mutation evidence; stop on calibration-policy redesign |
| 8 | `APT-ECO-INT-001` ecosystem/CAL boundary | Citlali + all product owners | Ultra | TolAPT/`toltec_beammap` round trip and CAL binding regression | all remaining falsification cases; no claim of CAL physical closure |

## 16. Owner decisions required before repair

Grant must decide or assign authority for:

1. the exact minimal raw readout manifest tuple and whether tune identity beyond
   observation/subobservation/scan/interface/channel is available and required;
2. acceptance of existing `uid` as v1 artifact-local row key only, including
   exact integer range/type and whether another existing exact field qualifies;
3. occurrence encoding: explicit opaque value or derivation from creation event
   plus output role;
4. semantic digest canonicalization/scope, physical-order treatment, and
   whether a separate byte transport hash is mandatory;
5. canonical relation serialization: embedded metadata versus an atomically
   bound producer-governed relation artifact;
6. protected/extension fields and the authoritative source of `fg`, `pg`,
   `ori`, and `loc`;
7. whether an authoritative persistent measured detector/resonator identity
   exists; absent proof, it remains omitted;
8. the design artifact/row namespace owner and lifecycle;
9. legacy profile names and which, if any, reduced-assurance profile is allowed
   for `existing_use_only` science;
10. the authoritative CAL task definition and the boundary between APT binding
    evidence and physical calibration closure; and
11. the exact owner-defined ALIGN B2 acceptance gate after the APT critical
    path is implemented.

## 17. Source and test evidence

The complete independent-core source manifest is frozen in the independent
core, section 15. The final-phase implementation/test manifest below records
the additional clean-current evidence inspected before candidate exposure. Git
blob IDs identify exact file contents.

### 17.1 TolProj clean current, revision `0fcd33ff9d805246a54d15d67751b762410f2e86`

| Source/test | Blob |
| --- | --- |
| `tolproj/utils.py` | `c3da9dbe238c5827c2d11196e23d4c975d9957cd` |
| `tolproj/steps/match_apts.py` | `72ab5751c0c2e12ebdf8976410c08c18ea3a4641` |
| `tolproj/legacy_scripts/make_matched_apt.py` | `83d27a671776342c85f4b16d9b2afb5b75a9faa5` |
| `tolproj/steps/calibrate_flxscale.py` | `7fe5d60087fd9a6cc2cb94aaf2b887585f0fcc84` |
| `tolproj/steps/setup_science_reductions.py` | `d73a9277cad7fb5362f2331516c44adab6cdc705` |
| `tolproj/steps/setup_pointing_reductions.py` | `38299af018f7adedce5bdbe1bbbb8fea77b484bd` |
| `tolproj/steps/common.py` | `9aac1853bb8bb0b264946157ad63daa4d457f173` |
| `tolproj/reduction_config.py` | `6af4abfc993bceb0568beed318ef1aab697f5834` |
| `tolproj/refactor_config.py` | `ca9f2d0123a64fcc8e580cd65669c882c5a9f31b` |
| `tolproj/templates/apt_Neptune_137389_60835.46_260123.ecsv` | `59425f962a0631fc90a593d2572294244c900d29` |
| `tests/test_make_matched_apt.py` | `7f3aba10b965512bd6bbcf4f2334434db0cb4450` |
| `tests/test_flux_calibration.py` | `8ac45570f5443a5d0008a5f466e5b732fc780aec` |
| `tests/test_science_scannums.py` | `bee799265346913eed006aabe8ff437df3bd44d2` |
| `tests/test_refactor_config.py` | `a742297cb77497d4b0a8ab96e6b4a8ddaba54b82` |
| `tests/test_beammap_pointings.py` | `eaea5ba99ae111f9dd9eefb17d8038b179f977b1` |

Existing tests cover current selection, unmatched/missing network behavior,
duplicate selected seed use, flux factors, and handoff path choice. They do not
cover reorder invariance, exact UID type/precision, complete ambiguity/unused
lineage, digests/tamper, or forged observation binding. They were not run.

### 17.2 TolAPT supported surface, revision `3a07cc551faf903da3e1d49d7d3a6b20381afc3d`

| Source/test | Blob |
| --- | --- |
| `src/tolapt/artifacts.py` | `2f75d3b472416d4dd601773ea3d3affe834ee82c` |
| `src/tolapt/io/design_reader.py` | `02ed45947267a87a10ba3e735d37688332db6e51` |
| `src/tolapt/io/measured_reader.py` | `6086c0b7f5c9a1045925040fa6dfa688f7ff2b67` |
| `src/tolapt/pipeline.py` | `f7f052047ed6ff69cb121dd263c5cd8980e01ff1` |
| `src/tolapt/dewarp/apply.py` | `424ea3cb9bcb8e657744f5d56882b433ba7ea377` |
| `src/tolapt/frequency_grouping.py` | `7f19ce1d24561e8726e58d7a79de5fb891364e77` |
| `src/tolapt/geometry.py` | `b182f735720153642c3436f1dca44246ef68c856` |
| `src/tolapt/stage_b_affine/pipeline.py` | `31439ca946801be615fd1582684eb2e0bb6df695` |
| `src/tolapt/matching.py` | `392c5b1dbcdbfa56cfa162d10829ae53d1f9dad7` |
| `src/tolapt/sanity.py` | `938dd95c52b76023d5bfb6a0af42817547f9286e` |
| `src/tolapt/tone_match_consistency.py` | `272ba9ba63ef42a8066aea50f41f98a7ad972d11` |
| `src/tolapt/hero_apt.py` | `8dda134af3dc20222b523a8ad84356c3d62016b3` |
| `src/tolapt/hero_overlay.py` | `80f110e5cb54f01c302421d16ed7db52eb5d4dd8` |
| `src/tolapt/beammap_priors.py` | `b5dac84aee3189e88b55c6a93c8f15104f489f54` |
| `tests/test_io_readers.py` | `70b95da175b9931a4bb838785800f53c4f42f108` |
| `tests/test_artifacts.py` | `512665e5b1d4845e001ea8ab89859a9d1dfa9d70` |
| `tests/test_matching.py` | `cf924de2bf5c25c466a4939074bac4c433f16029` |
| `tests/test_sanity.py` | `05feb97bd140744e1681c1888f9095d6604db6ac` |
| `tests/test_tone_match_consistency.py` | `37d4ac2a01db396ff610019804af9bb62d454ed3` |
| `tests/test_hero_apt.py` | `fa97ff9141a4498a7de54fecd56281a621713975` |
| `tests/test_hero_overlay.py` | `dc16a886b66e826f59d32e400d952a4b1aee5014` |
| `tests/test_beammap_priors.py` | `2754473267162938f834302b1da1097acbc5abd0` |

The inspected tests establish much of TolAPT's current matching/status science,
but not canonical input re-verification, row-order/precision invariance,
output digests, forged binding, supersession, or end-to-end round trip. They
were not run.

### 17.3 `toltec_beammap`, revision `958a2a15f43189846a24556a63ef908da789c7b8`

| Source/test | Blob |
| --- | --- |
| `src/toltec_beammap/apt_modifier.py` | `06332638a1581b93894736c99bd4d1e0db01854f` |
| `src/toltec_beammap/toltec_beammap.py` | `641461898bdb342b3955a097666e76ba82c23183` |
| `src/toltec_beammap/pipeline/process.py` | `dd427ff87891d0b82e7d18541d2255c27049d99f` |
| `src/toltec_beammap/pipeline/apt_robustness_audit.py` | `2f0dd9d39d41fa134b6cb0b24c4d533fd45efa0e` |
| `src/planetflux/apt.py` | `451673727dcb2ea90203bbed499995695567d8f5` |
| `tests/test_regressions.py` | `6bd484d1e89036749ce6c86f0c278f420f79a038` |

Existing tests cover current partial-array update, tone-to-UID fallback,
mtime regeneration, flagging, fit preference, and in-place flux refresh. They
do not cover duplicate fail-closed behavior, canonical digests, exact mapping,
angle-unit conformance, immutable lineage, or round trip. They were not run.

### 17.4 TolTECA reduction authority

At locally stored `origin/main`
`2791e6a1e6349ad1d3ac549a648f41cbc51b98c7`,
`tolteca/reduce/engines/citlali.py` is blob
`b963d6531ec7e0e3df893f88f17962476348da9f`;
`tolteca/utils/runtime_context.py` is blob
`a0b2357cad307dbd2143a177eb13e7f4c630e372`; and the workdir contract is
blob `a62d09ac440ae8fb1b1fad0c425a878262c9e7ba`. No direct APT identity test was
found or run. The checked-out `v3.x` ref contains no corresponding reduction
engine surface, so no v3 APT handoff behavior was inferred.

### 17.5 Citlali ingestion, CAL, and ALIGN final-phase evidence

All are at `46ad23888a40f5102cdfd50c06e49a549bdf8a20` and supplement the full
producer manifest in the core.

| Source/test | Blob |
| --- | --- |
| `include/citlali/core/engine/io.h` | `10caf2c72590df37d5afaa9edd287c3d03a21bf1` |
| `include/citlali/core/engine/detail/rawobs_collection_impl.h` | `8e0e6f0d02754063fe643ccba559985d9a95f4e6` |
| `src/citlali/core/engine/calib.cpp` | `c48adfef12ac243e6457d75be24a2fc6abc471c6` |
| `include/citlali/core/utils/ecsv_io.h` | `fca63b60738d194f807665f91b15bf1a6dc8cab2` |
| `include/citlali/core/engine/detail/todproc_raw_input_impl.h` | `b160fe50de02dc50889236d88ae062bbc43b5ca0` |
| `include/citlali/core/timestream/rtc/calibrate.h` | `e20f2355fa7ce8f24fe838cef2bbf5c72dff0fa3` |
| `include/citlali/core/pipeline/flxscale_correction.h` | `50f6dad2ed011684646da9f6ff820933c68df481` |
| `include/citlali/core/engine/detail/todproc_map_geometry_impl.h` | `debdbc90f99a0ddbd1719f21e5c5f7890508ee19` |
| `include/citlali/core/engine/detail/todproc_alignment_impl.h` | `2a2cabd505b14256fcae6d71b93005b528865bda` |
| `include/citlali/core/pipeline/timestream_alignment_helpers.h` | `aa58b0488f69c3e00d5422cfa10567b2bbe53d9a` |
| `tests/test_calib_apt_filtering.cpp` | `7a7938ad30d29270f27fd265efba31acca86118e` |
| `tests/test_config_scaffold.cpp` | `291b9cc94e21810ecff07651f03323bd6d3998f4` |
| `tests/test_session_failure_boundaries.cpp` | `75d06cf43d69d8e523a754dd04e0369cd0123b4a` |
| `validation/product_contracts.json` | `f335052e42a9331e0ded901790457fa9fe244dcd` |
| `include/citlali/core/utils/sha256.h` | `9f54343898536737b7db555036e85e4717fd9c3a` |
| `include/citlali/core/mapmaking/science_map_contract.h` | `2920a17694e5d4b4501a798ff243c5bce1d0a5ef` |
| `include/citlali/core/pipeline/config_source_manifest.h` | `bdf2e371cad20c3eb97fa2bb26330229c1f72820` |
| `include/citlali/core/pipeline/noise_execution_plan.h` | `5d7a29f4d9748b235f92676dbe8ad36b3949abe8` |
| `include/citlali/core/pipeline/noise_provenance.h` | `cbd1d2b356ceb6b5c081f34d8c378c085b037be7` |
| `include/citlali/core/pipeline/beammap_provenance.h` | `97669d812f524cec47b5d462b41d26ecd6ea00a3` |
| `include/citlali/core/pipeline/beammap_provenance_lifecycle.h` | `56b4acf796b09bf27bd2f564093089bc3a229f5c` |
| `include/citlali/core/pipeline/beammap_provenance_serialization.h` | `2cf4955149b2d12ba2ae0051594ee46757fce7c9` |
| `include/citlali/core/pipeline/atomic_yaml_output.h` | `c085ce39c23312e5d89f7ed1ec35c7090382bc0c` |
| `include/citlali/core/pipeline/product_index_file.h` | `2d2ce7c9ee3ebc0430883ac2c37aeb25419d2a08` |

Current tests cover structural filtering and temporal alignment failure
boundaries, not APT identity/digest/readout binding. They were not run.

## 18. Final stop

This report completes the documentation-only audit and proposed decomposition.
It does not accept an implementation, release ALIGN B2, close CAL, upgrade a
historical APT, or authorize production. The next action, if any, is Grant's
explicit decision on section 16 and approval or rejection of the bounded
`APT-PROD-001` task. No implementation task may start automatically.
