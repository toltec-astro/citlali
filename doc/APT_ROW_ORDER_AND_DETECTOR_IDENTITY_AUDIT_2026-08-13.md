# APT Row Order And Detector Identity Audit

Date: 2026-08-13

Status: final owner-authorized read-only audit checkpoint

Citlali basis: `origin/codex/repair-sci-cal-001-successor-8` at
`7cdc4152931f3d42f11f26ef62649fb755e1b553`

Audit branch: `codex/audit-apt-row-identity-20260813`

Implementation authorization: none

## 1. Decision

Physical APT table order is **not** detector identity and must not be used as
detector identity at any TolTEC software boundary.

The durable production rule is:

> Every boundary must name the scope of its stable key and carry an explicit,
> one-to-one mapping to the next scope. Presentation order, source-row order,
> detector-application order, and canonical-digest order are separate facts.

A sort order alone cannot repair the current boundary. Sorting may make output
deterministic, but it cannot establish which detector one row represents.

The audit confirmed production-blocking identity defects in each of these
owners:

- `toltec_beammap` can infer an APT UID from numeric namespace overlap or an
  array-wide local-tone dictionary and can silently apply a fit to the wrong
  row.
- TolAPT can generate measured IDs from presentation position, select an
  arbitrary exact-tie assignment, and use stale `meas_idx` values to bind a
  match to the wrong measured UID.
- TolProj persists seed presentation indices and can select exact-tone ties by
  presentation order; its refresh path can create two APT entries for one
  observation.
- TolTECA v2 can choose the last applicable APT and, on its discovered-APT
  compatibility path, replace every UID with the output row ordinal while
  deleting explicit mappings and provenance.
- Citlali's production C++ ingestion correctly joins by observation-local
  `(network, absolute tone frequency)`, but its Python v4 validator re-infers
  local-tone identity from source presentation order and rejects a legitimate
  same-network permutation. Its duplicate-tone flagger is also presentation-
  and network-order dependent.

The exact current Citlali candidate therefore must not be treated as resolving
APT row identity across the full production boundary. No production
authorization is made by this audit.

No owner decision about a universal or cross-campaign physical detector ID is
needed to repair these defects. This report deliberately does **not** promote
`uid`, `det_id`, `common_uid`, a local-tone index, or any row number to that
role. All required repairs can use scoped artifact and observation identities.

## 2. Repository and authority checkpoint

This was recorded before the audit branch, tests, synthetic fixtures, or this
document were created. Sibling repositories remained read-only throughout.

| Repository / authority | Branch or ref | Commit | Tree | Initial state |
| --- | --- | --- | --- | --- |
| `citlali-refactor` | requested `origin/codex/repair-sci-cal-001-successor-8` | `7cdc4152931f3d42f11f26ef62649fb755e1b553` | `ebc32f152b9400e00c994c0a78076953145d83a2` | clean; audit branch later created at the same commit |
| `tolapt` | live `codex/rework-foundation` | `3a07cc551faf903da3e1d49d7d3a6b20381afc3d` | `8e0e514982dcef92eff547b59e9752f1b185704c` | clean |
| `toltec_beammap` | local `main` | `958a2a15f43189846a24556a63ef908da789c7b8` | `c02b631a56554bdd35488d0d4dd7a7cdd0cf2fff` | tracked-clean, one local guidance commit ahead of `origin/main`, three preserved untracked scripts |
| `TolProj` | `main` | `0fcd33ff9d805246a54d15d67751b762410f2e86` | `6647092b4ca4376f8467278fb38e6ea3712fb878` | clean, one commit ahead of `origin/main` |
| TolTECA checkout, not v2 authority | `v3.x` | `8d05ecde7c116d52b7a80a84d21e0ade367f163a` | `04f44c9acdad0c91b572b857fec23a40c03aa880` | clean; not used as v2 implementation evidence |
| user-selected TolTECA v2 authority | local `origin/main` object | `2791e6a1e6349ad1d3ac549a648f41cbc51b98c7` | `3f3e4b8136bf5528b203cb3bb8b474233bb27a85` | inspected from the exact Git object; no checkout mutation |

The preserved, pre-existing `toltec_beammap` files were excluded from evidence:

- `scripts/audit_flux_ratio_calculation.py`
- `scripts/fit_a1400_correction.py`
- `scripts/plot_flux_ratio_vs_elevation.py`

The local Beammap commit `958a2a1` is repository guidance/metadata. It was used
only for operating constraints and routing, not as scientific or product-
contract evidence.

### 2.1 Live authority and precedence

The applicable repository-local instructions and live authorities were read in
full before substantive work:

- Citlali: repository `AGENTS.md`, `doc/REFACTOR_STATUS.md`,
  `doc/ARCHITECTURE.md`, `doc/SCIENTIFIC_CONVENTIONS.md`, the ADR index, the
  live Phase 5/build-integration authorities, and the frozen F009-B audit and
  acceptance evidence at their exact candidate ancestry.
- TolAPT: `AGENTS.md`, `docs/STATUS.md`, `README.md`,
  `docs/output_contract.md`, and the applicable hero release/application
  contracts. `docs/STATUS.md` selects `codex/rework-foundation`; supported
  runtime is `src/tolapt`; `reference/legacy` is historical.
- `toltec_beammap`: `AGENTS.md`, `docs/STATUS.md`, and `README.md`, with the
  local-guidance limitation above.
- TolProj: `AGENTS.md`, `docs/STATUS.md`, `README.md`,
  `docs/WORKFLOW_V0_2.md`, `docs/CITLALI_REFACTOR_CONFIG.md`, and the scoped
  hero trial document. Live `STATUS` and workflow contracts outrank dated trial
  evidence.
- TolTECA v2: the exact user-selected `origin/main` implementation and its
  numbered-workdir documentation. That object contains no `AGENTS.md`.
- Shared TolTEC context: the `toltec-codex` TolTEC context skill's repository,
  authority, glossary, scientific-conventions, software-boundary, APT/data-
  product, and status-routing references.

No incompatible live authorities were found. Two apparent ambiguities are
resolved without a scientific choice:

1. `toltec_beammap` local documentation does not establish equality between a
   numeric FITS `det_N`/EXTNAME coordinate and `BEAMMAP.UID`. Citlali's live
   convention explicitly calls `det_N` a map slot linked to, but not equal to,
   detector UID (`doc/SCIENTIFIC_CONVENTIONS.md:79-92`). Consumers must preserve
   both or require a producer mapping; this report does not choose one as a
   cross-campaign identity.
2. Citlali explicitly leaves future UID lifetime unresolved
   (`doc/SCIENTIFIC_CONVENTIONS.md:710-717`). The present repair contract scopes
   UID to its issuing artifact/package and therefore does not answer that
   scientific question by accident.

### 2.2 Scope ceiling, prohibitions, and stop conditions

The only authorized write is this file:

`doc/APT_ROW_ORDER_AND_DETECTOR_IDENTITY_AUDIT_2026-08-13.md`

No application, configuration, test, build-system, product-contract, or sibling
repository file was edited. No push, merge, Unity access, reduction, network
access, production-data claim, production authorization, or repair launch was
performed.

The audit would have stopped for owner input if evidence required choosing a
cross-campaign physical identity, if two live repository authorities required
incompatible behavior, or if the source/test/artifact path ceiling had to
expand. None occurred.

## 3. Evidence labels and limits

Findings use these labels:

- **Source-confirmed**: direct behavior in the exact inspected source.
- **Executable-confirmed**: exercised with repository tests or a small
  production-shaped synthetic fixture against the exact source/commit.
- **Contract gap**: required information is absent; no numerical behavior is
  inferred.
- **Historical**: useful evidence but not current production authority.
- **Unavailable**: could not be established without forbidden production data
  or external execution.

A complete production-data Beammap -> TolAPT -> selection/copy -> Citlali round
trip was **unavailable**. This report does not relabel independent synthetic
stage tests as a production round trip.

## 4. Normative terminology

Raw integers are not interchangeable identities. The same spelling has
different meanings at different boundaries.

| Term | Meaning | Kind | Permitted authority |
| --- | --- | --- | --- |
| design input source row | Physical row in one design serialization | positional provenance | Only `(design artifact, source row)`; never detector identity |
| measured input source row | Physical row in one measured serialization before filtering | positional provenance | Only `(measured artifact, source row)` |
| selected/matched output row | Physical row in a derived APT/table | presentation or declared application coordinate | Not stable without an explicit mapping |
| Citlali detector/readout row | Column of the admitted raw detector axis, ordered by admitted raw artifacts/networks and local detector column | application order | Meaningful for aligned numerical arrays within that observation |
| array ID | Band/topological identity carried by the APT | designed/instrument metadata | Validated value; not container index |
| network ID / `nw` | TolTEC readout-network identity | observational/readout metadata | Explicit ID within observation; never inferred from container position |
| network index | Dense position in a run-local collection | positional | Internal only |
| local-tone index / `kids_tone` | Detector/tone coordinate within one network/KMP artifact | positional readout coordinate | Meaningful only with network and issuing observation/artifact |
| tone frequency | Absolute measured readout frequency in an observation, or separately a design frequency in a design table | observational or designed value, according to producer | Matching evidence or an observation-local unique join component; not physical identity |
| TolAPT `meas_idx` | Row locator in the exact canonical measured table used by one run | positional | Must be checked against explicit measured ID; not identity |
| TolAPT `design_idx` | Row locator in the exact canonical design table used by one run | positional | Must be checked against explicit design ID; not identity |
| source index | Row locator in a named source serialization | provenance | Requires exact source artifact identity |
| `match_id` | Locator assigned to a match record in one TolAPT run | inferred/run-local positional label | A match edge is identified by its explicit endpoints, not this number alone |
| design `det_id` | Producer-defined design-row name in one design artifact | designed artifact-scoped ID | Stable only in its design artifact/version unless a stronger producer contract says otherwise |
| measured `det_id` | Currently often generated by TolAPT from presentation position | positional in current implementation | Must not be treated as durable until generated from/persisted with an explicit measured key |
| TolProj output `det_id` | Target measured/readout application index | positional application coordinate | Observation-local only |
| TolProj `det_id_right` | Seed/source presentation index created by its matcher | positional provenance | Exact seed serialization only; sign currently also denotes matchedness |
| Beammap EXTNAME `det_id` / `det_N` | Beammap container/HDU map slot | product-local positional coordinate | Requires Beammap product identity and an explicit slot map |
| `BEAMMAP.UID` | UID value written in a detector HDU header | product/APT-scoped tag | Equality with EXTNAME is not assumed |
| APT `uid` | Producer-issued detector tag used for joins | artifact/package scoped | Require exact issuing APT/package and uniqueness; never promote globally |
| `citlali_uid` | Beammap tool's inferred reference to a selected Citlali fit-QC/APT row | inferred artifact-scoped mapping | Valid only with exact fit-QC and APT identities |
| `common_uid` | TolAPT release/reference-scoped cross-observation registration target | inferred release-scoped ID | Requires release ID and explicit `(obsnum, local_uid)` registration |
| stable physical detector identity | A campaign-independent physical KID identity | scientific identity | **Not defined by the inspected repositories; not chosen here** |

### 4.1 Four orders

1. **Presentation order** is the row sequence used to display or serialize a
   metadata/product table. It may change without changing membership.
2. **Source-provenance order** records where a row came from in an exact source
   artifact. It changes when that source serialization is permuted and is useful
   for reconstruction, but it is not detector identity.
3. **Detector-application order** is the ordered detector axis used by raw
   timestream, map, fit, PTC, calibration-factor, flag, and related numerical
   arrays. It is semantically meaningful within the admitted observation and
   must stay aligned across all arrays.
4. **Canonical-digest order** is a deterministic serialization rule applied
   after key validation. It exists to make a digest reproducible; it does not
   create identity.

Exact membership and ordered sequence are distinct. Products must declare which
one a digest covers.

## 5. Current end-to-end behavior

### 5.1 Citlali Beammap construction

Citlali's internal Beammap APT is constructed directly from the raw detector
inventory (`include/citlali/core/pipeline/rawobs_detector_inventory.h:55-138`).
Networks are represented explicitly; detector rows follow the raw numerical
axis. Its `uid` is assigned `0..N-1` (`:110-113`) and is therefore run-local and
positional. Its acquisition mode says
`internal_raw_network_local_row_v1` (`:122-134`).

`assign_beammap_kids_tone_indices` currently increments by adjacent row and
resets when `nw` increases
(`include/citlali/core/engine/detail/beammap_setup_state_impl.h:6-20`). This is
valid only because the internal inventory is admitted in network-contiguous
application order; it is not a standalone APT-order contract.

Detector FITS extensions are named from the map slot `det_<i>_`
(`include/citlali/core/pipeline/map_layer_name.h:29-31`). Every APT value,
including `uid`, is independently written as `BEAMMAP.<key>`
(`include/citlali/core/engine/detail/beammap_map_product_headers.h:23-40`). Thus
the producer already exposes slot and UID as separate fields.

The legitimate ordered objects here are the detector axes of maps, fit buffers,
and APT-aligned numerical vectors. A standalone ECSV's physical row number is
not an external detector identity.

### 5.2 `toltec_beammap` extraction and APT update

The main extractor parses the numeric detector coordinate from FITS EXTNAME,
sorts it, and warning-overwrites duplicates
(`src/toltec_beammap/toltec_beammap.py:464-527`). Interactive/QC paths instead
key on `BEAMMAP.UID` and silently overwrite duplicate UIDs
(`pipeline/interactive_review.py:67-94`,
`pipeline/beammap_qc_dash.py:870-970`). Conflicting synthetic FITS files proved
that the two paths can expose different detector sets.

`run_apt_update` (`toltec_beammap.py:1748-1919`) is the unsafe application
boundary. In the absence of an explicit `uid`, it:

1. treats at least 90% numeric overlap between quick-fit `det_id` and APT `uid`
   as namespace equality;
2. otherwise builds an array-wide `{kids_tone: uid}` dictionary even though
   `kids_tone` is network-local;
3. accepts at least 90% overlap, drops unmatched quick rows, ignores extras, and
   ignores a present `citlali_uid`;
4. warning-overwrites duplicate keys; and
5. writes no selected-APT component digest or explicit binding record.

`apt_modifier.py:32-69,232-350` repeats warning/last-wins behavior, accepts
implicit partial membership, and preserves the source APT presentation order.

Legacy report diagnostics group rows by array, then compute nearest-tone
distance from physical adjacency without network grouping
(`ToltecAptDiagnostics.py:566-659,931-938`; used by
`pipeline/report.py:585-600`). The repository already contains a correct model:
`pipeline/apt_robustness_audit.py:196-220` groups by `(reduction,nw)`, sorts a
temporary frequency view, and writes results back to original indices.

### 5.3 TolAPT ingestion, assignment, and products

`read_measured` drops flagged rows, generates `mNNNNNN` from the surviving
presentation position, casts input UID to float, then drops non-finite
coordinate/tone rows without persisting an original source-row index
(`src/tolapt/io/measured_reader.py:69-139`). Therefore
`measured.enriched.ecsv` is complete only for surviving canonical rows, not for
raw input membership. It retains surviving unmatched rows with empty
`matched_design_id` and `match_id=-1`
(`src/tolapt/matching.py:1374-1407`).

Design ingestion has the parallel provenance gap: both raw and canonical design
readers drop non-finite coordinate/tone rows before checking `det_id`
uniqueness, retain no source-row/exclusion map, and can therefore hide a
duplicate ID when one duplicate is filtered
(`src/tolapt/io/design_reader.py:98-124,143-172`).

The matcher creates run-local `meas_index` and `design_index`, performs global
assignment, and writes explicit IDs and indices to `matches.ecsv`
(`matching.py:162-210,1077-1115`). With unique explicit IDs, ordinary
permutations change only the positional indices. Exact equal-cost assignments
have no stable tie breaker, however, so reversing either input can change the
authoritative endpoint pairing while only emitting a review flag.

Output orders are explicit presentation choices, not identities:

- `matches.ecsv` is sorted by `(block_id, measured_id)`, and `match_id` is the
  ordinal in that order (`matching.py:1077-1115`); and
- design/measured enriched tables preserve their respective canonical input
  presentation while attaching match fields by the run-local index
  (`matching.py:1348-1407`).

Several current consumers dereference `meas_idx`/`design_idx` without checking
their explicit endpoint IDs: `row_shift.py:688-695`,
`plots.py:3013-3036,3060-3075`, and `residual_profile.py:464-475`. They are safe
only while consuming the exact co-generated immutable tables; copied, reordered,
or stale components can silently change diagnostic values. By contrast,
`frequency_grouping.py:32-122` is a guardrail: it operates on temporary array
subsets and writes labels back through original indices, preserving canonical
measured presentation without making that order detector identity.

Frequency meanings are distinct:

- `measured.enriched.tone_freq` is the measured/raw APT frequency
  (`measured_reader.py:101-113`).
- `design.enriched.tone_freq` is design `fr * 1e9`
  (`design_reader.py:98-108`).
- `matches.ecsv` carries differences and scoring provenance, not an
  authoritative absolute measured or design tone (`matching.py:1077-1115`).

TolAPT's current production hero boundary is an immutable geometry overlay plus
an explicit `(obsnum, local_uid) -> common_uid` registration, not a full
observation-local Citlali APT. The common-detector catalog
(`hero_overlay.py:502-541`) is a distinct supporting artifact; projected
geometry is built in the overlay (`hero_overlay.py:660-837`) and released as
`hero_geometry_overlay_*.ecsv`. The historical
anchor-only `hero_anchor_apt` copies tone/readout metadata from selected base
measured rows and contains only that subset (`hero_apt.py:560-695`). Neither
alone supplies complete observation membership plus Citlali's readout binding.

The tone-match-consistency contract documents
`matches.meas_idx -> measured.enriched.uid -> matched APT det_id -> matched APT uid`
(`docs/output_contract.md:213-219`). Its implementation dereferences
`meas_idx` without checking that `matches.measured_id` equals the indexed
measured row (`tone_match_consistency.py:756-800,1990-1995`). Stale, missing, or
out-of-range mappings can therefore be skipped or rebound silently.

### 5.4 TolProj selection, matching, hero application, and copy/transport

TolProj library copy uses `shutil.copy2`, so bytes and row sequence are
preserved, but its manifest and seed selection do not store/verify an artifact
SHA-256 or canonical membership identity (`tolproj/utils.py:1033-1227,2084-2150`).
Selection is principally path/obsnum based.

The ordinary matcher creates target `kids_tone` per KMP/network, stacks networks
in sorted order, and assigns target `det_id=0..N-1`
(`tolproj/legacy_scripts/make_matched_apt.py:244-272`). It discards any seed
`det_id`, replaces it with seed row `0..N-1` (`:726-728`), matches within
network/frequency, records that positional value as `det_id_right`, joins it
back to the seed, and sorts the final rows by target `det_id`
(`:275-370,594-665`). The final `tone_freq` is observational target
`kids_f_out` (`:664`).

Accordingly:

- final row order is target detector/readout application order and is legitimate
  for aligned target arrays;
- `det_id_right` is seed serialization provenance, not detector identity;
- seed UID is copied through as an artifact-scoped tag; and
- equal-frequency candidate ties can choose the first seed presentation.

TolProj hero application is much stronger: it maps target row `uid` through an
observation-scoped registration to `common_uid`, then to explicit design
geometry, and preserves target application order
(`tolproj/steps/apply_hero.py:151-179,355-496`). It rejects many duplicate and
many-to-one mappings. Its `_uid_key` converts numeric-looking values through
float (`:733-744`), however, so large integers, leading zeros, and near-integer
lexical IDs can collide. It computes overlay/registration hashes for its
summary, but it does not verify release-declared hashes or record the input
matched-APT digest.

Flux calibration updates rows by explicit `array` and preserves order/columns,
but records paths rather than input/output component digests
(`steps/calibrate_flxscale.py:418-452,499-540`).

Science/pointing setup writes selected APT paths into observation YAML without
schema, membership, key, or digest validation. A concrete `science-refresh`
regex defect (`tolproj/cli.py:2262-2273,2396-2429`) can fail to recognize a
normal existing hero APT and append a fluxcal APT for the same observation.

### 5.5 TolTECA v2 selection and delivery

Numbered YAML precedence and list replacement are meaningful configuration
selection order, not detector order. The exact v2 implementation merges
applicable calibration entries in list order and silently lets the last
`array_prop_table` win
(`tolteca/reduce/engines/citlali.py:640-668` at `2791e6a...`). A discovered APT
is appended after configured entries and can override the configured APT
(`:671-731`).

Normal TolProj-configured APT paths are passed through unchanged. A separately
discovered `interface=="apt"` path invokes `_fix_apt`. That function has
dtype-dependent semantics (`:853-926`):

- all-float tables bypass conversion;
- one non-float column triggers a hard-coded numeric rewrite;
- `uid` becomes `0..N-1`;
- fields outside the whitelist, including TolAPT mappings and `det_id_right`,
  are dropped; and
- all original metadata is replaced by `Radesys: altaz`.

Fallback `_make_apt` sorts networks, stacks KMP rows, and assigns every UID
`-1` (`:929-1003`). It is an unmapped compatibility artifact, not a matched
production identity.

### 5.6 Citlali production ingestion, publication, and validation

Citlali C++ computes absolute raw tone frequencies for each explicit network,
rejects nonfinite/duplicate raw network-tone keys, and joins every raw
network/local-tone coordinate to exactly one selected APT `(nw,tone_freq)` row
(`src/citlali/core/engine/calib.cpp:569-719`). Missing, extra, duplicate, or
ambiguous acquisition keys fail. Output detector records follow raw application
order; the source row index is retained separately. This is the correct
identity direction, and the existing test
`calib_apt_binding.explicit_join_is_invariant_to_apt_row_permutation` confirms
it (`tests/test_calib_apt_filtering.cpp:297-323`).

The selected source APT is published as an exact byte copy with digest checks
and atomic replacement
(`include/citlali/core/pipeline/raw_timestream_provenance.h:807-887`). The
provenance declares `apt_row_order_authoritative: false` and serializes each
ordered detector row with its source-row index and raw network/local-tone/frequency
mapping (`:600-625`).

There are three remaining contract problems:

1. Python v4 validation re-computes `raw_network_local_tone` as the count of
   preceding source APT rows with the same network
   (`tools/baseline/audit_reduction_run.py:925-932`). This contradicts the C++
   frequency join and the serialized declaration that source order is not
   authoritative.
2. Exact artifact SHA and source-row association are embedded in the binding,
   calibration, and package identities (`calib.cpp:721-976` and
   `include/citlali/core/timestream/calibration_product.h:401-463`). Two
   presentation permutations with the same semantic detector mapping therefore
   receive different binding/CALID/PKGID. Exact-byte provenance should remain,
   but a separate presentation-invariant semantic binding digest is absent.
3. The required numeric UID column is loaded through an `int64 -> double` cast
   (`include/citlali/core/utils/ecsv_io.h:240-256`), and ingestion does not
   enforce UID uniqueness. This cannot be an exact artifact-scoped UID join for
   values beyond binary64 integer precision or for duplicate UIDs, even though
   UID is serialized as `int64` in the binding preimage (`calib.cpp:959-973`).

`Calib::setup` requires network and array groups to be contiguous
(`calib.cpp:1083-1134`). This is an internal application-order constraint after
the C++ join, not authority granted to the input ECSV order.

The duplicate-tone flagger is independently unsafe: it compares adjacent rows
globally, fails to take the absolute value for its first difference, and ignores
network boundaries
(`include/citlali/core/pipeline/rawobs_tone_frequency_inventory.h:102-129`).
Its flags feed RTC/PTC detector eligibility.

## 6. Required key scope and mapping at every boundary

The following table is normative for a repair. Every boundary persists both:

- an **artifact instance identity**, including SHA-256 of the exact bytes, for
  source reconstruction and tamper detection; and
- a **semantic component identity**, derived from the producer/issuer scope and
  canonical typed membership as defined in Section 10.1, for keys that must
  survive presentation-only reserialization.

An exact artifact instance maps explicitly to its verified semantic component.
The artifact SHA is never embedded in a presentation-invariant endpoint key.

| Boundary | Stable key scope | Required explicit mapping | Fields that remain redundant checks/provenance |
| --- | --- | --- | --- |
| raw/KMP -> Citlali internal detector axis | `(raw acquisition component ID, network ID, network-local detector column)` | exact raw artifact/interface/`RoachIndex` -> verified raw component -> admitted network/local detector; carry absolute measured tone | exact raw SHA, array, tone frequency, input file position |
| Citlali Beammap numerical axis -> FITS/ECSV | `(beammap component ID, producer slot ID)` | exact FITS/HDU locator -> component slot -> `BEAMMAP.UID` and full APT member | exact product SHA, EXTNAME integer, source raw coordinate, product row |
| Beammap product/quick-fit -> selected APT update | source `(beammap component ID, slot ID)`; target `(selected APT component ID, exact typed apt_uid)` | one-to-one source slot -> target APT UID, with exact selected Beammap, fit-QC, and APT artifact identities | EXTNAME `det_id`, `BEAMMAP.UID`, `citlali_uid`, array, network, local tone, tone frequency |
| design input -> TolAPT canonical design | `(design component ID, exact typed design_det_id)` | exact source artifact/row -> verified design component/member; exclusions retain source row and reason | exact artifact SHA, `design_idx`, input row order, design tone |
| measured input -> TolAPT canonical measured | `(observation identity, measured component ID, exact typed local_uid)` | exact source artifact/row -> verified measured component/member or explicit exclusion record | exact artifact SHA, `meas_idx`, generated display ID, array, network, local tone, measured tone |
| TolAPT measured -> design assignment | explicit measured key plus explicit design key | one match edge containing both endpoints, status, ambiguity state, evidence, and checked locators | `match_id`, `meas_idx`, `design_idx`, score/rank |
| cross-observation TolAPT registration | `(TolAPT release/package ID, common_uid)` and `(observation identity, local_uid)` | explicit one-to-one `(obsnum, local_uid) -> (release ID, common_uid)` | design key, array, network; these must agree when repeated |
| TolAPT overlay -> TolProj observation APT | target observation-local UID/readout key plus release-scoped common key | observation APT row -> registration edge -> accepted design geometry; persist both endpoints and decision | source row, target application row, `det_id_right`, measured tone |
| TolProj tone match -> matched APT | target raw key and selected seed APT key | one-to-one target raw coordinate -> seed scoped UID; retain target/seed source indices and ambiguity status | target `det_id`, seed `det_id_right`, separation, presentation order |
| matched -> fluxcal/hero APT | input key `(input component ID, exact typed uid)` | exact input artifact/row -> verified input member -> output member plus transformation identity; preserve target application axis | input/output artifact SHA, row index, array factor, common UID/design geometry |
| TolProj -> TolTECA | `(observation selector, selected product kind, selected APT component/package ID)` | exactly one configured APT selection per observation, including expected exact artifact SHA | YAML list position and path text |
| TolTECA -> Citlali | verified selected APT component plus optional compatibility-output component | lossless identity-preserving pass-through, or explicit source-row/key -> converted-row/key map with both hashes | dtype, discovery order, output row order |
| Citlali selected APT -> raw application axis | APT `(APT component ID, exact typed uid)` plus raw `(raw acquisition component ID, network, local tone)` | exact artifacts -> verified components, then complete one-to-one raw coordinate -> selected APT member, currently proven with unique `(nw, absolute measured tone)` | exact artifact SHAs, selected source row, APT presentation order |
| Citlali package -> validator | package member digests and serialized explicit mapping | recompute exact member identity and semantic mapping from persisted endpoints; never infer a locator from row order | package-local path, source-row provenance, presentation digest |

`uid` and `common_uid` in this table are always exact typed values. Conversion
through binary64, numeric-string normalization, stripping leading zeros, or
inventing a row ordinal is forbidden unless a versioned producer contract
explicitly defines that conversion and publishes a lossless map.

## 7. Counterexample disposition

Legend:

- **A**: accepted without changing the intended explicit mapping.
- **R**: rejected before application.
- **S**: accepted but silently changes, drops, or misbinds identity.
- **P**: accepted as an explicitly represented partial/unmatched result.
- **G**: contract gap; the fact is neither validated nor reconstructible.
- **U**: unavailable under this audit's production-data restrictions.

### 7.1 Citlali exact-candidate behavior

| Counterexample | C++ ingestion/application | v4 Python/package validation | Other identity effect |
| --- | --- | --- | --- |
| arbitrary permutation within one network | **A**; same raw-ordered detector/UID mapping | **R** when the permutation changes within-network source position | exact APT, row-association, binding, CALID, and PKGID change |
| interleaved networks, within-network order preserved | **A** | **A** in the synthetic validator fixture | exact artifact/binding identities still change |
| reversed same-network rows | **A** | **R**: “selected APT sibling row order differs from detector join” | proves validator/C++ contradiction |
| identical membership, different presentation | **A** semantic application | conditional **R** as above | no presentation-invariant semantic binding digest |
| missing APT `(nw,tone)` key | **R** | not reached | fail-closed |
| extra APT key/network/row | **R**; exact membership required | **R**; F009-B source coverage also rejects unused package-local row | fail-closed |
| duplicate/ambiguous `(nw,tone)` | **R** | not reached | fail-closed |
| duplicate UID on distinct unique tones | source shows **A/G**; UID uniqueness is not checked | no independent uniqueness check | unsafe for UID-addressed products |
| UID values above exact binary64 integer range | source shows **S/G** through `int64 -> double` | typed retained source text may disagree with numeric application value | exact UID cannot be guaranteed |
| stale, duplicate, or out-of-range source index | not a C++ input mapping | **R** | fail-closed for serialized locator tampering |
| forged component/package digest | not applicable at ingestion | **R** in existing v4 tamper tests | fail-closed |
| partial source coverage | **R** | **R** | fail-closed; no implicit partial policy |
| conflicting retained mapping/status fields | **R** for TolAPT manifest/status conflicts | **R** | fail-closed |
| permuted/reversed/interleaved duplicate-tone inventory | ingestion mapping unchanged | not checked here | **S** flags from physical adjacency; network boundary can create false flags |

The current C++ binding pass is important: it proves that fixing the Python
validator does not require sorting the selected APT or changing detector-
aligned numerical arrays.

### 7.2 TolAPT behavior

| Counterexample | Stage | Current result |
| --- | --- | --- |
| arbitrary within-network permutation | measured reader | **S**: generated `det_id` changes with presentation |
| interleaved networks | measured reader | **S**: generated `det_id` changes |
| reversed same-network rows | measured reader | **S**: generated `det_id` changes |
| unique explicit IDs, same membership/different presentation | main matcher | **A** endpoint pairing; `meas_idx`/`design_idx` change as locators |
| exact equal-cost assignment | global matcher | **S**: authoritative pairing flips with input order; review flag only |
| extra measured row | matcher | **P** as explicit unmatched surviving canonical row |
| flagged/nonfinite measured source row | reader | **S/G**: dropped without reconstructible source membership/exclusion row |
| missing measured UID | reader | **S/G**: accepted with positional generated ID |
| duplicate measured UID or `det_id` | reader/main matcher | **S/G**: accepted |
| UIDs `2^53` and `2^53+1` | measured reader | **S**: both become `9007199254740992.0` |
| duplicate design ID | design reader | **R** |
| nonfinite design source row | design reader | source-confirmed **S/G**: dropped without exclusion/source-row mapping |
| duplicate design ID with one nonfinite occurrence | design reader | source-confirmed **S/G**: filter runs before uniqueness and can hide the duplicate |
| duplicate pairwise UID or `(nw,kids_tone)` | pairwise matcher | **R** |
| exactly coincident pairwise reference tones | pairwise matcher | **S**: reciprocal mapping flips under reversal despite zero margin |
| stale in-range `meas_idx` | tone-match consistency | **S**: match can bind to another measured UID/common UID |
| stale in-range `meas_idx`/`design_idx` | row-shift/plot/residual consumers | source-confirmed **S/G**: positional row is used without endpoint agreement check |
| out-of-range/missing/partial mapping | tone-match consistency | **S**: silently yields no assignment |
| duplicate matched-APT local/common key | tone-match consistency | **S**: implicated rows are omitted and processing continues |
| conflicting array/network attributes | tone-match consistency | **S**: accepted |
| forged input digest in run manifest | hero anchor | **S**: digest is not verified |
| duplicate hero source UID | hero anchor | **R** |
| duplicate measured-to-design source for one hero design | hero anchor | **S**: first row wins |
| conflicting overlay local/common mapping | overlay registration | **R** |
| same overlay edge with conflicting ancillary metadata | overlay registration | **S**: first row wins |
| conflicting common-UID reference metadata | summary | **S**: `rows[0]` changes result under reversal |

The executed filtered-membership fixture was
`[101 good, 102 nonfinite-x, 201 flagged, 202 good]`. Output contained UIDs
`[101,202]`, IDs `[m000000,m000002]`, and no source-row column.

### 7.3 `toltec_beammap` behavior

| Counterexample | Current result |
| --- | --- |
| arbitrary quick-row permutation with unique explicit UID | **A**; update is keyed and source APT presentation is preserved |
| reversed same-network APT rows | update **A** with unique UID; legacy nearest-tone result **S** |
| identical membership, different presentation | **A/G**; output preserves each input order, no canonical membership digest |
| interleaved networks | **S** in legacy spacing and array-wide `kids_tone` inference |
| missing 1 of 10 target quick rows | **P/S**: accepted and target row flagged under implicit 90% policy |
| missing 2 of 10 target quick rows | **R** by threshold, not by identity contract |
| extra quick UID | **S**: ignored |
| duplicate quick UID | **S**: last wins; reversal changes output |
| duplicate APT UID | **S**: modifier accepts and updates both rows |
| conflicting quick `uid` and `det_id` | **S**: UID wins without consistency check |
| stale source index | **S/G**: preserved or ignored |
| forged component digest metadata | **S/G**: accepted; no recomputation |
| implicit array partial update | **P/G**: accepted without an exact target-set manifest |
| same EXTNAME `det_id=5`, different `BEAMMAP.UID=100/200` | **S**: core collapses to one; review exposes two |
| EXTNAME IDs 5/6, same `BEAMMAP.UID=100` | **S**: core exposes two; review collapses to one |
| duplicate network-local tone key in array-wide dictionary | **S**: exactly 90% case accepted; reversal changes selected UID |
| numeric APT UID 0-9, reverse `kids_tone` 9-0 | **S**: overlap heuristic chooses the wrong identity domain |
| explicit reversed `citlali_uid` but no UID | **S**: `citlali_uid` ignored and numeric heuristic used |
| same UID in different reductions/arrays | **S**: UID-only robustness join computes a cross-artifact ratio |
| duplicate APT basename in separate directories | **S**: report index keeps first path |

Conflicting FITS and duplicate-tone fixtures exercised the production extractor,
review/QC conventions, and update functions. They do not claim production-data
provenance.

### 7.4 TolProj and TolTECA v2 behavior

| Counterexample | TolProj | TolTECA v2 |
| --- | --- | --- |
| arbitrary within-network seed permutation, unique tones | **A** physical attributes; source `det_id_right` changes | configured path unchanged; discovered mixed-type path **S** rewrites UID |
| interleaved networks | **A**; grouped by network, output returns to target application order | discovered `_fix_apt` preserves sequence but **S** regenerates identity |
| reversed same-network seed rows | **A** unique pairing; seed locator changes | mixed-type path **S** UID binding |
| identical membership, different presentation | **G**: different source indices/bytes, no canonical membership digest | **G** |
| missing source network | **P**: target rows retained unmatched/bad; zero total matches **R** | nonexistent configured path can reach low-level resolution **G** |
| extra seed row | **S/P**: silently unused | unchecked |
| multiple target rows -> same seed row | **R** | not independently checked |
| duplicate seed UID on different tones | **S/G**: accepted and persisted | `_fix_apt` can hide it by assigning unique ordinals |
| exact equal-frequency candidates | **S**: presentation-first UID selected | unchecked |
| missing hero registration | **P**: row flagged bad with reason | not applicable |
| conflicting hero mapping | **R** | not applicable |
| stale negative `det_id_right` | hero marks unmatched | not applicable |
| forged positive `det_id_right` | **S** if UID registration succeeds | not applicable |
| forged release/component digest | **S/G**: declared release hash ignored | no verification |
| large/lexically distinct UID strings | **S**: hero `_uid_key` can collide | mixed path cannot preserve them |
| two applicable APT paths | `science-refresh` can create them | **S**: last applicable entry wins |
| configured APT plus discovered APT | no intended second selection | **S**: discovered trimmed APT overrides configured selection |
| mixed-type APT with UID `[7001,7002]` | passed by normal configured path | discovered path **S** -> `[0,1]`; mapping/provenance dropped |
| same mixed table reversed | passed by normal configured path | **S** -> `[0,1]` attached to reversed geometry |
| duplicate UID `[7001,7001]` | possible | **S** -> `[0,1]`, hiding duplicate |
| partial hero overlay | **P** with row reasons/bad flags | selected file passed through if configured |
| full production-data round trip | **U** | **U** |

## 8. Executable evidence

All Python used the repository-mandated `$HOME/tolteca/bin/python`. Plotting
tests used a headless backend and task-specific temporary cache directories.
Repository trees were unchanged by the tests.

### 8.1 Citlali

An existing sibling build was verified to originate from the same exact clean
candidate commit. Focused C++ tests:

```text
/Users/gwilson/.codex/worktrees/ed52/citlali-refactor/build/tests/citlali_test \
  --gtest_filter='calib_apt_binding.*:calib_apt_lineage.*:calib_unit_policy.*'
```

Result: `9 tests passed`.

The test proves:

- full APT reversal preserves raw-ordered UID and `kids_tone` application;
- raw network-file reorder preserves keyed rows;
- missing, extra, duplicate, mismatched, unavailable, and conflicting raw/APT
  acquisition identities reject; and
- exact APT and binding digests deliberately differ across presentation.

Focused Python v4 tests:

```text
$HOME/tolteca/bin/python -m unittest \
  tools.baseline.test_audit_reduction_run.AuditReductionRunTests.test_selected_apt_membership_binding_matrix \
  tools.baseline.test_audit_reduction_run.AuditReductionRunTests.test_rejects_unused_package_local_selected_apt_row \
  tools.baseline.test_audit_reduction_run.AuditReductionRunTests.test_rejects_missing_tampered_stale_conflicting_and_forged_v4_members
```

Result: `3 tests passed`.

A production-shaped two-network ECSV/provenance fixture then changed only
source presentation and regenerated the source-row locator as C++ does:

- reversed same-network rows: validator rejected with
  `v4 selected APT sibling row order differs from detector join`;
- interleaved networks with within-network sequence retained: accepted; and
- interleaved plus same-network reversal: rejected.

A temporary exact-header duplicate-tone executable produced:

```text
descending_first_pair: tones=[1.2e9,1.1e9,1.10001e9], flags=[1,1,1], count=3
unsorted_near_pair:    tones=[1.0e9,1.3e9,1.00001e9], flags=[0,0,0], count=0
cross_network_boundary: adjacent 1.2e9/1.20001e9 in different networks, middle flags set, count=2
```

These are respectively a sign error, a missed non-adjacent near pair, and a
false cross-network pair.

### 8.2 TolAPT

```text
env PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg \
  MPLCONFIGDIR=/tmp/tolapt-audit-mpl \
  XDG_CACHE_HOME=/tmp/tolapt-audit-cache TMPDIR=/tmp \
  $HOME/tolteca/bin/python -m pytest -p no:cacheprovider \
  tests/test_io_readers.py tests/test_matching.py tests/test_pipeline.py \
  tests/test_pairwise_tone_match.py tests/test_tone_match_consistency.py \
  tests/test_hero_apt.py tests/test_hero_overlay.py tests/test_tone_order.py
```

Result: `51 passed, 39 Plotly deprecation warnings in 32.38 s`.

Small Astropy fixtures exercised the dynamic counterexamples including the
two-by-two equal-cost flip, stale `meas_idx`, equal-tone pairwise flip,
duplicate maps, exact UID precision loss, measured filtered membership, and
forged manifest digest. Design filtering and the row-shift/plot/residual
locator behavior are source-confirmed and are not mislabeled as executed
counterexamples.

### 8.3 `toltec_beammap`

```text
MPLBACKEND=Agg \
MPLCONFIGDIR=/private/tmp/beammap_audit_mpl \
XDG_CACHE_HOME=/private/tmp/beammap_audit_cache \
PYTHONDONTWRITEBYTECODE=1 \
$HOME/tolteca/bin/python -m pytest -p no:cacheprovider -q
```

Result: `39 passed, 1 Matplotlib deprecation warning in 16.52 s`.

The supported CLI registered successfully under `python -m toltec_beammap
--help`. Temporary FITS/ECSV fixtures exercised the slot/header conflicts,
numeric-overlap misclassification, duplicate local-tone reversal, ignored
`citlali_uid`, missing/extra/partial cases, duplicate last-wins behavior, and
permuted nearest-tone spacing.

The tracked 5,372-row historical APT fixture had five detector-spacing
discrepancies between physical adjacency and correct network-local sorted
spacing. Its particular `<60 kHz` count happened to remain 188; that coincidence
is not a correctness guarantee.

### 8.4 TolProj and TolTECA v2

TolProj focused matcher/hero tests: `23 passed`.

```text
$HOME/tolteca/bin/python -m pytest -p no:cacheprovider \
  tests/test_make_matched_apt.py tests/test_apply_hero.py
```

Transport/setup/flux tests: `29 passed`.

```text
$HOME/tolteca/bin/python -m pytest -p no:cacheprovider \
  tests/test_science_scannums.py tests/test_flux_calibration.py \
  tests/test_beammap_pointings.py
```

Production-shaped fixtures exercised seed permutations/ties, unused and
duplicate seed rows, hero lexical UID collisions, forged release hashes,
refresh duplicate selection, fluxcal order preservation, and selected-path
transport.

The exact TolTECA v2 runtime-context test could not collect because the locally
installed modern `tollan.utils` lacks v2's `odict_from_list`; no false pass is
claimed. Exact v2 function bodies were instead extracted from Git object
`2791e6a...` and executed in bounded fixtures for configured/discovered
selection, list-order reversal, `_fix_apt` dtype behavior, UID rewrite, and
field/metadata loss.

## 9. Finding-by-finding disposition

Severity means:

- **Critical**: can silently bind scientific values to the wrong detector or
  silently replace the selected APT.
- **High**: can make identity non-reconstructible, accept unresolved identity,
  or reject a valid production mapping.
- **Moderate**: provenance/reproducibility or historical-path defect without a
  demonstrated current silent detector misbinding.
- **Pass/guardrail**: behavior to preserve.

### 9.1 Citlali owner

| ID | Severity | Finding and evidence | Disposition |
| --- | --- | --- | --- |
| CIT-01 | Pass | C++ validates raw interface/`RoachIndex`, network/tone cardinality, finite unique raw keys, and complete one-to-one `(nw,tone_freq)` APT coverage, then orders records by raw application axis (`calib.cpp:569-719`). | Preserve. Do not replace with sorting or row-position binding. |
| CIT-02 | High | Python computes local tone from preceding source rows (`audit_reduction_run.py:925-932`) and rejects a legitimate same-network permutation accepted by C++. | Confirmed validator defect; repair in Citlali. |
| CIT-03 | High | Duplicate-tone flags use global physical adjacency, ignore network, and contain a first-difference sign bug (`rawobs_tone_frequency_inventory.h:102-129`). Synthetic reversal/permutation/network-boundary cases changed flags. | Confirmed numerical-eligibility metadata defect; repair in Citlali without reordering arrays. |
| CIT-04 | High | Required int64 UID is cast to double (`ecsv_io.h:240-256`) and not checked for uniqueness, although downstream provenance describes it as `int64`. | Confirmed boundary contract defect; fail closed or retain exact typed UID. |
| CIT-05 | Moderate/high contract gap | Exact artifact SHA and source-row association are included in binding/CALID/PKGID, so semantically identical presentation permutations have different identities (`calib.cpp:721-976`; `calibration_product.h:401-463`). | Preserve exact artifact identity, but add a separately named presentation-invariant semantic mapping digest in a new schema. Do not redefine v4 in place. |
| CIT-06 | Pass, bounded | v4 publishes an exact digest-verified selected-APT copy and serializes explicit raw/application/source mappings (`raw_timestream_provenance.h:520-625,807-887`). F009-B rejects unused package-local rows. | Preserve. F009-B proves exact membership/tamper closure, not presentation invariance. |
| CIT-07 | Guardrail | Contiguous network/array groups and detector-aligned numerical vectors are internal application-order requirements after the explicit join (`calib.cpp:1083-1134`). | Do not independently sort metadata or numerical arrays. |

### 9.2 TolAPT owner

| ID | Severity | Finding and evidence | Disposition |
| --- | --- | --- | --- |
| TA-01 | High | Measured ingestion filters before/after positional ID creation, converts UID to float, and retains no complete source membership (`measured_reader.py:96-139`). Design ingestion likewise filters nonfinite rows before uniqueness and retains no exclusion/source-row map (`design_reader.py:98-124,143-172`). | Confirmed TolAPT input-membership defect. |
| TA-02 | High | Main matching permits optional/fallback IDs and exact global ties inherit positional solver order (`matching.py:162-210,985-1018,1952-1967`). Row-shift, plot, and residual consumers dereference positional indices without checking explicit endpoints. | Require explicit unique endpoints; unresolved equivalent optima must not become authoritative identity; every consumer must verify co-generated components and endpoint/locator agreement. |
| TA-03 | Critical | Tone-match consistency dereferences `meas_idx` and does not verify `measured_id` against the indexed row (`tone_match_consistency.py:756-800,1990-1995`). | Confirmed identity-corruption defect. |
| TA-04 | High | Duplicate handling ranges from rejection to last/first-wins or silent omission across main matcher, tone order, consistency, and hero paths. | All identity-bearing duplicates must be fatal before application. |
| TA-05 | High | Exactly coincident pairwise tones can flip reciprocal common-identity mappings under reference reversal despite zero ambiguity margin (`pairwise_tone_match.py:39-80,370-410,597-660`). | Withhold unresolved identity unless another explicit discriminator proves it. |
| TA-06 | Moderate | Common-UID summary uses `rows[0]`; overlay duplicate edges with conflicting ancillary metadata are first-wins (`tone_match_consistency.py:920-990`; `hero_overlay.py:544-598`). | Prove repeated metadata consistency or fail. |
| TA-07 | High | Run input hashes are recorded but hero consumers do not verify them; required product paths lack finalized hashes (`artifacts.py:197-265`; `hero_apt.py:846-870`). | Confirmed immutable-contract defect. |
| TA-08 | High joint gap | No TolAPT output alone contains full target observation membership, measured readout binding, and production overlay geometry. | TolAPT owns registration schema, TolProj owns observation application/output, Citlali owns admission. Require their explicit mapping, not a shared sort order. |
| TA-09 | Pass | Design-frequency prior packages validate unique design IDs, complete coverage, identity agreement, and file hashes (`design_frequency_prior_matching.py:99-141,250-305`). | Preserve and reuse this pattern. |
| TA-10 | Guardrail | Frequency grouping operates on temporary array subsets and writes results back to original measured indices (`frequency_grouping.py:32-122`). Enriched table order and sorted match presentation remain explicitly non-identity. | Preserve the write-back alignment pattern. |

### 9.3 `toltec_beammap` owner

| ID | Severity | Finding and evidence | Disposition |
| --- | --- | --- | --- |
| BM-01 | Critical | Core uses EXTNAME `det_id`, review/QC use `BEAMMAP.UID`, and APT update guesses UID via 90% numeric overlap or array-wide `kids_tone` (`toltec_beammap.py:464-570,1748-1919`). | Production-contract blocker owned by `toltec_beammap`. |
| BM-02 | High | Duplicate/missing/extra/partial mappings are warning/last-wins or threshold-based (`apt_modifier.py:32-69,232-350`). | Replace with exact one-to-one, declared target membership. |
| BM-03 | High | Legacy nearest-tone diagnostics treat presentation adjacency as frequency adjacency across networks (`ToltecAptDiagnostics.py:566-659,931-938`). | Repair using the repository's correct network-local sorted/write-back pattern. |
| BM-04 | High | Robustness, report, review, and comparison paths join by unscoped UID without redundant array/network/local-tone checks. | Scope joins to exact components and validate redundant fields. |
| BM-05 | High | First-sorted artifact selection, basename-only provenance, and approximate reusable-fit validation cannot reconstruct the binding. | Require unambiguous selection, full component identity, and exact reuse checks. |
| BM-06 | Moderate/historical | Tracked scripts/helpers retain row-position or unvalidated dictionary assumptions. | Repair before supported use or explicitly retire as historical. |
| BM-07 | Guardrail | IMAGE/WEIGHT/QUICKFIT/DET_HEADERS and TOD metadata share legitimate ordered numerical axes (`toltec_beammap.py:2012-2192` and dashboard consumers). | Preserve aligned axes; add explicit unique axis mapping. |

### 9.4 TolProj owner

| ID | Severity | Finding and evidence | Disposition |
| --- | --- | --- | --- |
| TP-01 | High | Ordinary matching overwrites seed `det_id` with row ordinal and persists it as `det_id_right`; exact frequency ties select presentation-first (`make_matched_apt.py:275-370,531-546,639-665,726-743`). | Persist explicit scoped endpoints/evidence and reject unresolved equal-best mapping. |
| TP-02 | High | APT library/seed selection copies bytes but does not store or verify a component digest; artifact replacement at a selected path remains admissible (`tolproj/utils.py:1033-1227,2084-2150`). | Add immutable component identity and verify it through selection. |
| TP-03 | High | Hero application has the right explicit mapping shape but `_uid_key` loses lexical/integer identity and release-declared hashes are not verified (`apply_hero.py:309-352,355-496,733-744`). | Preserve exact declared UID type; verify package/input/output identities. |
| TP-04 | Critical | `science-refresh` doubly escaped regexes miss normal APT selectors and can append a second product for one observation (`tolproj/cli.py:2262-2273,2396-2429`). TolTECA then selects last. | Repair selection normalization; refreshing must preserve an explicit product and yield exactly one APT. |
| TP-05 | Moderate/high | Fluxcal and science/pointing setup preserve useful row/application order but transport paths without exact input/output hashes, schema, or membership validation. | Add component identities and preflight; preserve numerical axis. |

### 9.5 TolTECA v2 owner

| ID | Severity | Finding and evidence | Disposition |
| --- | --- | --- | --- |
| TV2-01 | Critical | `_fix_apt` has dtype-dependent semantics, generates positional UIDs, drops explicit mappings/metadata, and can hide duplicate UID (`citlali.py:853-926`). | Replace with lossless transport or an explicit source-to-compatibility map; fail on ambiguity. |
| TV2-02 | Critical | Applicable calibration entries are list-order last-wins and discovered APT overrides configured selection (`citlali.py:640-760`). | Require exactly one applicable verified APT; configured selection must not be silently displaced. |
| TV2-03 | High | Fallback `_make_apt` assigns every UID `-1` (`citlali.py:929-1003`). | Label unmapped/synthetic, publish raw network/local-tone mapping and KMP identity, and prohibit matched-production use. |

## 10. Durable production contract

The recommended boundary artifact is a versioned detector-binding manifest,
conceptually `toltec.detector_binding.v1`. A repository may embed it in its
existing package manifest, but all fields below remain required.

### 10.1 Identity scopes

1. **Exact artifact instance identity**

   ```text
   artifact_instance_id = (artifact schema/version, sha256(exact bytes))
   ```

   Store path, size, and producer/version for provenance, but do not use path or
   basename as identity. This identity deliberately changes when presentation
   bytes change.

2. **Issuer scope, member key, and semantic component identity**

   ```text
   issuer_scope = (
       producer namespace,
       component schema/version,
       declared release/package/observation scope
   )
   issuer_member_key = (issuer_scope, exact typed producer-local member ID)
   membership_sha256 = sha256(
       canonical type-tagged member records sorted by issuer_member_key
   )
   component_id = (component schema/version, issuer_scope, membership_sha256)
   component_member_key = (component_id, exact typed producer-local member ID)
   ```

   The membership preimage contains the issuer scope, local member keys, and all
   contract-required semantic member fields. It contains neither an exact
   artifact SHA nor the `component_id` that it computes, so it is not circular
   and does not change under presentation-only reserialization. The manifest
   binds each exact artifact instance to the component ID obtained by parsing
   and canonicalizing that artifact. Substitution in either direction fails.

   An issuer scope cannot be a path, basename, timestamp alone, or inferred row
   order. It must be a producer-declared namespace plus the release, package, or
   observation that determines the lifetime of its local IDs.

3. **Design key**

   ```text
   (design_component_id, design_det_id_exact)
   ```

4. **Measured key**

   ```text
   (observation identity, measured_component_id, local_uid_exact)
   ```

   A producer that cannot supply `local_uid` must publish a versioned explicit
   mapping from its raw coordinate to a newly issued component-local ID. It must
   not synthesize a supposedly durable ID from presentation row.

5. **Readout/application coordinate**

   ```text
   (raw acquisition component_id, network_id, network_local_detector_column)
   ```

   Absolute measured tone frequency is required evidence and may be the current
   unique observation-local join component. It is not a cross-observation
   detector identity. The exact raw/KMP artifact instance and interface/
   `RoachIndex` mapping are retained and verified against this component.

6. **Match edge**

   ```text
   (measured_key, design_key)
   ```

   `match_id`, `meas_idx`, and `design_idx` are checked locators only.

7. **Cross-observation registration**

   ```text
   (observation identity, local_uid_exact)
       -> (TolAPT release/package identity, common_uid_exact)
   ```

8. **Beammap slot key**

   ```text
   (beammap_component_id, producer-issued slot ID)
   ```

   The exact Beammap artifact/HDU locator and EXTNAME are mapped to the component
   slot. `BEAMMAP.UID` is an explicit mapped/redundant value, not an assumed
   synonym.

9. **Source-row provenance key**

   ```text
   (source_artifact_instance_id, source_row_index)
   ```

   This reconstructs bytes; it is never the detector key.

### 10.2 Required mapping record

Every detector edge crossing a repository boundary must persist at least:

```text
mapping_schema_version
mapping_authority_and_method
source_artifact_schema, source_artifact_sha256, source_component_id
target_artifact_schema, target_artifact_sha256, target_component_id
artifact_to_component_verification_method
observation_id, subobsnum, scannum (when applicable)
source_stable_key (typed)
target_stable_key (typed)
source_row_index (provenance only)
target_row_index or detector_application_index (locator only)
array_id, network_id
network_local_tone_index (when available)
absolute_measured_tone_frequency_hz (when applicable)
design_component_id, design_det_id (when matched)
tolapt_release_id, common_uid (when registered)
match_status, ambiguity_status, exclusion_reason
```

Redundant array/network/local-tone/frequency/UID fields must agree with both
endpoints. The mapping must be injective in both directions for all applied
rows unless a versioned contract explicitly describes a many-to-one
scientific operation; none is authorized for detector identity here.

### 10.3 Membership and exclusion

Canonical measured/design products must account for every source row:

- included rows appear in the membership/mapping table;
- filtered rows appear in an exclusion table with source key/index and exact
  reason (`flagged`, nonfinite field name, unsupported schema, and so on); and
- counts in included plus excluded tables equal exact source membership.

An output may retain unmatched detector rows as bad/unused if its contract says
so, but `unmatched`, `excluded`, and `ambiguous` are explicit states. They must
not be silently dropped or assigned placeholder IDs that can collide with valid
rows.

### 10.4 Order rules

- Metadata/APT physical row order is presentation only.
- Source row order is exact-artifact provenance only.
- Raw detector/TOD/map/fit/factor vector order is meaningful application order.
- YAML precedence/list replacement is meaningful configuration selection order.
- A persisted application-axis map is mandatory before numerical arrays may use
  dense indices.
- Sorting for a digest never grants identity to that sort position.

### 10.5 Canonical digests

Use schema-versioned, length-delimited, type-tagged serialization. Strings remain
strings and are encoded exactly as declared; they are never parsed through
float. Signed integers are range-validated and serialized canonically as their
declared integer type. Finite frequencies are normalized to Hz and serialized
as an explicitly declared binary64 bit pattern or another single documented
round-trip encoding; nonfinite key fields reject.

Persist separate digests with separate names:

1. `artifact_sha256`: exact file bytes, therefore presentation sensitive.
2. `membership_sha256`: normalized semantic records sorted by
   `issuer_member_key`. Its preimage includes issuer scope and exact typed local
   IDs, but excludes exact artifact SHA, physical row, source index, and the
   resulting component ID. Row/source facts remain in separate provenance.
3. `component_id`: component schema/version, issuer scope, and the verified
   `membership_sha256`. It is presentation invariant.
4. `artifact_component_binding_sha256`: exact artifact instance ID, parser and
   canonicalization schema, and verified component ID. This proves which bytes
   instantiated the semantic component and changes when those bytes change.
5. `mapping_sha256`: validated edges sorted by typed component-scoped source
   key, then typed component-scoped target key. This covers exact semantic
   endpoint membership and match status, not artifact presentation.
6. `application_binding_sha256`: records serialized in the declared detector
   application order, each containing application coordinate plus mapped APT
   key. This order sensitivity is legitimate because it identifies a numerical
   axis, not source presentation.
7. `presentation_sha256` or the exact artifact digest: optional/diagnostic when
   a table's row presentation itself must be compared.
8. Package/CALID/PKGID identities: name which artifact and component digests
   they include.
   An exact package identity may include `artifact_sha256`; a semantic detector
   binding identity must use canonical membership/mapping/application digests
   and must not use source row position as detector meaning.

For Citlali specifically, canonical semantic application records are ordered by
the already-admitted raw application coordinate, normally explicit network
identity and network-local detector column within each verified raw acquisition
component. Exact raw and selected-APT SHAs and source-row maps remain separate
artifact/component bindings and provenance. This
preserves legitimate numerical ordering while making APT presentation
irrelevant to semantic binding.

### 10.6 Failure policy

Fail before scientific application on:

- duplicate source or target stable key;
- missing or extra member relative to an exact declared target set;
- stale locator that disagrees with its explicit stable key;
- missing, zero-margin/equivalent-best, or otherwise ambiguous identity edge;
- conflicting EXTNAME/header/UID, array, network, local-tone, frequency, design,
  release, or package facts;
- unknown, missing, stale, or mismatched component digest;
- more than one applicable APT for one observation;
- discovered/configured APT conflict;
- reuse of quick-fit/match/overlay output against a different source component;
  or
- a lossy compatibility conversion.

Partial processing is permitted only when a machine-readable target-set
manifest declares the exact subset in advance. Its membership must match
exactly. Non-target rows may be preserved, while targeted missing/extra/
ambiguous rows still fail. If a scientific workflow intentionally retains
unmatched rows, it must persist their explicit status and exclude them from
application.

## 11. Exact bounded repair contracts

No repair below was implemented or launched.

### 11.1 Citlali repair contract

#### CIT-R1: remove source-order inference from v4 validation

Smallest path ceiling:

- `tools/baseline/audit_reduction_run.py`
- `tools/baseline/test_audit_reduction_run.py`

Required behavior:

- use `selected_apt_source_row_index` only to retrieve/verify the exact source
  row;
- compare that row's explicit network/frequency/UID/retained values to the
  serialized detector join;
- validate `raw_network_local_tone` against the serialized raw artifact tone
  array at `(raw artifact, network, local tone)`, not by counting prior source
  rows; and
- retain exact full source membership coverage from F009-B.

Required tests: arbitrary within-network permutation, reverse same-network,
interleaved networks, identical membership/different presentation, stale and
duplicate source index, missing/extra/partial, and regenerated valid mapping.
Existing tamper/reordered-member tests remain and must be clearly distinguished
from a legitimate regenerated presentation permutation.

#### CIT-R2: make duplicate-tone detection network-local and order invariant

Smallest path ceiling:

- `include/citlali/core/pipeline/rawobs_tone_frequency_inventory.h`
- `tests/test_calib_apt_filtering.cpp` (or one already-registered focused test
  source, without a build-system expansion)

For each explicit network, create a temporary `(frequency, detector_index)`
view, sort finite frequencies, compute neighbor spacing, and map flags back to
the unchanged detector application axis. Validate empty/singleton/cardinality
cases. Test ascending, descending, arbitrary permutation, non-adjacent near
pair, exact duplicate, and adjacent frequencies in different networks.

#### CIT-R3: enforce exact typed, component-scoped UID

Smallest fail-closed ceiling:

- `src/citlali/core/engine/calib.cpp`
- `include/citlali/core/utils/ecsv_io.h` only if exact typed-column retention is
  implemented there
- `tests/test_calib_apt_filtering.cpp`
- `tools/baseline/audit_reduction_run.py`
- `tools/baseline/test_audit_reduction_run.py`

Require UID presence, finiteness/type validity, and uniqueness within the
verified selected-APT component and its exact artifact instance. Until the
application structure retains exact typed UID, reject
integers outside exact binary64 range and reject any disagreement between the
typed retained UID and numerical value. A durable implementation should retain
the exact typed UID separately from dense numerical arrays. Test duplicate UID
on distinct tones, `2^53`/`2^53+1`, string/numeric schema mismatch, permutation,
and selected-APT digest scope.

#### CIT-R4: separate exact artifact identity from semantic binding

New-schema path ceiling:

- `src/citlali/core/engine/calib.cpp`
- `include/citlali/core/timestream/calibration_product.h`
- `include/citlali/core/pipeline/raw_timestream_provenance.h`
- `tools/baseline/audit_reduction_run.py`
- `tests/test_calib_apt_filtering.cpp`
- `tests/test_calibration_product.cpp`
- `tools/baseline/test_audit_reduction_run.py`
- the governing scientific convention/status/ADR documentation required by the
  repository

Keep current v4 exact artifact and source-row digests unchanged. Add a new
schema/version with canonical membership/mapping/application digests defined in
Section 10.5. Prove that APT presentation permutation preserves semantic
binding while exact artifact/package serialization remains distinguishable.

No Citlali repair is required to reinterpret `det_N` as UID; doing so would be
wrong. Citlali already writes slot and `BEAMMAP.UID` separately. The consumer
must use that mapping.

### 11.2 TolAPT repair contract

#### TA-R1: lossless measured/design membership and source identity

Paths:

- `src/tolapt/io/measured_reader.py`
- `src/tolapt/io/design_reader.py`
- `src/tolapt/pipeline.py`
- `src/tolapt/artifacts.py`
- `tests/test_io_readers.py`
- `tests/test_pipeline.py`

Preserve exact UID/design-ID type and source-row index; require unique scoped
IDs before and after filtering; persist full membership or an exclusion table
with reasons; never issue a durable detector ID from row position. Tests must
cover measured and design permutations, every filter reason, and a duplicate ID
whose other occurrence would otherwise disappear before the uniqueness check.

#### TA-R2: explicit endpoints and ambiguity rejection

Paths:

- `src/tolapt/matching.py`
- `src/tolapt/sanity.py`
- `src/tolapt/tone_order.py`
- `src/tolapt/row_shift.py`
- `src/tolapt/plots.py`
- `src/tolapt/residual_profile.py`
- `tests/test_matching.py`
- `tests/test_sanity.py`
- `tests/test_tone_order.py`
- `tests/test_row_shift.py`
- `tests/test_plots.py`
- `tests/test_residual_profile.py`

Require unique measured/design keys; validate every ID/index pair; detect
equivalent optimum assignments and withhold/fail unresolved identity. Consumers
must resolve by explicit endpoint or prove the exact co-generated immutable
component identities before using a positional locator, then verify the
endpoint/locator agreement. Test stale in-range locators, independent
permutations, and missing/extra component rows.

#### TA-R3: repair tone-match-consistency joins

Paths:

- `src/tolapt/tone_match_consistency.py`
- `tests/test_tone_match_consistency.py`

Join `matches.measured_id` to unique `measured.enriched.det_id`; use `meas_idx`
only as a checked locator. Require declared mapping coverage, reject duplicate
and array/network conflicts, and aggregate reference metadata only after
consistency is proved.

#### TA-R4: reject pairwise zero-margin identity

Paths:

- `src/tolapt/pairwise_tone_match.py`
- `tests/test_pairwise_tone_match.py`

Coincident/equal-best candidate sets remain unresolved unless another explicit
discriminator proves the edge.

#### TA-R5: verify hero and immutable run artifacts

Paths:

- `src/tolapt/hero_apt.py`
- `src/tolapt/hero_overlay.py`
- `src/tolapt/artifacts.py`
- `tests/test_hero_apt.py`
- `tests/test_hero_overlay.py`
- `tests/test_artifacts.py`

Verify manifest hashes before reading, finalize hashes for required run
products, reject duplicate measured/design rows and conflicting duplicate
registration metadata, and expose one verifier used by downstream TolAPT
commands.

#### TA-R6: publish the boundary contract

Path:

- `docs/output_contract.md`

Specify scoped keys, complete/excluded membership, explicit endpoint mappings,
canonical digest order, and fail-closed policy. A new schema version is required
for strengthened registration/provenance; immutable prior releases remain
legacy inputs to a reporting validator.

### 11.3 `toltec_beammap` repair contract

#### BM-R1: explicit Beammap-slot -> APT binding

Smallest implementation ceiling:

- `src/toltec_beammap/toltec_beammap.py`
- `src/toltec_beammap/apt_modifier.py`
- `src/toltec_beammap/pipeline/interactive_review.py`
- `src/toltec_beammap/pipeline/beammap_qc_dash.py`
- new focused `tests/test_apt_identity_contract.py`

Each quick-fit row must carry/refer to Beammap product digest, HDU locator,
EXTNAME slot, `BEAMMAP.UID` when present, selected fit-QC digest, selected APT
digest/UID, array, network, local tone, and mapping method/schema. If producer
authority declares EXTNAME and UID equal, disagreement fails; otherwise the
producer mapping is mandatory. Remove numeric-overlap and array-wide local-tone
inference. Validate `citlali_uid` only with its issuing components. Test all
permutations, namespace collisions, duplicates, missing/extra/partial,
stale-index, forged-digest, and reversed-map cases.

#### BM-R2: network-local nearest-tone diagnostic

Paths:

- `src/toltec_beammap/ToltecAptDiagnostics.py`
- `tests/test_regressions.py`

Share or reproduce the correct group/sort/write-back method in
`pipeline/apt_robustness_audit.py`. Test empty, singleton, duplicate, reversed,
permuted, and interleaved-network inputs.

#### BM-R3: scoped joins and provenance

Paths:

- `src/toltec_beammap/pipeline/apt_robustness_audit.py`
- `src/toltec_beammap/pipeline/beammap_report.py`
- `src/toltec_beammap/pipeline/interactive_review.py`
- `src/toltec_beammap/pipeline/beammap_qc_dash.py`
- `scripts/apt_comparison_dashboard.py`
- new focused `tests/test_apt_join_contract.py`

All joins use component-scoped keys, prove uniqueness/coverage, validate
redundant readout fields, and explicitly preserve the desired left presentation
instead of accepting library-default join sorting.

#### BM-R4: artifact selection/reuse identity

Paths:

- `src/toltec_beammap/toltec_beammap.py`
- `src/toltec_beammap/pipeline/process.py`
- `src/toltec_beammap/pipeline/report.py`
- `scripts/process_beammaps.py`
- `scripts/summarize_planet_power.py`
- focused `tests/test_apt_provenance.py` or existing registered regression tests

Ambiguous candidates fail. Persist canonical full component identity, not
basename. Reused quick fits must match exact detector membership, source APT,
binding, and effective configuration.

#### BM-R5: historical paths

Repair or explicitly retire from supported production use:

- `scripts/compare_apts.py`
- `src/toltec_beammap/ToltecBeammapFits.py`
- `scripts/beammap_workflow_demo.py`

### 11.4 TolProj repair contract

#### TP-R1: explicit tone-match endpoints

Paths:

- `tolproj/legacy_scripts/make_matched_apt.py`
- `tolproj/steps/match_apts.py`
- `tests/test_make_matched_apt.py`

Retain exact target raw and seed component keys/indices, reject duplicate scoped
identity and equal-best ambiguity, persist the full match map and seed digest,
and test permutations, interleaving, reversal, missing/extra/duplicate,
ambiguity, and identical membership.

#### TP-R2: immutable library and selected seed

Paths:

- `tolproj/utils.py`
- new focused `tests/test_apt_library.py`

Store/verify artifact SHA-256 and canonical membership identity; reject silent
obsnum-key replacement; carry the selected digest into `project.yaml` and
`selection_report.json`.

#### TP-R3: exactly one transported APT

Paths:

- `tolproj/cli.py`
- `tolproj/steps/setup_science_reductions.py`
- `tolproj/steps/setup_pointing_reductions.py`
- `tests/test_portable_workflows.py`
- the relevant science/pointing setup tests

Fix `_obsnum_from_item` regexes, preserve explicitly selected product through
refresh, normalize/reject to exactly one APT per observation, and require
existence/schema/digest/key validation for matched, fluxcal, and hero products.
Mandatory regression: refreshing an explicitly hero-selected config cannot
append or switch to fluxcal.

#### TP-R4: hero identity/integrity

Paths:

- `tolproj/steps/apply_hero.py`
- `tests/test_apply_hero.py`

Compare local UIDs in TolAPT's exact declared lexical/type domain, verify
release-declared hashes, record/verify input matched-APT digest and scoped
identity, and test large integers, leading zeros, near-integer decimals,
stale/forged locators, incomplete mappings, and permutation.

#### TP-R5: fluxcal component provenance

Paths:

- `tolproj/steps/calibrate_flxscale.py`
- `tests/test_flux_calibration.py`

Record input/output APT hashes and component identities and validate scoped-key
uniqueness while preserving target application order.

### 11.5 TolTECA v2 repair contract

#### TV2-R1: lossless compatibility transport

Paths:

- `tolteca/reduce/engines/citlali.py`, especially `_fix_apt` at `853-926`
- new focused v2
  `tolteca/reduce/engines/tests/test_citlali_apt_transport.py`

Remove dtype-dependent identity behavior. Preserve exact UID, explicit mapping
columns, and producer metadata, or emit a lossless original-key/row -> converted
key/row map with source/output hashes. Reject duplicate/ambiguous identity.
Test float/mixed parity, arbitrary permutation, reversal, duplicate UID, all
TolAPT mapping fields, and metadata/digest preservation.

#### TV2-R2: unique applicable calibration

Paths:

- `tolteca/reduce/engines/citlali.py`, `_resolve_cal_items` and
  `_resolve_input_item` at `640-760`
- the focused v2 engine test above

Require exactly one applicable verified APT. Explicit configured calibration
must not be silently overridden by discovery. Reject overlapping configured/
discovered inputs, missing files, digest mismatch, and package conflicts. Reverse
list/discovery order in tests and prove invariant rejection/selection.

#### TV2-R3: synthetic fallback isolation

Paths:

- `tolteca/reduce/engines/citlali.py`, `_make_apt` at `929-1003`
- the focused v2 engine test above

Label fallback output as unmapped/synthetic, persist network/local-tone and
source KMP hashes, and disallow it from satisfying a matched production
calibration requirement.

## 12. Minimum cross-repository conformance suite

The repair is not complete when each repository merely adds a sort. Each owner
must run the same logical fixture family through its own public or production
boundary. A future shared fixture package should contain exact component hashes
and expected canonical digests, but it must be vendored/versioned by each owner
rather than fetched at test time.

### 12.1 Canonical valid fixture

Use at least two networks and four target detector coordinates:

```text
target/raw keys:
  (raw-A, nw=0, local=0, measured_hz=1_000_000_000) -> local_uid="m-A0"
  (raw-A, nw=0, local=1, measured_hz=1_100_000_000) -> local_uid="m-A1"
  (raw-B, nw=1, local=0, measured_hz=1_200_000_000) -> local_uid="m-B0"
  (raw-B, nw=1, local=1, measured_hz=1_300_000_000) -> local_uid="m-B1"

design keys:
  (design-component-v1, "d-A0"), (design-component-v1, "d-A1"),
  (design-component-v1, "d-B0"), (design-component-v1, "d-B1")

release registration:
  (obs-X,"m-A0") -> (release-R,"c-A0"), ...
```

The fixture must also include explicit Beammap component slots and exact
product/HDU locators, exact artifact hashes mapped to selected seed/output APT
component IDs, source-row locators, application indices, and one excluded source
row with a declared reason.

### 12.2 Valid presentation variants

Generate from identical membership and endpoints:

1. arbitrary permutation within network 0;
2. reverse all rows within each network;
3. interleave network 0 and network 1 rows;
4. reverse full table presentation;
5. reserialize with the same typed values and a different harmless metadata
   order, where the file format permits it.

Expected results:

- exact artifact SHA/presentation identity differs where bytes differ;
- each exact artifact instance is re-parsed and bound to the same verified
  semantic component ID;
- source-row provenance locators change and remain reconstructible;
- stable endpoint keys and edge set remain identical;
- canonical membership and mapping digests remain identical;
- detector application records and application digest remain identical because
  the admitted raw numerical axis did not change; and
- every stage either preserves the requested presentation or explicitly
  publishes its output presentation without using it as identity.

### 12.3 Invalid variants

Every applicable owning boundary must cover the cases it can create, transport,
or consume; the aggregate cross-repository suite must cover the complete list:

- missing source and missing target member;
- extra source and extra target member;
- duplicate source key;
- duplicate target key;
- duplicate network-local tone coordinate;
- duplicate UID within its issuing artifact;
- exact equal-cost/equal-frequency ambiguous candidates;
- stale in-range source/`meas_idx`/`design_idx` locator;
- out-of-range locator;
- forged artifact, membership, mapping, and package digest;
- partial target without a target-set manifest;
- declared partial target with exact target-set membership;
- conflicting array, network, local tone, frequency, UID, design, and release
  fields;
- EXTNAME/`BEAMMAP.UID` conflict;
- configured/discovered APT conflict;
- two configured APTs for one observation;
- mixed versus all-float transport;
- exact integer UIDs at `2^53`, `2^53+1`, leading-zero strings, and
  near-integer strings; and
- source component substitution at the same path/basename.

Expected result is deterministic fail-closed rejection before application,
except for a declared partial/unmatched policy that emits explicit row states.
Reversing any invalid fixture must not change which identity is applied; it may
only produce the same rejection with deterministic diagnostics.

### 12.4 Legitimate ordered-array fixture

Create a detector-aligned numerical vector with a distinct sentinel for every
raw application coordinate. After every accepted metadata permutation, verify
that each sentinel remains attached to the same explicit raw coordinate and APT
key. No test may independently sort one numerical vector or its metadata.

### 12.5 Required acceptance assertions

Each repaired repository must be able to assert all of the following:

```text
valid_permutation.endpoint_mapping == canonical.endpoint_mapping
valid_permutation.component_id == canonical.component_id
valid_permutation.membership_sha256 == canonical.membership_sha256
valid_permutation.mapping_sha256 == canonical.mapping_sha256
valid_permutation.application_binding_sha256 == canonical.application_binding_sha256
valid_permutation.artifact_sha256 != canonical.artifact_sha256  # when bytes differ
valid_permutation.artifact_component_binding_sha256 != canonical.artifact_component_binding_sha256
invalid_variant.applied_detector_count == 0                      # fail closed
```

For a declared partial operation, substitute the exact declared target set for
full membership and assert that every omitted row is explicitly non-target, not
missing.

## 13. Compatibility and migration

### 13.1 Current Citlali v4 production candidate

Do not silently change the meaning of existing v4 fields:

- `selected_apt_sha256` remains exact-byte identity.
- `selected_apt_row_association_sha256` remains exact source/application
  association under its current schema.
- the current binding, CALID, and PKGID remain presentation-sensitive exact
  identities as already emitted.

CIT-R1 can and should be repaired within the validator because it enforces the
already-declared C++ v4 contract that APT row order is not authoritative. It
must continue to reject tampered package members and incomplete source coverage.
CIT-R2 and a fail-closed CIT-R3 likewise tighten admission without relabeling
existing valid identities.

The presentation-invariant semantic binding in CIT-R4 requires a new named
schema/field set, not an in-place reinterpretation. During migration, a package
may carry both current exact v4 identities and the new canonical semantic
identity. Validators must state which one they checked.

The current candidate cannot be authorized across the production software
boundary until, at minimum, the critical Beammap, TolAPT consistency, TolProj
unique-selection, and TolTECA selection/transport defects are repaired and the
Citlali validator contradiction is closed. This audit supplies no production
authorization.

### 13.2 TolAPT and TolProj current release artifacts

Existing `tolapt.run.v1`, hero overlay/registration, and TolProj matched/hero/
fluxcal files remain immutable. Do not rewrite them to manufacture missing
provenance.

A compatibility validator may:

- compute and attach an external exact artifact hash;
- reconstruct a scoped edge only when all endpoints and redundant fields are
  unique and consistent; and
- label missing source membership, exact UID typing, or component mapping as
  `legacy_unverified` rather than treating row order as evidence.

Strengthened TolAPT registration and matched-output contracts require new
schema versions. TolProj should read legacy products through an explicit legacy
adapter, write only the new contract after migration, and reject identity-
sensitive use when a unique mapping cannot be reconstructed.

### 13.3 Beammap history

Historical `uid == row index` fixtures are not detector-identity authority.
Across the six inspected tracked APTs, the same numeric UID maps to different
`(nw,kids_tone)` coordinates in thousands of overlapping rows. Historical
products may receive an external artifact-scoped identity and source-row map;
they must remain `legacy_unverified` if EXTNAME/header/APT binding cannot be
proved.

Schema drift such as suffixed duplicate fields (`kids_tone_1`, `flag2_1`)
requires an explicit compatibility rule. A consumer may not silently choose the
first similarly named field.

### 13.4 Historical Citlali v1-v3 and redu66 evidence

Historical v1-v3/redu66 products remain evidence under their original
contracts. They are not retroactively promoted to the new detector-binding
contract. A historical parser must distinguish:

- exact bytes available;
- mapping reconstructible and verified;
- mapping reconstructible only with an owner-approved external map; and
- mapping unavailable/ambiguous.

Only the second state satisfies identity-sensitive reuse without additional
owner input. Row position, historical UID coincidence, or a successful old
reduction is insufficient by itself.

F009-B remains valid evidence that the exact v4 package validator rejects an
unused selected-APT row and forged/tampered package members. It does not prove
that a newly generated, semantically equivalent source permutation is valid;
that is CIT-R1's missing regression.

### 13.5 Migration sequence

1. Add fail-closed readers/verifiers and explicit mapping sidecars while
   preserving current files.
2. Repair TolProj/TolTECA exactly-one selection and lossless transport so no
   new product loses identity.
3. Repair Beammap and TolAPT producers to write scoped endpoints, ambiguity
   states, and component hashes.
4. Repair Citlali admission/validator and add the new semantic digest alongside
   current v4 exact identities.
5. Run the shared conformance fixture independently in every repository.
6. Only then run an owner-authorized, real production-data end-to-end
   round trip. Record it as separate evidence; do not infer it from this audit.
7. After a defined compatibility interval, reject legacy/unverified mappings
   for production scientific application while preserving historical readers.

## 14. Final checkpoint

| Required audit outcome | Disposition |
| --- | --- |
| Beammap construction/update and published order guarantee traced | Complete; numerical slot axis is meaningful, table row is not; current consumer/update binding is unsafe |
| TolAPT assignment, indices, IDs, output, consistency joins traced | Complete; explicit match edges exist, but measured identity, ambiguity, stale-index, duplicate, and integrity defects remain |
| TolProj selection/copy/hero/flux/YAML transport traced | Complete; copy is lossless but unhashed, ordinary seed locator is positional, hero mapping is explicit but type/integrity deficient, refresh can duplicate selection |
| TolTECA v2 boundary traced | Complete; normal configured path passes through, discovered mixed path is lossy, and last applicable APT silently wins |
| Citlali raw/APT binding and serialization traced | Complete; C++ application join is row-order independent and exact membership fail-closed |
| Python/C++ component/package validation traced | Complete; F009-B exact membership closes, but Python re-infers within-network source order |
| Order-sensitive numerical arrays separated from metadata tables | Complete |
| Mandatory synthetic counterexamples | Complete at each available local stage; full production-data round trip unavailable and not claimed |
| Stable key scope and explicit mapping at every boundary | Defined in Sections 6 and 10 |
| Canonical digest/preimage ordering | Defined in Section 10.5 |
| Duplicate/missing/extra/ambiguous/partial failure policy | Defined in Section 10.6 |
| Exact owner and bounded repair paths/tests | Defined in Section 11 |
| Current v4 and historical compatibility | Defined in Section 13 |
| Repairs implemented | None; prohibited by audit scope |
| Sibling repository mutations | None |
| Unity/network/production execution | None |

The durable conclusion is not “sort APTs.” It is “name each identity scope,
persist every boundary edge, validate it one-to-one, and keep presentation,
provenance, application, and digest order separate.”
