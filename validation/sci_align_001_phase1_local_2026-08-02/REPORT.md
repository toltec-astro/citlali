# SCI-ALIGN-001 bounded phase-one local evidence

Date: 2026-08-02

Branch: `codex/repair-sci-align-001`

Application candidate exercised: `c77105b9b1676ec1ec74a9d560765954c5f1d5dd`

Candidate parent: `b9787afa68cb45efd862c8892aabfa0806e8576f`

Dispatch evidence-only head: `bfffe0e60fa8ce05a75ae34b89383bceaadb8fc2`

Governing application ancestor: `9aae0e669384c5c0c0dda93debc194d6b8dac787`

Frozen coordination identity: `fff3e02b8e34ba1788014101bd0bd584569b0e80`

## Disposition

The bounded phase-one implementation and local non-reduction gates are
complete. The repair is **not accepted or production-authorized**. The two
completed local reductions are limited compatibility/engineering evidence,
not the preferred scientific evidence lane. Per the final coordinator/owner
clarification, no further local Citlali reduction was launched after the
already-running Beammap completed. Future reduction evidence defaults to a
human-run Unity campaign unless a later dispatch explicitly justifies a
local-only run.

No Unity connection, re-audit, merge, rebase, push, production status change,
polarization expansion, MAP/JINC/AST implementation change, 2-arcsec campaign,
or non-1x production claim occurred.

## Engineering invariants

These are implementation and deterministic-contract conclusions, not angular
science tolerances:

- Round-half-up slot assignment uses checked arithmetic and rejects exact
  half-cell ambiguity, collisions, malformed counters, overflow, unsupported
  edges, and unproved rate profiles.
- Requested, effective, observation-resolved, and realized interface offsets
  are distinct, additive with the approved sign, and applied exactly once.
- The governing assigned-time constructor remains
  `eigen_vectorxd_linspaced_9aae_gap_v1`. Generative and contract tests require
  exact zero ordinary governing-to-candidate assigned-time and coordinate
  displacement. No nonzero tolerance was introduced.
- Synthetic native-rate algebra covers 0.5x, 1x, 2x, and 4x. The two local
  observations are native 1x at 122.0703125 Hz; native observational 0.5x,
  2x, and 4x remain evidence-pending.
- Pointing preserves 12 compatibility outputs and records one additional
  partial identity internally. Beammap preserves 198 compatibility outputs
  and records 43 short identities internally; those 43 are excluded from
  science reductions.
- Pointing records 7,700 common-axis slots, 84,689 acquired original
  interface slots of 84,700 capacity, 11 unavailable slots, no synthesized
  slots, and 198 guarded original slots. Beammap records 383,702 common-axis
  slots, 4,220,705 acquired original interface slots of 4,220,722 capacity,
  17 unavailable slots, no synthesized slots, and no guarded original slots.
- Disabled raw/processed Beammap TOD streams now emit empty selected-window
  sequences rather than validating a nonexistent output selection against the
  admitted scan plan. This changes provenance status only; it does not alter
  the 198 admitted windows or science products.
- The 34.062668 microsecond Pointing half-cell boundary margin remains an
  engineering distance to the slot decision boundary. It is not sky-placement
  accuracy and is not an angular acceptance tolerance.

## Hold producer authority

Late owner-supplied producer evidence defines the Hold word bits as Pointing
0x02, External 0x04, ObsPgm 0x08, M1 0x10, M2 0x20, and M3 0x40. External was
contemplated but never implemented. A native sample is science-valid only when
the complete raw word is zero; any defined or unknown set bit fails closed as
a science-invalid sample.

The approved existing-use compatibility adapter remains whole-word linear
interpolation followed by nonzero, with outside-map-box handled separately.
It is algebraically behavior-preserving for the observed words
`{0,2,8,10,64,66,72,74}`. Raw bits remain typed and distinct internally. No
routine dense per-sample product was added, and no physical transition side or
event timing is claimed. The minimal coordination amendment is to bind
`hold_producer_authority.json` at SHA-256
`d6edb175c3aa62ccf92d9644675ece9c8db572a90146370a9c201c296f211c7e`
without changing scan results or production status.

## Local compatibility runs

Both successful candidate runs used the owner-supplied versioned suite and a
six-thread realized resource cohort. Input identities are frozen in
`local_run_manifest.json` and `selected_output_manifest.csv`.

| Fixture | Exit | Elapsed | Output | Error/critical log entries |
| --- | ---: | ---: | ---: | ---: |
| Pointing 152389 | 0 | 20.643148208 s | approximately 312 MiB | 0 |
| Beammap 148670 | 0 | 3569.753012291 s | approximately 12 GiB | 0 |

The output roots are preserved read-only under
`/private/tmp/citlali-sci-align-001-phase1-c77105b9b1676ec1ec74a9d560765954c5f1d5dd`.
Their run summaries, preparations, logs, runtime provenance, ALIGN provenance,
fit tables, and bounded source-crossing diagnostics are digest-bound in the
machine-readable manifests. These `/private/tmp` products are not committed
to the repository.

Two controlled Pointing attempts at the preceding `b9787afa` candidate and
the natural 12-thread local cohort failed repeatably in the required PTC
NetCDF write path after 4.67 s and 6.22 s. A six-thread retry completed. The
writer path is not an ALIGN repair and was not modified here. This is a
material operational owner-return item: choose a separate non-ALIGN I/O/
resource investigation or pin the exact campaign to six realized threads.
No further local reduction may be launched in this task to investigate it.

The existing 5% setup/total runtime ceiling is not evaluated because no
same-host exact-9aae control exists. The reported elapsed times are identities,
not acceptance claims.

## Measured angular compatibility

The historical products are not an exact-9aae control, so the following are
bounded compatibility measurements only.

The Beammap scale-aware scientific-equivalence comparator accepted:

- 5,234 detector identities exact;
- detector flags exact;
- comparable product sets exact;
- maximum fitted position change 0.0000408376 arcsec;
- maximum fitted FWHM change 0.0000511483 arcsec;
- good-detector kernel RMS relative P99 0.00000793848.

The Pointing fit-table comparison measured centroid displacements of
0.0234552, 0.0276122, and 0.0332304 arcsec for arrays 0, 1, and 2. The largest
absolute fitted major/minor FWHM change was 0.0216012 arcsec and the largest
absolute ellipticity change was 0.00229952. Exact signed components and
per-array PSF changes are in `comparison_disposition.json`.

The strict whole-product comparisons failed, as expected for historical
products from a different whole-application state. Pointing had 673 changed
records across 12 common products and lacked seven historical filtered
products in the realized candidate. Beammap had 15,537 changed records across
13 common products with no missing/extra comparable products. These broad
RTC/PTC/map differences cannot be attributed to ALIGN and cannot establish
the exact-zero acceptance criterion. They demonstrate why the later exact-SHA
control is mandatory.

For reference only, half/full 8.192 ms slot shifts correspond to
0.2048/0.4096 arcsec at 50 arcsec/s, 0.4096/0.8192 arcsec at 100 arcsec/s, and
0.8192/1.6384 arcsec at 200 arcsec/s. These are translations, not tolerances.
SCI-ALIGN owns correct per-sample assigned coordinates; SCI-MAP owns gridding
those coordinates. The existing one-arcsecond downstream products were used
only as bounded compatibility sentinels.

## Physical timestamp authority still unavailable

The detector producer has not established whether timestamps identify the
start, end, or effective temporal centroid of the nominal 8.192 ms
integration, nor that each cadence interval is a contiguous physical
integration. Therefore:

- assigned-time-to-physical-integration-centroid error is unresolved;
- absolute per-sample sky-placement correctness is unresolved;
- candidate-versus-governing exact slot identity would not resolve absolute
  correctness;
- no angular tolerance is derived solely from cadence, the residual maximum,
  or the half-cell boundary margin;
- Hold transition/event timing and left/right continuity remain unresolved.

## Non-reduction gates

- CLI and `citlali_test` targets built with candidate `c77105b9b` embedded.
- CTest: 629 enabled passed, zero failed; documented test 447 remains disabled.
- Baseline tools: 154/154 passed.
- Validation ledger: 60 records valid.
- Science-change ledger: 3 changes and 5 integration commits valid.
- Config tests: 123/123 passed.
- Full config preflight: 8/8 compact profiles, 100% surface coverage, and all
  authority/boundary audits passed.
- Touched public-header translation units compiled in `citlali_test`.
- Runner syntax, Ruff, and `git diff --check` passed.
- Phase 5 readiness remains correctly `preparing`, not promotion-ready.

## Owner return and smallest future request

Do not accept or advance the repair yet. The smallest future human reduction
request is exactly four Unity runs with identical host/allocation, realized
threads, configuration, dependencies, input digests, and storage class:

1. Pointing 152389 at exact `9aae0e669384c5c0c0dda93debc194d6b8dac787`.
2. Pointing 152389 at candidate `c77105b9b1676ec1ec74a9d560765954c5f1d5dd`.
3. Beammap 148670 at exact `9aae0e669384c5c0c0dda93debc194d6b8dac787`.
4. Beammap 148670 at candidate `c77105b9b1676ec1ec74a9d560765954c5f1d5dd`.

Require exact zero ordinary assigned-time/coordinate displacement, exact
unaffected products, only the approved OD5 compact internal identity/status
additions, unchanged accepted scan windows and source/centroid/PSF behavior,
and no repeatable setup or total runtime regression above 5%. Return to the
owner on any nonzero ordinary scientific change.

Separately, the owner must decide whether the 12-thread PTC write failure gets
a non-ALIGN investigation or the campaign is pinned to six realized threads.
Absolute sky correctness remains unavailable until detector timestamp
producer semantics are proved.

## Artifact index

- `local_run_manifest.json`: candidate, input, run, failure, and runtime
  identities.
- `selected_output_manifest.csv`: selected historical/candidate fit,
  provenance, and bounded diagnostic file digests.
- `comparison_disposition.json`: exact engineering versus measured angular
  versus unresolved physical-timestamp conclusions.
- `gate_results.json`: non-reduction gate results.
- `owner_decision_brief.json`: concise CAL/AST/MAP-safe return brief.
- `hold_producer_authority.json`: late producer semantics overlay.
- `changed_paths.tsv`: implementation changed-path inventory relative to
  dispatch head `bfffe0e60`.
- `*_historical_strict_comparison.{json,md}`: full historical triage tables.
- `beammap_historical_scientific_equivalence.{json,md}`: bounded accepted
  Beammap compatibility sentinel.
- `SHA256SUMS`: repository artifact digests.
