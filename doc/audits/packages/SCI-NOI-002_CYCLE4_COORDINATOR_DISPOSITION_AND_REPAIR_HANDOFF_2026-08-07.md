# SCI-NOI-002 Cycle 4 coordinator disposition and bounded repair handoff — 2026-08-07

Status: `coordinator_approved_cycle_4_repair_ready_for_frozen_dispatch`.
This record accepts the fresh Cycle 3 independent re-audit at audit commit
`b45da53708dcb05e22f284d6a815bab47caefa40` as the governing assessment of
exact application candidate
`390edf4f8c696551921c615f2439e956d240ec1d`. It authorizes one bounded
engineering-correctness successor repair. It does not integrate application
code, change estimator or filter mathematics, request Unity or astronomical
evidence, alter a configured realization count or default, launch a re-audit,
or expand production beyond `existing_use_only`.

## Accepted disposition

The Cycle 3 verdict is `amend; do not integrate`. There are no P0 or P2
findings and three P1 findings:

1. `SCI-NOI-002-C3RA-P1-001`: compact ECSV and NetCDF joins omit the required
   missingness field, and both active validators accept that omission.
2. `SCI-NOI-002-C3RA-P1-002`: successor-coadd expected and observed empirical
   identity/cardinality accounting contradicts the approved coadd output
   policy.
3. `SCI-NOI-002-C3RA-P1-003`: actual split detector-group Beammap files may
   contain multiple detector-map bundles and per-map realization sequences,
   while the package validator assumes one bundle and one realization scope
   sequence per file; unused per-array split files are also admitted.

These are engineering publication/validation defects under settled owner
decisions. No new scientific-owner decision is required.

## Binding Cycle 4 repair decisions

### C4-R001 — exact compact missingness parity

Add the required `missingness` field to the existing compact ECSV and NetCDF
joins. The production writers, C++ finalizer, Python auditor, and deterministic
fixtures must use the same exact field set and exact value for each product.
FITS `NOIMISS` remains the equivalent FITS authority. Missing, empty, extra,
duplicated, or wrong missingness must fail closed. Do not duplicate full
package semantics into products or create a new join framework.

### C4-R002 — mode-aware successor-coadd reconciliation

Preserve the approved successor-coadd output policy and existing calculations:

- successor coadds do not publish the empirical scatter, standardized-signal,
  realization, or related empirical companion family;
- a coadd with no empirical weight application is not an empirical NOI package
  map and must not increment expected or observed empirical-product
  cardinality or require an empirical FITS member;
- when the existing global nonprecision empirical scale is applied, the
  resulting scaled coefficient is a permitted standalone diagnostic identity,
  not evidence that a full empirical companion bundle exists, not a precision
  or significance product, and not an empirical-map bundle count;
- plan-derived expected counts, observed successful-publication counts, member
  identity, and final reconciliation must all describe that same mode-aware
  policy.

Do not add missing companion products merely to satisfy the validator. Do not
change coadd values, weighting, map selection, filenames, estimator meaning,
or D002's `existing_use_only` nonprecision restriction.

### C4-R003 — actual split-Beammap package reconciliation

Preserve the existing split detector-group Beammap layout, selection, and
numerical behavior. One per-array FITS file may contain zero, one, or multiple
selected detector-map bundles. Package validation must reconcile logical
products per selected detector map, not assume one logical bundle per file.

Each accepted logical map bundle must have an exact, unambiguous map identity
derived from its existing stored product identity/EXTNAME structure. Repeated
canonical product types are allowed only across distinct logical detector-map
bundles. Realization indices may restart only within a distinct logical map
bundle and must be unique and cardinality-correct inside that bundle. Duplicate
identities or realization scopes within the same logical bundle remain
invalid. Expected and observed counts remain counts of selected published
logical maps, not counts of files or HDUs.

A per-array split file containing no selected detector map and no NOI product
join must not be admitted to the NOI package inventory. Zero-selection
fallback and standard Beammap behavior remain unchanged. Do not change flag
selection, detector order, filenames, file partitioning, map values, Beammap
mathematics, or the default-disabled optional noise-map capability.

### C4-R004 — production-shape validation

Add deterministic writer-to-final-package fixtures for both successor-coadd
policy branches and actual split-Beammap shapes. Tests must pass through the
existing production writer/product representation and the final C++ and Python
reconciliation logic; isolated counter arithmetic or hand-built happy-path
documents are insufficient.

Required negative cases include wrong/missing compact missingness, a coadd
falsely claiming or requiring an empirical bundle, a scaled-only coadd treated
as a full bundle, an admitted empty split file, duplicate product identity
within one detector-map bundle, duplicate realization scope within one bundle,
and inconsistent selected-map cardinality. Required positive cases include
unscaled successor coadd, scaled-coefficient-only successor coadd, one- and
multi-detector split files, arrays with no selected detector excluded from NOI
membership, and C++/Python parity.

## Preserved dispositions and exclusions

- RA-B001 and RA-B003 remain not closed until Cycle 4 passes a fresh
  independent re-audit. RA-B002, RA-R001, and RA-R002 remain closed.
- RA-B004 remains `local_repair_pass_finding_open_conditioned`.
- F001, F002, and F008 remain closed. F003 remains open. F004 and F007 remain
  `open_conditioned` pending this repair and re-audit.
- F005 remains `open_conditioned` with parity status exactly
  `scope_blocked_not_applicable_pending_FLT`, owned by SCI-FLT-001.
- F006 remains open, `held_external`, and SCI-FRUIT-001-owned.
- D001–D008 remain settled. No count/default recommendation, physical-noise
  variance, precision, calibrated significance, aperture uncertainty, dense
  covariance, per-sample identity, sign stream, or auxiliary `r/I/Q/phase`
  substitution is authorized.
- No FLT, SRC, FRUIT, MODE, MAP, JINC, RTC, PTC, coadd, Beammap, or Wiener
  numerical algorithm change is authorized.
- The verified `SCI-NOI-002-XAUD-001` digest for future references is the
  64-hex value
  `dfcd59e9d59395ba84f7dfed1656690daae694872c2a1a40bf4f5c79f6abed3a`.
  Frozen Cycle 2 artifacts containing the 63-hex transcription remain
  byte-preserved.

## Exact continuation and return gate

Continue the existing isolated application repair branch
`codex/repair-sci-noi-002` from exact clean commit
`390edf4f8c696551921c615f2439e956d240ec1d`. The successor must be its child.
Never use audit commit `b45da53708dcb05e22f284d6a815bab47caefa40`,
the coordination line, or another audit branch as the application base.

The task must first verify the frozen manifest and return a scope checkpoint.
After coordinator continuation it may implement only C4-R001–R004, run the
specified proportional/full local gates, create one coherent repair commit,
and stop. No push, integration, Unity or astronomical evidence, production
action, canonical-ledger mutation, or re-audit launch is authorized from the
repair task.
