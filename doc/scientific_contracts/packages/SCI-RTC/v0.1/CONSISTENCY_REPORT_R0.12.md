# SCI-RTC v0.1/r0.12 consistency report

Date: 2026-08-21

Status: Implementation-blind candidate verification. This is not an owner
freeze, implementation-conformity assessment, validation result, performance
claim, science qualification, or production promotion.

## Source and boundary checks

- [x] Comparison baseline is approved r0.11 architecture commit
  `85e1e6c6865f74f1a97e99fab465714f43877c3d`.
- [x] Supplied review digest is
  `432b7cbeccdee0b9a41e75b15ee11cc2aced2a8c474248914c14d1885ab1309b`.
- [x] Only scientific-contract source, core, crosswalk, ledger, review records,
  verifier, and canonical PDFs were inspected or changed.
- [x] Conditioned-$x$ numerical behavior, CAL, PTC policy, SCI-VAL policy, and
  unrelated owner decisions remain outside the correction.

## Mechanical checks

- [x] Verifier passes sequential inventories, exact shared-core inclusion,
  decision counts, and candidate PDF hashes.
- [x] Inventory is 52 definitions, 44 equation tags, 12 assumptions, 143
  requirements, 108 predictions, 24 author decisions, and 103 owner entries.
- [x] Ledger states are 63 open, 1 conditional, 34 resolved, and 5 deferred.
- [x] Both PDFs compile without warnings and Poppler extraction finds r0.12,
  EQ-042, REQ-143, PRED-108, and the r0.12 end-of-core marker.
- [x] Every final PDF page has been rasterized and visually inspected.

Canonical PDF hashes and the all-page QA disposition are recorded in
`PDF_VISUAL_QA_AND_SOURCE_IDENTITY_R0.12.md`.
