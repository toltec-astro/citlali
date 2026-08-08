# SCI-NOI-002 Application Integration Decision — 2026-08-08

## Decision

The project owner and coordinator authorize the accepted bounded
`SCI-NOI-002` candidate for application-mainline integration. This is an
application integration and status-alignment decision, not a production
release, new estimator decision, realization-count recommendation, Unity
campaign, or closure of SCI-FLT-001 or SCI-FRUIT-001 work.

The dedicated `codex/integrate-sci-noi-002` branch was created from exact
current `origin/codex/refactor-mainline`
`d5015fe716971bf8ea617e8a187311bf5af05185` and advanced by exact fast-forward
to accepted repair tip `5b29e13548a6fec884c67b192dec20c92f0bbb62`.
Before this record and the two live status documents were edited, the branch
tree was exactly the audited application tree
`641c724f40a9fa9f322f09c703705239439d2374`. The commit containing this
record is a later documentation-only child of `5b29e135...`; it is the
integration-candidate tip, not a different application-source revision.

## Verified topology and authorities

- Current local and origin-tracking application mainline:
  `d5015fe716971bf8ea617e8a187311bf5af05185`.
- Accepted repair tip and pushed repair branch:
  `5b29e13548a6fec884c67b192dec20c92f0bbb62`.
- Merge-base and direct ancestor: `d5015fe716971bf8ea617e8a187311bf5af05185`.
- Mainline/candidate divergence count before integration: zero mainline-only,
  six candidate-only commits.
- Exact linear series:
  `0bc4d95d6bb2117442d0ccdb79c57e42e0b79989`,
  `d1d19145d`, `de18f0610`, `63efd8b08`, `390edf4f8`, and
  `5b29e1354`.
- Accepted independent Cycle 4 re-audit:
  `6de648f5ae2b37f5bc65162feae221f19bb84a5a`.
- Accepted coordinator closeout:
  `d03ef80b31f704859ef836e368801dc17d92e76e`.
- Cycle 4 report SHA-256:
  `aa03f3bc94fda349c4f1bcb16274ad9fb21195a3aef26a58383d12fce2af37e5`.
- Cycle 4 machine-readable result SHA-256:
  `da6ed16e6a93fd10539bc001014818e99a45361cf0b2442ecc32738e3a2f6b8c`.
- Cycle 4 ledger proposal SHA-256:
  `a42180d881f4db56870a9ecd5221a4da92338686e9b2f1209486eba3fa2305f1`.
- Coordinator closeout SHA-256:
  `2efc4d571dafaaf29450f4f9814e1b286193b7d44820161e1fa759cdcfeefcad`.

The exact fast-forward modifies 23 application, test, validator, and product-
contract paths relative to `d5015fe...`, with 7,068 insertions and 366
deletions. No audit or coordination branch was merged. There was no conflict,
resolution commit, patch reconstruction, or change to the audited application
bytes.

## Accepted disposition and limits

The bounded package axes are `approved`, `conformant`, `complete`, and
`existing_use_only`, with controlled verdict `retain` and bounded application
disposition `accept`. F001, F002, F003, F004, F007, and F008 are closed within
their recorded contracts. Cycle 3 P1-001 through P1-003 and repair findings
RA-B001, RA-B002, RA-B003, RA-R001, and RA-R002 are closed.

F005 and RA-B004 remain `open_conditioned`, with parity state
`scope_blocked_not_applicable_pending_FLT`, under SCI-FLT-001. F006 remains
open and `held_external` under SCI-FRUIT-001. F004 closure establishes
truthful descriptive and engineering identities, not calibrated significance
or catalog authority. F007 closure establishes requested/effective/completed
count truthfulness, not realization-count adequacy or a recommended default.

This integration makes no physical-noise variance, inverse-variance,
precision, significance, aperture-uncertainty, source-catalog, dense-
covariance, or auxiliary-channel substitution claim. Production remains
`existing_use_only` and requires a separate later decision.

## Validation basis

The accepted exact-candidate re-audit includes four of four required Release
targets; 40/40 focused core tests; 2/2 focused and 32/32 full science-product
tests; 88 baseline-auditor, 23 product-contract, 9 reduction-validator, and 7
science-change-ledger Python tests; 623/623 runnable CTests; and the complete
127-test configuration preflight with four mode kits, eight compact
compatibility cases, and no gap or required skip. The disabled external
pointing-corpus replay is unrelated and is not an NOI skip.

The paired exact C++/Python matrix covers compact ECSV/NetCDF missingness,
successor-coadd membership and counts in both scale branches, preservation of
configured standalone realization files, false empirical claims, split
Beammap zero/one/multiple logical maps, per-map scopes, cardinality, and
negative production shapes. The re-audit found no estimator, normalization,
realization-generation/sign, count/default, mapmaking/filter, output-selection
or layout, physical-variance, or significance scope expansion.

No astronomical or Unity reduction was required or claimed for the bounded
Cycle 4 closure. `validation/intended_science_changes.json` is therefore left
unchanged: its current validator requires every entry to cite an accepted
reduction, and neither weakening that policy nor attaching an unrelated
historical run is authorized. The externally visible product-contract and
schema corrections are instead bound by the exact audited
`validation/product_contracts.json`, the independent report/result/proposal,
and the canonical coordinator closeout. Any future change-ledger policy that
admits exact deterministic audit evidence is a separate framework decision.

## Integration boundary and owner handoff

After exact candidate `5b29e135...`, this branch changes only this handoff,
`doc/REFACTOR_STATUS.md`, and `doc/INTEGRATION_LEDGER.md`. It changes no
application source, test, configuration, numerical behavior, product
contract, validation registry, audit artifact, or coordination record.

The proportionate integration gates are exact tree identity, exact six-commit
ancestry, documentation-link integrity, full diff review, `git diff --check`,
the validation and science-change ledger validators, product-contract
validation, and clean committed state. The complete build/test/config gates
are not repeated because they already passed at the byte-identical exact
candidate and this child changes documentation only.

`codex/refactor-mainline` is deliberately not moved by this preparation task.
After review, the owner may push `codex/integrate-sci-noi-002` and fast-forward
mainline to the exact integration-candidate tip. No push, Unity action,
production expansion, or Conan-lane update is part of candidate preparation.
