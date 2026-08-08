# SCI-NOI-002 Cycle 4 independent re-audit — 2026-08-07

Status: `complete_pending_coordinator_ledger_integration_and_owner_push`.
Controlled verdict: `retain`. The exact Cycle 4 application candidate is
`conformant` within the frozen repair scope, and the applicable deterministic
validation is `complete`. Production remains `existing_use_only`.

This is a fresh, role-separated re-audit of exact application candidate
`5b29e13548a6fec884c67b192dec20c92f0bbb62`. It accepts the bounded repair:
C4-R001--C4-R004 are satisfied, and
`SCI-NOI-002-C3RA-P1-001` through `P1-003` close. The result does not itself
integrate application code, mutate canonical status or the audit ledger,
authorize production expansion, close external FLT/FRUIT work, or select a
realization count or default.

## Controlled outcome

| Axis | Controlled disposition |
| --- | --- |
| contract | `approved`; D001--D008 remain settled |
| implementation | `conformant` at exact candidate `5b29e135...` within the bounded Cycle 4 scope |
| validation | `complete` for the accepted local writer/finalizer/auditor and configuration gates |
| production | `existing_use_only`; expansion requires a separate coordinator/owner decision |
| verdict | `retain`; bounded application disposition `accept` |
| repair/re-audit chain | complete and ready for coordinator ledger/status review |

F003 and F004 close. F007 closes only for the implemented and re-audited
requested/effective/completed-count, incomplete-execution, and disabled-zero
contract; this is not count-adequacy evidence or a default recommendation.
F005 remains `open_conditioned` with parity status exactly
`scope_blocked_not_applicable_pending_FLT` under SCI-FLT-001. F006 remains
`open`, `held_external`, and SCI-FRUIT-001-owned. Correspondingly, RA-B001
and RA-B003 close; RA-B004 remains
`local_repair_pass_finding_open_conditioned`.

## Exact target, lineage, and changed-path accounting

| Fact | Exact value |
| --- | --- |
| Audit branch | `codex/reaudit-sci-noi-002-cycle4` |
| Worktree | `/Users/gwilson/.codex/worktrees/a65b/citlali-refactor` |
| Candidate / audit-branch entry | `5b29e13548a6fec884c67b192dec20c92f0bbb62` |
| Candidate parent | `390edf4f8c696551921c615f2439e956d240ec1d` |
| Candidate tree | `641c724f40a9fa9f322f09c703705239439d2374` |
| Candidate subject | `Repair SCI-NOI-002 Cycle 4 package reconciliation` |
| Locally stored repair ref | `origin/codex/repair-sci-noi-002` = exact candidate |
| Governing Cycle 3 audit commit | `b45da53708dcb05e22f284d6a815bab47caefa40` |
| Cycle 3 audit parent / tree | exact candidate parent `390edf4f...` / `7fa37d19ba874a2bd07af737c7f344ba21bd21f4` |
| Cycle 4 coordination commit | `b670c0163e02152953dbf44ab41bd299ad3ee768` |
| Coordination parent / tree | `afedbad3a0d7f8829555f84bc8a21974992a5dfc` / `ce61ff7c7fc264f7a4960d36f07251c8d9afc0f3` |

The dedicated audit branch started directly from the exact candidate, not from
an earlier audit, re-audit, or coordination branch. Clean entry and the local
repair ref were verified before evidence execution. The documentation commit
that contains this report is intentionally not self-referential; its commit
identity is returned to the coordinator after creation.

The parent-to-candidate diff changes exactly nine existing paths, with 1,401
additions, 130 deletions, and no added, deleted, or renamed file:

1. `include/citlali/core/engine/detail/beammap_map_product_writers_impl.h`
2. `include/citlali/core/pipeline/map_source_table_output.h`
3. `include/citlali/core/pipeline/mapdiag_netcdf_map_double_values.h`
4. `include/citlali/core/pipeline/noise_execution_plan.h`
5. `include/citlali/core/pipeline/noise_provenance.h`
6. `tests/test_config_scaffold.cpp`
7. `tests/test_science_map_fits_products.cpp`
8. `tools/baseline/audit_reduction_run.py`
9. `tools/baseline/test_audit_reduction_run.py`

No unauthorized application, test, configuration, or contract path appears in
the candidate diff. This re-audit adds only this report and its result and
ledger-update-proposal companions.

## Frozen authorities and verified digests

All values below were recomputed over exact Git-object bytes before candidate
evidence was accepted.

| Authority or evidence | Exact Git object | SHA-256 |
| --- | --- | --- |
| Audit framework README | `b670c016:doc/audits/README.md` | `d84f6d8b77f9c251fa427bbeb98a2ddd77d3393c90b1f6424cd5bb37f888cdad` |
| Audit-manager instructions | `b670c016:doc/audits/AUDIT_MANAGER_INSTRUCTIONS.md` | `31f85b597af0490d31718b837ec1955468de81b694b77649a5bac6f0fac8887d` |
| Cycle 4 coordinator disposition | `b670c016:doc/audits/packages/SCI-NOI-002_CYCLE4_COORDINATOR_DISPOSITION_AND_REPAIR_HANDOFF_2026-08-07.md` | `bb8060f704fbc2abccffdf1fb1b5388364568653bed0b82bbbad242f213aec10` |
| Cycle 4 repair prompt | `b670c016:doc/audits/prompts/SCI_NOI_002_CYCLE4_REPAIR_PROMPT.md` | `2beda29726d3da4f13da83e277e26200431536ae11199f27231cf6e7c6896a27` |
| Cycle 4 authority manifest | `b670c016:doc/audits/handoffs/SCI-NOI-002/SCI-NOI-002_CYCLE4_REPAIR_AUTHORITY_MANIFEST_2026-08-07.yaml` | `ed3b1cb2776d484670f21b4fe57d509583701b3f1cb4fc9feb746ba638d43b21` |
| Cycle 4 dispatch readiness | `b670c016:doc/audits/packages/SCI-NOI-002_CYCLE4_REPAIR_DISPATCH_READINESS_2026-08-07.md` | `3786c31c1617b8cebdffec67c26fe838b4f2cba7f919de7352f5531f1da57697` |
| Cycle 3 independent re-audit | `b45da537:doc/audits/packages/SCI-NOI-002_CYCLE3_INDEPENDENT_REAUDIT_2026-08-07.md` | `cef029918bc9923d2f20e479a1bfcda02027c658359c186835ceff2643b6a139` |
| Cycle 3 result | `b45da537:doc/audits/results/SCI-NOI-002_CYCLE3_REAUDIT_RESULT_2026-08-07.yaml` | `e801fb991c146d0af3f522edbbee3dcc45f22975b4a08868546b0df11dbe4ecf` |
| Cycle 3 ledger proposal | `b45da537:doc/audits/proposals/SCI-NOI-002_CYCLE3_REAUDIT_LEDGER_UPDATE_PROPOSAL_2026-08-07.yaml` | `f46ee2c6f90bab5f81ba34a1a0c3d5a91badaa1b11031ce1e46dd0f964e13451` |
| Inherited Cycle 2 authority manifest | `28784071:doc/audits/handoffs/SCI-NOI-002/SCI-NOI-002_CYCLE2_REPAIR_AUTHORITY_MANIFEST_2026-08-06.yaml` | `36602cd0ba779a3fe9d9419c3b0ed53d86e26eb4c887ec7b5d19c0442ad86202` |
| Owner decisions D001--D008 | `64ba8179:doc/audits/packages/SCI-NOI-002_OWNER_DECISION_BRIEF_2026-08-06.md` | `3520172cfc11e8e34f280f9ebdf147ea414c7a3a4ca6109bad55354a5ff3cf71` |

The verified SCI-NOI-002-XAUD-001 digest used for future references remains
`dfcd59e9d59395ba84f7dfed1656690daae694872c2a1a40bf4f5c79f6abed3a`.
Frozen Cycle 2 artifacts containing the historical 63-hex transcription remain
byte-preserved.

## Scientific contract and binding clarification

The estimator remains the empirically centered second moment of the exact
completed `source_imprinted_current` stack:

\[
\bar{Y}_p = \frac{1}{R}\sum_{r=1}^{R}Y_{r,p},\qquad
V_p = \frac{1}{R}\sum_{r=1}^{R}(Y_{r,p}-\bar{Y}_p)^2.
\]

It remains conditional finite-stack scatter, not iid-unbiased sample variance,
repeated physical-noise variance or covariance, inverse-variance precision,
calibrated significance, aperture uncertainty, a production calibration, or
evidence that a configured realization count is scientifically adequate.

The coordinator clarification is binding:

- successor coadds publish no formal-coefficient, empirical-scatter, or
  coefficient-standardized-signal empirical companion FITS bundle and add no
  empirical-map count for such a bundle;
- already configured standalone coadd realization files remain published,
  remain NOI members, retain dedicated realization counters, and preserve
  output selection, order, names, layout, and persistence; and
- those standalone realization files are not the prohibited companion bundle.

The candidate and this re-audit preserve all estimator equations,
normalization, realization generation/sign, configured counts/defaults,
mapmaking/filter/Beammap/coadd numerical behavior, output
selection/order/name/layout, and auxiliary-channel behavior.

## Independent finding-to-implementation trace

### C4-R001 / SCI-NOI-002-C3RA-P1-001 — exact compact missingness

The Cycle 3 finding is accepted as the governing defect report. The candidate:

1. adds `missingness: nonfinite_unavailable` to production ECSV source-table
   metadata in `map_source_table_output.h`;
2. adds ordered `missingness=nonfinite_unavailable` to all three production
   NOI Mapdiag NetCDF records in
   `mapdiag_netcdf_map_double_values.h`;
3. makes the final C++ package validator require the exact ECSV map and exact
   ordered NetCDF record in `noise_provenance.h`;
4. makes the active Python auditor require exactly one raw ECSV missingness key
   and the same exact ECSV values, and require the same ordered NetCDF record;
   and
5. supplies positive and missing/empty/wrong/duplicate/swapped fixtures on
   both validator paths.

The ECSV field set is exactly `package_id`, `provenance_id`, `column`,
`product_identity`, `product_version`, `semantic_digest`, `digest_kind`,
`missingness`, `scope`, `validity`, and `restriction`. NetCDF uses
`variable` instead of `column`, in that exact order after schema marker
`citlali_noise_product_join_v1`.

Disposition: C4-R001 is satisfied and
`SCI-NOI-002-C3RA-P1-001` is closed.

### C4-R002 / SCI-NOI-002-C3RA-P1-002 — successor-coadd reconciliation

The candidate makes expected, observed, member, and final package accounting
describe one mode-aware policy:

- a coadd stage rejects a nonzero empirical-companion count;
- coadd scientific-map and realization generation/write counts remain
  recorded;
- unscaled coadd data has no NOI join and is not an NOI member;
- a scaled coadd data product may carry exactly one standalone
  `global_nonprecision_scaled_coefficient` identity, with empirical-map
  contribution zero; and
- separately configured coadd realization files remain members with two
  realization joins in the exercised configuration.

The production fixture writes an observation and actual successor coadd through
final C++ publication for both scale branches. It proves the observation's one
empirical logical map remains one, the coadd adds zero, four realization image
writes remain overall, and package membership is three files unscaled or four
files scaled. The active Python fixture independently accepts an isolated
coadd package with empirical count zero and two realizations in both branches,
adds only the scaled identity in the scaled branch, and rejects a false
empirical count/bundle claim.

Disposition: C4-R002 is satisfied and
`SCI-NOI-002-C3RA-P1-002` is closed.

### C4-R003 / SCI-NOI-002-C3RA-P1-003 — split Beammap reconciliation

The candidate preserves the existing per-array file layout while reconciling
logical selected detector maps:

- only array file indices reached by selected detector maps enter NOI
  membership;
- logical map identity is derived from existing product identity, EXTNAME, and
  realization scope encoding;
- canonical product identities may repeat across distinct logical maps but not
  within one;
- realization scopes are unique and exactly zero-based within each logical
  map and may restart only after logical-map identity changes;
- empirical cardinality counts complete logical map bundles rather than files
  or HDUs; and
- an admitted file with no NOI join remains invalid.

Production fixtures pass both actual shapes: one selected map in each of two
array files, and two selected maps in one array file with the unused array
excluded. Both shapes have selected/empirical logical-map count two and
realization-image count four. An inconsistent selected count, deliberately
admitted empty file, duplicate identity, and duplicate realization scope fail
closed. The active Python fixtures independently enforce the same identities,
counts, membership, per-map restarts, and failures.

Disposition: C4-R003 is satisfied and
`SCI-NOI-002-C3RA-P1-003` is closed.

### C4-R004 — production-shape and paired-parity evidence

The coordinator accepted paired exact C++ and Python fixtures for this bounded
re-audit. No persisted C++-to-Python artifact, new harness, helper, schema, or
fixture subsystem was required or authorized. The assessment compared actual
identities, field sets/order/values, counts, membership, selected-map
cardinality, and failure semantics.

| Contract case | Exact paired assessment | Result |
| --- | --- | --- |
| Exact ECSV missingness | Both require the exact 11-field contract, one `sig2noise` column, and `missingness=nonfinite_unavailable`. | pass |
| ECSV missing/empty/wrong/duplicate | Both reject each class; Python also proves raw duplicate-key detection before YAML-map collapse. | pass |
| Exact NetCDF missingness | Both require the same schema marker, exact ordered 11 fields, three variable/value bindings, and product identities. | pass |
| NetCDF missing/empty/wrong/extra | Both reject missing, empty, wrong, duplicated/extra missingness, duplicate scope, and swapped variable identity. | pass |
| Unscaled successor coadd | Coadd empirical increment is zero, data has no NOI join, and two configured realization joins remain members. | pass |
| Scaled-only successor coadd | Exactly one standalone `global_nonprecision_scaled_coefficient` join is admitted with empirical increment zero; realizations remain. | pass |
| False coadd empirical claim | C++ rejects nonzero coadd companion accounting; Python rejects the semantic and empirical-inventory mismatch. | pass |
| One Beammap per array file | Two selected maps across two files yield empirical count two, realization count four, and exact data/realization membership. | pass |
| Multiple Beammaps in one file | Two logical maps in one file retain distinct identities and independent zero-based realization scopes. | pass |
| Zero-map array file | Empty physical files remain outside membership; deliberate admission fails for no NOI join. | pass |
| Duplicate identity | Repetition within one logical map is rejected; repetition across distinct maps is accepted. | pass |
| Duplicate realization scope | Duplicate scope within one map is rejected; restart across distinct maps is accepted. | pass |
| Selected-map cardinality mismatch | Final reconciliation rejects reported/expected cardinality inconsistent with observed logical maps. | pass |

Disposition: C4-R004 is satisfied.

## Deterministic execution provenance and results

The independent build tree was configured as Release with tests enabled. Two
infrastructure-only stop classes preceded the successful run: creation/fetch
of the absent independent build tree, including the declared kidscpp update,
and correction of the canonical default tests-disabled profile. The successful
command was:

```sh
BUILD_TESTS=ON make local-bootstrap
```

The authorized network permission was limited to update/fetch operations for
public dependency repositories already declared by the candidate and their
declared nested dependencies. Test-profile fetches included
`google/benchmark` tag `v1.6.0` and `google/googletest` `main`, both
declared through the existing test dependency graph. Configuration completed
with `CMAKE_BUILD_TYPE=Release` and `CITLALI_BUILD_TESTS=ON`; all four
required targets were present. No package installation, arbitrary URL,
dependency version override, credential use, source edit, or tracked-file
mutation occurred. CMake developer/deprecation warnings were non-fatal and no
unexpected error-level message occurred.

| Exact gate | Exact result |
| --- | --- |
| build `citlali_cli citlali_test citlali_science_map_fits_products_test citlali_safety_test -j 8` | pass; 4/4 targets |
| recorded focused `citlali_test` filter | 40/40 pass; 0 failed/skipped |
| recorded two-test science-product filter | 2/2 pass |
| full `citlali_science_map_fits_products_test` | 32/32 pass |
| `tools/baseline/test_audit_reduction_run.py` | 88/88 pass |
| product-contract validator unit suite | 23/23 pass |
| reduction validator unit suite | 9/9 pass |
| science-change-ledger validator unit suite | 7/7 pass |
| validation-ledger validator | valid; 60 records |
| science-change-ledger validator | valid; 3 changes, 5 integration commits |
| full CTest | 624 registered; 623/623 runnable pass; 0 failed; 1 unrelated disabled |
| full config preflight `--require-all` | pass; 127/127 tests, 4/4 mode kits, 8/8 compatibility cases, 0 skipped |
| config contract coverage | 261 covered, 17 profile-owned, 0 gaps, 100%; 591 leaves, 589 executable, 2 non-executable, 15 authorities |
| exact `390edf4f..5b29e135` `git diff --check` | pass; no output |

The disabled CTest is
`citlali::MapFitterLifecycle.ExactProductSequence`. Its source explicitly
classifies it as an opt-in replay of an external 2026-07-23 pointing corpus
requiring three replay environment variables. It is non-required, unrelated to
NOI, and not a skip or failure of this re-audit. No required case skipped.

Focused, full-target, CTest, and preflight counts overlap by design; they are
reported command by command and are not presented as a unique-test aggregate.

## Finding and decision dispositions

### Cycle 4 and Cycle 3

| Record | Final disposition |
| --- | --- |
| C4-R001 | satisfied |
| C4-R002 | satisfied |
| C4-R003 | satisfied |
| C4-R004 | satisfied |
| SCI-NOI-002-C3RA-P1-001 | closed |
| SCI-NOI-002-C3RA-P1-002 | closed |
| SCI-NOI-002-C3RA-P1-003 | closed |

### Earlier repair decisions

| Decision | Final disposition |
| --- | --- |
| RA-B001 | `closed`: exact compact joins and actual coadd/Beammap membership, identity, digest, and cardinality reconcile |
| RA-B002 | `closed`: prior disabled/no-work and literal enabled-zero closure remains supported |
| RA-B003 | `closed`: plan expectations and observed successful-publication counts now reconcile for the actual shapes |
| RA-B004 | `local_repair_pass_finding_open_conditioned`; parity remains `scope_blocked_not_applicable_pending_FLT` |
| RA-R001 | `closed`; one compatible stored physical plane remains unchanged |
| RA-R002 | `closed`; Mapdiag names, calculations, values, order, comments, joins, and restrictions remain exact |

Cycle 2 repair findings P1-003 and P1-004 now close because their exact
package-reconciliation and compact-join criteria pass. P1-005 remains
`local_repair_pass_finding_open_conditioned`. All other previously closed
Cycle 2 repair findings remain closed.

### Original findings

| Finding | Final disposition and boundary |
| --- | --- |
| F001 | `closed`; retained conditional finite-stack estimator identity |
| F002 | `closed`; retained global nonprecision diagnostic restriction |
| F003 | `closed`; exact package-level provenance, integrity, compact joins, and actual member shapes pass |
| F004 | `closed`; distinct descriptive/engineering identities and not-significance restrictions pass; no SCI-SRC significance/catalog claim is inferred |
| F005 | `open_conditioned`; parity `scope_blocked_not_applicable_pending_FLT`, owner SCI-FLT-001 |
| F006 | `open`, manifest state `held_external`, owner SCI-FRUIT-001 |
| F007 | `closed` for exact requested/effective/completed and incomplete-execution behavior; no count adequacy/default recommendation or production claim |
| F008 | `closed`; proportional local evidence closure remains supported |

D001--D008 remain `settled_unchanged` within their recorded qualifications.
F003 closure does not authorize dense covariance, per-sample identity/sign
ledgers, or duplicated full semantics in detached products. F004 closure does
not authorize calibrated significance, source-selection tail probability, or
catalog completeness. F007 closure does not establish convergence adequacy for
Science=10, Pointing=5, 64, or any other count.

## Regression, scope, and remaining limitations

No candidate regression, scientific scope expansion, or new finding was
identified. The following remain unchanged:

- no estimator, normalization, realization generation, or sign change;
- no configured count or default selection;
- no mapmaking, filtering, Beammap, coadd, FRUIT, MODE, MAP, JINC, RTC, PTC,
  Wiener, or auxiliary-channel numerical change;
- no output selection, order, filename, layout, or file-partition change;
- no physical-noise variance/covariance, inverse-variance precision,
  calibrated significance, aperture uncertainty, dense covariance,
  per-sample identity/sign ledger, or `r/I/Q/phase` substitution; and
- no production-size performance, I/O, astronomical, or convergence claim.

Production remains `existing_use_only`. F005 requires the separately owned
SCI-FLT-001 response/support contract before its blocked parity can become
applicable. F006 remains wholly owned by SCI-FRUIT-001. Any count-adequacy
recommendation, production expansion, application integration, canonical
ledger/status mutation, recipient dispatch, or external campaign requires a
separate coordinator/owner action.

No Unity host was accessed, no Citlali or astronomical reduction was run, and
no external scientific/operational evidence was requested. The only network
activity in the re-audit was the explicitly authorized declared-dependency
bootstrap described above. No delegation occurred.

## Companion artifacts and acyclic digest binding

This report is accompanied by:

- `doc/audits/results/SCI-NOI-002_CYCLE4_REAUDIT_RESULT_2026-08-07.yaml`;
  and
- `doc/audits/proposals/SCI-NOI-002_CYCLE4_REAUDIT_LEDGER_UPDATE_PROPOSAL_2026-08-07.yaml`.

The established acyclic pattern is used: the proposal records this report's
SHA-256; the result records the report and proposal SHA-256 values; neither
file contains its own digest. The result digest and containing audit commit are
returned out of band to the coordinator. The proposal is noncanonical until
the coordinator reviews and applies or supersedes it.

Stop for coordinator review and owner push. Do not push, integrate, expand
production, or mutate canonical ledger/status state from this audit task.
