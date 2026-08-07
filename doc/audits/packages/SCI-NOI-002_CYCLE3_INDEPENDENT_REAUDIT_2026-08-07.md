# SCI-NOI-002 Cycle 3 independent re-audit — 2026-08-07

## Outcome

This is a fresh independent re-audit of the exact cumulative Cycle 3
application candidate
`390edf4f8c696551921c615f2439e956d240ec1d`, not an audit of only its last
commit. The complete `d1d19145..390edf4f` Cycle 3 diff and its interactions
with the complete `0bc4d95d..d1d19145` prior repair were inspected.

The recommendation is **amend; do not integrate this candidate**. There are
no P0 findings, three P1 findings, and no new P2 findings. The candidate fixes
the publication-order and iteration-lifecycle faults, literal enabled-zero
handling, selected Beammap bookkeeping count, duplicate compatible FITS plane,
v2 aggregate digest, the approved local mixed-validity classification, and
Mapdiag regression coverage. It does not yet implement the exact package
contract for all output shapes:

1. ECSV and NetCDF compact joins omit the required `missingness` field, and
   both the C++ and Python validators accept that omission.
2. Successor coadds are counted as empirical package products even when their
   approved output policy publishes no empirical companion; when an empirical
   weight is used, the actual scaled-coefficient-only join is rejected by the
   package validator.
3. Split detector-group Beammap files can contain repeated per-detector
   empirical bundles and restarted realization scopes in one per-array file,
   while the package validator permits only one bundle and one scope sequence
   per file. Split files for arrays with no selected detector are also admitted
   even though they contain only a primary HDU.

All standard gates pass. The three failures are independently reproduced by
deterministic negative fixtures and by direct source-to-writer-to-validator
trace; passing existing tests is therefore not sufficient for closure.

| Audit axis | Cycle 3 recommendation |
| --- | --- |
| contract | approved; frozen estimator and decision authority remain coherent |
| implementation | nonconformant in three package-identity/join cases |
| validation | incomplete/adverse; standard gates pass but required negative production-shape fixtures fail |
| production | `existing_use_only`; no new scientific or operational readiness claim |
| verdict | `amend`; bounded successor repair and fresh independent re-audit required |

No owner or coordinator scientific decision is needed. D001–D008 remain
settled. F005 remains `open_conditioned` with parity status exactly
`scope_blocked_not_applicable_pending_FLT`. F006 remains `open`,
`held_external`, and SCI-FRUIT-001-owned.

## Exact target, ancestry, and independence

The dedicated branch is `codex/reaudit-sci-noi-002-cycle3`. Its entry state
was clean and exactly the candidate. Application/audit/coordination histories
were not merged or rebased into one another.

| Role | Commit | Tree | Exact parent |
| --- | --- | --- | --- |
| Cycle 2 repair target / Cycle 3 base | `d1d19145df574571a894772fdc9410c86cba1041` | `826f601e2c3447765e5d1b25285ac365ee3fd120` | `0bc4d95d6bb2117442d0ccdb79c57e42e0b79989` |
| Cycle 3B1 | `de18f061000255f7f042393ba4f68be6e1f211ee` | `47bc04cd8c64698d510b0cc3f74a69892eb25304` | `d1d19145df574571a894772fdc9410c86cba1041` |
| Cycle 3B2 | `63efd8b08a599d2d56a1716e3cbb2d3686d62b9f` | `e70a37c2a6b0f010034abc13367b7b6b01d3dab4` | `de18f061000255f7f042393ba4f68be6e1f211ee` |
| final Cycle 3 candidate | `390edf4f8c696551921c615f2439e956d240ec1d` | `a82cdad542494c261d9095105813e157436766c8` | `63efd8b08a599d2d56a1716e3cbb2d3686d62b9f` |

`git rev-list --count d1d19145..390edf4f` returned three, and the merge base
of the endpoints is exactly `d1d19145df574571a894772fdc9410c86cba1041`.
The locally verified remote-tracking identities were:

- `origin/codex/repair-sci-noi-002` =
  `390edf4f8c696551921c615f2439e956d240ec1d`;
- `origin/codex/reaudit-sci-noi-002-cycle2` =
  `00d39c0499d947c9bcc926b6f2f133e7cdbcbaba`.

Cycle 2 independent authority is the separate documentation-only commit
`00d39c0499d947c9bcc926b6f2f133e7cdbcbaba`, tree
`53c494fa8de242412d3e8a1126a2a97633751c1b`, directly based on
`d1d19145df574571a894772fdc9410c86cba1041`. Its frozen artifacts were
rehashed exactly:

| Cycle 2 independent artifact | SHA-256 |
| --- | --- |
| `doc/audits/packages/SCI-NOI-002_CYCLE2_INDEPENDENT_REAUDIT_2026-08-06.md` | `c689ac09f90c904909d8bd77c4b95c454ea2875134c40a87dc8f1c8db8dbb7a6` |
| `doc/audits/results/SCI-NOI-002_CYCLE2_REAUDIT_RESULT_2026-08-06.yaml` | `f67885ea38dacf6ab8e370de85aee2277a03a440b492d95a9e59fc6a6bcda93f` |
| `doc/audits/proposals/SCI-NOI-002_CYCLE2_REAUDIT_LEDGER_UPDATE_PROPOSAL_2026-08-06.yaml` | `369a5b1dae42531a26a9a928d77c3ca4fd470579b7d4fa7f2b17860f13d533fb` |

The earlier frozen coordination authority is exact commit
`287840715c8e4ae778ce57b7166de86e7b7dfa9c`, tree
`d56ddbd4583a7cf9dd7ee5a27095db469b632cd4`, parent
`455862c7ddb45265583e99f238d21c9528b0835b`. The following digests were
recomputed over exact Git-object bytes:

| Frozen authority/evidence | SHA-256 |
| --- | --- |
| Cycle 2 repair prompt | `42ba64a459f8496d4f167cc9fd32a0634023b37252249becd850b64b626e889d` |
| Cycle 2 authority manifest | `36602cd0ba779a3fe9d9419c3b0ed53d86e26eb4c887ec7b5d19c0442ad86202` |
| Cycle 2 owner decision and repair handoff | `3b8587564418632bcf9cddd417ebb30d08374127fd75420179c7b6050187559d` |
| Cycle 2 dispatch readiness | `1ef66a8436d14d95dfb0fcf369706e743f7a3bf3248a72b4849cebed7e623dfa` |
| first repair prompt | `45fc19d6ccf0e55aa1c2a1189d97f72ffb9b51027c659580d7cbcc9415c4bc71` |
| first repair authority manifest | `6f4b84995c8cb118bdb182b9189c4b95464c4fe7e414b762debb8786e825ce79` |
| first repair dispatch readiness | `246c751397b2d138939372c7943e8187ba36658bbb7a5b8f96bb01edca0f4804` |
| owner decision brief | `3520172cfc11e8e34f280f9ebdf147ea414c7a3a4ca6109bad55354a5ff3cf71` |
| original exact-application final audit | `2874ffe950aed769f73277ed8f60ecab8860692d24e7c541f05a47a041a8a40d` |
| frozen independent mathematical core | `36781b766a2f57c9a3bd7e173ee8f1d85cba7f3d08afe2e67a403166f6b6d72d` |
| original ledger proposal | `5574d8e34fcfba8f4709d5848e79732ff3557a9817be25604375b9f3d4ec278d` |
| prior independent re-audit report | `a66ef3f17976a7149ef04d2fec08e2c1faa2947c4b883fddd651a3eb57e44517` |
| prior re-audit proposal | `6f9410fc2015bccba49001e337b063660387edd39f1879c0d5de7100dd1c970e` |
| SCI-NOI-001 R3 bounded ensemble evidence | `27263ab3bf29ac8f098463455e540f13e783241a688ef2bc5cb15b1f2a4319da` |
| original audit prompt | `74cd002580f1bd92eb6c5030eb8a9fdf19711db7fb6f9359f26ac21245a12220` |
| original audit dispatch readiness | `f3bb7c9429388388e731a0732781d316bd6648cc70511391a33f6c3ec27e48f6` |
| SCI-NOI-002 inbox manifest | `1de9dbc5f9aca42f5a9f9b1f05b7b14c092ee9b443946d0c52eeb27c8da117b0` |
| SCI-NOI-002-XAUD-001 | `dfcd59e9d59395ba84f7dfed1656690daae694872c2a1a40bf4f5c79f6abed3a` |
| SCI-NOI-002-XAUD-002 | `9eb6c778409d344ce73387f44ac4a5429a89d43b8e768c19bf3da1ed6967c1e5` |

The independent-core commit is
`f08a6da2ceebff03f498386f374980d13c5146a6`, directly based on application
commit `d5015fe716971bf8ea617e8a187311bf5af05185`. The original final-audit
commit is `4f1fec36f7802f3b5e8ac067377679946930983c`; the first independent
re-audit application commit is
`e4cdf7b3ac42f536497a3249c0499d6e2de2f8c1`, directly based on first repair
`0bc4d95d6bb2117442d0ccdb79c57e42e0b79989`.

The manifest-listed outgoing proposals were verified but not treated as
returned evidence: FLT-001 XAUD-003
`47b5533ba88e3dfac19c5beda2e92ff84d149bf3d50a41ec3740fa1b9615d9d7`,
FLT-002 XAUD-001
`655054b9253bfb0023f3598c2606ab15ea4d29d404a3d5ef4345c26c078c1249`,
SRC-001 XAUD-002
`258325b407757dcc716ec4d53db300e56bb896a93f993dc500f1080050399544`,
MODE-001 XAUD-003
`6b12c60277f9dd408a21f1b31e3470aaebc84ed746a39b105ceb9669224e11b8`,
FRUIT-001 XAUD-002
`000583fe2ab82adb926b9c8c9d9f78829eb88c1d980a1208bdc3e3c45a9a7fcd`,
and BEAM-001 XAUD-003
`ebaf770d7269474f0709b20c6faceb357a67ae0e4668002679ad47d7292cb312`.

### Frozen-artifact provenance corrections

The independently verified SCI-NOI-002-XAUD-001 digest is the 64-hex value
ending in `...bed3a` shown above. The frozen Cycle 2 report and result contain
a 63-hex transcription ending in `...bed3`. Those frozen artifacts were not
rewritten. Their own file digests remain the frozen values in the Cycle 2
table. The source XAUD object and the canonical inbox manifest already contain
the correct digest, so no separate competing handoff is required. The
companion Cycle 3 ledger-update proposal narrowly records the correction for
future references.

There was no separate pre-repair Cycle 3 Git authority manifest. The
coordinator's continuation directive for this re-audit explicitly confirms
that all 13 cumulative Cycle 3 paths were authorized through the staged Cycle
3 checkpoint, 3A, 3B1, and 3B2 directives under the older manifest's
direct-reference expansion rule. That confirmation is procedural scope
authority only; it is not correctness or closure evidence. The added
`reduction_iteration_setup.h` scope was limited to moving/resetting NOI
publication after existing iteration output-layout creation and before buffer
preparation. This missing staged manifest is process defect
`SCI-NOI-002-C3RA-PROC-001`, not an application-code finding. Future cycles
should freeze staged Git authority before implementation.

## Independently bound contract

The estimator remains the empirically centered second moment of the exact
completed `source_imprinted_current` stack. For completed stack size (R),

\[
\bar{Y}_p = \frac{1}{R}\sum_{r=1}^{R}Y_{r,p},\qquad
V_p = \frac{1}{R}\sum_{r=1}^{R}(Y_{r,p}-\bar{Y}_p)^2.
\]

This is conditional finite-stack scatter. It is not iid-unbiased sample
variance, physical-noise variance or covariance, inverse-variance precision,
calibrated significance, aperture uncertainty, a production calibration, or
proof that the configured realization count is scientifically adequate.

The binding repair decisions are:

- **RA-B001:** explicit current-iteration successfully published membership;
  in-root, lexical, regular, all-component non-symlink identity; unique
  lexical and resolved paths; full-file hashes; exact FITS/ECSV/NetCDF joins;
  injective canonical aggregate digest; atomic final completion.
- **RA-B002:** disabled/effective-no-work packages retain the request and have
  available-zero realized state with no stack-derived FITS/NetCDF members;
  non-stack source ECSV remains distinguishable and allowed. Literal requested
  `enabled=true, n_noise_maps=0` is invalid regardless of mapmaking state.
- **RA-B003:** plan expectations and observed successful-publication counters
  are separate and iteration-scoped. Partial or failed work cannot be
  complete. No persistent per-sample identity/sign ledger is authorized.
- **RA-B004:** only the approved local mixed validity classification and
  fail-closed NaN behavior are in Cycle 3 scope. No response/support invention
  and no FLT mathematical change are authorized.
- **RA-R001:** one stored compatible coefficient-standardized physical plane;
  no `sig2noise_pixel_*` physical duplicate; canonical identity and joins must
  remain truthful without changing the calculation or in-memory field.
- **RA-R002:** Mapdiag calculations, compatible stored variable names, value
  bindings, order, and restrictions remain exact. Only truthful descriptions
  and regression strength may change.

Owner decisions D001–D008 remain settled and are not reopened by this audit:

| Decision | Bound disposition retained |
| --- | --- |
| D001 | retain `1/R` only as conditional finite-stack scatter for the completed source-imprinted stack; no physical-noise or significance claim |
| D002 | retain the global scale only as an `existing_use_only` nonprecision engineering diagnostic; invalid support is unavailable |
| D003 | authoritative provenance is package-level with exact compact product joins; no redundant full metadata, dense covariance, or per-sample ledger |
| D004 | distinct S/N-like identities and restricted legacy aliases; no calibrated-significance claim |
| D005 | filtered/aperture uncertainty remains conditioned on exact operator/response/support authority and FLT review |
| D006 | FRUIT remains `existing_use_only`, external to NOI, with no algorithm, threshold, or default change |
| D007 | requested/effective/completed counts remain distinct; no count/default recommendation or silent requested-as-completed claim |
| D008 | proportional deterministic fixtures are the current gate; astronomical evidence remains separately authorized and is not claimed here |

For original findings, allowed disposition vocabulary is `closed`, `open`, or
`open_conditioned`; `held_external` is a manifest state attached to an open
finding, not a synonym for closed. For Cycle 2 repair findings, this report
uses `closed` only when the exact closure criterion is met, `not_closed` when
it is not, and `local_repair_pass_finding_open_conditioned` when a bounded
subrepair passes but the authority explicitly keeps the original finding
open. Decision status uses the same distinction and never promotes
`existing_use_only` to validation or production completion.

## Complete diff and path accounting

`git diff --check` passes for both `d1d19145..390edf4f` and
`0bc4d95d..390edf4f`. The cumulative Cycle 3 diff modifies 13 existing paths,
adds 2,668 lines, deletes 384 lines, and adds/deletes/renames no file. Every
hunk was inspected.

| Finding scope | Authorized cumulative Cycle 3 paths (additions/deletions) |
| --- | --- |
| P1-001/P1-002 publication order and iteration lifecycle | `include/citlali/core/cli/reduction_execution.h` (0/3); `include/citlali/core/pipeline/reduction_iteration_setup.h` (9/0); `include/citlali/core/pipeline/noise_execution_plan.h` (47/3); `include/citlali/core/pipeline/noise_provenance.h` (567/70); lifecycle coverage in `tests/test_config_scaffold.cpp` and auditor files |
| P1-003/P1-004 and P2-002 membership, joins, reconciliation, digest | `include/citlali/core/pipeline/noise_provenance.h`; `tools/baseline/audit_reduction_run.py` (432/39); `tools/baseline/test_audit_reduction_run.py` (562/39); `tests/test_config_scaffold.cpp` (899/143); `validation/product_contracts.json` (17/35) |
| P1-005 bounded filtered validity | `include/citlali/core/pipeline/map_image_output_helpers.h` (9/16); `tests/test_science_map_fits_products.cpp` (84/14) |
| P1-006 selected split Beammap count | `include/citlali/core/engine/detail/beammap_map_product_writers_impl.h` (14/10); `include/citlali/core/pipeline/noise_execution_plan.h`; `tests/test_config_scaffold.cpp` |
| P1-007 literal enabled-zero | `include/citlali/core/pipeline/noise_execution_plan.h`; C++ and Python tests |
| P2-001 one compatible physical plane | `include/citlali/core/pipeline/map_image_output_helpers.h`; `tests/test_science_map_fits_products.cpp`; `validation/product_contracts.json` |
| P2-003 / RA-R002 exact Mapdiag regression | `include/citlali/core/engine/detail/mapdiag_output_impl.h` (7/3); `include/citlali/core/pipeline/mapdiag_netcdf_map_double_values.h` (21/9); `tests/test_config_scaffold.cpp` |

Nine paths in the complete prior repair were not re-edited in Cycle 3 and were
still inspected at the candidate because their interactions are required:

- writer/member hooks:
  `include/citlali/core/engine/detail/lali_output_impl.h`,
  `map_filter_execution_impl.h`, `pointing_output_impl.h`, and
  `source_table_output_impl.h`;
- output identity and compatibility:
  `include/citlali/core/pipeline/fits_image_hdu_names_wcs.h`,
  `fits_image_metadata_keys.h`, and `fits_image_units_kernels.h`;
- request serialization:
  `include/citlali/core/pipeline/noise_config_serialization.h`;
- the prior bounded fail-closed numerical change:
  `src/citlali/core/mapmaking/map.cpp`.

The complete `0bc4d95d..d1d19145` prior repair is 21 paths, 2,380 additions,
and 547 deletions. The combined `0bc4d95d..390edf4f` state is 21 paths, 4,707
additions, and 590 deletions; the union of per-commit touched paths is 22.

No unauthorized application path was found. The 13 paths are coordinator-
authorized direct-reference scope, while correctness remains independently
assessed here.

## Scoped repairs that meet their criteria

### Publication lifecycle and multi-iteration FRUIT interaction

`begin_reduction_iteration` now calls `prepare_iteration_output_layout_if_needed`
before `begin_noise_product_publication` and before observation-buffer
preparation (`reduction_iteration_setup.h:33-42`). Each iteration resets
expected and realized state, all counters, completion basis, and explicit
members (`noise_execution_plan.h:222-246`). It invalidates only the current
iteration root's complete, temporary, and pending authorities
(`noise_provenance.h:270-295`).

The final CLI completion uses the final iteration's mapmaking/noise plans and
the current final iteration root (`reduction_execution.h:288-299`). Member
validation occurs before a pending sidecar is written, then members are
revalidated before the pending file is atomically renamed; failure removes
pending and temporary-pending artifacts (`noise_provenance.h:1038-1132`).
Saved prior FRUIT iteration roots are not invalidated by beginning a new saved
iteration. Reused unsaved roots correctly lose stale authority at the next
iteration boundary. These facts close Cycle 2 P1-001 and P1-002 without
changing FRUIT recurrence, activation, feedback, or convergence algorithms.

### Disabled packages and literal enabled-zero

Both C++ and Python reject the literal request `enabled=true` with
`n_noise_maps <= 0` before effective mapmaking suppression. Disabled plans
initialize every observed counter to available zero, and final semantics reject
any stack-derived FITS/NetCDF member while allowing the explicitly non-stack
source ECSV. This closes P1-007 and RA-B002.

### P1-005 bounded local change

Cycle 3 changes only the mixed case where supported scatter values exist but
none is finite: it now returns `scatter_unavailable_or_nonfinite` and remains
fail-closed, while empty support remains `support_invalid`
(`map_image_output_helpers.h:576-600`). The existing upstream NaN path remains
in `map.cpp`; no response is derived, no support authority is invented, and no
FLT operator or filter mathematics changes. Therefore the local requested
repair passes, while F005 and RA-B004 remain conditioned exactly as directed.

### One physical plane, digest v2, selected count, and Mapdiag

- `sig2noise_*` is the sole compatible stored coefficient-standardized
  physical plane. The `sig2noise_pixel_*` HDU write was removed; the in-memory
  `sig2noise_pixel` field and its calculation are unchanged.
- C++ and Python both sort the member inventory canonically and hash the same
  length-prefixed v2 preimage. The adversarial newline golden vector is
  `sha256:9fa4aab8f2b41bb83a412019cf8ac158dbf332905cdd0ee711075158ea00863e`.
  Noncanonical order and duplicate lexical/resolved identities are rejected.
- Split Beammap observed bookkeeping uses the number of selected maps. Zero
  selection retains the old standard-writer fallback. Flag selection,
  deduplication/order, map values, and numerical work are unchanged. This
  closes the narrow Cycle 2 P1-006 count defect, although the new split-file
  package finding below prevents RA-B001/B003 closure.
- Mapdiag's 43 stored value bindings and compatible names remain unchanged.
  The new exact sentinel fixture detects swapped/duplicate bindings and checks
  order, comments, joins, and restrictions. Only compact comment construction
  and conditional member admission changed.

## New findings

### P1-001 — compact ECSV and NetCDF joins omit required missingness

The binding compact join includes package/provenance/identity/version,
semantic digest, digest kind, missingness, scope, validity, and restriction.
FITS implements `NOIMISS` and validates it. ECSV production metadata in
`map_source_table_output.h:103-125`, the C++ expected map in
`noise_provenance.h:587-606`, and the Python expected map in
`audit_reduction_run.py:1532-1558` omit `missingness`. NetCDF comment
production in `mapdiag_netcdf_map_double_values.h:52-70`, C++ parsing in
`noise_provenance.h:643-653`, and Python `NOISE_NETCDF_JOIN_KEYS` at
`audit_reduction_run.py:1560-1571` also omit it.

The validators enforce exact equality to their incomplete schemas, so they
actively accept the omission. The deterministic fixture printed:

```text
ECSV missingness present: False validator accepted: True
NetCDF missingness present: False validator accepted: True
```

This is not a cosmetic omission: a detached compact product does not carry its
missing/nonfinite interpretation. It keeps Cycle 2 P1-004 and RA-B001 open.

### P1-002 — successor coadd identity/cardinality accounting contradicts output policy

The approved product registry forbids empirical companion HDUs in successor
coadds. Production implements that policy with
`publish_empirical_product_companions = expected && !coadd_product`
(`map_image_output_helpers.h:418-432`). Nevertheless:

- `record_noise_map_output_stage` counts the coadd buffer's
  `noise_variance.size()` whenever products are enabled
  (`noise_execution_plan.h:388-476`);
- `expected_noise_counts` includes each coadd output stage in
  `empirical_product_map_count` (`noise_execution_plan.h:479-562`);
- when empirical weights are not applied,
  `noise_data_fits_have_package_join` correctly declines to admit the coadd
  data file, so the package has fewer empirical FITS products than its observed
  count;
- when empirical weights are applied, the ordinary coadd weight carries only
  `global_nonprecision_scaled_coefficient`, while
  `validate_noise_fits_joins` rejects every empirical member that does not
  contain exactly one formal coefficient plus one conditional scatter
  (`noise_provenance.h:445-477`).

The direct fixtures produced:

```text
Scaled-only coadd bundle: incomplete or mixed empirical FITS noise-product bundle
Published count without empirical FITS: noise package empirical FITS inventory does not match observed empirical product maps
```

Thus both valid coadd policy branches can fail final package publication.
Existing tests exercise coadd writes and counter arithmetic separately, not
the actual coadd writer shape through final package reconciliation. This keeps
Cycle 2 P1-003, RA-B001, and RA-B003 open.

### P1-003 — split detector-group Beammap files cannot satisfy package identity rules

Split output creates one output vector from every base per-array path, then
writes every selected detector map into the path selected by
`arrays_to_maps(i)` (`beammap_map_product_writers_impl.h:179-211` and
`fits_image_write_slots.h:12-22`). Therefore one per-array FITS file can
contain multiple detector map bundles. Every bundle repeats the formal and
scatter canonical identities with the same `raw_map_pixel` scope, and every
detector's noise bundle restarts realization scopes at
`realization_map_index_0`.

The validator instead rejects a second non-realization identity and duplicate
realization scope, and assigns empirical cardinality one per file rather than
one per selected map (`noise_provenance.h:354-477`). A split vector also
contains per-array files for arrays with no selected detector; primary headers
are created for all of them, every path is recorded, and the validator rejects
the empty file as having no package join. The deterministic repeated-bundle
fixture produced:

```text
Repeated Beammap bundle: duplicate non-realization FITS noise-product identity formal_nonprecision_coefficient_snapshot
```

The selected-map counter itself is correct, and zero-selection fallback is
preserved; this new failure is the incompatibility between actual selected
multi-map file layout and package identity/cardinality semantics. It keeps
Cycle 2 P1-003/P1-004 and RA-B001/RA-B003 open.

## Finding and decision dispositions

### Cycle 2 repair findings

| Finding | Disposition | Reason |
| --- | --- | --- |
| P1-001 publication begins before layout | `closed` | begin occurs after the current iteration layout and before buffers |
| P1-002 session-scoped publication | `closed` | members, counters, invalidation, and final authority are reset/bound to the current/final iteration |
| P1-003 package product reconciliation | `not_closed` | successor coadd and actual split Beammap shapes do not reconcile |
| P1-004 exact joins/member identity | `not_closed` | path/symlink/ordering/digest portions pass; ECSV/NetCDF missingness and split bundle identity do not |
| P1-005 filtered response/support validity | `local_repair_pass_finding_open_conditioned` | approved mixed classification passes; actual response/support remains external to this repair |
| P1-006 split Beammap selected count | `closed` | selected count and zero fallback are exact; adjacent package-shape defect is new P1-003 |
| P1-007 literal enabled-zero | `closed` | rejected before effective suppression in C++ and Python |
| P2-001 duplicate compatible plane | `closed` | only `sig2noise_*` remains physically stored |
| P2-002 aggregate digest/order | `closed` | injective length prefix, canonical order, uniqueness, and parity pass |
| P2-003 Mapdiag exact fixture | `closed` | names/calculations/bindings unchanged and exact regression is complete |

### Repair decisions

| Decision | Disposition |
| --- | --- |
| RA-B001 | `not_closed`: exact compact joins and actual coadd/Beammap package membership still fail |
| RA-B002 | `closed`: disabled/no-work semantics, allowed non-stack ECSV distinction, and literal enabled-zero are exact |
| RA-B003 | `not_closed`: iteration lifecycle passes, but coadd/Beammap observed identity/cardinality reconciliation does not |
| RA-B004 | `local_repair_pass_finding_open_conditioned`: no response/support/FLT invention; F005 remains conditioned |
| RA-R001 | `closed`: one compatible stored physical plane with unchanged calculation/in-memory authority |
| RA-R002 | `closed`: exact stored names, values, comments, joins, order, and restrictions are regression-covered |

### Original findings

| Finding | Disposition |
| --- | --- |
| F001 | `closed`; closure remains supported |
| F002 | `closed`; closure remains supported |
| F003 | `open`; residual package identity/cardinality defects remain |
| F004 | `open_conditioned`; compact metadata and split-map identity remain incomplete |
| F005 | `open_conditioned`; parity status exactly `scope_blocked_not_applicable_pending_FLT`, owner SCI-FLT-001 |
| F006 | `open`, manifest state `held_external`, owner SCI-FRUIT-001; no FRUIT algorithm/configuration decision changed |
| F007 | `open_conditioned`; coadd and Beammap realized publication truth remains incomplete |
| F008 | `closed`; closure remains supported |

## Numerical, compatibility, storage, performance, and architecture assessment

No estimator equation, normalization, realization generation, sign assignment,
mapmaking calculation, filtered operator, coadd calculation, selection rule,
default, or realization-count choice changed in Cycle 3. The prior repair's
authorized `R < 2` fail-closed NaN behavior remains. The only Cycle 3
scientific-value effect is the approved mixed filtered-scatter classification
and NaN preservation. No response/support authority was created.

No FLT, SRC, FRUIT, MODE, MAP, JINC, RTC, or PTC algorithm changed. No dense
covariance, persistent per-sample identity/sign stream, auxiliary `r/I/Q/phase`
substitution, or primary-x semantic replacement was added. Mapdiag values are
unchanged. Beammap selected maps and their order are unchanged. The only
approved FITS layout change is removal of the duplicate `sig2noise_pixel_*`
physical HDU; one full plane per map is saved.

Publication validation performs two full member-validation/hash passes around
pending publication. That is linear in package bytes and outside estimator hot
loops. It is a deliberate integrity cost, but no large-production performance
measurement was run. The unresolved coadd and Beammap cases can make an
otherwise successful reduction fail at required final provenance publication;
that is an operational correctness consequence, not a numerical-selection
change.

The changes preserve requested/effective/realized separation and do not add
public cross-cutting `Engine` state. The new iteration hook stays at the
existing layout/lifecycle boundary. No architecture scope creep was found.

## Deterministic local verification

All commands used `/Users/gwilson/tolteca/bin/python` where Python was needed.
No Unity host or external service was contacted.

The candidate was configured in a fresh local `build/` directory using the
repository's known Homebrew dependencies and previously cached FetchContent
source trees. The system's unversioned Eigen was 5.0.1, so the known Eigen 3
installation was selected explicitly. Newer CMake required the non-source
compatibility option `-DCMAKE_POLICY_VERSION_MINIMUM=3.5`. The effective final
configure commands were:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCITLALI_BUILD_TESTS=ON \
  -DBUILD_TESTING=ON \
  -DEigen3_DIR=/opt/homebrew/opt/eigen@3/share/eigen3/cmake \
  '-DCMAKE_PREFIX_PATH=/opt/homebrew;/opt/homebrew/opt/eigen@3;/opt/homebrew/opt/libomp;/opt/homebrew/opt/netcdf;/opt/homebrew/opt/hdf5;/opt/homebrew/opt/ccfits;/opt/homebrew/opt/cfitsio;/opt/homebrew/opt/fftw;/opt/homebrew/opt/boost'
cmake -S . -B build -DCMAKE_POLICY_VERSION_MINIMUM=3.5
```

The first command reached cached `glog` and stopped on the new CMake minimum-
policy rule; the second completed configuration. Earlier exploratory defaults
also exposed, then resolved, the unavailable Conan helper and Eigen 5
incompatibility. These were local toolchain setup conditions, not candidate
test failures. Cached dependency headers reported Tula `f30f81d-dirty` and
KIDs `04088da-dirty`; the Citlali binary identity was exactly
`v4.0.0-3638-g390edf4f8`.

| Exact gate | Result |
| --- | --- |
| `git diff --check d1d19145..390edf4f` | pass |
| `git diff --check 0bc4d95d..390edf4f` | pass |
| `cmake --build build --target citlali_cli citlali_test citlali_science_map_fits_products_test citlali_safety_test -j 8` | pass, 324/324 build steps |
| focused core filter `config_scaffold.*noise*:*mapdiag*:*beammap*:pipeline_execution.*noise*:*iteration*` | 59/59 pass |
| full `citlali_science_map_fits_products_test` | 30/30 pass |
| `ctest --test-dir build --output-on-failure` | 620 registered; 619/619 runnable pass; one existing disabled `MapFitterLifecycle.ExactProductSequence` |
| `tools/baseline/test_audit_reduction_run.py` | 84/84 pass |
| product-contract/reduction/science-ledger validator unit suites | 39/39 pass |
| `tools/config/run_config_preflight.py --require-all` | 127/127 unit checks; 4 mode kits, 8 compatibility cases, 100% compact coverage, all boundaries drift-free |
| `validation/product_contracts.json` parse | pass; 10 contracts |
| validation-ledger validator | pass; 60 records |
| science-change-ledger validator | pass; 3 changes and 5 integration commits |
| C++/Python digest golden vector | pass; exact v2 digest shown above |
| deterministic compact-join/coadd/Beammap negative fixture | adverse as expected; all three P1 findings reproduced |

The negative fixture was an inline Python invocation of the active auditor's
fixture writers and `ecsv_noise_member_joins`, `netcdf_noise_member_joins`,
`fits_noise_member_joins`, and `noise_package_integrity_errors`. It checked
field presence before validation, repeated a complete empirical FITS bundle,
constructed the actual scaled-coefficient-only coadd join, and set a nonzero
empirical observed count without an empirical FITS member. Its exact result
was:

```text
ECSV missingness present: False validator accepted: True
NetCDF missingness present: False validator accepted: True
Repeated Beammap bundle: duplicate non-realization FITS noise-product identity formal_nonprecision_coefficient_snapshot
Scaled-only coadd bundle: incomplete or mixed empirical FITS noise-product bundle
Published count without empirical FITS: noise package empirical FITS inventory does not match observed empirical product maps
```

## Limitations

No Unity build, Unity query, astronomical reduction, or astronomical
validation was performed or claimed. No production-size performance or I/O
benchmark was run. The build reused locally cached third-party source trees;
the exact Citlali candidate was clean and identified in the binary, but this is
not a pristine dependency-supply-chain attestation. The application defects
above are deterministic format/lifecycle contradictions and do not depend on
astronomical data to reproduce.

The audit did not modify application code, tests, configuration, the canonical
ledger, or canonical handoff registry. It did not repair findings, launch a
repair, integrate, push, contact Unity, or request external evidence.

## Closure criteria and recommendation

A bounded successor repair should start from exact candidate
`390edf4f8c696551921c615f2439e956d240ec1d` and do only the following:

1. add and exactly validate compact `missingness` in ECSV and NetCDF with
   C++/Python parity;
2. define mode-aware exact expected/observed identity and cardinality for
   successor coadd output policy, including the no-companion and
   scaled-coefficient-only cases;
3. reconcile actual split detector-group Beammap per-array multi-map bundles,
   realization scopes, selected-map cardinality, and unused split-file
   admission without changing selection, filenames, or numerical outputs;
4. add deterministic production-writer-to-final-package fixtures for both
   coadd branches, multi-detector/single-array Beammap, selected arrays with an
   unused base slot, and missingness omission; then rerun every gate above.

Closure additionally requires no regression in the already passing lifecycle,
disabled/no-work, literal-zero, one-plane, v2 digest, bounded P1-005, selected-
count, or Mapdiag contracts. F005 must remain `open_conditioned` with exact
parity status `scope_blocked_not_applicable_pending_FLT`; F006 must remain
`open`/`held_external`/SCI-FRUIT-001-owned. No Unity or astronomical claim is
needed for these deterministic repair criteria, but the production status
must remain `existing_use_only` unless separately authorized evidence changes
it.

The companion result and ledger/provenance-correction proposal are
documentation-only. Stop for coordinator review; do not integrate, push,
launch repair, or update the canonical ledger from this audit task.
