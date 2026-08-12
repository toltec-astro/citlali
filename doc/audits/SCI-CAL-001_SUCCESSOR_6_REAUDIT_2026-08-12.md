# SCI-CAL-001 successor-6 independent re-audit

Date: 2026-08-12

Auditor role: fresh, role-separated independent technical auditor

Candidate ref: `origin/codex/repair-sci-cal-001-successor-6`

Candidate commit: `211e2f16f6354609de3ce6c6ee526d8aa4c6c59c`

Candidate parent: `5dfc414a13fe69e6b063608906d87e3b30491ec7`

Candidate tree: `5ed203711ad5242aafb029373b416afdd1232081`

Parent-to-candidate binary patch SHA-256:
`2ff5928b48c516e70500e6f3190366e4e05243a3ba6327e18cad3251e1858019`

Candidate changed paths: 34 modified, zero added, deleted, or renamed

Frozen coordination commit:
`2f132ae6ab01660b3ea51bd31e69e7065c25ec0d`

## Executive disposition

The candidate is **not a conforming completion of successor-6**. The bounded
F005 implementation is locally closed, and the accepted F002, F003, F004,
and F006 closures remain preserved. F007, F008, and local F009 remain open
and nonconformant:

- F007's canonical response component omits coefficient-defining sample rate
  from FIR and IIR-highpass state and omits reduced-observation identity from
  the actual-notch serialization. Different realized operators can therefore
  share a response identity. The candidate test covers only notch sample
  rate.
- F008 orders the canonical package first but does not reopen and validate it
  before dependent publication. CAL-linked FITS products are created directly
  at their final paths, and the Beammap ECSV publication path lacks the
  required in-writer reopen validation and durable atomic-replace sequence.
- F009 recomputes a self-declared hash tree but does not bind its factor basis
  to the selected-APT bytes, its requested response digest to the requested
  configuration, or its observation identity to the observation directory.
  A self-consistent detector-`flxscale` forgery with an unchanged exact sibling
  digest was accepted with zero errors. Seven pre-existing validation profiles,
  including three active profiles, were also weakened by a basename exclusion
  that suppresses nested selected-APT members.

The exact axes are:

| Axis | State | Independent basis |
|---|---|---|
| Scientific contract | `approved` | No new scientific choice was made, and the accepted F002/F003/F004/F006 boundaries remain intact. |
| Implementation | `nonconformant` | F007, F008, and local F009 retain production-path or consumer counterexamples. |
| Validation/readiness | `in_progress` | Fresh compilation and bounded focused tests pass, but the mandatory stop prevented the remaining non-waived matrix from being claimed reproducible. The historical fixture gate remains failed and owner-waived, never passed. |
| Production | `fail_closed` | No production authorization is supplied. |
| Verdict | `amend` | Return the bounded findings for coordinator and owner review; do not integrate or launch repair automatically. |

F001 and F010 remain open and conditioned because no newly authorized
external or observational evidence was supplied.

## READY checkpoint, governing clarification, and role separation

Before substantive candidate source, diff, test-body exposure, branch
creation, edits, build, or tests, the mandatory checkpoint established:

- the exact local commit, sole parent, tree, standard binary-patch digest,
  local repair ref, remote-tracking ref, and live-origin repair ref;
- a clean detached audit worktree and index at the candidate, physically
  separate from the repair worktree;
- immutable frozen coordination objects and artifact digests;
- a count-only 34-path candidate inventory and an initially unresolved
  conflict with the frozen handoff's original 33-path ceiling;
- absence of the proposed audit branch locally and at origin; and
- a two-file, documentation-only audit-artifact ceiling plus all audit-only
  prohibitions and stop conditions.

The repair branch was checked out separately at
`/Users/gwilson/.codex/worktrees/a3a3/citlali-refactor`. This audit worktree,
`/Users/gwilson/.codex/worktrees/db19/citlali-refactor`, was detached and clean
at the candidate. A fresh read-only `git ls-remote` returned the exact live
origin candidate. The standard patch command reproduced the supplied digest
without displaying patch bytes.

The frozen handoff and machine ledger each authorize 33 paths. Their exact
SHA-256 values are:

| Frozen artifact | Git blob | SHA-256 |
|---|---|---|
| Successor-6 bounded repair handoff | `8daff98432fd6fbca0609839f9b56aea0f1340ce` | `3de68392de61544349bae808661ea9b0ca31719c6388c4f4caffa055a21974de` |
| Successor-6 repair finding ledger | `f583452c1ac9755e00df0e6151255ea2bcdeb0dd` | `417f501d1961c4b6c7528c4f2e75e4fb0face22d24568fa66ecc560e8ab5074a` |
| Successor-5 owner acceptance | `12233290d471421996e7d61c2589b9c41f4552d3` | `825f67f874f0ec444f2d3250b08174ef9c1b9ceaf35e87e8a33e236af991ea5f` |
| Accepted successor-5 re-audit report | `7d409ed4dbeeb43e9ecc72d914f29b08ba493799` | `7f3a484cf5d446647313659a3d6d3103805837ecbb3a9d77034d49bb5762234a` |
| Accepted successor-5 local evidence | `4041e989e93bd6b49ab9cfe450d453eea18d9c99` | `e6d8d721c02d22683a0ca8500efcd66d1e53c00b85d9529f92bd1c7ccbc64206` |

The coordinator then supplied the project owner's later, higher-precedence
repair-time authority for exactly two additional paths:

1. `tests/test_config_scaffold.cpp`, only to replace the stale
   source-mutating fallback fixture with fail-closed/no-mutation expectations.
2. `tools/config/audit_raw_timestream_execution_reads.py`, only to classify
   `begin_reduced_observation` as a legitimate executor operation and update
   the stable census count/digest without weakening the gate.

That clarification supersedes only the obsolete 33-path ceiling. The final
ceiling is 35: the candidate changes 32 original-allowlist paths plus those two
later-authorized paths. The originally authorized
`include/citlali/core/engine/detail/beammap_setup_impl.h` was not changed.
No other path expansion or scientific, architectural, gate, audit-only, or
stop-boundary change was authorized. The canonical name-status inventory has
SHA-256
`2693bfa966c0a198b5b8276d6cd19834b87d252371164591f0ea1b248f898e5f`.

Only after coordinator approval was
`codex/reaudit-sci-cal-001-successor-6-20260812` created directly at the exact
candidate.

## Candidate scope and changed-path assessment

The candidate modifies 34 paths with 2,877 insertions and 292 deletions:

- `doc/REFACTOR_STATUS.md`
- `include/citlali/core/engine/detail/beammap_setup_metadata_impl.h`
- `include/citlali/core/engine/detail/map_phdu_output_helpers.h`
- `include/citlali/core/engine/detail/ptc_line_audit_impl.h`
- `include/citlali/core/engine/detail/tod_file_output_impl.h`
- `include/citlali/core/pipeline/calibration_product_admission.h`
- `include/citlali/core/pipeline/flxscale_correction.h`
- `include/citlali/core/pipeline/initial_observation_setup.h`
- `include/citlali/core/pipeline/raw_observation_outputs.h`
- `include/citlali/core/pipeline/raw_timestream_execution_plan.h`
- `include/citlali/core/pipeline/raw_timestream_observation_shadow.h`
- `include/citlali/core/pipeline/raw_timestream_provenance.h`
- `include/citlali/core/pipeline/raw_timestream_provenance_lifecycle.h`
- `include/citlali/core/pipeline/reduction_observation_calibration.h`
- `include/citlali/core/pipeline/reduction_observation_inputs.h`
- `include/citlali/core/pipeline/reduction_observation_pipeline.h`
- `include/citlali/core/pipeline/tod_metadata_mapmaker_tau.h`
- `include/citlali/core/timestream/calibration_product.h`
- `include/citlali/core/timestream/rtc/rtcproc.h`
- `src/citlali/core/engine/calib.cpp`
- `tests/test_calibration_product.cpp`
- `tests/test_config_scaffold.cpp`
- `tests/test_science_map_fits_products.cpp`
- `tools/baseline/audit_reduction_run.py`
- `tools/baseline/examples/sci_cal_001_raw_timestream_provenance_v4.yaml`
- `tools/baseline/examples/sci_cal_001_selected_calibration_apt.ecsv`
- `tools/baseline/test_audit_reduction_run.py`
- `tools/baseline/test_compare_reduction_products.py`
- `tools/baseline/test_validate_product_contract.py`
- `tools/baseline/test_validation_profiles.py`
- `tools/baseline/validate_product_contract.py`
- `tools/config/audit_raw_timestream_execution_reads.py`
- `validation/product_contracts.json`
- `validation/validation_profiles.json`

All 34 are modified files; no addition, deletion, or rename is hidden in the
candidate patch. `git diff --check` passes. The two later owner-authorized
paths match their bounded purposes: the C++ fixture now expects the fallback
to fail without mutating source `flxscale`, and the census tool adds only the
executor classification plus the expected 88-record digest
`1a60280fbde4749b0c753f305e98f2917bcbd4ba570e02f20468169f57218f96`.

No change was found in RTC filter coefficient arithmetic, notch detection or
selection, PTC weighting arithmetic, naive mapmaking arithmetic, map
normalization, atmosphere arithmetic, or the fixed atmosphere artifacts. The
authorized once-only observation correction changes the intended signal and
weight recipients; no unrelated numerical behavior change was found.

## Finding-by-finding disposition

### F001 — open, conditioned, unchanged

The candidate supplies no newly authorized SCI-ALIGN, SCI-AST, exact-SHA
Unity, astronomical-standard, or observational evidence. F001 remains open
and conditioned without reinterpretation.

### F002 — bounded closure preserved

The accepted fixed-DJF25 artifacts are byte-identical to the parent:

| Artifact | SHA-256 |
|---|---|
| Fixed operator contract | `7a064ff768a3de4f427f1338d94ef6cb9026d248f3c3c816fc3dfc96d156e36a` |
| Operator node table | `fd688a4cd3f46585b08631bc63a562aed482feb9b24ec9ee0071b70db7eb8a5f` |
| Generated node header | `d322bdc863ccb1292325c739865f772ef53f4e9f4101967752027ea0a2413262` |
| Production atmosphere operator | `3fd4352d05e77e07c1e354b7e4124733505064667968676d8d4e94315017d584` |

This preserves the accepted structural closure only; it makes no new
atmospheric-fidelity or uncertainty claim.

### F003 — bounded closure preserved

The typed configuration validation and reduction execution boundary are
byte-identical to the parent. Unsupported calibrated units still fail at the
startup/admission boundary, while the intentionally uncalibrated request
remains supported.

### F004 — bounded closure preserved

The APT-load/filter/lineage/join boundary is unchanged. `calib.h` and
`tests/test_calib_apt_filtering.cpp` are byte-identical to the parent. The
bounded `calib.cpp` rewrite constructs and commits observation-local flux
state without changing selected-APT filtering, ordered-row association, or
lineage ownership.

### F005 — locally closed within the authorized technical claim

The successor-5 lifecycle defects are repaired on the actual production path:

- initial setup loads and validates selected-APT metadata without applying a
  correction against an empty carrier
  (`initial_observation_setup.h:26-30`);
- reduction observation input creates observation-local all-ones flux state
  before applying the correction
  (`reduction_observation_inputs.h:26-30,96-99`);
- `Calib::calc_flux_calibration` validates every array and atomically commits
  local carrier/summary state, leaving prior state untouched on failure
  (`calib.cpp:1040-1080`);
- correction composition occurs in local temporaries and rejects missing
  initialization, duplicate application, cardinality mismatch, non-finite or
  non-positive operands, overflow, and exact-zero underflow before mutation
  (`flxscale_correction.h:35-53,77-114`);
- admission persists the exact applied scalar, `applied_once` state, source
  identity, recipient identity, and carrier equality, and revalidates the
  composed product (`calibration_product_admission.h:40-51,336-364` and
  `calibration_product.h:718-739,791-815,863-880`); and
- admission precedes TOD/map scientific product creation
  (`observation_setup_impl.h:263-281`).

The selected APT remains read-only. The fallback is an internal ones carrier,
not a write into source APT `flxscale` or sensitivity. The correction state is
serialized into raw observation/realized provenance, map PHDU, TOD metadata,
and Beammap metadata, while its canonical preimage contains scalar, state,
source, and recipient.

The unchanged runtime calibrator applies signal factor `a` once. Independent
source tracing confirms approximate precision weights scale as `1/a^2`, full
weights derive inverse variance from already calibrated samples, and hybrid
and validated modes combine branches with the same inverse-square behavior.
The unchanged naive map and map normalization paths therefore carry nonzero
signal to map `a` and `noise_variance_I` to `a^2` when the production
recipient is exercised. Constant weighting remains intentionally
dimensionless/nonprecision and is not a blocker.

The pre-stop focused product/atmosphere/config/flxscale run passed 30/30. Its
green result supports the trace but is not the basis for treating the
production lifecycle as correct.

### F006 — bounded closure preserved

Typed validation, runtime calibration setup, and complete product admission
still allow calibrated production output only as top-of-atmosphere
point-source-peak `mJy/beam`. No additional calibrated unit was introduced.

### F007 — open and nonconformant

Several lifecycle pieces are materially improved. Immutable requested YAML is
now hashed; reduced observation and fruit iteration are recorded at
observation begin; join registration uses the reduced observation; fruit
transitions clear prior join state; repeat same-state finalization is
idempotent; conflicts fail closed; homogeneous/heterogeneous joins are
checked; and TOD-only dispatch exists.

The response identity nevertheless remains incomplete:

1. `calibration_response_identity` serializes FIR frequency and term/Gibbs
   fields but not the application sample rate
   (`calibration_product_admission.h:100-104`). FIR coefficients explicitly
   depend on sample rate (`rtc/filter.h:49-97`).
2. IIR-highpass serialization similarly omits sample rate
   (`calibration_product_admission.h:155-165`), while the coefficient path
   depends on it (`rtc/filter.h:294-330`).
3. The fixed/actual notch record does bind sample rate, but the candidate's
   sample-rate test covers only that notch route
   (`test_science_map_fits_products.cpp:1487-1517`). With FIR or highpass
   active and notch/line-audit inactive, identical serialized Hz settings at
   different sample rates describe different operators with the same
   response identity.
4. `RTCAppliedResponseNotch` carries and populates
   `reduced_observation_identity` (`rtcproc.h:285-300,523-539`), but the
   canonical actual-notch serialization omits it
   (`calibration_product_admission.h:167-196`).

The overall CALID contains other components that may incidentally distinguish
some observations; that does not repair the contracted response component.
The component is required to identify the realized operator completely before
it is joined into CALID/package identity. F007 therefore remains open.

### F008 — open and nonconformant

The package pipeline does invoke canonical-package publication before
dependents (`reduction_observation_pipeline.h:32-60`), and the TOD helper is a
useful partial closure: it stages, calls NetCDF `sync`, closes, reopens,
validates CALID/PKGID, renames, and removes staged/final artifacts on failure
(`tod_file_output_impl.h:18-55`).

Three required production paths remain incomplete:

1. The canonical raw YAML returns immediately after its helper writes it;
   production does not reopen, parse, and recompute its identities before
   dependents proceed (`raw_timestream_provenance.h:602-712`). The YAML helper
   flushes, closes, and renames but supplies no durable file/directory
   synchronization (`atomic_yaml_output.h:11-29`).
2. CAL-linked FITS products are opened directly at their final path in
   overwrite mode (`fits_io.h:39-62`). CALID/PKGID are written to the PHDU
   before later HDU creation (`map_phdu_output_helpers.h:154-221`,
   `lali_output_impl.h:65-85`, and Beammap writers at
   `beammap_map_product_writers_impl.h:103-125`). A later failure can leave a
   partial, final-path artifact that already claims the calibration join.
3. Beammap ECSV stages and renames but does not reopen and validate within the
   production writer; it removes a pre-existing final before rename
   (`ecsv_io.h:55-86`). The candidate test reopens only after the writer has
   returned, so it does not make validation part of publication.

No global transaction is required or recommended by this audit. The defect is
the missing per-artifact stage/synchronize/reopen/validate/atomic-replace
discipline. F008 remains open.

### Local F009 — open and nonconformant

The candidate correctly implements several bounded structural pieces:

- exact hexadecimal float formatting, including normal/subnormal anchors;
- isolated vector, factor, CALID, component, and package recomputation;
- a new `sci-cal-001-production-candidate-2026-08-12` epoch that remains
  `preparing`, while the active epoch is unchanged;
- four preparing current-production profiles with no accepted baseline;
- exact-v4 current contract admission; and
- generic historical recognition of v1-v4, while v1-v3 and `redu66` remain
  historical/test evidence and fail the current-production v4 contract.

Historical product-contract bodies match the pre-CAL authority at
`46ad23888a40f5102cdfd50c06e49a549bdf8a20`; the candidate adds only the new
current family/contracts to that normalized historical content. Accepted-run
records were not rewritten, and no epoch/profile was promoted.

The integrity consumer still trusts unauthoritative self-declarations.

#### Self-consistent selected-APT/factor forgery

Starting from the production-shaped v4 fixture, the audit changed the first
declared detector `flxscale` from exact hex `0x1p+0` to `0x1p+1`, then used the
consumer's canonical routines to recompute the vector digest, factor-state
digest, factor component, CALID, package identity, and observation/realized
joins. The declared exact sibling digest was left unchanged. The semantic
auditor returned an empty error list:

```text
errors=[]
selected_APT_digest=8c18f6268523f7db45b18c9d6eb5eb7c99f355b46f1a0eedd5c858fa9e9430aa
forged_detector_flxscale=[0x1p+1]
FACTORID=6a479a545b9a03fa085440ea9b9104660ad7c4cafb9bc2d6a3a0d3d57830a7a9
CALID=50c9ae03504e543fae70036577c09a3420f00025e2661f9dc812b06e438c38e1
PKGID=e27a96c6242bb44d4adacc6c2d1203e27bc788babbc6ef2a5b94fedbc799e0c6
```

The exact sibling fixture contains `flxscale=1.0`, but the consumer never
parses those ECSV bytes and binds their values to the factor basis. It verifies
the file digest, then recomputes identities only from the YAML's claimed
bases (`raw_timestream_provenance.h:364-525` and
`audit_reduction_run.py:652-1104`). This is a direct tamper/forgery false
acceptance.

#### Requested-state and copied-observation ownership gaps

The producer hashes immutable requested YAML
(`calibration_product_admission.h:54-85`). The consumer instead treats the
serialized requested-state provenance as nonempty opaque text and hashes it;
the test fixture substitutes `sha256(obsnum)` rather than requested YAML
(`test_audit_reduction_run.py:455-461`). It therefore does not independently
recompute requested response state from a serialized authoritative preimage.

The path passed to the consumer is used only to derive the sibling APT path
(`audit_reduction_run.py:1277-1281`). No check binds the package observation
to `provenance_path.parent`. An internally valid observation `000042` package
can therefore be copied under `000043` with the same sibling and remain
admissible. The successor-6 multi-observation test makes distinct fixture
documents but does not add this path-ownership check.

The producer also copies a broad calibration state from observation into
realized provenance (`raw_timestream_observation_shadow.h:220-285` and
`raw_timestream_provenance_lifecycle.h:70-152`). The consumer joins only CALID
and PKGID at `audit_reduction_run.py:1430-1444`; an in-memory mismatch of
factor-state digest, response identity, raw identity, target unit, schema, and
correction state across observation/realized sections was accepted with zero
errors.

#### Pre-existing profile weakening

Relative to pre-CAL authority `46ad23888`, the candidate adds
`selected_calibration_apt.ecsv` to seven pre-existing comparison-profile
exclusion lists. Relative to its immediate parent, it broadens
`*/selected_calibration_apt.ecsv` to the basename form. The comparator matches
both full path and basename (`compare_reduction_products.py:149-150`), so the
new form suppresses every nested selected-APT package member.

Affected profiles are active phase-4 point, OOF, and Beammap; preparing
phase-5 point and OOF; and preparing SCI-MAP-001 point and OOF. This rewrites
and weakens accepted/pre-existing profiles contrary to the registry's
immutable-snapshot/successor-epoch policy and the frozen no-profile-weakening
boundary. The later two-path owner clarification did not authorize it. This is
an independent stop-level authority contradiction.

The candidate status entry consequently overstates complete independent
recomputation and attack rejection. Local F009 remains open.

### F010 — open, conditioned, unchanged

No newly authorized evidence changes F010. It remains open and conditioned.

## Historical readiness-fixture gate truth

The candidate records the special gate truth correctly in
`doc/REFACTOR_STATUS.md:438-448`: `phase5_readiness.py --verify-fixtures` is
**failed and owner-waived**, never passed. Beammap and OOF retain their
recorded states; point has six and science has twelve historical product
errors because immutable historical FITS products contain the forbidden
`sig2noise_pixel_I` extension. That waiver is limited to pre-existing
historical product drift and does not admit the extension for new/current
products.

Static candidate-versus-parent evidence confirms these governing objects are
byte-identical:

| Object | Git blob | SHA-256 |
|---|---|---|
| `validation/accepted_runs.json` | `8e5fa67ddbdab94bbf4e0770039d08f806224971` | `4a134dcdd14e0444d96875547f628a3353574cc66574dd9a559bcf59dafb94bb` |
| `validation/phase5_validation_readiness.json` | `0b592054fbf12af152a156ad3830481618d2bd06` | `b9daf6ab3973d2d35968ab27d2b7c75eca8534d2baeb6af9bb43725261f04755` |
| `tools/baseline/phase5_readiness.py` | `794379673c77024d9e7b8b4e0fe63f4d909d4764` | `3a27fa5279c75432aa0939cbcc2add2db4d30df92379f6bf511ff281202b2af7` |

No historical fixture path is changed by the candidate. Historical contract
bodies match the pre-CAL authority after removal of the newly added current
family/contracts. `sig2noise_pixel_I` remains explicitly forbidden in the
new/current product checks.

The audit did **not** rerun the write-producing fixture gate after the
non-waived stop findings. Its authoritative status is therefore recorded as
`failed_owner_waived_not_rerun`, never as passed.

## Build and gate truth

A fresh Release/test configuration was created in this audit worktree using
the repository's `BUILD_TESTS=ON make local-bootstrap` route. The first
sandboxed attempt reached dependency acquisition and failed only because DNS
was unavailable. The approved network retry completed configuration. A fresh
build of `citlali_cli`, `citlali_test`, `citlali_safety_test`,
`citlali_science_map_truth_test`, and
`citlali_science_map_fits_products_test` passed. The built CLI reports
`v4.0.0-3647-g211e2f16f`.

Built-artifact SHA-256 values are:

| Artifact | SHA-256 |
|---|---|
| `build/CMakeCache.txt` | `dbc941439cae24e5263b0f375474f76e11e52c03a3681216d7f06d16ccc73819` |
| `build/bin/citlali` | `00485a616e58d310fd01428464eb0d7076bb16ea6ea0f7b8386d1cebc464a193` |
| `build/tests/citlali_test` | `41de1953418ae623eb88186ded6d8ede284305a67885bf1ae357885752f75b0d` |
| `build/tests/citlali_safety_test` | `f2a29587e3c2714927f2ad2967718647670ce8d83ddc63267d287aeced62fd0f` |
| `build/tests/citlali_science_map_fits_products_test` | `e2da585cf558e4c2f2e90c961afb2063681e9c816e390d42095416be90f50ee7` |
| `build/tests/citlali_science_map_truth_test` | `13dacd93eacb26b0075366c48e7ee3cd7ca0900aef572af9eee6874808b05b69` |

The bounded runs completed before the stop were:

| Gate | Result | Exact truth |
|---|---|---|
| Candidate identity, patch digest, 34-path inventory, and `git diff --check` | pass | Exact supplied identities reproduced. |
| Fresh Release/test configuration and five required build targets | pass | Initial DNS-only infrastructure failure; approved retry and build passed. |
| Focused product/atmosphere/config/flxscale C++ selection | pass | 30/30; no candidate failure. |
| Focused response/join/package/TOD/Beammap FITS-product selection | pass | 14/14; these happy-path tests do not cover the reported identity and publication counterexamples. |
| TOD-only finalized-header scaffold test | pass | 1/1; scaffold dispatch is not an end-to-end atomic publication proof. |
| Historical Phase-5 fixture verification | failed, owner-waived | Six point plus twelve science errors from pre-existing historical `sig2noise_pixel_I`; not rerun by this audit and never counted as pass. |

After F007/F008/F009 produced non-waived failures and an authority
contradiction, the explicit stop rule controlled. Full CTest, full standalone
test binaries, baseline unittest/pytest suites, validation and science
ledgers, config preflight, standalone raw-execution census, ordinary Phase-5
readiness, profile listing, atmosphere generator, and session-exit audit were
not run by this audit. The candidate status claims those gates passed in its
repair worktree, but this audit does not relabel those claims as independently
reproduced. Therefore the answer to “are all non-waived gates reproducible?”
is **not established before the mandatory stop**, not “yes.”

Green candidate tests do not override the source-level and executable
consumer counterexamples above.

## Audit-only boundaries and handoff

This audit made no application, configuration, test, build-system,
validation-product, or canonical-coordination edit. It made no repair, push,
merge, Unity access or request, astronomical reduction, external contact,
downstream task, scientific redesign, architecture expansion, or production
authorization. Build output is ignored local state. The only committed paths
are this report and its local-evidence YAML.

Disposition: stop for coordinator verification and project-owner review. Do
not integrate, repair, re-audit, push, merge, or launch downstream work
automatically.
